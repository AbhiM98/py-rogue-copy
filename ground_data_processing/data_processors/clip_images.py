"""Clip images from the ground data."""
from typing import List

import boto3
import cv2
import numpy as np
from S3MP.keys import KeySegment, replace_key_segments_at_relative_depth
from S3MP.mirror_path import MirrorPath

from ground_data_processing.utils.peak_finding_utils import binary_search_peaks
from ground_data_processing.utils.s3_constants import (
    ROGUES_BUCKET_KEY,
    DataFiles,
    DataFolders,
    Framerates,
)
from ground_data_processing.utils.video_utils import (
    get_ffmpeg_reader_trimmed,
    get_frame_count,
    get_video_ffmpeg_reader,
)


def clip_images_from_video_with_exg_data(
    vid_mp: MirrorPath, n_images, desired_tol=4, e_tol=10, overwrite=False
):
    """Clip images from a video with a binary search."""
    camera = vid_mp.local_path.stem
    npy_mp = vid_mp.get_sibling(DataFiles.EXG_SLC_20PX_NPY)

    if not npy_mp.exists_on_s3():
        print(f"{npy_mp.s3_key} does not exist on S3.")
        return

    if not overwrite:
        r80_pct_idx = int(n_images * 0.8)
        img_80_pct_key = replace_key_segments_at_relative_depth(
            vid_mp.s3_key,
            [
                KeySegment(0, f"{camera} {DataFolders.RAW_IMAGES}"),
                KeySegment(1, f"{r80_pct_idx:02d}.png"),
            ],
        )
        img_80_pct_mp = MirrorPath.from_s3_key(img_80_pct_key)
        if img_80_pct_mp.exists_on_s3():
            print(
                f"80% of images already present on S3 (image {img_80_pct_key} exists)."
            )
            return
    else:
        # Delete everything in folder
        folder_key = replace_key_segments_at_relative_depth(
            vid_mp.s3_key, [KeySegment(0, f"{camera} {DataFolders.RAW_IMAGES}")]
        )
        print(f"Deleting {folder_key}")
        s3_resource = boto3.resource("s3")
        bucket = s3_resource.Bucket(ROGUES_BUCKET_KEY)
        bucket.objects.filter(Prefix=folder_key).delete()

    npy_mp.download_to_mirror()
    vid_mp.download_to_mirror()

    npy_data = np.load(npy_mp.local_path)

    peaks = binary_search_peaks(npy_data, n_images, desired_tol, e_tol)

    vid_iter, iter_len = get_video_ffmpeg_reader(vid_mp.local_path, use_tqdm=True)
    plant_idx = 0
    save_folder_mp = vid_mp.get_sibling(f"{camera} {DataFolders.RAW_IMAGES}")
    for frame_idx, frame in enumerate(vid_iter):
        if frame_idx in peaks:
            out_mp = save_folder_mp.get_child(f"{plant_idx:02d}.png")
            out_mp.save_local(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), upload=True, overwrite=True
            )
            plant_idx += 1


def clip_frames_from_video_with_frame_indices(
    vid_mp: MirrorPath,
    frame_indices: List[int],
    output_folder_mp: MirrorPath,
    reduction: int = 1,  # how many times smaller to make image (1 = no reduction)
    overwrite: bool = True,
):
    """Clip frames from a video to the specified MirrorPath."""
    vid_mp.download_to_mirror()
    img_idx = 0

    n_frames = get_frame_count(vid_mp.local_path)
    start_frame = min(frame_indices)
    vid_iter = get_ffmpeg_reader_trimmed(
        vid_mp.local_path,
        start_frame,
        n_frames,
        fps=Framerates.fps60.fps,
        use_tqdm=False,
    )
    # print()
    # print(n_frames)
    # print(frame_indices)

    if overwrite:
        output_folder_mp.delete_all()

    output_folder_mp.local_path.mkdir(parents=True, exist_ok=True)
    for frame_idx, frame in enumerate(vid_iter):
        if frame_idx + start_frame in frame_indices:
            out_mp = output_folder_mp.get_child(f"{img_idx:03d}.png")
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # pad image to change aspect ratio
            height, width = img.shape[:2]
            if height > width:
                new_width = int(height**2 / width)
                pad_width = int((new_width - width) / 2)
                img = np.pad(
                    img, ((0, 0), (pad_width, pad_width), (0, 0)), mode="constant"
                )
                img = cv2.resize(img, (height, width))
            if reduction > 1:
                img = cv2.resize(
                    img, (int(img.shape[1] / reduction), int(img.shape[0] / reduction))
                )
            out_mp.save_local(img, upload=True, overwrite=True)
            img_idx += 1
