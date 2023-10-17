"""DS (decasecond) splits for longer-form video."""
import itertools

import numpy as np
import psutil
from S3MP.mirror_path import KeySegment, get_matching_s3_mirror_paths

from ground_data_processing.data_processors.thresh_split_video import (
    split_video_at_bounds,
)
from ground_data_processing.utils.ffmpeg_utils import FFmpegFlags, FFmpegProcessManager
from ground_data_processing.utils.processing_utils import print_processing_info
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFiles,
    DataFolders,
    Framerates,
)
from ground_data_processing.utils.video_utils import get_frame_count

if __name__ == "__main__":
    print_processing_info()
    """
    This script is used to split videos into decasecond chunks.
    Mostly boilerplate.
    """
    FFmpegProcessManager.max_processes = psutil.cpu_count(logical=False)
    # if you turn off "copy" do not forget to reduce the max_processes or you will chew up all your memory
    FFmpegFlags.set_output_flag("c", "copy")
    FFmpegFlags.set_output_flag("loglevel", "quiet")

    FPS = Framerates.fps60.fps
    DS_S = 10  # decasecond
    OVERWRITE = True

    # TODO get this implemented
    # USE_OFFSETS = False

    FRAMES_PER_DS = DS_S * FPS

    segments = [
        KeySegment(0, "2023-field-data"),
        KeySegment(1, "Williamsburg_Strip_Trial"),
        KeySegment(2, "2023-07-17"),
        KeySegment(3, "row-25"),
        # KeySegment(6, "DS 000"),
        KeySegment(5, "bottom.mp4", is_file=True),
    ]

    bottom_vid_mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(bottom_vid_mps)} bottom videos to split.")
    for bottom_vid_mp in bottom_vid_mps:
        offset_json_mp = bottom_vid_mp.get_sibling(DataFiles.OFFSETS_JSON)
        if offset_json_mp.exists_on_s3():
            offset_data = offset_json_mp.load_local()
        else:
            offset_data = {camera: 0 for camera in CameraViews}

        # This gets the count of relevant frames in each video and splits into evenly-sized bounds
        split_bounds_by_video = []
        vid_mps = []
        for camera in CameraViews:
            vid_mp = offset_json_mp.get_sibling(f"{camera}.mp4")
            vid_mp.download_to_mirror(overwrite=OVERWRITE)
            vid_mps.append(vid_mp)
            n_vid_frames = get_frame_count(vid_mp.local_path)
            vid_offset = offset_data[camera]

            split_frames = list(
                np.arange(vid_offset, n_vid_frames, FRAMES_PER_DS).astype(int)
            ) + [n_vid_frames]
            split_bounds_by_video.append(list(itertools.pairwise(split_frames)))

        # Trim trailing bounds as that data will not contain all cameras thus is trivial
        min_n_bounds = min(len(bounds) for bounds in split_bounds_by_video)
        split_bounds_by_video = [
            bounds[:min_n_bounds] for bounds in split_bounds_by_video
        ]

        # With bounds trimmed, each video will have the same number of DS splits
        # Thus folder creation can be done independent of the camera
        print(f"Creating {min_n_bounds+1} DS folders...")
        root_ds_mp = vid_mp.get_sibling(DataFolders.DS_SPLITS)
        root_ds_mp.local_path.mkdir(parents=True, exist_ok=True)
        ds_output_folder_mps = [
            root_ds_mp.get_child(f"DS {i:03d}") for i in range(min_n_bounds)
        ]
        for ds_output_folder_mp in ds_output_folder_mps:
            ds_output_folder_mp.local_path.mkdir(parents=True, exist_ok=True)

        # Now that the DS folders are created, we can split the videos
        # for camera, vid_split_bounds, vid_mp in zip(CameraViews, split_bounds_by_video, vid_mps):
        for camera, vid_split_bounds, vid_mp in zip(
            CameraViews, split_bounds_by_video, vid_mps
        ):
            output_vid_mps = [
                ds_output_folder_mp.get_child(f"{camera}.mp4")
                for ds_output_folder_mp in ds_output_folder_mps
            ]
            assert len(output_vid_mps) == len(vid_split_bounds)
            split_video_at_bounds(
                vid_mp, output_vid_mps, vid_split_bounds, framerate=FPS, overwrite=True
            )

        FFmpegProcessManager.wait_for_all_processes_to_finish()

    print("Done splitting videos.")
