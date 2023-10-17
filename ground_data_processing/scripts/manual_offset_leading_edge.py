"""Detect offsets in videos."""
import itertools
import json
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from S3MP.mirror_path import KeySegment, MirrorPath, get_matching_s3_mirror_paths
from tqdm import tqdm

from ground_data_processing.measurements.frame_measurements import (
    get_normalized_frame_diff,
)
from ground_data_processing.scripts.generate_framewise_data import (
    calc_framewise_data_and_upload_json,
)
from ground_data_processing.scripts.npy_plot_sync import (
    get_leading_edge,
    get_leading_edge_mean_prop,
    moving_average,
)
from ground_data_processing.utils.multiprocessing_utils import MultiprocessingManager
from ground_data_processing.utils.plot_utils import (
    multi_image_viewer_with_nav_buttons,
    plot_1d_data_with_markers,
)
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFiles,
    Resolutions,
)
from ground_data_processing.utils.video_utils import get_frames_between_indices


def detect_offset_in_video_group(video_paths: List[str], output_json_path: str) -> None:
    """Detect offset in a group of videos."""
    video_names = [vid.split("/")[-1].split(".")[0] for vid in video_paths]
    bottom_idx = np.argwhere([CameraViews.BOTTOM == vid for vid in video_names])[0][0]

    processes = []

    MultiprocessingManager.wait_for_all_queued_processes_to_finish(processes)
    offsets = MultiprocessingManager.get_ret_vals(processes)
    print(offsets)
    bottom_offset = offsets[bottom_idx]
    offset_dict = {
        vid_name: offsets[idx] - bottom_offset
        for idx, vid_name in enumerate(video_names)
    }
    output_mp = MirrorPath.from_local_path(output_json_path)
    with open(output_mp.local_path, "w") as fd:
        json.dump(offset_dict, fd)

    output_mp.upload_from_mirror()


if __name__ == "__main__":
    IN_RES = Resolutions.r4k
    FRAMERATE = 59.94
    proc_only = True
    OVERWRITE_LEADING_FRAMES = True
    GENERATE_ONLY = proc_only
    OVERWRITE_OFFSET_JSON = True
    BUFFER = 480
    OVERWRITE_INPUT_DATA = True
    segments = [
        KeySegment(0, "2023-field-data"),
        KeySegment(1, "Waterman_Strip_Trial"),
        KeySegment(2, "2023-07-18"),
        # KeySegment(3, "row-30"),
        # KeySegment(6, "DS 000"),
        KeySegment(5, "bottom.mp4", is_file=True),
    ]

    bottom_vid_mps = get_matching_s3_mirror_paths(segments)
    processes = []

    for bottom_vid_mp in tqdm(bottom_vid_mps):
        offset_mp = bottom_vid_mp.get_sibling(DataFiles.OFFSETS_JSON)
        root_folder_mp = bottom_vid_mp.get_parent()
        if not OVERWRITE_OFFSET_JSON and offset_mp.exists_on_s3():
            print(f"Offset for {root_folder_mp.s3_key} exists on S3, skipping.")
            continue

        vid_mps = [offset_mp.get_sibling(f"{cam_view}.mp4") for cam_view in CameraViews]
        if len(vid_mps) != 3:
            print(
                f"Skipping {root_folder_mp.s3_key} because it has {len(vid_mps)} videos"
            )
            continue
        print(f"\nProcessing {root_folder_mp.s3_key}")

        [
            vid_mp.download_to_mirror(overwrite=OVERWRITE_INPUT_DATA)
            for vid_mp in vid_mps
        ]

        slice_idxs = []
        offset_data = {}
        for vid_mp in vid_mps:
            cam = vid_mp.local_path.stem
            frame_diff_mp = vid_mp.get_sibling(
                f"{cam}{DataFiles.Suffixes.NORM_FRAME_DIFF_JSON}"
            )
            result = None
            if not frame_diff_mp.exists_on_s3():
                # Generate
                print("Generating frame diff for", vid_mp.local_path)
                result = calc_framewise_data_and_upload_json(
                    vid_mp.local_path,
                    frame_diff_mp.s3_key,
                    [get_normalized_frame_diff],
                    [itertools.pairwise],
                )

            if result is not None:
                if result[1] == -1:
                    continue
            frame_diff_data = np.array(
                frame_diff_mp.load_local(download=True, overwrite=OVERWRITE_INPUT_DATA)[
                    "data"
                ]
            )
            tmp_dir = Path(
                vid_mp.local_path.as_posix().replace(
                    vid_mp.local_path.suffix, "_leading_frames/"
                )
            )
            if OVERWRITE_LEADING_FRAMES or not tmp_dir.exists():
                if tmp_dir.exists():
                    # Delete all files.
                    for file in tmp_dir.glob("*"):
                        file.unlink()
                else:
                    tmp_dir.mkdir()

                frame_diff_data = moving_average(frame_diff_data, 240)
                leading_edge = get_leading_edge(frame_diff_data, 0.4)

                end_edge = (
                    get_leading_edge_mean_prop(frame_diff_data[leading_edge:], 1.0)
                    + leading_edge
                )
                # Adjust buffer
                leading_edge = max(0, leading_edge - BUFFER)
                end_edge = min(len(frame_diff_data), end_edge + BUFFER)

                # TODO rejection GUI
                plot_1d_data_with_markers(
                    frame_diff_data,
                    [leading_edge, end_edge],
                    marker=["o", "x"],
                    show=False,
                )
                frames = get_frames_between_indices(
                    vid_mp.local_path, leading_edge, end_edge
                )
                # vid_rez = get_resolution(vid_mp.local_path)
                resize_rez = Resolutions.r720p.as_tuple()
                # if vid_rez[0] > vid_rez[1]:  # Handle portrait videos.
                #     resize_rez = resize_rez[::-1]
                for idx, frame in tqdm(
                    enumerate(frames), total=end_edge - leading_edge
                ):
                    frame = cv2.resize(frame, resize_rez)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Draw 2px red line down center of each frame.
                    frame = cv2.line(
                        frame,
                        (frame.shape[1] // 2, 0)
                        if cam != CameraViews.NADIR
                        else (0, frame.shape[0] // 2),
                        (frame.shape[1] // 2, frame.shape[0])
                        if cam != CameraViews.NADIR
                        else (frame.shape[1], frame.shape[0] // 2),
                        (255, 0, 0),
                        2,
                    )
                    cv2.imwrite(str(tmp_dir / f"{idx + leading_edge:05d}.jpg"), frame)

            filenames = list(os.listdir(tmp_dir))
            filenames.sort()
            filenames = [str(tmp_dir / fn) for fn in filenames]
            frames = []

            # Show slices to user and have them select the correct one.
            if not GENERATE_ONLY:
                first_img_idx = 0
                for idx, filename in enumerate(filenames):
                    fn_path = Path(filename)
                    number = int(fn_path.stem)
                    if number % 240 != 0:
                        first_img_idx = idx
                        break
                offset_val = multi_image_viewer_with_nav_buttons(
                    filenames, frames, first_img_idx
                )

                print(offset_val)
                if len(offset_val) == 0:
                    # prompt user for frame idx
                    print(f"Video path: {str(vid_mp.local_path)}")
                    offset_val = int(input("Enter offset: "))
                else:
                    offset_val = int(offset_val[0])
                if CameraViews.BOTTOM in vid_mp.local_path.as_posix():
                    offset_data[CameraViews.BOTTOM] = offset_val
                elif CameraViews.NADIR in vid_mp.local_path.as_posix():
                    offset_data[CameraViews.NADIR] = offset_val
                elif CameraViews.OBLIQUE in vid_mp.local_path.as_posix():
                    offset_data[CameraViews.OBLIQUE] = offset_val

        # Aggregate data
        if not GENERATE_ONLY:
            print("Offsets:", offset_data)
            offset_mp.save_local(
                offset_data, upload=True, overwrite=OVERWRITE_OFFSET_JSON
            )
            # tmp_dirs = vid_mps[0].local_path.parent.glob("*_leading_frames")
            # for tmp_dir in tmp_dirs:
            #     tmp_dir.rmdir()
