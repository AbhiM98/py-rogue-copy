"""Split video based on threshold."""
import json
import string
from typing import List

import ffmpeg
import numpy as np
from S3MP.keys import KeySegment, replace_key_segments_at_relative_depth
from S3MP.mirror_path import MirrorPath

from ground_data_processing.data_processors.generate_exg_npy import NPJSONEncoder
from ground_data_processing.utils.ffmpeg_utils import (
    FFmpegFlags,
    FFmpegProcessManager,
    FFmpegProcessS3Upload,
)
from ground_data_processing.utils.peak_finding_utils import (
    binary_search_threshold_min_size,
)
from ground_data_processing.utils.processing_utils import get_processing_info_as_json
from ground_data_processing.utils.s3_constants import CameraViews, DataFiles

"""
Signature for key_fn:
    def key_fn(vid_key, split_idx):
"""


def pass_split_key_fn(vid_key, split_idx):
    """Key fn for pass splitting."""
    pass_char = string.ascii_uppercase[split_idx]
    pass_name = f"Pass {pass_char}"
    camera = vid_key.split("/")[-1]
    return replace_key_segments_at_relative_depth(
        vid_key,
        [
            KeySegment(0, pass_name),
            KeySegment(1, camera),
        ],
    )


def plot_split_key_fn(vid_key, split_idx):
    """Key fn for plot splitting."""
    camera = vid_key.split("/")[-1]
    return replace_key_segments_at_relative_depth(
        vid_key,
        [
            KeySegment(0, "Rel Plots"),
            KeySegment(1, f"{split_idx:02d}"),
            KeySegment(2, camera),
        ],
    )


def split_video_on_threshold_search(
    vid_mp: MirrorPath,
    data_file: DataFiles,
    n_splits,
    split_key_fn,
    thresh_max=1.1,
    frame_buffer=60,
    use_offset=False,
    overwrite=False,
):
    """Split a video based on binary search of threshold_data."""
    camera = vid_mp.local_path.stem
    data_mp = vid_mp.get_sibling(data_file)

    if not data_mp.exists_on_s3():
        print(f"{data_mp.s3_key} does not exist.")
        return

    offset = 0
    if use_offset:
        offset_mp = vid_mp.get_sibling(DataFiles.OFFSETS_JSON)
        if not offset_mp.exists_on_s3():
            print(f"{offset_mp.s3_key} does not exist.")
            return
        offset_data = offset_mp.load_local(download=True)
        offset = offset_data[camera] - offset_data[CameraViews.BOTTOM]

    split_keys = [
        split_key_fn(vid_mp.s3_key, split_idx) for split_idx in range(n_splits)
    ]
    split_mps = [MirrorPath.from_s3_key(split_key) for split_key in split_keys]
    if not overwrite and all(split_mp.exists_on_s3() for split_mp in split_mps):
        print(f"All splits exist for {vid_mp.s3_key}.")
        return

    vid_mp.download_to_mirror()
    json_data = data_mp.load_local(download=True)
    if data_mp.local_path.suffix == ".json":
        npy_data = np.array(json_data["data"])
    else:
        npy_data = np.array(json_data)

    data_len = len(npy_data)
    no_exg_bounds = binary_search_threshold_min_size(
        npy_data, data_len, n_splits + 1, thresh_max
    )

    clip_bounds = []

    for idx, split_mp in enumerate(split_mps):
        current_bound = [
            no_exg_bounds[idx][1] - frame_buffer,
            no_exg_bounds[idx + 1][0] + frame_buffer,
        ]
        current_bound = [val + offset for val in current_bound]
        if current_bound[0] < 0:
            raise ValueError("Offset adjust walks off of start of video.")
        current_bound[0] = max(current_bound[0], 0)
        current_bound[1] = min(current_bound[1], data_len + offset)
        clip_bounds.append(current_bound)

        # Split npy files down pre-emptively
        clipped_exg_mp = split_mp.get_sibling(data_file)
        clipped_exg_data = npy_data[current_bound[0] : current_bound[1]]
        if data_mp.local_path.suffix == ".json":
            clipped_exg_json = get_processing_info_as_json()
            clipped_exg_json["data"] = clipped_exg_data
            clipped_exg_mp.local_path.parent.mkdir(parents=True, exist_ok=True)
            with clipped_exg_mp.local_path.open("w") as f:
                json.dump(clipped_exg_json, f, cls=NPJSONEncoder)
            clipped_exg_mp.upload_from_mirror(overwrite=overwrite)
        else:
            clipped_exg_mp.save_local(
                clipped_exg_data, upload=True, overwrite=overwrite
            )

    split_video_at_bounds(vid_mp, split_mps, clip_bounds, overwrite=overwrite)
    # # Create clips
    # for idx, split_mp in enumerate(split_mps):
    #     start_timestamp = clip_bounds[idx][0] / 119.88
    #     end_timestamp = clip_bounds[idx][1] / 119.88
    #     duration = end_timestamp - start_timestamp
    #     stream = ffmpeg.input(
    #         str(vid_mp.local_path), ss=start_timestamp, **FFmpegFlags.input_flags
    #     ).output(str(split_mp.local_path), to=duration, **FFmpegFlags.output_flags)
    #     # stream = ffmpeg.input(vid_path, **FFmpegFlags.input_flags).trim(start_frame=clip_bounds[idx][0], end_frame=clip_bounds[idx][1]).setpts("PTS-STARTPTS").output(clip_paths[idx], **FFmpegFlags.output_flags)
    #     # Default behavior is always overwrite local.
    #     stream = stream.overwrite_output().run()
    #     split_mp.upload_from_mirror()
    # FFmpegProcessManager.add_process(FFmpegProcessS3Upload(stream, split_keys[idx]))


def split_video_at_bounds(
    input_mp: MirrorPath,
    output_mps: List[MirrorPath],
    bounds: List[List[int]],
    framerate: float = 119.88,
    overwrite: bool = False,
):
    """Split video at specified bounds."""
    assert len(output_mps) == len(bounds)
    # procs = []
    for idx, output_mp in enumerate(output_mps):
        start_timestamp = bounds[idx][0] / framerate
        end_timestamp = bounds[idx][1] / framerate
        duration = end_timestamp - start_timestamp
        stream = ffmpeg.input(
            str(input_mp.local_path), ss=start_timestamp, **FFmpegFlags.input_flags
        ).output(str(output_mp.local_path), to=duration, **FFmpegFlags.output_flags)
        stream = stream.overwrite_output()
        FFmpegProcessManager.add_process(
            FFmpegProcessS3Upload(stream, output_mp.s3_key)
        )
    # Wait at this level in hopes of solving occasional upload issues
    FFmpegProcessManager.wait_for_all_processes_to_finish()
