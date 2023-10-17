"""Utilities for calculating excess green."""
from pathlib import Path
from typing import Callable, Iterable

from ground_data_processing.utils.image_utils import (
    excess_green,
    prop_nonzero,
    slice_center_segment,
)
from ground_data_processing.utils.plot_utils import constant_plot_thread
from ground_data_processing.utils.video_utils import get_video_ffmpeg_reader


def get_prop_exg_per_frame_slice(
    vid_path: Path, slice_width: int = 20, display_freq: int = None
):
    """Get the proportion of excess green per frame."""
    if display_freq:
        plot_thread_queue = constant_plot_thread()
    #    vid_iter, iter_len = get_video_iterator(vid_path, use_tqdm=True)
    vid_iter, iter_len = get_video_ffmpeg_reader(vid_path, use_tqdm=True)
    prop_exg_per_frame = []
    for idx, frame in enumerate(vid_iter):
        rgb_slice = slice_center_segment(frame, slice_width)
        exg_seg = excess_green(rgb_slice)
        # exg_seg = min_val_exg(rgb_slice, min_val=200, channel=-1)
        # exg_seg = harsh_exg(rgb_slice, channel=-1)
        flattened_exg = exg_seg.reshape(-1, 1)
        prop_exg_per_frame.append(prop_nonzero(flattened_exg))
        if display_freq and idx % display_freq == 0:
            plot_thread_queue.put(prop_exg_per_frame)
    return prop_exg_per_frame


def get_prop_per_frame_slice_generalized(
    iter_fn: Callable, preproc_fn: Callable, seg_fn: Callable, display_freq: int = None
):
    """Get the proportion of excess green per frame."""
    vid_iter: Iterable = iter_fn()

    if display_freq:
        plot_thread_queue = constant_plot_thread()
    prop_per_frame = []
    for idx, frame in enumerate(vid_iter):
        # if idx >= max_idx:
        #     break
        preproc = preproc_fn(frame)
        seg = seg_fn(preproc)
        flattened_seg = seg.reshape(-1, 1)
        prop_per_frame.append(prop_nonzero(flattened_seg))
        if display_freq and idx % display_freq == 0:
            plot_thread_queue.put(prop_per_frame)
    return prop_per_frame
