"""Frame vidff analysis on videos."""
import itertools

import cv2
import numpy as np

from ground_data_processing.utils.image_utils import excess_green, slice_center_segment
from ground_data_processing.utils.iter_utils import get_value_per_frame_generalized
from ground_data_processing.utils.video_utils import get_video_iterator

# Measurements on single frames


def distance_to_stem(frame):
    """Extract distance to the stem from the bottom of the frame."""
    frame_height, _, _ = frame.shape
    img_slice = slice_center_segment(frame, 20)
    img_slice = excess_green(img_slice)
    row_sums = np.sum(img_slice, axis=1)
    nonzero_counts = np.argwhere(row_sums > 1)
    if nonzero_counts.size > 1:
        return frame_height - np.max(nonzero_counts)
    return frame_height


# Measurements on frame pairs
def get_normalized_frame_diff(frame_tup):
    """Get normalized frame diff (0-1.0) between two frames."""
    return np.mean(cv2.absdiff(*frame_tup)) / 255.0


def get_normalized_frame_diff_from_video(vid_path, plot_interval=50):
    """Get normalized frame diff (0-1.0) between adjacent frames in a video."""
    vid_iter, _ = get_video_iterator(vid_path, use_tqdm=True)
    vid_iter = itertools.pairwise(vid_iter)
    return np.array(
        get_value_per_frame_generalized(
            vid_iter, get_normalized_frame_diff, display_freq=plot_interval
        )
    )
