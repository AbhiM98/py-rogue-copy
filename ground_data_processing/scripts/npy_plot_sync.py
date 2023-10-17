"""Synchronize npy plots and view relevant frames."""
import matplotlib.pyplot as plt
import numpy as np
from S3MP.mirror_path import get_matching_s3_mirror_paths

from ground_data_processing.utils.absolute_segment_groups import ProductionFieldSegments
from ground_data_processing.utils.plot_utils import plot_1d_data_with_markers
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFiles,
    DataTypes,
    Fields,
    Framerates,
)
from ground_data_processing.utils.video_utils import get_frame_at_index


def get_leading_edge(npy_data, transition_prop=0.5):
    """Get the leading edge from some data."""
    boundary = transition_prop * np.max(npy_data)
    return np.where(npy_data > boundary)[0][0]


def get_leading_edge_mean_prop(npy_data, mean_prop=0.5):
    """Get the leading edge from some data."""
    boundary = mean_prop * np.mean(npy_data)
    boundary = np.min([boundary, np.max(npy_data)])
    return np.where(npy_data > boundary)[0][0]


def moving_average(x, w):
    """Calculate moving average of data."""
    return np.convolve(x, np.ones(w), "valid") / w


if __name__ == "__main__":
    CAMERA_VIEW = CameraViews.BOTTOM
    FRAMERATE = Framerates.fps120

    frame_diff_names = [
        f"{cam}{DataFiles.Suffixes.NORM_FRAME_DIFF_NPY}" for cam in CameraViews
    ]

    segments = [
        ProductionFieldSegments.FIELD_DESIGNATION(Fields.PROD_FIELD_ONE),
        ProductionFieldSegments.DATA_TYPE(DataTypes.VIDEOS),
        ProductionFieldSegments.DATE("6-27"),
        ProductionFieldSegments.ROW_DESIGNATION("Row 1"),
        ProductionFieldSegments.OG_VID_FILES(incomplete_name=".mp4"),
    ]
    matching_mps = get_matching_s3_mirror_paths(segments)
    for bot_vid_mp in matching_mps:
        cam_fd_mps = [bot_vid_mp.get_sibling(fd_n) for fd_n in frame_diff_names]
        vid_mps = [bot_vid_mp.get_sibling(f"{cam}.mp4") for cam in CameraViews]
        for idx, (mp, vid_mp) in enumerate(zip(cam_fd_mps, vid_mps)):
            plt.figure(2 * idx)
            mp_data = mp.load_local(download=True)
            mp_data = moving_average(mp_data, 240)
            leading_edge = get_leading_edge(mp_data, 0.3)
            end_edge = (
                get_leading_edge_mean_prop(mp_data[leading_edge:], 1.0) + leading_edge
            )

            vid_mp.download_to_mirror()
            leading_frame = get_frame_at_index(vid_mp.local_path, leading_edge)
            plt.imshow(leading_frame)
            plt.figure(2 * idx + 1)
            plot_1d_data_with_markers(mp_data, [leading_edge, end_edge], show=False)
        plt.show()
