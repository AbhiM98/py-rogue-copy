"""Load an npy plot and peak-filter to identify the plots."""
from typing import List

import numpy as np
from S3MP.mirror_path import MirrorPath, get_matching_s3_mirror_paths

from ground_data_processing.data_processors.thresh_split_video import (
    split_video_at_bounds,
)
from ground_data_processing.utils.absolute_segment_groups import VideoSegments
from ground_data_processing.utils.ffmpeg_utils import FFmpegFlags, FFmpegProcessManager
from ground_data_processing.utils.peak_finding_utils import (
    correct_splits,
    find_closest_dominant_fft_frequency,
    get_n_best_split_bounds,
)
from ground_data_processing.utils.rogues_key_utils import (
    plot_trial_prefix_segment_builder,
)
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFiles,
    Framerates,
    Resolutions,
)


def fft_clip_video(
    data_mp: MirrorPath,
    n_splits: int,
    n_search_freqs: int,
    n_top_peaks: int,
    frame_buffer: int,
    vid_names: List[str] = CameraViews,
    overwrite: bool = False,
    debug: bool = False,
):
    """
    Use FFT to clip a video into n_splits.

    Args:
        data_mp: MirrorPath to the npy data.
        n_splits: Number of splits to make.
        n_search_freqs: Number of frequencies to search for dominant peaks.
        n_top_peaks: Number of top peaks to use. General 1.5x n_splits works well.
        frame_buffer: Number of frames to buffer around each split.
        vid_names: Names of the videos to clip.
    """
    vid_mps = [data_mp.get_sibling(f"{vid_name}.mp4") for vid_name in vid_names]
    vid_mps = [v_mp for v_mp in vid_mps if v_mp.exists_on_s3()]
    [v_mp.download_to_mirror() for v_mp in vid_mps]
    exg_data = data_mp.load_local()
    if data_mp.local_path.suffix == ".json":
        exg_data = np.array(exg_data["data"])

    data_len = len(exg_data)
    inverted_data = exg_data * -1
    freq_comp = find_closest_dominant_fft_frequency(
        inverted_data, n_splits, n_search_freqs
    )
    if freq_comp.freq < n_splits:
        raise ValueError(f"Could not find {n_splits} dominant frequencies.")
    crossings = freq_comp.get_phase_splits()
    corrected_splits = correct_splits(crossings, inverted_data, n_top_peaks)
    best_split_bounds = get_n_best_split_bounds(exg_data, corrected_splits, n_splits)

    if debug:
        import matplotlib.pyplot as plt

        plt.plot(exg_data)
        for split in corrected_splits:
            plt.axvline(split, color="red")
        plt.show()

    clip_bounds = []
    for idx in range(n_splits):
        current_bound = [
            best_split_bounds[idx][0] - frame_buffer,
            best_split_bounds[idx][1] + frame_buffer,
        ]
        current_bound[0] = max(current_bound[0], 0)
        current_bound[1] = min(current_bound[1], data_len)
        clip_bounds.append(current_bound)

    for vid_mp in vid_mps:
        vid_name = vid_mp.local_path.stem
        output_mps = [
            vid_mp.replace_key_segments(
                [
                    VideoSegments.PLOT_SPLIT("Rel Plots"),
                    VideoSegments.PLOT_SPLIT_IDX(f"{idx:02d}"),
                    VideoSegments.PLOT_SPLIT_FILES(f"{vid_name}.mp4"),
                ]
            )
            for idx in range(n_splits)
        ]
        if all(output_mp.exists_on_s3() for output_mp in output_mps) and not overwrite:
            continue
        for output_mp in output_mps:
            output_mp.local_path.parent.mkdir(parents=True, exist_ok=True)
        split_video_at_bounds(
            vid_mp,
            output_mps,
            clip_bounds,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    FFmpegProcessManager.max_processes = 12
    FFmpegFlags.set_output_flag("c", "copy")
    FFmpegFlags.set_output_flag("loglevel", "quiet")

    FRAMERATE = Framerates.fps120
    FRAME_BUFFER_S = 0.3
    FRAME_BUFFER = FRAME_BUFFER_S * FRAMERATE.fps
    # CAMERA = CameraViews.BOTTOM
    N_PLOTS = 22
    DEBUG = True

    base_segments = plot_trial_prefix_segment_builder(planting_number=1)

    segments = [
        *base_segments,
        VideoSegments.DATE("7-06"),
        VideoSegments.ROW_DESIGNATION("4, 9"),
        VideoSegments.RESOLUTION(f"{Resolutions.r4k}@{FRAMERATE}"),
        VideoSegments.ROW_SPLIT("Pass B"),
        VideoSegments.ROW_SPLIT_FILES(DataFiles.EXG_SLC_20PX_JSON),
    ]
    exg_mps = get_matching_s3_mirror_paths(segments)

    for exg_mp in exg_mps:
        fft_clip_video(
            exg_mp,
            N_PLOTS,
            n_search_freqs=10,
            n_top_peaks=30,
            frame_buffer=FRAME_BUFFER,
            vid_names=[cam.value for cam in CameraViews],
            debug=DEBUG,
        )
    FFmpegProcessManager.wait_for_all_processes_to_finish()
    print("done!")
