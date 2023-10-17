"""View an npy plot with splits."""
import matplotlib.pyplot as plt
from S3MP.keys import get_matching_s3_keys
from S3MP.mirror_path import MirrorPath

from ground_data_processing.utils.absolute_segment_groups import VideoSegments
from ground_data_processing.utils.ffmpeg_utils import FFmpegFlags
from ground_data_processing.utils.peak_finding_utils import (
    correct_splits,
    find_closest_dominant_fft_frequency,
    get_n_best_split_bounds,
)
from ground_data_processing.utils.rogues_key_utils import (
    plot_trial_prefix_segment_builder,
    rogues_key_video_plot_segment_builder,
)
from ground_data_processing.utils.s3_constants import CameraViews, DataFiles, Framerates

if __name__ == "__main__":
    FFmpegFlags.set_output_flag("c", "copy")
    FFmpegFlags.set_output_flag("loglevel", "quiet")
    CAMERA_VIEW = CameraViews.BOTTOM
    FRAMERATE = Framerates.fps120

    base_segments = plot_trial_prefix_segment_builder(planting_number=1)
    segments = [
        *rogues_key_video_plot_segment_builder(
            date="7-06", row_number=6, existing_segments=base_segments
        ),
        VideoSegments.ROW_SPLIT_FILES(DataFiles.EXG_SLC_20PX_NPY),
    ]
    # Use these to swap between plots / passes
    N_SPLITS = 22
    N_PEAKS = 30
    # N_SPLITS = 2
    # N_PEAKS = 2
    matching_keys = get_matching_s3_keys(segments)
    print(matching_keys[0])
    npy_mp = MirrorPath.from_s3_key(matching_keys[0])

    npy_data = npy_mp.load_local(download=True, overwrite=True)
    print(npy_data.shape)
    data_len = len(npy_data)
    inverted_data = npy_data * -1
    freq_comp = find_closest_dominant_fft_frequency(inverted_data, N_SPLITS, 10)
    crossings = freq_comp.get_phase_splits()

    corrected_splits = correct_splits(crossings, inverted_data, N_PEAKS)

    best_split_bounds = get_n_best_split_bounds(npy_data, corrected_splits, N_SPLITS)

    best_split_bounds = [list(x) for x in best_split_bounds]
    # extra_bound = [best_split_bounds[0][1], 27000]
    # best_split_bounds.insert(1, extra_bound)
    # best_split_bounds[-1][0] = 27000
    print(best_split_bounds)

    # Plot the data
    x_count, y_data = zip(*enumerate(npy_data))
    # plt.plot(freq_x, freq_y, label="Closest Frequency")
    # plt.plot(x_count, y_data, label="Data")
    # for split in corrected_splits:
    #     plt.axvline(split, color="red", label="Split")
    # plt.show()
    for bound_min, bound_max in best_split_bounds:
        # plt.plot(x_count, y_data)
        plt.plot(x_count[bound_min:bound_max], y_data[bound_min:bound_max])
        plt.plot(x_count[bound_min], y_data[bound_min], "x")
    plt.show()
