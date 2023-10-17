"""Load an npy plot and peak-filter to identify the passes."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
from S3MP.mirror_path import get_matching_s3_mirror_paths

from ground_data_processing.data_processors.generate_exg_npy import generate_exg_json
from ground_data_processing.data_processors.thresh_split_video import (
    split_video_at_bounds,
)
from ground_data_processing.utils.absolute_segment_groups import ProductionFieldSegments
from ground_data_processing.utils.ffmpeg_utils import FFmpegFlags
from ground_data_processing.utils.peak_finding_utils import (
    correct_splits,
    find_closest_dominant_fft_frequency,
    get_n_best_split_bounds,
)
from ground_data_processing.utils.processing_utils import print_processing_info
from ground_data_processing.utils.rogues_key_utils import (
    plot_trial_prefix_segment_builder,
)
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFiles,
    DataTypes,
    Fields,
    Framerates,
)

if __name__ == "__main__":
    print_processing_info()

    # FFmpegFlags.set_output_flag("c", "copy")
    FFmpegFlags.set_output_flag("loglevel", "quiet")
    FPS = Framerates.fps120.fps
    FRAME_BUFFER_S = 1
    FRAME_BUFFER = FRAME_BUFFER_S * FPS
    OVERWRITE = True
    # CAMERA = CameraViews.BOTTOM
    base_segments = plot_trial_prefix_segment_builder()

    segments = [
        # VideoSegments.DATE("6-28"),
        # VideoSegments.ROW_DESIGNATION("9, 4"),
        # VideoSegments.RESOLUTION(f"{Resolutions.r4k}@{FRAMERATE}"),
        # VideoSegments.OG_VID_FILES(f"{CAMERA}.mp4"),
        ProductionFieldSegments.FIELD_DESIGNATION(Fields.FOUNDATION_FIELD_TWO),
        ProductionFieldSegments.DATA_TYPE(DataTypes.VIDEOS),
        ProductionFieldSegments.DATE("7-08"),
        ProductionFieldSegments.ROW_DESIGNATION("Row 1, 16"),
        ProductionFieldSegments.RESOLUTION("trevor-test"),
        ProductionFieldSegments.OG_VID_FILES("bottom.mp4"),
        # KeySegment(0, "Farmer City 2022"),
        # KeySegment(1, "Strip Trial"),
        # KeySegment(2, "Planting 1"),
        # KeySegment(3, "Videos"),
        # KeySegment(4, "6-21"),
        # KeySegment(5, "Row 3b, 4a GPS TEST"),
        # KeySegment(7, is_file=True, name="bottom.mp4")
    ]
    bottom_vid_mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(bottom_vid_mps)} matching S3 path(s).")

    for bottom_vid_mp in bottom_vid_mps:
        pass_a_folder_mp = bottom_vid_mp.get_sibling("Pass A")
        pass_b_folder_mp = bottom_vid_mp.get_sibling("Pass B")
        if (
            pass_a_folder_mp.exists_on_s3()
            and pass_b_folder_mp.exists_on_s3()
            and not OVERWRITE
        ):
            print(f"Skipping {bottom_vid_mp.s3_key} because both passes exist.")
            continue
        pass_a_folder_mp.local_path.mkdir(parents=True, exist_ok=True)
        pass_b_folder_mp.local_path.mkdir(parents=True, exist_ok=True)

        exg_json_mp = bottom_vid_mp.get_sibling(
            f"bottom{DataFiles.Suffixes.CENTER_EXG_SLC_JSON}"
        )
        if not exg_json_mp.exists_on_s3():
            print(f"ExG JSON not found for {bottom_vid_mp.s3_key}, generating...")
            generate_exg_json(bottom_vid_mp)
        exg_json = exg_json_mp.load_local()
        exg_data = np.array(exg_json["data"])

        offset_mp = bottom_vid_mp.get_sibling(DataFiles.OFFSETS_JSON)
        offset_data = offset_mp.load_local()
        print(offset_data)

        bot_start_point = offset_data["bottom"]
        data_len = len(exg_data)

        inverted_data = exg_data * -1
        n_splits = 2
        n_search_freqs = 30
        n_top_peaks = 6
        freq_comp = find_closest_dominant_fft_frequency(
            inverted_data, n_splits, n_search_freqs
        )
        if freq_comp.freq < n_splits:
            raise ValueError(f"Could not find {n_splits} dominant frequencies.")
        crossings = freq_comp.get_phase_splits()
        corrected_splits = correct_splits(crossings, inverted_data, n_top_peaks)
        best_split_bounds = get_n_best_split_bounds(
            exg_data, corrected_splits, n_splits
        )
        print(best_split_bounds)
        bounds_flattened = list(itertools.chain.from_iterable(best_split_bounds))

        # bounds_flattened = binary_search_peaks(shifted_data, 4, start_prominence=0.5)
        # no_exg_bounds = binary_search_threshold_min_size(npy_data, data_len // 2, 4)
        # bounds_flattened = list(itertools.chain.from_iterable(no_exg_bounds))

        bounds_flattened[0] = bot_start_point

        raw_clip_a_frame_bounds = [
            bounds_flattened[0] - FRAME_BUFFER,
            bounds_flattened[1] + FRAME_BUFFER,
        ]
        raw_clip_b_frame_bounds = [
            bounds_flattened[2] - FRAME_BUFFER,
            bounds_flattened[3] + FRAME_BUFFER,
        ]

        # print(no_exg_bounds)
        print(bounds_flattened)
        print(raw_clip_a_frame_bounds)
        print(raw_clip_b_frame_bounds)
        plt.plot(exg_data)
        for bound in bounds_flattened:
            plt.axvline(bound, color="r")
        plt.show()

        for camera in CameraViews:
            video_mp = bottom_vid_mp.get_sibling(f"{camera}.mp4")
            video_mp.download_to_mirror(overwrite=OVERWRITE)

            clip_a_mp = pass_a_folder_mp.get_child(f"{camera}.mp4")
            clip_b_mp = pass_b_folder_mp.get_child(f"{camera}.mp4")

            start_point = offset_data[camera]
            offset = start_point - bot_start_point
            print(f"Offset: {offset}")

            current_clip_a_frame_bounds = [
                val + offset for val in raw_clip_a_frame_bounds
            ]
            current_clip_b_frame_bounds = [
                val + offset for val in raw_clip_b_frame_bounds
            ]
            current_clip_a_frame_bounds[0] = max(current_clip_a_frame_bounds[0], 0)
            current_clip_b_frame_bounds[1] = min(
                current_clip_b_frame_bounds[1], data_len + offset
            )

            print(clip_a_mp.local_path)
            print(clip_b_mp.local_path)
            print(current_clip_a_frame_bounds)
            print(current_clip_b_frame_bounds)
            split_video_at_bounds(
                video_mp,
                [clip_a_mp, clip_b_mp],
                [current_clip_a_frame_bounds, current_clip_b_frame_bounds],
                overwrite=OVERWRITE,
            )
