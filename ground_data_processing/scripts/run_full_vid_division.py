"""Download a video from s3, resize it, and upload it back to s3."""
import concurrent.futures
from typing import List

import psutil
from S3MP.mirror_path import get_matching_s3_mirror_paths

from ground_data_processing.data_processors.generate_exg_npy import generate_exg_json
from ground_data_processing.data_processors.thresh_split_video import (
    pass_split_key_fn,
    split_video_on_threshold_search,
)
from ground_data_processing.scripts.clip_images_col_spread import clip_single_vid
from ground_data_processing.scripts.clip_images_hp_sum import spread_col_sum_frame_proc
from ground_data_processing.scripts.generate_framewise_data import (
    calc_framewise_data_and_upload_json,
)
from ground_data_processing.scripts.identify_and_clip_plots import fft_clip_video
from ground_data_processing.utils.absolute_segment_groups import VideoSegments
from ground_data_processing.utils.ffmpeg_utils import FFmpegFlags, FFmpegProcessManager
from ground_data_processing.utils.processing_utils import print_processing_info
from ground_data_processing.utils.rogues_key_utils import (
    plot_trial_prefix_segment_builder,
)
from ground_data_processing.utils.s3_constants import CameraViews, DataFiles, VideoRez

if __name__ == "__main__":
    print_processing_info()
    """ WARNING ::: THIS SCRIPT WILL SPIN UP max_processes * num_videos_found PROCESSES."""
    FFmpegProcessManager.max_processes = (
        5  # x 6 = 30 (probably don't want more than that)
    )
    """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
    # FFmpegFlags.set_output_flag("c", "copy")
    FFmpegFlags.set_output_flag("loglevel", "quiet")
    # FFmpegFlags.set_output_flag("c:v", "libx265")
    # FFmpegFlags.set_output_flag("preset", "slow")
    # FFmpegFlags.set_output_flag("crf", 0)

    overwrite_exg = [False, False, True]
    overwrite_pass_videos = False
    overwrite_plot_videos = True
    overwrite_images = True
    ROW_SPLIT_THRESH_VAL = 0.2
    base_segments = plot_trial_prefix_segment_builder(planting_number=2)
    general_segments = [
        *base_segments,
        VideoSegments.DATE("7-06"),
        VideoSegments.ROW_DESIGNATION("10, 3"),
        VideoSegments.RESOLUTION(VideoRez.r4k_120fps),
        VideoSegments.OG_VID_FILES(incomplete_name=".mp4"),
    ]
    og_vid_mps = get_matching_s3_mirror_paths(general_segments)
    bot_vid_mps = [
        og_vid_mp
        for og_vid_mp in og_vid_mps
        if CameraViews.BOTTOM.value in og_vid_mp.key_segments[-1].name
    ]

    n_procs = psutil.cpu_count(logical=False)
    current_proc_futures: List[concurrent.futures.Future] = []

    # Step 1: Generate exg jsons
    print(f"Generating ExG JSON(s) for {len(bot_vid_mps)} video(s).")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as proc_executor:
        for vid_mp in bot_vid_mps:
            current_proc = proc_executor.submit(
                generate_exg_json, vid_mp, DataFiles.EXG_SLC_20PX_JSON, overwrite_exg[0]
            )
            current_proc_futures.append(current_proc)
        for pf in current_proc_futures:
            if pf.exception():
                print(pf.exception())

    # Step 2, Split passes from all videos at OG depth
    print(f"Splitting into passes for {len(og_vid_mps)} video(s).")
    current_proc_futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as proc_executor:
        for vid_mp in og_vid_mps:
            split_proc = proc_executor.submit(
                split_video_on_threshold_search,
                vid_mp,
                DataFiles.EXG_SLC_20PX_JSON,
                2,
                pass_split_key_fn,
                ROW_SPLIT_THRESH_VAL,
                60,
                True,
                overwrite=overwrite_pass_videos,
            )
            current_proc_futures.append(split_proc)
        for pf in current_proc_futures:
            if pf.exception():
                print(pf.exception())

    # Step 3, Generate ExG for all videos at split pass depth
    current_proc_futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as proc_executor:
        # Regen with new splits
        pass_vid_segments = [seg.__copy__() for seg in general_segments]
        # We replace the previous incomplete match
        pass_vid_segments[-1] = VideoSegments.ROW_SPLIT_FILES(incomplete_name=".mp4")
        pass_vid_mps = get_matching_s3_mirror_paths(pass_vid_segments)

        bot_pass_vid_mps = [
            pass_vid_mp
            for pass_vid_mp in pass_vid_mps
            if CameraViews.BOTTOM.value in pass_vid_mp.key_segments[-1].name
        ]
        print(
            f"Generating ExG JSON(s) for {len(bot_pass_vid_mps)} split pass video(s)."
        )
        for vid_mp in bot_pass_vid_mps:
            current_proc = proc_executor.submit(
                generate_exg_json, vid_mp, DataFiles.EXG_SLC_20PX_JSON, overwrite_exg[1]
            )
            current_proc_futures.append(current_proc)
        for pf in current_proc_futures:
            if pf.exception():
                print(pf.exception())

    # Step 4, Split passes into plots
    print(f"Splitting into plots for {len(pass_vid_mps)} split pass video(s).")
    current_proc_futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as proc_executor:
        for pass_vid_mp in pass_vid_mps:
            plot_split_proc = proc_executor.submit(
                fft_clip_video,
                pass_vid_mp.get_sibling(DataFiles.EXG_SLC_20PX_JSON),
                n_splits=22,
                n_search_freqs=300,
                n_top_peaks=30,
                frame_buffer=0.3 * 120,
                vid_names=[cam.value for cam in CameraViews],
                overwrite=overwrite_plot_videos,
            )
            current_proc_futures.append(plot_split_proc)
        FFmpegProcessManager.wait_for_all_processes_to_finish()
        for pf in current_proc_futures:
            if pf.exception():
                print(pf.exception())

    # Step 5, Generate spread column sum for all videos at split plot depth
    current_proc_futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as proc_executor:
        # Regen with new splits
        plot_vid_segments = [seg.__copy__() for seg in general_segments]
        # We replace the previous incomplete match
        plot_vid_segments[-1] = VideoSegments.PLOT_SPLIT_FILES(incomplete_name=".mp4")
        plot_vid_mps = get_matching_s3_mirror_paths(plot_vid_segments)

        bot_plot_vid_mps = [
            plot_vid_mp
            for plot_vid_mp in plot_vid_mps
            if f"{CameraViews.BOTTOM}.mp4" in plot_vid_mp.s3_key
        ]
        print(
            f"Generating spread column sum JSON(s) for {len(bot_plot_vid_mps)} split plot video(s)."
        )
        for vid_mp in bot_plot_vid_mps:
            data_mp = vid_mp.get_sibling(
                f"{CameraViews.BOTTOM}{DataFiles.Suffixes.SPREAD_COL_SUM_JSON}"
            )
            current_proc = proc_executor.submit(
                calc_framewise_data_and_upload_json,
                vid_mp.local_path,
                data_mp.s3_key,
                [spread_col_sum_frame_proc],
                iter_wrappers=None,
                overwrite=overwrite_exg[2],
            )
            current_proc_futures.append(current_proc)

        for pf in current_proc_futures:
            if pf.exception():
                print(pf.exception())

    # Step 6, Split plots into images
    print(f"Splitting into images for {len(plot_vid_mps)} split plot video(s).")
    current_proc_futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as proc_executor:
        split_processes = []
        for plot_vid_mp in plot_vid_mps:
            data_mp = plot_vid_mp.get_sibling(
                f"{CameraViews.BOTTOM}{DataFiles.Suffixes.SPREAD_COL_SUM_JSON}"
            )
            img_split_proc = proc_executor.submit(
                clip_single_vid,
                data_mp,
                flipped=False,
                overwrite=overwrite_images,
                make_thumbnail=True,
            )

        proc_executor.shutdown(wait=True)

    print("Done.")
