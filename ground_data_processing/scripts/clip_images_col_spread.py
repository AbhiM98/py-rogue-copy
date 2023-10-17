"""View an npy plot."""
import concurrent
import concurrent.futures
import gzip
import os
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import psutil
from S3MP.mirror_path import KeySegment, MirrorPath, get_matching_s3_mirror_paths
from scipy.signal import find_peaks
from scipy.spatial import KDTree

from ground_data_processing.data_processors.clip_images import (
    clip_frames_from_video_with_frame_indices,
)
from ground_data_processing.data_processors.stem_peak_tracker import (
    KDDimSpec,
    NPPeakSet,
    kd_trace,
)
from ground_data_processing.scripts.generate_framewise_data import (
    calc_framewise_data_and_upload_json,
)
from ground_data_processing.utils.image_utils import harsh_exg_col_sum
from ground_data_processing.utils.processing_utils import print_processing_info
from ground_data_processing.utils.s3_constants import CameraViews, DataFolders

FRAME_SCALE = 2
COL_SCALE = 200
SPREAD_SCALE = 50
CENTER_BAND_WIDTH = 0.02 * COL_SCALE
# Vertical deviation in the figure
MIN_COL_DEV = 0 * COL_SCALE
COL_DEV = 0.04 * COL_SCALE
MIN_FRAME_DEV = 1 * FRAME_SCALE  # Picking a peak from the same frame is not allowed
FRAME_DEV = 2 * FRAME_SCALE
SPREAD_DEV = 0.6 * SPREAD_SCALE
# Removal thresholds
OVERLAP_THRESH = 0.6  # Green
LONG_HOMIE_THRESH = 0.0 * COL_SCALE  # Red
MIN_COL_EXTENT = 0.15 * COL_SCALE  # Red
SPREAD_VAL_THRESH = 0.1 * SPREAD_SCALE  # Blue
LOW_PROM_THRESH = 0.0  # Orange


def clip_single_vid(
    data_mp: MirrorPath,
    flipped: bool = False,
    overwrite: bool = False,
    make_thumbnail: bool = False,
):
    """Clip images from a single video."""
    if not overwrite:
        raw_img_folder_mp = data_mp.get_sibling("bottom Raw Images")
        if raw_img_folder_mp.exists_on_s3():
            return
    mp_data = data_mp.load_local(download=True, overwrite=True)
    if data_mp.local_path.suffix == ".json":
        mp_data = mp_data["data"]
        adj_data = []
        for idx, frame_peaks in enumerate(mp_data):
            for vals in zip(*frame_peaks):
                adj_data.append([idx, *vals])
        mp_data = np.array(adj_data)

    """
    Spread Col Sum
    Dims
    0 : Frame
    1 : Column (x-index of peak column sum value)
    2 : Prominence (of the peak)
    3 : Spread value (of the peak)
    4 : Width (of the peak)
    """
    np_peak_set = NPPeakSet(mp_data)
    # np_peak_set = np_peak_set.frame_idx_subset(0, 1000)
    np_peak_set.peak_data[:, 0] *= FRAME_SCALE
    # np_peak_set = np_peak_set.column_idx_subset(1000, 3000)
    np_peak_set = np_peak_set.scale_dimension_to_range(
        dim_to_scale=1, max_val=COL_SCALE
    )
    np_peak_set = np_peak_set.scale_dimension_to_range(
        dim_to_scale=3, max_val=SPREAD_SCALE
    )
    # np_peak_set.peak_data[:, 0] *= 5
    # np_peak_set = np_peak_set.scale_dimension_on_other(dim_to_scale=0, ref_dim=0, scale_factor=2.5)

    kdtree = KDTree(np_peak_set.peak_data[:, (0, 1, 3)])

    # Pick out center band points
    height = np.max(np_peak_set.peak_data[:, 1])
    half_height = height // 2
    center_band_peak_set = np_peak_set.column_idx_subset(
        half_height - CENTER_BAND_WIDTH, half_height + CENTER_BAND_WIDTH
    )
    # Constrain to no more than one peak per frame
    center_band_peak_set.peak_data = center_band_peak_set.peak_data[
        np.unique(center_band_peak_set.peak_data[:, 0], return_index=True)[1]
    ]

    center_band_indices_by_spread_val = np.argsort(center_band_peak_set.peak_data[:, 3])
    center_band_indices_by_spread_val = center_band_indices_by_spread_val[::-1]

    # Vertical deviation in the figure
    dim_specs = [
        KDDimSpec(0, MIN_FRAME_DEV, FRAME_DEV),
        KDDimSpec(1, MIN_COL_DEV, COL_DEV),
        KDDimSpec(3, -SPREAD_DEV, SPREAD_DEV),
    ]
    if flipped:
        dim_specs[0] = dim_specs[0].flip()
        # dim_specs[0].min_bound *= -1
        # dim_specs[0].max_bound *= -1

    homie_groups = []
    for center_band_idx in center_band_indices_by_spread_val:
        peak = center_band_peak_set.peak_data[center_band_idx]
        leading_points = kd_trace(kdtree, np_peak_set.peak_data, peak, dim_specs)
        trailing_points = kd_trace(
            kdtree, np_peak_set.peak_data, peak, [d_s.flip() for d_s in dim_specs]
        )
        homies = np.array(trailing_points[::-1] + [peak] + leading_points)
        homie_groups.append(NPPeakSet(np.array(homies)))

    homie_groups = sorted(
        homie_groups, key=lambda h_g: len(h_g.peak_data), reverse=True
    )
    overlapping_idxs = []
    overlapping_frens = []
    short_idxs = []
    small_spread_idxs = []
    low_prom_idxs = []
    for overlap_idx, homie_group in enumerate(homie_groups):
        if len(homie_group) < LONG_HOMIE_THRESH:
            short_idxs.append(overlap_idx)
        col_extent = np.max(homie_group.peak_data[:, 1]) - np.min(
            homie_group.peak_data[:, 1]
        )
        if col_extent < MIN_COL_EXTENT:
            short_idxs.append(overlap_idx)
        if np.mean(homie_group.peak_data[:, 3]) < SPREAD_VAL_THRESH:
            small_spread_idxs.append(overlap_idx)
        if np.max(homie_group.peak_data[:, 2]) < LOW_PROM_THRESH:
            low_prom_idxs.append(overlap_idx)

        if overlap_idx in (
            overlapping_idxs + short_idxs + small_spread_idxs + low_prom_idxs
        ):
            continue
        for other_idx, other_homie_group in enumerate(homie_groups[overlap_idx + 1 :]):
            if len(homie_group & other_homie_group) > OVERLAP_THRESH * len(
                other_homie_group
            ):
                overlapping_idxs.append(other_idx + overlap_idx + 1)
                overlapping_frens.append(overlap_idx)

    overlapping_homies = []
    short_homies = []
    small_spread_homies = []
    low_prom_homies = []
    ballin_homies = []

    for overlap_idx, homie_group in enumerate(homie_groups):
        # sort homie groups into categories
        if overlap_idx in overlapping_idxs:
            # these are the overlapping frames
            overlapping_homies.append(homie_group)
        if overlap_idx in short_idxs:
            # these are the short frames (likely stalks in background)
            short_homies.append(homie_group)
        if overlap_idx in small_spread_idxs:
            # these are the small spread frames (likely stalks in background)
            small_spread_homies.append(homie_group)
        if overlap_idx in low_prom_idxs:
            # these are the low prominence frames (likely stalks in background)
            low_prom_homies.append(homie_group)
        if overlap_idx not in (
            overlapping_idxs + short_idxs + small_spread_idxs + low_prom_idxs
        ):
            # if not in any of the above categories, then it's a good frame
            ballin_homies.append(homie_group)

    # sort again for plotting
    overlapping_homies = [
        h_g for idx, h_g in enumerate(homie_groups) if idx in overlapping_idxs
    ]
    short_homies = [h_g for idx, h_g in enumerate(homie_groups) if idx in short_idxs]
    small_spread_homies = [
        h_g for idx, h_g in enumerate(homie_groups) if idx in small_spread_idxs
    ]
    low_prom_homies = [
        h_g for idx, h_g in enumerate(homie_groups) if idx in low_prom_idxs
    ]

    ballin_homies = [
        h_g
        for idx, h_g in enumerate(homie_groups)
        if idx
        not in (overlapping_idxs + short_idxs + small_spread_idxs + low_prom_idxs)
    ]

    # For each remaining homie group, find the point closest to the center of the column range
    column_center = COL_SCALE / 2
    homie_centers = []
    for homie_group in ballin_homies:
        closest_point_idx = np.argmin(
            np.abs(homie_group.peak_data[:, 1] - column_center)
        )
        closest_point = homie_group.peak_data[closest_point_idx]
        homie_centers.append(int(closest_point[0] / FRAME_SCALE))

    homie_centers = sorted(homie_centers)
    # print(*enumerate([h_c * 2 for h_c in homie_centers]), sep="\n")

    plt.figure("Prominence")
    np_peak_set.scatter_with_colored_dimension(2)
    plt.figure("Width")
    # Clip all values above 100 in dim 4 to 100
    np_peak_set.peak_data[:, 4] = np.clip(np_peak_set.peak_data[:, 4], 0, 1000)
    np_peak_set.scatter_with_colored_dimension(4, log_color=False)

    # Area Plot
    np_peak_set.peak_data[:, 4] = np.log10(np_peak_set.peak_data[:, 4])
    norm_prom = np_peak_set.peak_data[:, 2] / np.max(np_peak_set.peak_data[:, 2])
    norm_width = np_peak_set.peak_data[:, 4] / np.max(np_peak_set.peak_data[:, 4])
    peak_area = np.multiply(norm_prom, norm_width).reshape(-1, 1)
    np_peak_set.peak_data = np.concatenate((np_peak_set.peak_data, peak_area), axis=1)
    plt.figure("Area")
    np_peak_set.scatter_with_colored_dimension(5, log_color=False)

    # Compound Score Plot
    for i in range(2, 6):
        np_peak_set.peak_data[:, i] /= np.max(np_peak_set.peak_data[:, i])

    compound_score = np.sum(np_peak_set.peak_data[:, 2:6], axis=1).reshape(-1, 1)
    np_peak_set.peak_data = np.concatenate(
        (np_peak_set.peak_data, compound_score), axis=1
    )
    plt.figure("Compound Score")
    np_peak_set.scatter_with_colored_dimension(6, log_color=False)

    plt.figure("Spread")
    np_peak_set.scatter_with_colored_dimension(3)

    center_band_points = center_band_peak_set.peak_data[:, :2]
    plt.scatter(center_band_points[:, 0], center_band_points[:, 1], c="red")
    # plt.show()

    for homie_group in overlapping_homies:
        homie_group.plot(color="green", linestyle="-", marker="x")
    for homie_group in short_homies:
        homie_group.plot(color="red", linestyle="-", marker="x")
    for homie_group in small_spread_homies:
        homie_group.plot(color="blue", linestyle="-", marker="x")
    for homie_group in low_prom_homies:
        homie_group.trace_plot(color="orange")
    for homie_group in ballin_homies:
        homie_group.plot(color="black", linestyle="-", marker="x")

    # plt.show()
    n_homies = np.unique(homie_centers).shape[0]
    skipped = False
    if n_homies < 8:
        if not flipped:
            return clip_single_vid(
                data_mp,
                flipped=True,
                overwrite=overwrite,
                make_thumbnail=make_thumbnail,
            )
        skipped = True
        return (
            os.path.dirname(data_mp.s3_key),
            f"found {n_homies} stalk-like frames{', flipped' if flipped else ''}{', skipped' if skipped else ''}",
        )
    # Pull out frames
    for cam_view in CameraViews:
        vid_mp = data_mp.get_sibling(f"{cam_view}.mp4")
        if not vid_mp.exists_on_s3():
            print(f"Video {vid_mp} does not exist. Skipping image clipping.")
            skipped = True
            continue
        vid_mp.download_to_mirror()
        output_folder_mp = vid_mp.get_sibling(f"{cam_view} {DataFolders.RAW_IMAGES}")
        [mp.delete_all() for mp in output_folder_mp.get_children_on_s3()]
        clip_frames_from_video_with_frame_indices(
            vid_mp, homie_centers, output_folder_mp
        )
        if make_thumbnail:
            thumbnail_mp = vid_mp.get_sibling(f"{cam_view} Thumbnails")
            [mp.delete_all() for mp in thumbnail_mp.get_children_on_s3()]
            clip_frames_from_video_with_frame_indices(
                vid_mp, homie_centers, thumbnail_mp, reduction=4
            )

    return (
        os.path.dirname(data_mp.s3_key),
        f"found {n_homies} stalk-like frames{', flipped' if flipped else ''}{', skipped' if skipped else ''}",
    )


def clip_images_from_exg_framestack(
    video_mp: MirrorPath,
    exg_framestack_mp: MirrorPath,
    output_folder_mp: MirrorPath,
    overwrite: bool = False,
    make_thumbnail: bool = True,
) -> tuple[str, str]:
    """
    Clip images from a video's exg framestack.

    Args:
    vid_mp (MirrorPath): MirrorPath to the video file.
    exg_framestack_mp (MirrorPath): MirrorPath to the exg framestack.
    output_folder_mp (MirrorPath): MirrorPath to the output folder.
    overwrite (bool): Whether to overwrite existing files.
    make_thumbnail (bool): Whether to make thumbnails (default is True).

    Returns:
    path (str): The path to the output folder.
    output (str): The output message.
    """
    output = ""

    if not exg_framestack_mp.exists_in_mirror():
        exg_framestack_mp.download_to_mirror()

    # unzip with gzip and load pickle object
    print(f"Loading {exg_framestack_mp.local_path}...")
    exg_framestack = None
    with gzip.open(exg_framestack_mp.local_path, "rb") as f:
        exg_framestack = pickle.load(f)

    exg_framestack = exg_framestack["data"]
    # x_data = np.array([x[0] for x in exg_framestack])
    exg_framestack = np.array([x[1] for x in exg_framestack])

    # subtract the average x from each y value
    # exg_framestack = np.array([x - np.mean(y) for x,y in zip(exg_framestack, x_data)])

    # print("Plotting EXG Framestack...")
    # # plot exg framestack
    # plt.figure("EXG Framestack")
    # plt.imshow(exg_framestack, cmap='viridis')

    # plt.figure("EXG Framestack, x-axis")
    # plt.imshow(x_data, cmap='viridis')

    # window on either side of center to look for plants
    window_size = 10
    center = len(exg_framestack[0]) // 2

    exg_framestack_window = exg_framestack[
        :, center - window_size : center + window_size
    ]
    exg_framestack_window_scaled = (
        (exg_framestack_window - np.min(exg_framestack_window))
        / (np.max(exg_framestack_window) - np.min(exg_framestack_window))
        * 255
    )
    exg_framestack_window_scaled = np.uint8(exg_framestack_window_scaled)

    # plt.figure("EXG Framestack Window")
    # plt.imshow(exg_framestack_window_scaled, cmap='gray')
    # plt.show()

    # get the location of the maximum in each row
    row_sum = np.sum(exg_framestack_window, axis=1) / len(exg_framestack_window[0])
    # reshape to 1D array
    row_sum = row_sum.reshape(-1)

    # get peaks from row_sum
    threshold = 0.15
    # row_sum[row_sum < 0] = 0
    peaks = find_peaks(row_sum, prominence=0, width=0)
    peaks = find_peaks(
        row_sum, prominence=np.mean(peaks[1]["prominences"] * threshold), width=0
    )

    output = f"found {len(peaks[0])} peaks"

    # plt.figure("EXG Framestack Window Max")
    # plt.plot(row_sum)
    # # plt.plot(row_sum_conv)
    # plt.scatter(peaks[0], row_sum[peaks[0]], c='red')
    # [plt.annotate(f"{i}", (x, row_sum[x])) for i,x in enumerate(peaks[0])]
    # plt.show()

    # extract the frames where there are peaks
    print("Extracting frames...")
    peak_frames = peaks[0]
    if not overwrite:
        return (os.path.dirname(video_mp.s3_key), output)
    for cam_view in CameraViews:
        vid_mp = exg_framestack_mp.get_sibling(f"{cam_view}.mp4")
        if not vid_mp.exists_on_s3():
            print(f"Video {vid_mp} does not exist. Skipping image clipping.")
            # skipped = True
            continue
        vid_mp.download_to_mirror()
        output_folder_mp = vid_mp.get_sibling(f"{cam_view} {DataFolders.RAW_IMAGES}")
        [mp.delete_all() for mp in output_folder_mp.get_children_on_s3()]
        try:
            clip_frames_from_video_with_frame_indices(
                vid_mp, peak_frames, output_folder_mp, overwrite=overwrite
            )

            if make_thumbnail:
                thumbnail_mp = vid_mp.get_sibling(f"{cam_view} Thumbnails")
                [mp.delete_all() for mp in thumbnail_mp.get_children_on_s3()]
                clip_frames_from_video_with_frame_indices(
                    vid_mp, peak_frames, thumbnail_mp, reduction=4, overwrite=overwrite
                )
        except KeyError:
            print(f"Could not clip images from {vid_mp}.")
            # skipped = True
            continue

    return (os.path.dirname(video_mp.s3_key), output)


if __name__ == "__main__":
    print_processing_info()

    CAMERA_VIEW = CameraViews.BOTTOM
    OVERWRITE_EXG = False
    OVERWRITE_CLIPS = True
    FLIPPED = True
    segments = [
        KeySegment(0, "2023-field-data"),
        KeySegment(1, "Williamsburg_Strip_Trial"),
        KeySegment(2, "2023-07-17"),
        KeySegment(3, "row-30"),
        # KeySegment(6, "DS 000"),
        KeySegment(7, "bottom.mp4", is_file=True),
    ]

    matching_mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(matching_mps)} matching mirror paths")

    MAX_SLOPE = 100
    MIN_SLOPE = 40
    MIN_GROUP_SIZE = 20
    H_TOL = 0.01
    V_TOL = 0.01
    QUERY_SIZE = 25

    n_procs = psutil.cpu_count(logical=False)
    proc_executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_procs)
    all_proc_futures: List[concurrent.futures.Future] = []

    for bottom_vid_mp in matching_mps:
        spread_col_sum_mp = bottom_vid_mp.get_sibling("bottom_exg.gz")
        if OVERWRITE_EXG or not spread_col_sum_mp.exists_on_s3():
            print(
                f"Generating spread col sum file: {os.path.dirname(spread_col_sum_mp.s3_key)}"
            )
            # pf = proc_executor.submit(
            #     calc_framewise_data_and_upload_json,
            #     bottom_vid_mp.local_path,
            #     spread_col_sum_mp.s3_key,
            #     [spread_col_sum_frame_proc],
            # )
            pf = proc_executor.submit(
                calc_framewise_data_and_upload_json,
                bottom_vid_mp.local_path,
                spread_col_sum_mp.s3_key,
                [harsh_exg_col_sum],
            )
            all_proc_futures.append(pf)

    all_proc_futures_except = []
    all_proc_futures_no_except = []
    for pf in all_proc_futures:
        if pf.exception():
            all_proc_futures_except.append(pf)
        else:
            all_proc_futures_no_except.append(pf)

    if len(all_proc_futures_no_except) > 0:
        try:
            all_proc_futures_no_except.sort(key=lambda pf: pf.result()[0])
        except Exception:
            pass

    for pf in all_proc_futures_no_except:
        try:
            print(
                f"{os.path.dirname(pf.result()[0])} generated successfully with {pf.result()[1]} frames processed."
            )
        except Exception:
            pass

    for pf in all_proc_futures_except:
        pf.result()

    all_proc_futures: List[concurrent.futures.Future] = []
    print("Overwriting clips.")
    for bottom_vid_mp in matching_mps:
        spread_col_sum_mp = bottom_vid_mp.get_sibling("bottom_exg.gz")
        # pf = proc_executor.submit(
        #     clip_single_vid,
        #     spread_col_sum_mp,
        #     overwrite=OVERWRITE_CLIPS,
        #     make_thumbnail=True
        # )
        pf = proc_executor.submit(
            clip_single_vid,
            spread_col_sum_mp,
            overwrite=OVERWRITE_CLIPS,
            make_thumbnail=True,
            flipped=FLIPPED,
        )
        all_proc_futures.append(pf)

    all_proc_futures_no_except = [pf for pf in all_proc_futures if not pf.exception()]
    all_proc_futures_except = [pf for pf in all_proc_futures if pf.exception()]
    all_proc_futures_no_except.sort(key=lambda pf: pf.result()[0])
    for pf in all_proc_futures_no_except:
        print(f"{pf.result()[0]}: {pf.result()[1]}")

    for pf in all_proc_futures_except:
        raise pf.exception()

    proc_executor.shutdown(wait=True)
