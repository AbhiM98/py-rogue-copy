"""Extract the frames where a plant is located using peak finding."""
import concurrent
import concurrent.futures
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil
from S3MP.mirror_path import MirrorPath, get_matching_s3_mirror_paths
from scipy.spatial import KDTree
from tqdm import tqdm

from ddb_tracking.grd_constants import ProcessFlags
from ddb_tracking.grd_structure import GRDPlantGroup
from ddb_tracking.utils.s3_utils import get_n_files_in_folder
from ground_data_processing.data_processors.clip_images import (
    clip_frames_from_video_with_frame_indices,
)
from ground_data_processing.data_processors.stem_peak_tracker import (
    KDDimSpec,
    NPPeakSet,
    kd_trace,
)
from ground_data_processing.params import RowParams
from ground_data_processing.scripts.clip_images_col_spread import (
    clip_images_from_exg_framestack,
)
from ground_data_processing.scripts.generate_framewise_data import (
    VideoProcessingExceptionError,
    calc_framewise_data_and_upload_json,
)
from ground_data_processing.utils.image_utils import harsh_exg_col_sum
from ground_data_processing.utils.lambda_utils import invoke_lambda
from ground_data_processing.utils.processing_utils import print_processing_info
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFolders,
    LambdaFunctionNames,
    ProcessingSteps,
)

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
    data_mp: MirrorPath, flipped: bool = False, overwrite: bool = False
) -> Tuple[int, List[int]]:
    """Clip images from a single video."""
    if not overwrite:
        raw_img_folder_mp = data_mp.get_sibling("bottom Raw Images")
        if raw_img_folder_mp.exists_on_s3():
            return None, None
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
        if overlap_idx in overlapping_idxs:
            overlapping_homies.append(homie_group)
        if overlap_idx in short_idxs:
            short_homies.append(homie_group)
        if overlap_idx in small_spread_idxs:
            small_spread_homies.append(homie_group)
        if overlap_idx in low_prom_idxs:
            low_prom_homies.append(homie_group)
        if overlap_idx not in (
            overlapping_idxs + short_idxs + small_spread_idxs + low_prom_idxs
        ):
            # we ball
            ballin_homies.append(homie_group)

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
    return n_homies, homie_centers


def highest_image_count_clip_single_vid(
    data_mp: MirrorPath,
    flipped: bool = False,
    overwrite: bool = False,
    make_thumbnail: bool = False,
):
    """Clip images from a single video in both directions, and use the one with the most images."""
    # Run clip_single_vid both flipped and not flipped, and use the one with the most images
    try:
        n_homies, homie_centers = clip_single_vid(
            data_mp, flipped=False, overwrite=overwrite
        )
    except Exception as e:
        print(f"Error clipping unflipped video: {e}")
        n_homies = None
    try:
        n_homies_flipped, homie_centers_flipped = clip_single_vid(
            data_mp, flipped=True, overwrite=overwrite
        )
    except Exception as e:
        print(f"Error clipping flipped video: {e}")
        n_homies_flipped = None

    flipped = False
    if n_homies_flipped > n_homies:
        flipped = True
        n_homies = n_homies_flipped
        homie_centers = homie_centers_flipped

    if n_homies is None or n_homies < 8:
        if not overwrite:
            # When overwrite is False, this will be returned if the images already exist
            return (
                os.path.dirname(data_mp.s3_key),
                f"skipped, existing images detected, overwrite={overwrite}",
            )
        else:
            return (os.path.dirname(data_mp.s3_key), "skipped, no images detected.")

    # Pull out frames
    for cam_view in CameraViews:
        vid_mp = data_mp.get_sibling(f"{cam_view}.mp4")
        if not vid_mp.exists_on_s3():
            print(f"Video {vid_mp} does not exist. Skipping image clipping.")
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
        f"found {n_homies} stalk-like frames{', flipped' if flipped else ''})",
    )


def extract_frames_from_ds_split(
    grd_plant_group: GRDPlantGroup, overwrite: bool = False
):
    """Extract frames from a GRDPlantGroup."""
    print_processing_info()

    matching_mps = get_matching_s3_mirror_paths(
        grd_plant_group.videos.bottom_mp.key_segments
    )
    print(f"Found {len(matching_mps)} matching mirror paths")

    n_procs = psutil.cpu_count(logical=False)
    proc_executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_procs)
    all_proc_futures: List[concurrent.futures.Future] = []

    for bottom_vid_mp in matching_mps:
        # spread_col_sum_mp = bottom_vid_mp.get_sibling(f"bottom{DataFiles.Suffixes.SPREAD_COL_SUM_JSON}")
        spread_col_sum_mp = bottom_vid_mp.get_sibling("bottom_exg.gz")
        if overwrite or not spread_col_sum_mp.exists_on_s3():
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
                use_two_frames=True,
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
    for bottom_vid_mp in matching_mps:
        # spread_col_sum_mp = bottom_vid_mp.get_sibling(f"bottom{DataFiles.Suffixes.SPREAD_COL_SUM_JSON}")
        spread_col_sum_mp = bottom_vid_mp.get_sibling("bottom_exg.gz")
        # pf = proc_executor.submit(
        #     highest_image_count_clip_single_vid,
        #     spread_col_sum_mp,
        #     overwrite=overwrite,
        #     make_thumbnail=True
        # )
        pf = proc_executor.submit(
            clip_images_from_exg_framestack,
            bottom_vid_mp,
            spread_col_sum_mp,
            bottom_vid_mp.get_sibling(f"bottom {DataFolders.RAW_IMAGES}"),
            overwrite=overwrite,
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


def extract_frames_from_row(row_params: RowParams):
    """Extract frames for each DS Split in a row."""
    empty_plant_group_idxs = []
    send_message = False
    for i, grd_plant_group in enumerate(
        tqdm(
            row_params.grdrow.plant_groups,
            desc=f"\nExtracting Frames for Row {row_params.row_number}",
        )
    ):
        # Skip if frames are already present
        n_files_present = get_n_files_in_folder(
            grd_plant_group.image_directories.bottom_mp.s3_key,
            ".png",
            client=row_params.s3_client,
        )
        if n_files_present > 0 and (
            not row_params.overwrite or row_params.ds_split_numbers
        ):
            print(
                f"Skipping DS Split {grd_plant_group.ds_split_number} because {n_files_present} images are already present."
            )
            continue

        # Extract frames
        try:
            extract_frames_from_ds_split(
                grd_plant_group, overwrite=row_params.overwrite
            )
            send_message = True
        except VideoProcessingExceptionError as e:
            # If DS Split failed, remove and send a notification.
            print(
                f"WARNING: Failed to extract frames for DS Split {grd_plant_group.ds_split_number} with error: {e}, skipping and removing from database..."
            )

            # Send SNS message
            params = {
                "name": "rogues-task-failure",
                "message": f"WARNING: Row {row_params.row_number} from {row_params.field_name} on {row_params.date} failed during {ProcessingSteps.EXTRACT_FRAMES}: {e} for DS Split {grd_plant_group.ds_split_number}. Removing from database...",
            }
            invoke_lambda(LambdaFunctionNames.SEND_SNS, params)

            empty_plant_group_idxs.append(i)
            continue

        # Update plant group w/the number of images extracted
        n_images = get_n_files_in_folder(
            grd_plant_group.image_directories.bottom_mp.s3_key,
            ".png",
            client=row_params.s3_client,
        )
        if n_images > 0:
            row_params.grdrow.plant_groups[i].n_images = n_images
        else:
            print(
                f"WARNING: No images extracted for DS Split {grd_plant_group.ds_split_number}, removing from database..."
            )
            empty_plant_group_idxs.append(i)

    # Remove empty plant groups
    for i in sorted(empty_plant_group_idxs, reverse=True):
        del row_params.grdrow.plant_groups[i]
    # Update number of DS splits
    row_params.grdrow.n_ds_splits = len(row_params.grdrow.plant_groups)

    # Update database
    row_params.update_process_flag_and_push_to_ddb(ProcessFlags.FRAMES_EXTRACTED, True)

    if send_message:
        # Notify slack channel that we can begin labeling
        params = {
            "name": "rogues-inference-complete",
            "message": f"Row {row_params.row_number} from {row_params.field_name} on {row_params.date} is ready for labeling. Number of DS splits: {row_params.grdrow.n_ds_splits}. S3 Path: {row_params.grdrow.full_row_video_mps.get_root_mp().get_child(DataFolders.DS_SPLITS).s3_key}",
        }
        invoke_lambda("rogues-prod-send-sns-message-to-name", params)
