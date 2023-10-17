"""Prepares the data for paddle inference on an unfiltered plot."""

import concurrent
import concurrent.futures
from typing import Dict, List

import cv2
import numpy as np
import psutil
from S3MP.keys import KeySegment
from S3MP.mirror_path import MirrorPath, get_matching_s3_mirror_paths
from tqdm import tqdm

from ground_data_processing.utils.absolute_segment_groups import (
    ProductionFieldSegments,
    ProductionFieldWithSplitPassSegments,
)
from ground_data_processing.utils.keymaps import (
    PLOT_TRIAL_RAW_DATA_TO_UNFILTERED_INFERENCE_KEYMAP,
    PROD_RAW_DATA_TO_UNFILTERED_INFERENCE_KEYMAP,
    PROD_SPLIT_ROW_RAW_DATA_TO_UNFILTERED_INFERENCE_KEYMAP,
)
from ground_data_processing.utils.relative_segment_groups import (
    DateAndRowSegments,
    DSSplitSegments,
    align_segment_depths,
)
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFolders,
    Fields,
    InferenceMethods,
)


def crop_square_from_img_center(img, resize_size, rotate=False, ccw=False):
    """Crop a square from the center of an image."""
    height, width = img.shape[:2]
    if height < width and rotate:
        # special case for oblique images
        if ccw:
            img = np.rot90(img, k=3)
        else:
            img = np.rot90(img)
        height, width = img.shape[:2]

    if width == height:
        return cv2.resize(img, (resize_size, resize_size))
    elif width > height:
        left = (width - height) / 2
        right = left + height
        top = 0
        bottom = height
    else:  # oblique
        # pad the image on both sides to make the image square
        img = np.pad(
            img,
            ((0, 0), (int((height - width) / 2), int((height - width) / 2)), (0, 0)),
            mode="constant",
        )
        return cv2.resize(img, (resize_size, resize_size))
    img = img[int(top) : int(bottom), int(left) : int(right)]
    if resize_size:
        return cv2.resize(img, (resize_size, resize_size))
    return img


def run_preproc_on_single_image(
    img_mp: MirrorPath,
    inference_method: InferenceMethods,
    base_output_folder_mp: MirrorPath,
    preproc_kwargs: Dict,
    save_kwargs: Dict,
    downsample: bool,
    downsample_pct: float,
):
    """Run preprocessing on a single image."""
    img = img_mp.load_local()
    if inference_method == InferenceMethods.PADDLE_SLICE:
        preproc_img = img
    elif inference_method == InferenceMethods.SQUARE_CROP:
        preproc_img = crop_square_from_img_center(img, **preproc_kwargs)
    else:
        raise ValueError(f"Invalid inference method: {inference_method}")

    if downsample:
        preproc_img = cv2.resize(
            preproc_img, (0, 0), fx=downsample_pct / 100, fy=downsample_pct / 100
        )
    output_img_mp = base_output_folder_mp.get_child(img_mp.local_path.name)
    output_img_mp.save_local(preproc_img, **save_kwargs)


SPLIT_PASS_REPL_SEGS = [
    *align_segment_depths(DateAndRowSegments.values_as_list(), 2),
    *align_segment_depths(DSSplitSegments.values_as_list()[1:], 5),
]
SINGLE_PASS_REPL_SEGS = [
    *align_segment_depths(DateAndRowSegments.values_as_list(), 2),
    *align_segment_depths(DSSplitSegments.values_as_list()[2:], 5),
]

if __name__ == "__main__":
    CROP_SIZE = 1024
    INFERENCE_METHOD = InferenceMethods.SQUARE_CROP
    DOWNSAMPLE = False
    DOWNSAMPLE_PCT = 50  # 50% of original size
    OVERWRITE = True
    segments = [
        KeySegment(0, "uav_test"),
    ]
    is_split_pass = any(Fields.FOUNDATION_FIELD_TWO in seg.name for seg in segments)
    is_plot_trial = any(Fields.FC_2022 in seg.name for seg in segments)
    if is_plot_trial:
        unfiltered_inf_keymap = PLOT_TRIAL_RAW_DATA_TO_UNFILTERED_INFERENCE_KEYMAP
    else:
        if is_split_pass:
            unfiltered_inf_keymap = (
                PROD_SPLIT_ROW_RAW_DATA_TO_UNFILTERED_INFERENCE_KEYMAP
            )
            ds_split_seg_group = ProductionFieldWithSplitPassSegments
        else:
            unfiltered_inf_keymap = PROD_RAW_DATA_TO_UNFILTERED_INFERENCE_KEYMAP
            ds_split_seg_group = ProductionFieldSegments
        segments.extend(
            [
                ds_split_seg_group.DS_SPLIT(DataFolders.DS_SPLITS),
                ds_split_seg_group.DS_SPLIT_IDX(incomplete_name="DS"),
            ]
        )

    root_ds_folder_mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(root_ds_folder_mps)} plot folders.")

    n_procs = psutil.cpu_count(logical=False)
    proc_executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_procs)
    all_proc_futures: List[concurrent.futures.Future] = []

    for root_ds_folder_mp in tqdm(root_ds_folder_mps):
        # duplicates_json_mp = root_ds_folder_mp.get_child(DataFiles.DUPLICATES_JSON)
        # if duplicates_json_mp.exists_on_s3():
        #     print(f"Skipping {root_ds_folder_mp.s3_key}...")
        #     continue
        print(f"Processing {root_ds_folder_mp.s3_key}...")
        plot_folder_mp = root_ds_folder_mp.get_child(
            f"{CameraViews.NADIR} {DataFolders.RAW_IMAGES}"
        )
        if not plot_folder_mp.exists_on_s3():
            print(f"Folder {root_ds_folder_mp.s3_key} not present on S3, skipping.")
            continue

        base_output_folder_mp = unfiltered_inf_keymap.apply(plot_folder_mp)
        preproc_img_folder_mp = base_output_folder_mp.get_child(
            "nadir Raw Images"
        ).get_child(DataFolders.PREPROCESSED_IMAGES)
        if OVERWRITE and preproc_img_folder_mp.get_children_on_s3():
            print(f"Overwriting {preproc_img_folder_mp.s3_key}...")
            [mp.delete_all() for mp in preproc_img_folder_mp.get_children_on_s3()]

        print(f"Output folder: {preproc_img_folder_mp.s3_key}\n")
        for img_mp in tqdm(plot_folder_mp.get_children_on_s3()):
            pf = proc_executor.submit(
                run_preproc_on_single_image,
                img_mp,
                INFERENCE_METHOD,
                preproc_img_folder_mp,
                preproc_kwargs={
                    "resize_size": CROP_SIZE,
                    "rotate": True,
                    "ccw": True,
                },
                save_kwargs={
                    "upload": True,
                    "overwrite": True,
                },
                downsample=DOWNSAMPLE,
                downsample_pct=DOWNSAMPLE_PCT,
            )
            all_proc_futures.append(pf)

        # Check for exceptions
        for pf in all_proc_futures:
            if pf.exception():
                raise pf.exception()

        all_proc_futures = []
        plot_folder_mp = root_ds_folder_mp.get_child(
            f"{CameraViews.OBLIQUE} {DataFolders.RAW_IMAGES}"
        )
        if not plot_folder_mp.exists_on_s3():
            print(f"Folder {root_ds_folder_mp.s3_key} not present on S3, skipping.")
            continue

        base_output_folder_mp = unfiltered_inf_keymap.apply(plot_folder_mp)
        preproc_img_folder_mp = base_output_folder_mp.get_child(
            "oblique Raw Images"
        ).get_child(DataFolders.PREPROCESSED_IMAGES)
        if OVERWRITE and preproc_img_folder_mp.get_children_on_s3():
            print(f"Overwriting {preproc_img_folder_mp.s3_key}...")
            [mp.delete_all() for mp in preproc_img_folder_mp.get_children_on_s3()]

        print(f"Output folder: {preproc_img_folder_mp.s3_key}\n")
        for img_mp in tqdm(plot_folder_mp.get_children_on_s3()):
            pf = proc_executor.submit(
                run_preproc_on_single_image,
                img_mp,
                INFERENCE_METHOD,
                preproc_img_folder_mp,
                preproc_kwargs={
                    "resize_size": CROP_SIZE,
                    "rotate": True,
                    "ccw": False,
                },
                save_kwargs={
                    "upload": True,
                    "overwrite": True,
                },
                downsample=DOWNSAMPLE,
                downsample_pct=DOWNSAMPLE_PCT,
            )
            all_proc_futures.append(pf)

        # Check for exceptions
        for pf in all_proc_futures:
            if pf.exception():
                raise pf.exception()
    proc_executor.shutdown(wait=True)
