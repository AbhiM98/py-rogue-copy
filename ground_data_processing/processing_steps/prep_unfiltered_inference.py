"""Prepares the data for paddle inference on an unfiltered plot."""

import concurrent
import concurrent.futures
from typing import Dict, List

import cv2
import numpy as np
import psutil
from S3MP.mirror_path import MirrorPath
from tqdm import tqdm

from ddb_tracking.grd_constants import ProcessFlags
from ddb_tracking.grd_structure import MultiViewMP
from ground_data_processing.params import RowParams
from ground_data_processing.utils.s3_constants import (
    CROP_SIZE,
    CameraViews,
    DataFolders,
    DataTypes,
    InferenceMethods,
    ModelNames,
)

INFERENCE_METHOD = InferenceMethods.SQUARE_CROP
DOWNSAMPLE = False
DOWNSAMPLE_PCT = 50  # 50% of original size


def crop_square_from_img_center(
    img,
    resize_size,
    nadir_crop_height=None,
    nadir_crop_width=None,
    rotate=False,
    ccw=False,
):
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
    elif width > height:  # nadir
        if nadir_crop_height:
            # crop the image from the center at the specified height
            top = height / 2 - nadir_crop_height // 2
            bottom = height / 2 + nadir_crop_height // 2
            left = width / 2 - nadir_crop_height // 2
            right = width / 2 + nadir_crop_height // 2
            if nadir_crop_width:
                # crop the image from the center at the specified width
                left = width / 2 - nadir_crop_width // 2
                right = width / 2 + nadir_crop_width // 2
                img = img[int(top) : int(bottom), int(left) : int(right)]
                # Pad the nadir_crop_width to make the image square
                img = np.pad(
                    img,
                    (
                        (0, 0),
                        (
                            int((nadir_crop_height - nadir_crop_width) / 2),
                            int((nadir_crop_height - nadir_crop_width) / 2),
                        ),
                        (0, 0),
                    ),
                    mode="constant",
                )
                return cv2.resize(img, (resize_size, resize_size))
        else:
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
    return cv2.resize(img, (resize_size, resize_size))


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
    img = img_mp.load_local(download=False, load_fn=cv2.imread)
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


def run_preproc_on_camera_view(
    row_params: RowParams,
    root_ds_folder_mp: MirrorPath,
    camera_view: CameraViews,
    proc_executor: concurrent.futures.ProcessPoolExecutor = None,
):
    """Run preprocessing on a single camera view."""
    print(f"Processing {camera_view} camera...")
    if proc_executor is None:
        n_procs = psutil.cpu_count(logical=False)
        proc_executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_procs)
    all_proc_futures: List[concurrent.futures.Future] = []

    # Check if raw images exist
    plot_folder_mp = root_ds_folder_mp.get_child(
        f"{camera_view} {DataFolders.RAW_IMAGES}"
    )
    if not plot_folder_mp.exists_on_s3():
        print(f"Folder {root_ds_folder_mp.s3_key} not present on S3, skipping.")
        return

    # Use seperate models for bottom vs nadir/oblique views
    model_name = (
        ModelNames.SOLO_V2_SEPT_05_MODEL
        if camera_view == CameraViews.BOTTOM
        else ModelNames.SOLO_V2_DEC_11_MODEL
    )

    # Store at the DS split level
    base_output_folder_mp = (
        root_ds_folder_mp.get_child(DataTypes.UNFILTERED_MODEL_INFERENCE)
        .get_child(model_name)
        .get_child(INFERENCE_METHOD)
    )
    preproc_img_folder_mp = base_output_folder_mp.get_child(
        f"{camera_view} {DataFolders.RAW_IMAGES}"
    ).get_child(DataFolders.PREPROCESSED_IMAGES)

    # Check if preprocessed images already exist, and if the number of images present
    # matches the number of images in the raw folder
    n_images_extected = len(list(plot_folder_mp.get_children_on_s3()))
    n_images_present = len(list(preproc_img_folder_mp.get_children_on_s3()))
    print(
        f"Found {n_images_present} images in {camera_view} view, expected {n_images_extected}."
    )

    # Make note of preprocessed paths in database object
    grd_plant_group = row_params.get_grd_plant_group_from_ds_split_mp(root_ds_folder_mp)
    if grd_plant_group.preprocessed_images is None:
        grd_plant_group.preprocessed_images = MultiViewMP.from_root_mp(
            base_output_folder_mp,
            f" {DataFolders.RAW_IMAGES}/{DataFolders.PREPROCESSED_IMAGES}",
        )
    else:
        # Ensure the camera view has the correct path
        match camera_view:
            case CameraViews.NADIR:
                grd_plant_group.preprocessed_images.nadir_mp = preproc_img_folder_mp
            case CameraViews.OBLIQUE:
                grd_plant_group.preprocessed_images.oblique_mp = preproc_img_folder_mp
            case CameraViews.BOTTOM:
                grd_plant_group.preprocessed_images.bottom_mp = preproc_img_folder_mp

    if not row_params.overwrite and n_images_present == n_images_extected:
        print(f"Skipping {camera_view} (already processed).")
        return  # Skip if already processed

    # Overwrite existing files if specified
    if row_params.overwrite or n_images_extected != n_images_present:
        print(f"Overwriting {preproc_img_folder_mp.s3_key}...")
        [mp.delete_all() for mp in preproc_img_folder_mp.get_children_on_s3()]

    # Run preprocessing on each image
    print(f"Output folder: {preproc_img_folder_mp.s3_key}\n")
    for img_mp in tqdm(plot_folder_mp.get_children_on_s3()):
        match camera_view:
            case CameraViews.NADIR:
                pf = proc_executor.submit(
                    run_preproc_on_single_image,
                    img_mp,
                    INFERENCE_METHOD,
                    preproc_img_folder_mp,
                    preproc_kwargs={
                        "resize_size": CROP_SIZE,
                        "nadir_crop_height": row_params.nadir_crop_height,
                        "nadir_crop_width": row_params.nadir_crop_width,
                    },
                    save_kwargs={
                        "save_fn": cv2.imwrite,
                        "upload": True,
                        "overwrite": True,
                    },
                    downsample=DOWNSAMPLE,
                    downsample_pct=DOWNSAMPLE_PCT,
                )
            case CameraViews.OBLIQUE | CameraViews.BOTTOM:
                pf = proc_executor.submit(
                    run_preproc_on_single_image,
                    img_mp,
                    INFERENCE_METHOD,
                    preproc_img_folder_mp,
                    preproc_kwargs={
                        "resize_size": CROP_SIZE,
                        "rotate": False,
                        "ccw": False,
                    },
                    save_kwargs={
                        "save_fn": cv2.imwrite,
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


def prep_unfiltered_inference(row_params: RowParams):
    """Prepare the data for paddle inference on an unfiltered plot."""
    if row_params.ds_split_numbers_mps:
        # Filter to only the specified DS split indices
        root_ds_folder_mps = []
        root_ds_folder_mps = row_params.ds_split_numbers_mps
        print(f"Found {len(root_ds_folder_mps)} plot folders: {root_ds_folder_mps}")
    else:
        root_ds_folder_mps = row_params.ds_split_mps
        print(f"Found {len(root_ds_folder_mps)} plot folders.")

    n_procs = psutil.cpu_count(logical=False)
    proc_executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_procs)

    for root_ds_folder_mp in tqdm(root_ds_folder_mps):
        # Run preprocessing on nadir and oblique camera views
        print(f"Processing {root_ds_folder_mp.s3_key}...")
        for camera_view in CameraViews:
            run_preproc_on_camera_view(
                row_params=row_params,
                root_ds_folder_mp=root_ds_folder_mp,
                camera_view=camera_view,
                proc_executor=proc_executor,
            )
    proc_executor.shutdown(wait=True)

    # Update database with process flag and the preprocessed image paths
    row_params.update_process_flag_and_push_to_ddb(ProcessFlags.FRAMES_PREPARED, True)


# if __name__ == "__main__":
#     # load image
#     img = cv2.imread("./034.png")
#     # run nadir crop
#     nadir_crop_img = crop_square_from_img_center(
#         img,
#         resize_size=1024,
#         nadir_crop_height=1600,
#         nadir_crop_width=1000,
#     )
#     print(nadir_crop_img.shape)
#     # save image
#     cv2.imwrite("./034_crop.png", nadir_crop_img)
