"""Run paddle inference on the images in the plot folder."""
import shutil
import subprocess
from pathlib import Path

from S3MP.mirror_path import MirrorPath
from tqdm import tqdm

from ddb_tracking.grd_constants import ProcessFlags
from ddb_tracking.grd_structure import MultiViewMP
from ground_data_processing.params import RowParams
from ground_data_processing.utils.lambda_utils import invoke_lambda
from ground_data_processing.utils.processing_utils import print_processing_info
from ground_data_processing.utils.s3_constants import (
    DataFolders,
    Directories,
    LambdaFunctionNames,
    PythonEntrypoints,
)

N_DS_SPLITS_PER_LAMBDA = 2


def run_paddle_inference(row_params: RowParams):
    """Run paddle inference on the images in a row.

    Requires having the `PaddleDetection` repo setup with the SoloV2 model config/params files present.
    Must be run from py-rogues-detection or the PaddleDetection directories.
    """
    print_processing_info()

    cwd = Path.cwd()
    python_path = PythonEntrypoints.DEFAULT  # python path on rogues-dev-trev-3
    match cwd.name:
        case Directories.PADDLE_DETECTION:
            # If we're already in the PaddleDetection directory, we're good to go
            prefix = ""
        case Directories.PY_ROGUE_DETECTION:
            # Check if PaddleDetection is present
            paddle_dir = cwd.parent / Directories.PADDLE_DETECTION
            if not paddle_dir.exists():
                raise ValueError(
                    f"Could not find PaddleDetection directory at {paddle_dir}"
                )
            prefix = f"{paddle_dir}/"
        case Directories.APP:
            # Used when running in a docker container
            # Check if PaddleDetection is present
            paddle_dir = cwd / Directories.PADDLE_DETECTION
            if not paddle_dir.exists():
                raise ValueError(
                    f"Could not find PaddleDetection directory at {paddle_dir}"
                )
            prefix = f"{paddle_dir}/"
            # Use paddle-venv python environment
            python_path = PythonEntrypoints.PADDLE_VENV
        case _:
            raise ValueError(
                f"Invalid current working directory '{cwd.name}', expected one of {[str(dir_name) for dir_name in Directories]}"
            )

    config_path = f"{prefix}configs/solov2/solov2_r101_vd_fpn_3x_coco.yml"
    weights_path = f"{prefix}output/solov2_r101_vd_fpn_3x_coco/best_model.pdparams"
    draw_thresh = 0.25

    if not row_params.grdrow.frames_prepared:
        print(f"Skipping Row {row_params.row_number} because frames are not prepared.")
        return

    for grd_plant_group in tqdm(row_params.grdrow.plant_groups):
        print(f"Processing DS Split {grd_plant_group.ds_split_number}...")

        # Store the inferenced image paths in the database object
        grd_plant_group.inferenced_images = MultiViewMP.from_root_mp(
            grd_plant_group.preprocessed_images.get_root_mp(),
            f" {DataFolders.RAW_IMAGES}/{DataFolders.ANNOTATED_IMAGES}",
        )

        # Inference each camera view
        for (
            camera,
            preproc_folder_mp,
        ) in grd_plant_group.preprocessed_images.as_dict().items():
            print(f"Processing {camera} view...")

            if not preproc_folder_mp.exists_on_s3():
                print(f"{camera} folder not present on S3, skipping.")
                continue

            annotated_folder_mp = grd_plant_group.inferenced_images.as_dict()[camera]
            if row_params.overwrite and list(annotated_folder_mp.get_children_on_s3()):
                # Delete old images
                [mp.delete_all() for mp in annotated_folder_mp.get_children_on_s3()]
                annotated_folder_mp.local_path.mkdir(parents=True, exist_ok=True)

            inference_mask_json_mp = annotated_folder_mp.get_child("segm.json")
            if inference_mask_json_mp.exists_on_s3():
                print(f"Skipping {camera} (already processed).")
                continue  # Skip if already processed

            for img_mp in preproc_folder_mp.get_children_on_s3():
                img_mp.download_to_mirror(overwrite=row_params.overwrite)

            # Run inference and block until done
            subprocess.run(
                [
                    *python_path,
                    "-u",
                    f"{prefix}tools/infer.py",
                    "-c",
                    config_path,
                    "-o",
                    f"weights={weights_path}",
                    "--infer_dir",
                    preproc_folder_mp.local_path,
                    "--output_dir",
                    annotated_folder_mp.local_path,
                    "--draw_threshold",
                    str(draw_thresh),
                    "--save_results",
                    "true",
                ]
            )

            # Upload the inference mask json
            new_inference_json_path = (
                inference_mask_json_mp.local_path.parent.parent / f"segm-{camera}.json"
            )
            shutil.move(
                inference_mask_json_mp.local_path,
                new_inference_json_path,
            )
            inference_mask_json_mp = MirrorPath.from_local_path(new_inference_json_path)
            inference_mask_json_mp.upload_from_mirror(overwrite=row_params.overwrite)

            # Upload the annotated images
            for img_path in annotated_folder_mp.local_path.iterdir():
                img_mp = MirrorPath.from_local_path(img_path)
                img_mp.upload_from_mirror(overwrite=row_params.overwrite)

    # Upload the database object
    row_params.update_process_flag_and_push_to_ddb(ProcessFlags.FRAMES_INFERENCED, True)


def run_paddle_inference_lambda(row_params: RowParams):
    """Invoke the paddle inference lambda function."""
    if row_params.ds_split_numbers:
        # Use the provided DS splits
        ds_split_numbers = row_params.ds_split_numbers

        if not row_params.overwrite:
            # Check if the other splits have their inference paths set already
            modified = False
            for plant_group in row_params.grdrow.plant_groups:
                if plant_group.ds_split_number in ds_split_numbers:
                    # Skip the provided DS splits
                    continue
                if (
                    plant_group.inferenced_images is None
                    and plant_group.preprocessed_images is not None
                ):
                    # Check if inferenced folder is present on S3
                    annotated_multi_view_mp = MultiViewMP.from_root_mp(
                        plant_group.preprocessed_images.get_root_mp(),
                        f" {DataFolders.RAW_IMAGES}/{DataFolders.ANNOTATED_IMAGES}",
                    )
                    # Check if the folder is present on S3, and if so, add it to the databases
                    if annotated_multi_view_mp.nadir_mp.exists_on_s3():
                        plant_group.inferenced_images = annotated_multi_view_mp
                        print(
                            f"Found inferenced images for DS Split {plant_group.ds_split_number}, adding to database."
                        )
                        modified = True
            if modified:
                # Update the database
                row_params.push_to_ddb()

    else:
        # Use all DS splits
        ds_split_numbers = [
            plant_group.ds_split_number
            for plant_group in row_params.grdrow.plant_groups
        ]

    ds_split_numbers_list = [
        ds_split_numbers[i : i + N_DS_SPLITS_PER_LAMBDA]
        for i in range(0, len(ds_split_numbers), N_DS_SPLITS_PER_LAMBDA)
    ]

    # Split the DS splits into groups of N_DS_SPLITS_PER_LAMBDA
    # ie for 54 splits and 8 splits per, we'll invoke the lambda 6 times with 8 splits each and once with 6 splits
    for ds_split_numbers in ds_split_numbers_list:
        # Params to pass to the lambda function
        params = {
            "field_name": row_params.field_name,
            "date": row_params.date,
            "row_number": str(row_params.row_number),
            # Convert to comma-separated string
            "ds_split_numbers": ",".join(
                [str(ds_split_number) for ds_split_number in ds_split_numbers]
            ),
            "nadir_crop_height": str(row_params.nadir_crop_height)
            if row_params.nadir_crop_height
            else "",
            "overwrite": str(row_params.overwrite),
            "rerun": str(row_params.rerun),
        }
        # TODO: detect env
        print(f"Invoking paddle inference lambda for DS Splits {ds_split_numbers}")
        # Run async so that if the lambda needs to retry, we can exit and open up space on ecs
        invoke_lambda(LambdaFunctionNames.RUN_INFERENCE, params, run_async=True)
