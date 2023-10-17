"""Run paddle inference on the images in a row. A modified version of run_paddle_inference.py in ground_data_processing dir."""
import argparse
import shutil
import subprocess
import sys
from typing import List

from S3MP.mirror_path import MirrorPath
from tqdm import tqdm
from utils.ecs_utils import (
    safely_reset_autoscaling_group_desired_and_max_capacity,
    safely_set_scale_in_protection,
)
from utils.sns_utils import (
    ROGUES_INFERENCE_COMPLETE,
    ROGUES_TASK_FAILURE,
    publish_message_to_name,
)

from ddb_tracking.grd_api import get_grd_row, put_grd_row
from ddb_tracking.grd_constants import DataFolders
from ddb_tracking.grd_structure import GRDRow, MultiViewMP


def run_paddle_inference(
    grdrow: GRDRow,
    ds_split_numbers: List[int] = None,
    overwrite: bool = False,
    nadir_only: bool = False,
):
    """Run paddle inference on the images in a row.

    Requires having the `PaddleDetection` repo setup with the SoloV2 model config/params files present.
    """
    python_path = ["conda", "run", "-n", "paddle-venv", "python"]
    prefix = "./PaddleDetection/"  # Prefix for running in docker container
    config_path = f"{prefix}configs/solov2/solov2_r101_vd_fpn_3x_coco.yml"
    coco_config_path = f"{prefix}configs/datasets/coco_instance.yml"
    draw_thresh = 0.25

    # Infer all DS splits if none are specified
    if ds_split_numbers is None:
        plant_groups = grdrow.plant_groups
        print("Processing all DS Splits...")
    else:
        plant_groups = [
            plant_group
            for plant_group in grdrow.plant_groups
            if plant_group.ds_split_number in ds_split_numbers
        ]
        print(f"Processing only DS Splits {ds_split_numbers}...")

    # Run inference for each plant group
    for grd_plant_group in tqdm(plant_groups):
        print(f"Processing DS Split {grd_plant_group.ds_split_number}...")

        # Store the inferenced image paths in the database object
        grd_plant_group.inferenced_images = MultiViewMP(
            bottom_mp=grd_plant_group.preprocessed_images.bottom_mp.get_sibling(
                DataFolders.ANNOTATED_IMAGES
            ),
            nadir_mp=grd_plant_group.preprocessed_images.nadir_mp.get_sibling(
                DataFolders.ANNOTATED_IMAGES
            ),
            oblique_mp=grd_plant_group.preprocessed_images.oblique_mp.get_sibling(
                DataFolders.ANNOTATED_IMAGES
            ),
        )

        # Inference each camera view
        for (
            camera,
            preproc_folder_mp,
        ) in grd_plant_group.preprocessed_images.as_dict().items():
            print(f"Processing {camera} view...")

            if not preproc_folder_mp.exists_on_s3():
                print(
                    f"{camera} folder {preproc_folder_mp.s3_key} not present on S3, skipping."
                )
                continue

            # Check if preprocessed images already exist, and if the number of images present
            # matches the number of images in the raw folder
            annotated_folder_mp = grd_plant_group.inferenced_images.as_dict()[camera]
            n_images_extected = len(list(preproc_folder_mp.get_children_on_s3()))
            n_images_present = len(list(annotated_folder_mp.get_children_on_s3()))
            print(
                f"Found {n_images_present} images in {camera} view, expected {n_images_extected}."
            )

            if not overwrite and n_images_present == n_images_extected:
                print(f"Skipping {camera} (already processed).")
                continue  # Skip if already processed

            # Skip if not nadir and all images are present, even if overwrite is True
            if (
                nadir_only
                and camera != "nadir"
                and n_images_present == n_images_extected
            ):
                print(f"Skipping {camera} (not nadir).")
                continue

            if overwrite or n_images_present != n_images_extected:
                # Delete old images
                [mp.delete_all() for mp in annotated_folder_mp.get_children_on_s3()]
                annotated_folder_mp.local_path.mkdir(parents=True, exist_ok=True)

            # Use the correct weights
            if camera == "bottom":
                weights_path = f"{prefix}output/solov2_r101_vd_fpn_3x_coco/solo_v2_sept_05_2023_stalk_tiller_braces.pdparams"
                # Save the stalk_tiller_braces.yml as coco_instance.yml
                shutil.copyfile(
                    f"{prefix}configs/datasets/stalk_tiller_braces.yml",
                    coco_config_path,
                )
            else:
                weights_path = f"{prefix}output/solov2_r101_vd_fpn_3x_coco/solo_v2_dec_11_2022_leaf_instance.pdparams"
                # Save the leaf_instance.yml as coco_instance.yml
                shutil.copyfile(
                    f"{prefix}configs/datasets/leaf_instance.yml", coco_config_path
                )

            # Default location of the inference mask json
            inference_mask_json_mp = annotated_folder_mp.get_child("segm.json")
            # New location of the inference mask json (one level above the anno folder)
            new_inference_json_path = (
                inference_mask_json_mp.local_path.parent.parent / f"segm-{camera}.json"
            )
            new_inference_mask_json_mp = MirrorPath.from_local_path(
                new_inference_json_path
            )

            for img_mp in preproc_folder_mp.get_children_on_s3():
                img_mp.download_to_mirror(overwrite=overwrite)

            print("Running inference...")
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
            shutil.move(
                inference_mask_json_mp.local_path,
                new_inference_json_path,
            )
            new_inference_mask_json_mp.upload_from_mirror(overwrite=overwrite)

            # Upload the annotated images
            for img_path in annotated_folder_mp.local_path.iterdir():
                img_mp = MirrorPath.from_local_path(img_path)
                img_mp.upload_from_mirror(overwrite=overwrite)

    # Update the grd row with the ds splits, and check if all ds splits have been inferenced
    get_and_update_grd_row(grdrow, ds_split_numbers)


def get_and_update_grd_row(grdrow: GRDRow, ds_split_numbers: List[int] = None):
    """Get the grd row and update only the specified ds splits to ensure data consistency."""
    # Update the grd row with the ds splits
    if ds_split_numbers is None:
        # Update all ds splits
        current_grdrow = grdrow
    else:
        # Get the most recent grd row from the database
        current_grdrow = get_grd_row(grdrow.field_name, grdrow.row_number, grdrow.date)
        for ds_split_number in ds_split_numbers:
            # Update the plant group at this index
            # Find the index of the plant group with this ds split number
            plant_group_index = next(
                (
                    index
                    for (index, plant_group) in enumerate(grdrow.plant_groups)
                    if plant_group.ds_split_number == ds_split_number
                ),
                None,
            )
            if plant_group_index is None:
                print(
                    f"WARNING: DS Split {ds_split_number} not found in grd row {grdrow.row_number} from {grdrow.field_name} on {grdrow.date}, skipping..."
                )
                continue

            current_grdrow.plant_groups[plant_group_index] = grdrow.plant_groups[
                plant_group_index
            ]

    # Check if all ds splits have been inferenced by seeing if the inferenced image paths are present
    if all(
        plant_group.inferenced_images for plant_group in current_grdrow.plant_groups
    ):
        # All ds splits have been inferenced, mark the row as inferenced
        current_grdrow.set_frames_inferenced(True)
        # Publish message to SNS
        publish_message_to_name(
            ROGUES_INFERENCE_COMPLETE,
            f"Row {grd_row.row_number} from {grd_row.field_name} on {grd_row.date} has finished inference.",
        )

    put_grd_row(current_grdrow)


def setup_cuda():
    """Run cuda setup script."""
    # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html
    print("Setting up cuda v11.2...")
    subprocess.run(
        [
            "echo",
            "wget",
            "https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run",
        ]
    )
    subprocess.run(["echo", "sudo", "sh", "cuda_11.2.0_460.27.04_linux.run"])
    subprocess.run(["echo", "chmod", "+x", "cuda_11.2.0_460.27.04_linux.run"])
    subprocess.run(
        [
            "echo",
            "sudo",
            "sh",
            "cuda_11.2.0_460.27.04_linux.run",
            "--silent",
            "--override",
            "--toolkit",
            "--samples",
            "--toolkitpath=/usr/local/cuda-11.2",
            "--samplespath=/usr/local/cuda",
            "--no-opengl-libs",
        ]
    )
    subprocess.run(
        [
            "echo",
            "sudo",
            "ln",
            "-s",
            "/usr/local/cuda-11.2",
            "/usr/local/cuda",
        ]
    )
    # https://github.com/PaddlePaddle/Paddle/issues/48681
    subprocess.run(
        [
            "echo",
            "export",
            "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/miniconda3/envs/paddle-venv/lib/",
        ]
    )
    print("Done setting up cuda v11.2")


if __name__ == "__main__":
    """Parse command line args and run inference."""
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--field_name", type=str, default=None)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--row_number", type=str, default=None)
    parser.add_argument(
        "--ds_split_numbers", type=str, default=None
    )  # Comma separated list of ints
    parser.add_argument("--nadir_crop_height", type=str, default=None)  # "2000"
    parser.add_argument("--nadir_crop_width", type=str, default=None)  # "2000"
    parser.add_argument(
        "--overwrite", type=str, default="False"
    )  # True if flag is present, False otherwise
    parser.add_argument(
        "--rerun", type=str, default="False"
    )  # True if flag is present, False otherwise
    args = parser.parse_args()
    print(args)

    if any([not args.field_name, not args.date, not args.row_number]):
        raise ValueError(
            "Must provide field_name, date, and row number for inferencing."
        )

    # Validate date format
    if len(args.date) < len("YYYY-MM-DD"):
        raise ValueError(f"Date {args.date} must be in YYYY-MM-DD format.")

    # Convert to boolean
    args.overwrite = str(args.overwrite).lower() == "true"
    args.rerun = str(args.rerun).lower() == "true"

    # Convert ds_split_numbers to list of ints
    if args.ds_split_numbers is not None and args.ds_split_numbers != "":
        args.ds_split_numbers = [
            int(ds_split_number) for ds_split_number in args.ds_split_numbers.split(",")
        ]

    # HACK: If nadir_crop_height is specified, we don't need to re-infer oblique
    nadir_only = False
    # if args.nadir_crop_height is not None and args.nadir_crop_height != "":
    #     nadir_only = True

    # Protect self from scale in
    safely_set_scale_in_protection(protect_from_scale_in=True)

    try:
        # Get the row and check if it has already been inferenced
        grd_row = get_grd_row(args.field_name, args.row_number, args.date)
        if not grd_row:
            raise ValueError(
                f"Row {args.row_number} from {args.field_name} on {args.date} does not exist in the database."
            )
        if not grd_row.frames_prepared:
            raise ValueError(
                f"Row {args.row_number} from {args.field_name} on {args.date} has not been prepared for inference."
            )
        if grd_row.frames_inferenced:
            raise ValueError(
                f"Row {args.row_number} from {args.field_name} on {args.date} has already been inferenced."
            )

        # Run inference
        print(
            f"Starting inference on row {grd_row.row_number} from {grd_row.field_name} on {grd_row.date}..."
        )
        run_paddle_inference(
            grd_row,
            ds_split_numbers=args.ds_split_numbers,
            overwrite=args.overwrite,
            nadir_only=nadir_only,
        )
    except Exception as e:
        # Send SNS message
        publish_message_to_name(
            ROGUES_TASK_FAILURE,
            f"Row {grd_row.row_number} from {grd_row.field_name} on {grd_row.date} DS Splits {args.ds_split_numbers} failed during inference. Error: {e}",
        )

    # Disable scale in protection
    safely_set_scale_in_protection(protect_from_scale_in=False)
    # Reset autoscaling group desired and max capacity to zero if this is the only active task
    safely_reset_autoscaling_group_desired_and_max_capacity()
