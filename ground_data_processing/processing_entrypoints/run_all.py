"""Run all rogues processing jobs in order."""
import argparse
import sys

from tqdm import tqdm

from ground_data_processing.params import FieldParams, RowParams
from ground_data_processing.processing_entrypoints.base import error_handler
from ground_data_processing.processing_steps.clip_images_col_spread import (
    extract_frames_from_row,
)
from ground_data_processing.processing_steps.ds_splits import generate_ds_splits
from ground_data_processing.processing_steps.generate_offset_from_gps_timestamp import (
    generate_offset_from_gps_timestamp,
)
from ground_data_processing.processing_steps.prep_unfiltered_inference import (
    prep_unfiltered_inference,
)
from ground_data_processing.processing_steps.run_paddle_inference import (
    run_paddle_inference_lambda,
)
from ground_data_processing.utils.ecs_utils import (
    safely_reset_autoscaling_group_desired_and_max_capacity,
    safely_set_scale_in_protection,
)
from ground_data_processing.utils.s3_constants import ProcessingSteps

FUNCTION_MAPPINGS = {
    ProcessingSteps.GENERATE_GPS_OFFSETS: generate_offset_from_gps_timestamp,
    ProcessingSteps.GENERATE_DS_SPLITS: generate_ds_splits,
    ProcessingSteps.EXTRACT_FRAMES: extract_frames_from_row,
    ProcessingSteps.PREP_INFERENCE: prep_unfiltered_inference,
    ProcessingSteps.RUN_INFERENCE: run_paddle_inference_lambda,  # Invoke a new job using a different docker image for inference
}

if __name__ == "__main__":
    """Parse command line args and run all processing steps."""
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

    if any([not args.field_name, not args.date]):
        raise ValueError("Must provide field_name and date.")

    # Validate date format
    if len(args.date) < len("YYYY-MM-DD"):
        raise ValueError(f"Date {args.date} must be in YYYY-MM-DD format.")

    # Convert overwrite to boolean
    args.overwrite = str(args.overwrite).lower() == "true"
    args.rerun = str(args.rerun).lower() == "true"

    # Convert ds_split_numbers to list of ints
    if args.ds_split_numbers is not None and args.ds_split_numbers != "":
        print(
            "WARNING: ds_split_numbers is not currently supported for run_all.py. Running the full pipeline for all DS splits.."
        )

    # Convert nadir_crop_height to int, or set to default
    if args.nadir_crop_height is not None and args.nadir_crop_height != "":
        args.nadir_crop_height = int(args.nadir_crop_height)
    else:
        args.nadir_crop_height = None

    # Convert nadir_crop_width to int, or set to default
    if args.nadir_crop_width is not None and args.nadir_crop_width != "":
        args.nadir_crop_width = int(args.nadir_crop_width)
    else:
        args.nadir_crop_width = None

    # Set scale in protection
    safely_set_scale_in_protection(True)

    if args.row_number:
        # Run for a single row
        args.row_number = int(args.row_number)
        row_params = RowParams(
            field_name=args.field_name,
            date=args.date,
            row_number=args.row_number,
            processing_step=ProcessingSteps.RUN_ALL,
            nadir_crop_height=args.nadir_crop_height,
            nadir_crop_width=args.nadir_crop_width,
            overwrite=args.overwrite,
            rerun=args.rerun,
        )
        for display_name, process_function in FUNCTION_MAPPINGS.items():
            try:
                if display_name not in row_params.all_incomplete_processes:
                    print(
                        f"Skipping '{display_name}' for row {args.row_number}: already complete."
                    )
                    continue
                print(
                    f"\nRunning '{display_name}' for row {args.row_number} of {args.field_name} on {args.date}\n"
                )
                process_function(row_params)
            except Exception as e:
                # If error occurs, send notification and stop processing this row
                error_handler(row_params, display_name, e)
                break
    else:
        # Run for all rows in the field
        field_params = FieldParams(
            field_name=args.field_name,
            date=args.date,
            processing_step=ProcessingSteps.RUN_ALL,
            nadir_crop_height=args.nadir_crop_height,
            nadir_crop_width=args.nadir_crop_width,
            overwrite=args.overwrite,
            rerun=args.rerun,
        )
        for display_name, process_function in FUNCTION_MAPPINGS.items():
            print(
                f"\nRunning '{display_name}' for all rows of {args.field_name} on {args.date}\n"
            )
            for row_params in tqdm(
                field_params.row_params,
                desc=f"\n{args.field_name} '{display_name}' progress",
            ):
                try:
                    if display_name not in row_params.all_incomplete_processes:
                        print(
                            f"Skipping '{display_name}' for row {args.row_number}: already complete."
                        )
                        continue
                    print(f"Running '{display_name}' for row {row_params.row_number}")
                    process_function(row_params)
                except Exception as e:
                    # If error occurs, send notification and stop processing this row
                    error_handler(row_params, display_name, e)
                    continue

    # Disable scale in protection
    safely_set_scale_in_protection(False)
    # Reset autoscaling group desired and max capacity to zero if this is the only active task
    safely_reset_autoscaling_group_desired_and_max_capacity()
