"""Base structure for ground rogues processing jobs."""
import argparse
import sys
from typing import Callable

from tqdm import tqdm

from ground_data_processing.params import FieldParams, RowParams
from ground_data_processing.utils.ecs_utils import (
    safely_reset_autoscaling_group_desired_and_max_capacity,
    safely_set_scale_in_protection,
)
from ground_data_processing.utils.lambda_utils import invoke_lambda
from ground_data_processing.utils.s3_constants import (
    LambdaFunctionNames,
    ProcessingSteps,
)


def error_handler(row_params: RowParams, display_name: ProcessingSteps, e: Exception):
    """Print error and send a notification via sns."""
    print(f"Error running '{display_name}' for row {row_params.row_number}: {e}")

    # Send SNS message
    params = {
        "name": "rogues-task-failure",
        "message": f"Row {row_params.row_number} from {row_params.field_name} on {row_params.date} failed during {display_name}. Error: {e}",
    }
    invoke_lambda(LambdaFunctionNames.SEND_SNS, params)


def base_entrypoint(
    process_function: Callable, display_name: ProcessingSteps
) -> RowParams:
    """Parse command line args and run a processing step."""
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
        args.ds_split_numbers = [
            int(ds_split_number) for ds_split_number in args.ds_split_numbers.split(",")
        ]

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
        print(
            f"\nRunning '{display_name}' for row {args.row_number} of {args.field_name} on {args.date}\n"
        )
        try:
            row_params = RowParams(
                field_name=args.field_name,
                date=args.date,
                row_number=args.row_number,
                processing_step=display_name,
                ds_split_numbers=args.ds_split_numbers,
                nadir_crop_height=args.nadir_crop_height,
                nadir_crop_width=args.nadir_crop_width,
                overwrite=args.overwrite,
                rerun=args.rerun,
            )
            process_function(row_params)
            return row_params
        except Exception as e:
            # If error occurs, send notification and stop processing this row
            error_handler(row_params, display_name, e)
            return None
    else:
        # Run for all rows in the field
        print(
            f"\nRunning '{display_name}' for all rows of {args.field_name} on {args.date}\n"
        )
        if args.ds_split_numbers is not None:
            print(
                "WARNING: ds_split_numbers argument will be ignored when running for all rows."
            )
        field_params = FieldParams(
            field_name=args.field_name,
            date=args.date,
            processing_step=display_name,
            nadir_crop_height=args.nadir_crop_height,
            nadir_crop_width=args.nadir_crop_width,
            overwrite=args.overwrite,
            rerun=args.rerun,
        )
        for row_params in tqdm(
            field_params.row_params,
            desc=f"\n{args.field_name} '{display_name}' progress",
        ):
            try:
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
