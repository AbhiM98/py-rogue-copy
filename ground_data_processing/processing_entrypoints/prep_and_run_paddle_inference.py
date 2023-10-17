"""Run Unfiltered Inference Prep, followed by invoking the Paddle Inference Lambda."""
from ground_data_processing.processing_entrypoints.base import base_entrypoint
from ground_data_processing.processing_steps.prep_unfiltered_inference import (
    prep_unfiltered_inference,
)
from ground_data_processing.processing_steps.run_paddle_inference import (
    run_paddle_inference_lambda,
)
from ground_data_processing.utils.s3_constants import ProcessingSteps


def main():
    """Parse command line args and run.

    Args:
        --field_name: Name of the field to process.
        --date: Date of the field to process. MM-DD format.
        --row_number: Row number to process. If not provided, process all rows in the field.
        --ds_split_numbers: DS split numbers to process. If not provided, process all DS splits in the row. List[int]
        --overwrite: If "True" is present, overwrite existing files.
        --rerun: If "True" is present, execute in rerun mode.

    Example usage:
    python ./ground_data_processing/processing_entrypoints/prep_and_run_paddle_inference.py --field_name "Foundation Field 2 (Dennis Zuber)" --date "2022-05-19"
    """
    # Run prep_unfiltered_inference
    row_params = base_entrypoint(
        prep_unfiltered_inference, ProcessingSteps.PREP_AND_RUN_INFERENCE
    )

    if row_params:
        # Invoke run-paddle-inference lambda
        run_paddle_inference_lambda(row_params)


if __name__ == "__main__":
    main()
