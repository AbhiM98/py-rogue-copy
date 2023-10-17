"""Run Paddle Inference."""
from ground_data_processing.processing_entrypoints.base import base_entrypoint
from ground_data_processing.processing_steps.run_paddle_inference import (
    run_paddle_inference,
)
from ground_data_processing.utils.s3_constants import ProcessingSteps


def main():
    """Parse command line args and run.

    NOTE: PaddleDetection must be installed and the model must be present before running this script.

    Args:
        --field_name: Name of the field to process.
        --date: Date of the field to process. MM-DD format.
        --row_number: Row number to process. If not provided, process all rows in the field.
        --ds_split_numbers: DS split numbers to process. If not provided, process all DS splits in the row. List[int]
        --overwrite: If "True" is present, overwrite existing files.
        --rerun: If "True" is present, execute in rerun mode.

    Example usage:
    python ./ground_data_processing/processing_entrypoints/run_inference.py --field_name "Foundation Field 2 (Dennis Zuber)" --date "2022-05-19"
    """
    base_entrypoint(run_paddle_inference, ProcessingSteps.RUN_INFERENCE)


if __name__ == "__main__":
    main()
