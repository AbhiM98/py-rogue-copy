"""Generate an offset file from GPS timestamps in the video files."""
from ground_data_processing.processing_entrypoints.base import base_entrypoint
from ground_data_processing.processing_steps.generate_offset_from_gps_timestamp import (
    generate_offset_from_gps_timestamp,
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
    python ./ground_data_processing/processing_entrypoints/generate_gps_offsets.py --field_name "Foundation Field 2 (Dennis Zuber)" --date "2022-05-19"
    """
    base_entrypoint(
        generate_offset_from_gps_timestamp, ProcessingSteps.GENERATE_GPS_OFFSETS
    )


if __name__ == "__main__":
    main()
