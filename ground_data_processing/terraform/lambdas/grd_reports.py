"""Reports for the ground rogues database."""
import tempfile

from utils.s3_utils import upload_file_to_s3

from ddb_tracking.grd_api import key_query_grd_rows
from ddb_tracking.grd_constants import GRD_S3_BUCKET_NAME


def write_report_csv(event, context):
    """Write a report CSV."""
    col_names = [
        "Field Name",
        "Row Number",
        "Date",
        "Videos Split",
        "Frames Extracted",
        "Extract Timestamp",
        "Frames Prepared",
        "Inference Complete",
        "Inference Timestamp",
    ]

    field_names = []
    row_numbers = []
    dates = []
    videos_split = []
    frames_extracted = []
    extract_timestamps = []
    frames_prepared = []
    inference_completes = []
    inference_timestamps = []

    # Get all rows
    grd_rows = key_query_grd_rows()
    print(len(grd_rows))
    # Use only 2023 rows
    grd_rows = [row for row in grd_rows if row.date_row_number.startswith("2023")]
    # Sort by date_row_number, most recent first
    # grd_rows.sort(key=lambda x: (x.field_name, x.date_row_number))
    grd_rows.sort(key=lambda x: x.date_row_number, reverse=True)

    rows = [col_names]
    for grd_row in grd_rows:
        field_names.append(grd_row.field_name)
        row_numbers.append(str(grd_row.row_number))
        dates.append(grd_row.date)
        videos_split.append(str(grd_row.videos_split.value))
        frames_extracted.append(str(grd_row.frames_extracted.value))
        extract_timestamps.append(grd_row.frames_extracted.timestamp)
        frames_prepared.append(str(grd_row.frames_prepared.value))
        inference_completes.append(str(grd_row.frames_inferenced.value))
        inference_timestamps.append(grd_row.frames_inferenced.timestamp)

    rows.extend(
        list(
            zip(
                field_names,
                row_numbers,
                dates,
                videos_split,
                frames_extracted,
                extract_timestamps,
                frames_prepared,
                inference_completes,
                inference_timestamps,
            )
        )
    )
    filename = "2023-field-data/rogues_report.csv"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    with open(temp.name, "w") as fd:
        for row in rows:
            fd.write(",".join([str(x) for x in row]) + "\n")
    upload_file_to_s3(temp.name, GRD_S3_BUCKET_NAME, filename)
    print("Done")


if __name__ == "__main__":
    write_report_csv(None, None)
