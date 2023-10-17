"""Lambdas for s3 monitoring."""
# NOTE: for simplicity, this uses ddb_tracking and S3MP found in the
# terraform folder. The alternative would be to use the hacky approach
# from mfstand which involves bundling in a bunch of dependencies into the zip,
# which is more trouble that it's worth for this use case. Also note that the S3MP
# and ddb_tracking modules used here are slightly different than the ones found elsewhere,
# since imports that depend on packages not natively supported by AWS Lambda have been removed.
# This mostly just affects some constants and typing and all of the core functionality should be the same.

from urllib.parse import unquote_plus

from S3MP.mirror_path import MirrorPath
from utils.lambda_utils import invoke_lambda
from utils.sns_utils import ROGUES_GRD_ADDED, publish_message_to_name

from ddb_tracking.grd_api import get_grd_row_from_full_video_s3_key, put_grd_row
from ddb_tracking.grd_constants import CAMERA_VIEWS
from ddb_tracking.grd_structure import GRDRow, MultiViewMP

BLACKLISTED_FOLDERS = ["2023-rgbd-data"]


def invoke_processing_lambda(grd_row: GRDRow):
    """Invoke the processing lambda for a row."""
    if should_process_row(grd_row):
        print("Invoking processing lambda...")
        params = {
            "field_name": grd_row.field_name,
            "date": grd_row.date,
            "row_number": str(grd_row.row_number),
        }
        ret = invoke_lambda("run_all", params, run_async=True)
        print(ret)

    publish_message_to_name(
        ROGUES_GRD_ADDED,
        f"ROGUES GRDROW ADDED: {grd_row.field_name} {grd_row.date} timestamp {grd_row.version_tracker.get_latest_version().timestamp_str} row {grd_row.row_number}",
    )


def should_process_row(grd_row: GRDRow) -> bool:
    """Check if a row should be processed (not blacklisted)."""
    # Check if the root folder is blacklisted
    grd_root_folder = grd_row.full_row_video_mps.bottom_mp.s3_key.split("/")[0]
    print(f"Checking if {grd_root_folder} is blacklisted...")
    if grd_root_folder in BLACKLISTED_FOLDERS:
        print(
            f"Row {grd_row.row_number} of field {grd_row.field_name} on {grd_row.date} is blacklisted. Skipping..."
        )
        return False
    return True


def s3_monitoring(event, context):
    """Lambda for s3 monitoring."""
    s3_key = unquote_plus(event["Records"][0]["s3"]["object"]["key"])
    file_name = s3_key.split("/")[-1]

    if any(f"{view}.mp4" == file_name for view in CAMERA_VIEWS):
        # File is a raw camera view, check if all views are present
        root_mp = MirrorPath.from_s3_key(s3_key).get_parent()
        if "DS Split" in root_mp.s3_key:
            print(f"Root path {root_mp.s3_key} is a DS Split, skipping...")
            return

        multiview_mp = MultiViewMP.from_root_mp(root_mp)
        if not multiview_mp.all_exists():
            print(f"Not all camera views are present for {s3_key}. Skipping...")
            return

        # Check if row exists in database
        print(s3_key)
        new_grd_row = GRDRow.from_s3_key(s3_key)
        existing_grd_row = get_grd_row_from_full_video_s3_key(s3_key)
        if existing_grd_row is None:
            # Row doesn't exist, add it
            print("Adding new row to database...")
            put_grd_row(new_grd_row)
            # Invoke the processing lambda
            invoke_processing_lambda(new_grd_row)
        elif existing_grd_row.version_tracker and new_grd_row.version_tracker:
            # Row exists and is versioned, check if new version is already in database
            if existing_grd_row.version_tracker.contains_version(
                new_grd_row.version_tracker.get_latest_version()
            ):
                print("Row version already exists in database. Skipping...")
                return
            else:
                # Add new version to database
                print("Adding new row version to database...")
                # Will add version and set it if it is the latest version
                existing_grd_row.add_and_set_version(
                    new_grd_row.version_tracker.get_latest_version()
                )
                put_grd_row(existing_grd_row)
                if (
                    existing_grd_row.version_tracker.get_latest_version()
                    == new_grd_row.version_tracker.get_latest_version()
                ):
                    print(
                        "New version is the latest version, invoking processing lambda..."
                    )
                    # Invoke the processing lambda
                    invoke_processing_lambda(new_grd_row)
        else:
            print("Unversioned Row already exists in database. Skipping...")

    else:
        print(f"{file_name} is not a raw camera view. Skipping...")
