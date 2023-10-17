"""S3 Utils for GRD."""
import os

import boto3

from ddb_tracking.grd_constants import GRD_S3_BUCKET_NAME


def get_n_files_in_folder(
    folder_path: str, ext: str, bucket_name: str = GRD_S3_BUCKET_NAME, client=None
) -> int:
    """Get the number of files in an S3 subfolder with the specified extension."""
    if not folder_path.endswith("/"):
        folder_path += "/"
    if not client:
        client = boto3.client("s3")
    try:
        result = client.list_objects(
            Bucket=bucket_name, Prefix=folder_path, Delimiter="/"
        )
        keys = [
            os.path.basename(os.path.normpath(o.get("Key")))
            for o in result.get("Contents")
        ]
    except TypeError:
        # No objects in folder
        return 0

    return len([key for key in keys if key.endswith(ext)])


def get_n_folders_in_folder(
    folder_path: str, bucket_name: str = GRD_S3_BUCKET_NAME, client=None
) -> int:
    """Get the number of folders in an S3 subfolder."""
    if not folder_path.endswith("/"):
        folder_path += "/"
    if not client:
        client = boto3.client("s3")

    result = client.list_objects(Bucket=bucket_name, Prefix=folder_path, Delimiter="/")
    try:
        return len([o.get("Prefix") for o in result.get("CommonPrefixes")])
    except TypeError:
        # No folders in folder
        return 0


if __name__ == "__main__":
    prefix = "Foundation Field 2 (Dennis Zuber)/Videos/7-08/"
    print(get_n_folders_in_folder(prefix))
