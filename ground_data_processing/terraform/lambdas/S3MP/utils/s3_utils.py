"""Utilities for working with S3."""
import warnings
from pathlib import Path

from S3MP.global_config import S3MPConfig


def s3_list_single_key(
    key: str,
    bucket=None,
    client=None,
):
    """List details of a single key on S3."""
    bucket = bucket or S3MPConfig.bucket
    client = client or S3MPConfig.s3_client
    return client.list_objects_v2(
        Bucket=bucket.name, Prefix=key, Delimiter="/", MaxKeys=1
    )


def s3_list_child_keys(
    key: str,
    bucket=None,
    client=None,
):
    """List details of all child keys on S3."""
    if not key.endswith("/"):
        warnings.warn(f"Listing child keys of {key} - key does not end with '/'")
    bucket = bucket or S3MPConfig.bucket
    client = client or S3MPConfig.s3_client
    return client.list_objects_v2(Bucket=bucket.name, Prefix=key, Delimiter="/")


def download_key(
    key: str,
    local_path: Path,
    bucket=None,
    client=None,
) -> None:
    """Download a key from S3."""
    bucket = bucket or S3MPConfig.bucket
    client = client or S3MPConfig.s3_client
    if key_is_file_on_s3(key, bucket, client):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(
            bucket.name,
            key,
            str(local_path),
            Callback=S3MPConfig.callback,
            Config=S3MPConfig.transfer_config,
        )
    else:
        for obj in s3_list_child_keys(key, bucket, client)["Contents"]:
            download_key(obj["Key"], local_path / obj["Key"].replace(key, ""))


def upload_to_key(
    key: str,
    local_path: Path,
    bucket=None,
    client=None,
) -> None:
    """Upload a file or folder to a key on S3."""
    bucket = bucket or S3MPConfig.bucket
    client = client or S3MPConfig.s3_client
    if local_path.is_file():
        client.upload_file(
            str(local_path),
            bucket.name,
            key,
            Callback=S3MPConfig.callback,
            Config=S3MPConfig.transfer_config,
        )
    else:
        for child in local_path.iterdir():
            upload_to_key(f"{key}/{child.name}", child, bucket, client)


def key_exists_on_s3(
    key: str,
    bucket=None,
    client=None,
) -> bool:
    """Check if a key exists on S3."""
    bucket = bucket or S3MPConfig.bucket
    client = client or S3MPConfig.s3_client
    res = s3_list_single_key(key, bucket, client)
    return "Contents" in res or "CommonPrefixes" in res


def key_is_file_on_s3(
    key: str,
    bucket=None,
    client=None,
) -> bool:
    """Check if a key is a file on S3, returns false if it is a folder. Raises an error if the key does not exist."""
    bucket = bucket or S3MPConfig.bucket
    client = client or S3MPConfig.s3_client
    if not key_exists_on_s3(key, bucket, client):
        raise ValueError("Key does not exist on S3")
    res = s3_list_single_key(key, bucket, client)
    # Handle case of trailing slash, but still verify
    if (
        key[-1] == "/"
        and len(res["Contents"]) == 1
        and res["Contents"][0]["Key"] == key
    ):
        return False
    return "Contents" in res


def key_size_on_s3(
    key: str,
    bucket=None,
    client=None,
) -> int:
    """Get the size of a key on S3. Raises an error if the key does not exist."""
    bucket = bucket or S3MPConfig.bucket
    client = client or S3MPConfig.s3_client
    if not key_exists_on_s3(key, bucket, client):
        raise ValueError("Key does not exist on S3")
    res = s3_list_single_key(key, bucket, client)
    return res["Contents"]["Size"] if "Contents" in res else 0


def delete_child_keys_on_s3(
    key: str,
    bucket=None,
    client=None,
) -> None:
    """Delete all keys that are children of a key on S3."""
    bucket = bucket or S3MPConfig.bucket
    client = client or S3MPConfig.s3_client
    for obj in s3_list_child_keys(key, bucket, client)["Contents"]:
        client.delete_object(Bucket=bucket.name, Key=obj["Key"])


def delete_key_on_s3(
    key: str,
    bucket=None,
    client=None,
) -> None:
    """Delete a key on S3."""
    bucket = bucket or S3MPConfig.bucket
    client = client or S3MPConfig.s3_client
    if not key_exists_on_s3(key, bucket, client):
        return
    if key_is_file_on_s3(key, bucket, client):
        client.delete_object(Bucket=bucket.name, Key=key)
    else:
        delete_child_keys_on_s3(key, bucket, client)
