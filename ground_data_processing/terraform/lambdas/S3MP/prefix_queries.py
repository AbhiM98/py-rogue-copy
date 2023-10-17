"""S3 prefix queries.."""
from __future__ import annotations

from typing import List

from S3MP.global_config import S3MPConfig


def get_prefix_paginator(folder_key: str, bucket_key: str = None, delimiter: str = "/"):
    """Get a paginator for a specified prefix."""
    if not bucket_key:
        bucket_key = S3MPConfig.default_bucket_key
    if folder_key != "" and folder_key[-1] != "/":
        folder_key += "/"
    s3_client = S3MPConfig.s3_client
    paginator = s3_client.get_paginator("list_objects_v2")
    return paginator.paginate(Bucket=bucket_key, Prefix=folder_key, Delimiter=delimiter)


def get_files_within_folder(folder_key: str, key_filter: str = None) -> List[str]:
    """Get files within a folder."""
    for page in get_prefix_paginator(folder_key):
        if "Contents" in page:
            for obj in page["Contents"]:
                obj = obj["Key"].replace(folder_key, "")
                if key_filter and key_filter not in obj:
                    continue
                yield obj


def get_folders_within_folder(folder_key: str, key_filter: str = None) -> List[str]:
    """Get folders within folder."""
    for page in get_prefix_paginator(folder_key):
        if "CommonPrefixes" in page:
            for obj in page["CommonPrefixes"]:
                obj = obj["Prefix"].replace(folder_key, "")
                if key_filter and key_filter not in obj:
                    continue
                yield obj
