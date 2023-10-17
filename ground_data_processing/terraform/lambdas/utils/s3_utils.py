"""S3 Utils."""

import boto3


def get_s3_resource_if_not_exists(s3_resource=None):
    """Get an S3 resource if one is not provided."""
    if s3_resource is None:
        s3_resource = boto3.resource("s3")
    return s3_resource


def upload_file_to_s3(path: str, bucket_name: str, key: str, s3_resource=None):
    """Upload a file to S3."""
    s3_resource = get_s3_resource_if_not_exists(s3_resource)
    bucket = s3_resource.Bucket(bucket_name)
    bucket.upload_file(path, key)
