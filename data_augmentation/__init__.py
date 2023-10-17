"""Data augmentation package."""
import os

from S3MP.global_config import S3MPConfig

# Set global configuration
S3MPConfig.default_bucket_key = "cvdb-data"
MIRROR_ROOT = "/".join(os.getcwd().split("/")[:3])  # /home/user/
MIRROR_ROOT = f"{MIRROR_ROOT}/s3_mirror/"
S3MPConfig.mirror_root = MIRROR_ROOT
