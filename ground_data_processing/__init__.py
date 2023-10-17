"""Ground data processing module."""
import os

from S3MP.global_config import S3MPConfig

# Set global configuration
ROGUES_BUCKET_KEY = "sentera-rogues-data"
S3MPConfig.default_bucket_key = ROGUES_BUCKET_KEY
MIRROR_ROOT = "/".join(os.getcwd().split("/")[:3])  # /home/user/
MIRROR_ROOT = f"{MIRROR_ROOT}/s3_mirror/"
S3MPConfig.mirror_root = MIRROR_ROOT
