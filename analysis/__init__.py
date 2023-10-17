"""Initialize the features of the module."""
import os

from S3MP.global_config import S3MPConfig

# Set global configuration
ROGUES_BUCKET_KEY = "sentera-rogues-data"
S3MPConfig.default_bucket_key = ROGUES_BUCKET_KEY
MIRROR_ROOT = "/".join(os.getcwd().split("/")[:2])  # /home/user/
MIRROR_ROOT = f"{MIRROR_ROOT}/s3_mirror/"
S3MPConfig.mirror_root = MIRROR_ROOT

# set up some additional directories
if not os.path.exists("jsons"):
    os.makedirs("jsons")
if not os.path.exists("s3_mirror"):
    os.makedirs("s3_mirror")
if not os.path.exists("models"):
    os.makedirs("models")
