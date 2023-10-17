"""Set global values for S3MP module."""
import tempfile
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import boto3


def get_config_file_path() -> Path:
    """Get the location of the config file."""
    root_module_folder = Path(__file__).parent.resolve()
    return root_module_folder / "config.ini"


class Singleton(type):
    """Singleton metaclass."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Get instance of class."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class S3MPConfig(metaclass=Singleton):
    """Singleton class for S3MP globals."""

    # Boto3 Objects
    _s3_client = None
    _s3_resource = None
    _bucket = None

    # Config Items
    default_bucket_key: str = "sentera-rogues-data"
    _mirror_root: Path = None

    # Other Items
    transfer_config = None
    callback: Callable = None
    use_async_global_thread_queue: bool = True

    @property
    def s3_client(self):
        """Get S3 client."""
        if not self._s3_client:
            self._s3_client = boto3.client("s3")
        return self._s3_client

    @property
    def s3_resource(self):
        """Get S3 resource."""
        if not self._s3_resource:
            self._s3_resource = boto3.resource("s3")
        return self._s3_resource

    @property
    def bucket(self, bucket_key: str = None):
        """Get bucket."""
        if bucket_key:
            return self.s3_resource.Bucket(bucket_key)
        elif self._bucket is None:
            if self.default_bucket_key is None:
                raise ValueError("No default bucket key set.")
            self._bucket = self.s3_resource.Bucket(self.default_bucket_key)
        return self._bucket

    @property
    def mirror_root(self) -> Path:
        """Get mirror root."""
        if self._mirror_root is None:
            print(
                "Mirror Root not set, a temporary directory will be used as the mirror root."
            )
            self._mirror_root = Path(tempfile.gettempdir())
        return self._mirror_root

    def load_config(self, config_file_path: Path = None):
        """Load the config file."""
        config_file_path = config_file_path or get_config_file_path()
        config = ConfigParser()
        config.read(config_file_path)

        if "DEFAULT" not in config:
            return

        if "default_bucket_key" in config["DEFAULT"]:
            self.default_bucket_key = config["DEFAULT"]["default_bucket_key"]

        if "mirror_root" in config["DEFAULT"]:
            self._mirror_root = Path(config["DEFAULT"]["mirror_root"])

    def save_config(self, config_file_path: Path = None):
        """Write config file."""
        config_file_path = config_file_path or get_config_file_path()
        config = ConfigParser()
        config["DEFAULT"] = {}
        if self.default_bucket_key:
            config["DEFAULT"]["default_bucket_key"] = self.default_bucket_key
        if self._mirror_root:
            config["DEFAULT"]["mirror_root"] = str(self._mirror_root)
        with open(config_file_path, "w") as configfile:
            config.write(configfile)


S3MPConfig = S3MPConfig()
S3MPConfig.load_config()
