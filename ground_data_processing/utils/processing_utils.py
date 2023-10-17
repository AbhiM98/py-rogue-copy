"""Utilities for running processing tasks."""
import datetime
import inspect
from pathlib import Path
from subprocess import PIPE, Popen

import ground_data_processing._version
from ddb_tracking.grd_constants import TimeZones


def get_version_str():
    """Return the version string for the ground_data_processing package."""
    version_str = ground_data_processing._version.__version__
    git_rev = (
        Popen("git rev-parse --short HEAD", shell=True, stdout=PIPE)
        .stdout.read()
        .decode("utf-8")
        .strip()
    )
    return f"{version_str}-rev{git_rev}"


def get_datetime_str():
    """Return the current datetime string."""
    return datetime.datetime.now(tz=TimeZones.CENTRAL).strftime("%Y-%m-%d %H:%M:%S")


def print_processing_info(ret_info: bool = True):
    """Print the current version and datetime."""
    version_str = get_version_str()
    datetime_str = get_datetime_str()
    print("Running with:")
    print(f"Version: {version_str}")
    print(f"Date: {datetime_str}\n")
    if ret_info:
        return version_str, datetime_str


def format_processing_info_as_json(version_str, datetime_str):
    """Return the current version and datetime as a JSON object."""
    return {"version": version_str, "datetime": datetime_str}


def get_processing_info_as_json():
    """Return the current version and datetime as a JSON object."""
    return format_processing_info_as_json(get_version_str(), get_datetime_str())


def get_data_json_skeleton():
    """Return a skeleton JSON object for the data JSON file."""
    return {
        "version": get_version_str(),
        "datetime": get_datetime_str(),
        "script_fn": Path(inspect.stack()[1][0].f_code.co_filename).name,
        "data": None,
    }
