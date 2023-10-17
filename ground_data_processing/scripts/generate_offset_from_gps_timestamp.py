"""Generate an offset file from GPS timestamps in the video files."""
import json
import os
from collections import namedtuple
from datetime import datetime, timedelta
from typing import List

from S3MP.mirror_path import KeySegment, get_matching_s3_mirror_paths
from tqdm import tqdm

import ground_data_processing.utils.gps_utils as gps
from ground_data_processing.utils.s3_constants import DataFiles


def get_gps_blocks(vid_path: str) -> List[namedtuple]:
    """
    Get the gps stream blocks as a list of dictionaries.

    Args:
    vid_path str: local path to the video file

    Returns:
    List[GPSData]: a list of gps blocks as namedtuples
    """
    stream = gps.extract_gpmf_stream(vid_path)
    gps_blocks = gps.extract_gps_blocks(stream)
    return list(map(gps.parse_gps_block, gps_blocks))


def get_first_timestamp_with_microseconds(gps_blocks: list[namedtuple]) -> tuple:
    """
    Return the first stable gps timestamp and the number of microseconds since the start of the video.

    Args:
    gps_blocks: list of gps blocks

    Returns:
    tuple: (first stable gps timestamp, length of video in microseconds at that timestamp)
    """
    # drop None values
    gps_blocks = [block for block in gps_blocks if block is not None]

    date = gps_blocks[-1].timestamp.split(" ")[0]  # yyyy-mm-dd
    for block in gps_blocks:
        if date not in block.timestamp:
            continue
        return (
            datetime.strptime(block.timestamp, "%Y-%m-%d %H:%M:%S.%f"),
            block.microseconds,
        )
    return (None, None)


def get_alignment_offset(local_vid_paths: list[str]) -> dict:
    """
    Parse the gps information from the video files and return a dictionary of the offsets.

    Args:
    local_vid_paths: list of paths to the video files on local

    Returns:
    dictionary of offsets
    """
    blocks = [get_gps_blocks(vid_path) for vid_path in local_vid_paths]
    timestamps = [get_first_timestamp_with_microseconds(block) for block in blocks]
    print(timestamps)

    vid_start_times = [
        timestamp[0] - timedelta(microseconds=int(timestamp[1]))
        for timestamp in timestamps
    ]

    last_start_time = max(vid_start_times)
    # last_camera = vid_start_times.index(last_start_time)

    return {
        os.path.basename(vid_path).split(".")[0]: int(
            ((last_start_time - vid_start_times[i]).total_seconds() + 1) * 119.88
        )
        for i, vid_path in enumerate(local_vid_paths)
    }


if __name__ == "__main__":
    # some constants
    OVERWRITE = True

    # define segments for OG vid files
    segments = [
        KeySegment(0, "2023-field-data"),
        KeySegment(1, "Waterman_Strip_Trial"),
        KeySegment(2, "2023-06-20"),
        KeySegment(3, "row-15"),
        KeySegment(5, incomplete_name=".mp4", is_file=True),
    ]

    # download the videos from s3
    mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(mps)} videos to process.")
    print(*[mp.s3_key for mp in mps], sep="\n")

    # guard against overwriting
    offset_mp = mps[0].get_sibling(DataFiles.OFFSETS_JSON)
    if offset_mp.exists_on_s3() and not OVERWRITE:
        raise ValueError("Offset json already exists on s3, aborting.")
    if offset_mp.exists_in_mirror():
        [mp.download_to_mirror() for mp in tqdm(mps)]

    offset = get_alignment_offset([mp.local_path for mp in mps])
    if any(offset[x] < 120 for x in offset):
        diff = 120 - min(offset[x] for x in offset)
        for x in offset:
            offset[x] += diff
    print("offsets: ", offset)

    # upload the offsets to s3
    print("writing offsets to s3")
    print("local path: ", offset_mp.local_path)
    with open(offset_mp.local_path, "w") as fd:
        json.dump(offset, fd)
    offset_mp.upload_from_mirror(overwrite=OVERWRITE)
