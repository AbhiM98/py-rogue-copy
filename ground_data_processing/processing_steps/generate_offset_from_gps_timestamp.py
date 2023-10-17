"""Generate an offset file from GPS timestamps in the video files."""
import json
import os
from collections import namedtuple
from datetime import datetime, timedelta
from typing import List

from tqdm import tqdm

import ground_data_processing.utils.gps_utils as gps
from ddb_tracking.grd_constants import ProcessFlags
from ground_data_processing.params import RowParams
from ground_data_processing.utils.s3_constants import DataFiles, Framerates


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

    vid_start_times = [
        timestamp[0] - timedelta(microseconds=int(timestamp[1]))
        for timestamp in timestamps
    ]

    last_start_time = max(vid_start_times)
    # last_camera = vid_start_times.index(last_start_time)

    return {
        os.path.basename(vid_path).split(".")[0]: int(
            ((last_start_time - vid_start_times[i]).total_seconds() + 1)
            * Framerates.fps60.fps
        )
        for i, vid_path in enumerate(local_vid_paths)
    }


def generate_offset_from_gps_timestamp(row_params: RowParams):
    """Generate an offset file from GPS timestamps in the video files for a row."""
    # download the videos from s3
    mps = row_params.grdrow.full_row_video_mps.as_list()
    print(f"Expecting {len(mps)} videos to process.")
    print(*[mp.s3_key for mp in mps], sep="\n")

    # guard against overwriting
    offset_mp = mps[0].get_sibling(DataFiles.OFFSETS_JSON)
    # Workaround to avoid overwriting manually-created offset files on reruns
    if offset_mp.exists_on_s3() and (not row_params.overwrite or row_params.rerun):
        print("Offset json already exists on s3, skipping.")
        if not row_params.grdrow.videos_aligned:
            # Update database if videos aren't yet marked as aligned.
            row_params.update_process_flag_and_push_to_ddb(
                ProcessFlags.VIDEOS_ALIGNED, True
            )
        return
    [mp.download_to_mirror() for mp in tqdm(mps, desc="Downloading videos: ")]

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
    offset_mp.upload_from_mirror(overwrite=row_params.overwrite)

    # update database
    row_params.update_process_flag_and_push_to_ddb(ProcessFlags.VIDEOS_ALIGNED, True)
