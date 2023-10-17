"""Generate the prop exg npy plot for a video."""
import concurrent.futures
import functools
import gzip
import itertools
import pickle
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import psutil
from S3MP.mirror_path import MirrorPath, get_matching_s3_mirror_paths

from ground_data_processing.measurements.frame_measurements import (
    get_normalized_frame_diff,
)
from ground_data_processing.scripts.clip_images_hp_sum import spread_col_sum_frame_proc
from ground_data_processing.utils.absolute_segment_groups import (
    RootSegments,
    VideoSegments,
)
from ground_data_processing.utils.image_utils import (
    excess_green,
    get_img_phash_flat_8,
    get_img_phash_flat_64,
    prop_nonzero,
    slice_center_segment,
)
from ground_data_processing.utils.iter_utils import get_value_per_frame_generalized
from ground_data_processing.utils.processing_utils import (
    get_data_json_skeleton,
    print_processing_info,
)
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFiles,
    Fields,
    Framerates,
    TrialTypes,
)
from ground_data_processing.utils.video_utils import (
    get_ffmpeg_reader_trimmed,
    get_frame_count,
)


class VideoProcessingExceptionError(Exception):
    """Exception for video processing."""

    pass


def calc_framewise_data_and_upload_json(
    vid_path_local: Path,
    output_s3_key: str,
    frame_proc: Callable,
    iter_wrappers: Iterable[Callable] = None,
    overwrite: bool = True,
    use_two_frames: bool = False,
):
    """Calculate the framewise value and upload the npy."""
    # if not output_s3_key.endswith('.json'):
    #     raise ValueError('output_s3_key must end with .json')
    json_data = get_data_json_skeleton()
    vid_mp = MirrorPath.from_local_path(vid_path_local)
    vid_mp.download_to_mirror()
    output_mp = MirrorPath.from_s3_key(output_s3_key)
    if output_mp.exists_on_s3() and not overwrite:
        print(f"Output already exists on s3: {output_s3_key}")
        return (output_mp.s3_key, -1)

    # Catch DS Split making videos that are too small to have
    # relevant data
    try:
        n_frames = get_frame_count(vid_mp.local_path)
    except KeyError as e:
        print(e)
        raise VideoProcessingExceptionError(f"Error processing video {vid_mp.s3_key}")

    vid_iter = functools.partial(
        get_ffmpeg_reader_trimmed,
        vid_mp.local_path,
        start_frame=0,
        end_frame=n_frames,
        fps=Framerates.fps60.fps,
    )
    vid_iter2 = None
    if use_two_frames:
        vid_iter2 = functools.partial(
            get_ffmpeg_reader_trimmed,
            vid_mp.local_path,
            start_frame=1,
            end_frame=n_frames,
            fps=59.94,
        )
    vpf = get_value_per_frame_generalized(
        [vid_iter, vid_iter2] if use_two_frames else [vid_iter],
        frame_proc,
        iter_wrappers=iter_wrappers,
    )
    json_data["data"] = vpf

    print("Writing file...")
    # with open(output_mp.local_path, 'w') as f:
    #     json.dump(json_data, f, cls=NPJSONEncoder)
    # pickle gz takes 1/3 the time of json
    with gzip.open(output_mp.local_path, "wb") as f:
        pickle.dump(json_data, f)

    print("Uploading to s3...")
    output_mp.upload_from_mirror(overwrite=overwrite)

    return (output_mp.s3_key, n_frames)


if __name__ == "__main__":
    print_processing_info()
    # CAMERA_VIEW = CameraViews.NADIR
    SLICE_WIDTH = 20
    N_VIDEO_THREADS = psutil.cpu_count(logical=False)
    FRAME_RATE = 59.94

    # SUFFIX = DataFiles.Suffixes.CENTER_EXG_SLC
    # SUFFIX = DataFiles.Suffixes.SPREAD_COL_SUM_JSON
    SUFFIX = DataFiles.Suffixes.NORM_FRAME_DIFF_JSON
    # SUFFIX = DataFiles.Suffixes.FRAME_PHASH_8BIT
    iter_wrappers = None
    if SUFFIX == DataFiles.Suffixes.CENTER_EXG_SLC_NPY:
        frame_procs = [
            functools.partial(slice_center_segment, width=20),
            functools.partial(excess_green),
            prop_nonzero,
        ]
    elif SUFFIX in [
        DataFiles.Suffixes.NORM_FRAME_DIFF_NPY,
        DataFiles.Suffixes.NORM_FRAME_DIFF_JSON,
    ]:
        frame_procs = [get_normalized_frame_diff]
        iter_wrappers = [itertools.pairwise]
    elif SUFFIX == DataFiles.Suffixes.SPREAD_COL_SUM_JSON:
        frame_procs = [spread_col_sum_frame_proc]
    elif SUFFIX == DataFiles.Suffixes.FRAME_PHASH_64BIT:
        frame_procs = [get_img_phash_flat_64]
    elif SUFFIX == DataFiles.Suffixes.FRAME_PHASH_8BIT:
        frame_procs = [get_img_phash_flat_8]

    OVERWRITE = False
    segments = [
        RootSegments.PLOT_NAME_AND_YEAR(Fields.FC_2022),
        RootSegments.TRIAL_TYPE(TrialTypes.STRIP_TRIAL),
        VideoSegments.OG_VID_FILES(incomplete_name=".mp4"),
    ]
    # print(segments)
    matching_mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(matching_mps)} matching mirror paths.")

    n_procs = psutil.cpu_count(logical=False)
    proc_executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_procs)
    all_proc_futures: List[
        Tuple[
            MirrorPath,
            concurrent.futures.Future,
        ]
    ] = []

    for vid_mp in matching_mps:
        # with FileSizeTQDMCallback(vid_mp, is_download=True) as tqdm_cb:
        vid_mp.download_to_mirror(overwrite=OVERWRITE)
        for cam_view in CameraViews:
            if cam_view in vid_mp.s3_key:
                output_npy_name = f"{cam_view}{SUFFIX}"
                break
        else:
            print("Matching cam view not found.")
            continue

        output_npy_mp = vid_mp.get_sibling(output_npy_name)
        if output_npy_mp.exists_on_s3() and not OVERWRITE:
            continue
        pf = proc_executor.submit(
            calc_framewise_data_and_upload_json,
            vid_mp.local_path,
            output_npy_mp.s3_key,
            frame_procs,
            iter_wrappers=iter_wrappers,
            overwrite=OVERWRITE,
        )
        all_proc_futures.append((vid_mp, pf))

    for mp, pf in all_proc_futures:
        if pf.exception():
            print(f"Exception in {mp.s3_key}: {pf.exception()}")

    # Wait for all processes to exit
    proc_executor.shutdown(wait=True)
