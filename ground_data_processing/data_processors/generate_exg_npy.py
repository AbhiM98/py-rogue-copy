"""Generate the prop exg npy plot for a video."""
import functools
import itertools
import json
from decimal import Decimal
from typing import Callable, List

import numpy as np
from S3MP.mirror_path import MirrorPath

from ground_data_processing.utils.exg_utils import get_prop_exg_per_frame_slice
from ground_data_processing.utils.image_utils import (
    excess_green,
    prop_nonzero,
    slice_center_segment,
)
from ground_data_processing.utils.iter_utils import get_value_per_frame_generalized
from ground_data_processing.utils.multiprocessing_utils import (
    MultiprocessingManager,
    queue_wrap_process,
)
from ground_data_processing.utils.processing_utils import get_data_json_skeleton
from ground_data_processing.utils.s3_constants import DataFiles
from ground_data_processing.utils.video_utils import (
    get_ffmpeg_reader_trimmed,
    get_frame_count,
    get_n_divisions,
)


class NPJSONEncoder(json.JSONEncoder):
    """DynamoDB encoder to handle decimal cases."""

    def default(self, o):
        """Convert to float."""
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NPJSONEncoder, self).default(o)


def generate_exg_npy(
    vid_mp: MirrorPath, npy_fn=DataFiles.EXG_SLC_20PX_NPY, overwrite=False
):
    """Generate the ExG npy for a video."""
    npy_mp = vid_mp.get_sibling(npy_fn)

    if npy_mp.exists_on_s3() and not overwrite:
        print(f"{npy_mp.s3_key} already exists on S3.")
        return

    vid_mp.download_to_mirror()

    # TODO slice width selectors
    center_slice_prop = get_prop_exg_per_frame_slice(vid_mp.local_path, 20)
    npy_mp.save_local(center_slice_prop, upload=True)


def generate_exg_json(
    vid_mp: MirrorPath, json_fn=DataFiles.EXG_SLC_20PX_JSON, overwrite=False
):
    """Generate the ExG json for a video."""
    json_mp = vid_mp.get_sibling(json_fn)

    if json_mp.exists_on_s3() and not overwrite:
        print(f"{json_mp.s3_key} already exists on S3.")
        return

    vid_mp.download_to_mirror()

    json_data = get_data_json_skeleton()
    json_data["data"] = get_prop_exg_per_frame_slice(vid_mp.local_path, 20)
    with json_mp.local_path.open("w") as f:
        json.dump(json_data, f, cls=NPJSONEncoder)
    json_mp.upload_from_mirror(overwrite=overwrite)


def multiproc_generate_npy_generalized(
    vid_mp: MirrorPath,
    npy_fn=None,
    n_threads=10,
    frame_rate: float = 119.88,
    overwrite=False,
    proc_fns: List[Callable] = None,
    iter_wrappers: List[Callable] = None,
):
    """Generalized ExG npy generation."""
    npy_mp = vid_mp.get_sibling(npy_fn)

    if npy_mp.exists_on_s3() and not overwrite:
        print(f"{npy_mp.s3_key} already exists on S3.")
        return

    vid_mp.download_to_mirror()

    # TODO slice width selectors
    n_frames = get_frame_count(vid_mp.local_path)
    divisions = get_n_divisions(n_frames, n_threads)
    processes = []
    for start, end in itertools.pairwise(divisions):
        vid_iter_fn = functools.partial(
            get_ffmpeg_reader_trimmed, vid_mp.local_path, start, end, frame_rate
        )
        processes.append(
            queue_wrap_process(
                get_value_per_frame_generalized, vid_iter_fn, proc_fns, iter_wrappers
            )
        )
    MultiprocessingManager.wait_for_all_queued_processes_to_finish(
        processes, print_freq=15
    )
    prop_exg_per_frame = np.concatenate(MultiprocessingManager.get_ret_vals(processes))
    npy_mp.save_local(prop_exg_per_frame, upload=True)


def multiproc_generate_exg_npy(
    vid_mp: MirrorPath,
    npy_fn=DataFiles.EXG_SLC_20PX_NPY,
    n_threads=10,
    frame_rate: float = 119.88,
    overwrite=False,
):
    """Generate the ExG npy for a video, multithreaded."""
    multiproc_generate_npy_generalized(
        vid_mp,
        npy_fn,
        n_threads,
        frame_rate,
        overwrite,
        [
            functools.partial(slice_center_segment, width=20),
            functools.partial(excess_green),
            prop_nonzero,
        ],
    )
