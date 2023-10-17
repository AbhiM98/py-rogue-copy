"""DS (decasecond) splits for longer-form video."""
import itertools

import numpy as np
import psutil
from S3MP.mirror_path import get_matching_s3_mirror_paths

from ddb_tracking.grd_constants import ProcessFlags
from ground_data_processing.data_processors.thresh_split_video import (
    split_video_at_bounds,
)
from ground_data_processing.params import RowParams
from ground_data_processing.utils.ffmpeg_utils import FFmpegFlags, FFmpegProcessManager
from ground_data_processing.utils.processing_utils import print_processing_info
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFiles,
    DataFolders,
    Framerates,
)
from ground_data_processing.utils.video_utils import get_frame_count

FPS = Framerates.fps60.fps
DS_S = 10  # decasecond
FRAMES_PER_DS = DS_S * FPS


def generate_ds_splits(row_params: RowParams):
    """Split videos into decasecond chunks."""
    print_processing_info()

    FFmpegProcessManager.max_processes = psutil.cpu_count(logical=False)
    # FFmpegProcessManager.set_max_processes(5)
    # FFmpegFlags.set_output_flag("c", "copy")
    FFmpegFlags.set_output_flag("loglevel", "quiet")

    # We can skip this step if we're rerunning only speific DS splits or if we're not rerunning at all
    if (
        (row_params.ds_split_numbers not in ["", None] and row_params.rerun)
        or not row_params.rerun
    ) and row_params.grdrow.full_row_video_mps.bottom_mp.get_sibling(
        DataFolders.DS_SPLITS
    ).exists_on_s3():
        print("DS splits already exist, skipping. To rerun, use --rerun flag.")
    else:
        bottom_vid_mps = get_matching_s3_mirror_paths(
            row_params.grdrow.full_row_video_mps.bottom_mp.key_segments
        )
        print(f"Found {len(bottom_vid_mps)} bottom videos to split.")
        for bottom_vid_mp in bottom_vid_mps:
            offset_json_mp = bottom_vid_mp.get_sibling(DataFiles.OFFSETS_JSON)
            if offset_json_mp.exists_on_s3():
                offset_data = offset_json_mp.load_local()
            else:
                print("No offset json found, assuming no offset.")
                offset_data = {camera: 0 for camera in CameraViews}

            # This gets the count of relevant frames in each video and splits into evenly-sized bounds
            split_bounds_by_video = []
            vid_mps = []
            for camera in CameraViews:
                vid_mp = offset_json_mp.get_sibling(f"{camera}.mp4")
                vid_mp.download_to_mirror(overwrite=row_params.overwrite)
                vid_mps.append(vid_mp)
                n_vid_frames = get_frame_count(vid_mp.local_path)
                vid_offset = offset_data[camera]

                split_frames = list(
                    np.arange(vid_offset, n_vid_frames, FRAMES_PER_DS).astype(int)
                ) + [n_vid_frames]
                split_bounds_by_video.append(list(itertools.pairwise(split_frames)))

            # Trim trailing bounds as that data will not contain all cameras thus is trivial
            min_n_bounds = min(len(bounds) for bounds in split_bounds_by_video)
            split_bounds_by_video = [
                bounds[:min_n_bounds] for bounds in split_bounds_by_video
            ]

            # With bounds trimmed, each video will have the same number of DS splits
            # Thus folder creation can be done independent of the camera
            print(f"Creating {min_n_bounds+1} DS folders...")
            root_ds_mp = vid_mp.get_sibling(DataFolders.DS_SPLITS)
            root_ds_mp.local_path.mkdir(parents=True, exist_ok=True)
            ds_output_folder_mps = [
                root_ds_mp.get_child(f"DS {i:03d}") for i in range(min_n_bounds)
            ]
            for ds_output_folder_mp in ds_output_folder_mps:
                ds_output_folder_mp.local_path.mkdir(parents=True, exist_ok=True)

            # Now that the DS folders are created, we can split the videos
            for camera, vid_split_bounds, vid_mp in zip(
                CameraViews, split_bounds_by_video, vid_mps
            ):
                print(f"Splitting {camera} video...")
                output_vid_mps = [
                    ds_output_folder_mp.get_child(f"{camera}.mp4")
                    for ds_output_folder_mp in ds_output_folder_mps
                ]
                assert len(output_vid_mps) == len(vid_split_bounds)
                split_video_at_bounds(
                    vid_mp,
                    output_vid_mps,
                    vid_split_bounds,
                    framerate=FPS,
                    overwrite=row_params.overwrite,
                )

            FFmpegProcessManager.wait_for_all_processes_to_finish()

    # Update database
    # Ensure that the number of DS splits on S3 is correct
    n_splits = len(
        row_params.grdrow.full_row_video_mps.bottom_mp.get_sibling(
            DataFolders.DS_SPLITS
        ).get_children_on_s3()
    )
    print(f"Found {n_splits} DS splits on S3, updating database...")
    row_params.set_n_ds_splits(n_splits)
    row_params.update_process_flag_and_push_to_ddb(ProcessFlags.VIDEOS_SPLIT, True)
    print("Done splitting videos.")
