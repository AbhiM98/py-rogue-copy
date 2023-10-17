"""Download a video from s3, resize it, and upload it back to s3."""
from S3MP.keys import get_matching_s3_keys
from S3MP.mirror_path import MirrorPath
from tqdm import tqdm

from ground_data_processing.utils.absolute_segment_groups import VideoSegments
from ground_data_processing.utils.image_utils import img_resize
from ground_data_processing.utils.rogues_key_utils import (
    plot_trial_prefix_segment_builder,
    rogues_key_video_plot_segment_builder,
)
from ground_data_processing.utils.s3_constants import Framerates, Resolutions
from ground_data_processing.utils.video_utils import vid_resize

if __name__ == "__main__":
    IN_RES = Resolutions.r4k
    # TODO util for this
    OUT_RES = Resolutions.r720p
    OUT_WIDTH = OUT_RES.width
    FRAMERATE = Framerates.fps120
    plot_segments, reverse_pass_flag = rogues_key_video_plot_segment_builder(
        date="6-21",
        row_number=7,
        plot_idx=16,
    )
    segments = [
        *plot_trial_prefix_segment_builder(),
        *plot_segments,
    ]

    matching_keys = get_matching_s3_keys(segments)
    data_mps = [MirrorPath.from_s3_key(key) for key in matching_keys if ".png" in key]
    # matching_keys = filter_keys_by_str(matching_keys, CameraViews.OBLIQUE)

    print(f"{len(data_mps)} matching keys found.")

    for data_mp in tqdm(data_mps):
        data_mp.download_to_mirror()
        resized_mp = data_mp.replace_key_segments(
            VideoSegments.RESOLUTION(f"{OUT_RES}@{FRAMERATE}")
        )
        match data_mp.s3_key.split(".")[-1]:
            case "mp4":
                vid_resize(
                    data_mp.local_path, resized_mp.local_path, OUT_WIDTH, overwrite=True
                )
            case "png" | "jpg":
                img_resize(
                    data_mp.local_path, resized_mp.local_path, OUT_WIDTH, overwrite=True
                )

        resized_mp.upload_from_mirror()
