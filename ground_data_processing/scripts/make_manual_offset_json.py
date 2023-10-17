"""Make manual offset json."""
import json

from S3MP.keys import get_matching_s3_keys
from S3MP.mirror_path import MirrorPath

from ground_data_processing.utils.absolute_segment_groups import (
    ProductionFieldSegments,
    VideoSegments,
)
from ground_data_processing.utils.s3_constants import CameraViews, DataFiles, Fields

if __name__ == "__main__":
    CAMERA_VIEW = CameraViews.BOTTOM

    segments = [
        ProductionFieldSegments.FIELD_DESIGNATION(Fields.FOUNDATION_FIELD_TWO),
        ProductionFieldSegments.ROW_DESIGNATION("Row 7, 10"),
        ProductionFieldSegments.OG_VID_FILES(f"{CameraViews.BOTTOM}.mp4"),
        # *plot_trial_prefix_segment_builder(),
        # VideoSegments.DATE("7-06"),
        # VideoSegments.ROW_DESIGNATION("4, 9"),
        # VideoSegments.RESOLUTION(VideoRez.r4k_120fps),
        # VideoSegments.OG_VID_FILES(f"{CAMERA_VIEW}.mp4"),
    ]
    matching_keys = get_matching_s3_keys(segments)
    bot_vid_mp = MirrorPath.from_s3_key(matching_keys[0])
    offset_mp = bot_vid_mp.replace_key_segments(
        [VideoSegments.OG_VID_FILES(DataFiles.OFFSETS_JSON)]
    )

    true_framerate = 119.88
    bot_timestamp = 45.802
    nad_timestamp = 62.964
    obl_timestamp = 54.462
    nad_offset = nad_timestamp - bot_timestamp
    obl_offset = obl_timestamp - bot_timestamp
    offset_data = {
        CameraViews.BOTTOM: 0,
        CameraViews.NADIR: round(nad_offset * true_framerate),
        CameraViews.OBLIQUE: round(obl_offset * true_framerate),
    }
    with open(offset_mp.local_path, "w") as fd:
        json.dump(offset_data, fd)
    offset_mp.upload_from_mirror()
