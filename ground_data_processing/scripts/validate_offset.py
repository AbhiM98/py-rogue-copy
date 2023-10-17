"""Validate the offsets amongst cameras."""
import matplotlib.pyplot as plt
from S3MP.mirror_path import get_matching_s3_mirror_paths

from ground_data_processing.utils.absolute_segment_groups import ProductionFieldSegments
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFiles,
    DataTypes,
    Fields,
)
from ground_data_processing.utils.video_utils import get_frame_at_index

if __name__ == "__main__":
    segments = [
        ProductionFieldSegments.FIELD_DESIGNATION(Fields.FOUNDATION_FIELD_TWO),
        ProductionFieldSegments.DATA_TYPE(DataTypes.VIDEOS),
        ProductionFieldSegments.DATE("7-08"),
        ProductionFieldSegments.OG_VID_FILES("bottom.mp4"),
    ]

    bot_vid_mps = get_matching_s3_mirror_paths(segments)

    for bot_vid_mp in bot_vid_mps:
        offset_json_mp = bot_vid_mp.get_sibling(DataFiles.OFFSETS_JSON)
        if not offset_json_mp.exists_on_s3():
            continue
        offset_data = offset_json_mp.load_local(download=True)

        for idx, cam_view in enumerate(CameraViews):
            vid_mp = bot_vid_mp.get_sibling(f"{cam_view}.mp4")
            vid_mp.download_to_mirror_if_not_present()
            frame_idx = offset_data[cam_view]
            img = get_frame_at_index(vid_mp.local_path, frame_idx)
            plt.figure(idx)
            plt.imshow(img)
        plt.show()
