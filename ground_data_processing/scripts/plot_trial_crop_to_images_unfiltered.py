"""Crop images for processing."""
import io
import traceback
from pathlib import Path

import cv2
from S3MP.keys import get_matching_s3_keys
from S3MP.mirror_path import MirrorPath
from tqdm import tqdm

from ground_data_processing.utils.absolute_segment_groups import (
    ImageSegments,
    RootSegments,
)
from ground_data_processing.utils.plot_layout import get_rogue_and_base_type
from ground_data_processing.utils.rogues_key_utils import (
    plot_trial_prefix_segment_builder,
    rogues_key_video_plot_segment_builder,
)
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataTypes,
    Dates,
    Framerates,
    Resolutions,
    get_week_idx_from_date_str,
)

if __name__ == "__main__":
    base_segments = plot_trial_prefix_segment_builder()

    OG_WIDTH = 3840
    OG_HEIGHT = 2160
    CROP_HEIGHT = 2160
    CROP_WIDTH = CROP_HEIGHT
    CROP = False

    X_OFFSET = (OG_WIDTH - CROP_WIDTH) // 2
    Y_OFFSET = (OG_HEIGHT - CROP_HEIGHT) // 2

    IN_RES = Resolutions.r4k
    FRAMERATE = Framerates.fps120
    OVERWRITE = True

    ROW_PLOT_TUPLES = []
    # ROW_START = 3
    # ROW_END = 10
    # for row, plot_idx in itertools.product(range(ROW_START, ROW_END + 1), range(1, 22+1)):
    #     ROW_PLOT_TUPLES.append((row, plot_idx))
    # ROW_PLOT_TUPLES.extend([(9, idx) for idx in range(1, 22+1)])
    ROW_PLOT_TUPLES = [(6, 2), (6, 4), (6, 6)]
    DATE = "6-16"

    for ROW_NUMBER, PLOT_NUMBER in tqdm(ROW_PLOT_TUPLES):
        try:
            WEEK = get_week_idx_from_date_str(DATE, Dates.FarmerCity.ALL_DATES) + 1
            ROGUE_TYPE, BASE_TYPE = get_rogue_and_base_type(ROW_NUMBER, PLOT_NUMBER)
            print()
            print(
                f"WEEK: {WEEK}\nROW_NUMBER: {ROW_NUMBER}, PLOT_NUMBER: {PLOT_NUMBER}\nROGUE_TYPE: {ROGUE_TYPE}, BASE_TYPE: {BASE_TYPE}"
            )
            # print(f"{ROGUE_TYPE}, {BASE_TYPE}")

            segments, pass_reverse_flag = rogues_key_video_plot_segment_builder(
                date=DATE,
                row_number=ROW_NUMBER,
                plot_number=PLOT_NUMBER,
                plot_split_file=f"{CameraViews.BOTTOM}.mp4",
                return_reverse_flag=True,
                existing_segments=base_segments,
            )

            matching_keys = get_matching_s3_keys(segments)
            bot_vid_mp = MirrorPath.from_s3_key(matching_keys[0])

            image_output_mp = bot_vid_mp.replace_key_segments(
                [
                    RootSegments.DATA_TYPE(DataTypes.IMAGES),
                    ImageSegments.ROGUE_TYPE(ROGUE_TYPE),
                    ImageSegments.BASE_TYPE(BASE_TYPE),
                    ImageSegments.ROW_AND_RANGE(
                        f"Row{ROW_NUMBER:02d}Col{PLOT_NUMBER:02d}"
                    ),
                    ImageSegments.WEEK_NUMBER(f"Week {WEEK}"),
                    ImageSegments.RESOLUTION(f"{IN_RES}"),
                ]
            )
            camera_output_mps = [
                image_output_mp.get_child(cam_view) for cam_view in CameraViews
            ]
            camera_input_mps = [
                bot_vid_mp.get_sibling(f"{cam_view} Raw Images")
                for cam_view in CameraViews
            ]

            stdout_dump = io.StringIO()
            # Download images hack
            for cam_input_mp in camera_input_mps:
                cam_input_mp.download_to_mirror()  # this handles folders

            if OVERWRITE:
                for cam_mp in camera_output_mps:
                    children = cam_mp.get_children_on_s3()
                    for child in children:
                        child.delete_s3()
            else:
                # TODO optimize with get_n_children or smthn
                children = camera_output_mps[0].get_children_on_s3()
                if len(children) > 10:
                    print("Skipping")
                    continue

            image_idx = 1  # renaming for simplicity

            for cam_idx, input_cam_mp in enumerate(tqdm(camera_input_mps)):
                images = list(Path(input_cam_mp.local_path).glob("*.png"))
                images = sorted(
                    images, key=lambda x: int(x.stem), reverse=pass_reverse_flag
                )
                for idx, image_file in enumerate(tqdm(images)):
                    img = cv2.imread(str(image_file.absolute()))
                    if CROP:
                        img = img[
                            Y_OFFSET : Y_OFFSET + CROP_HEIGHT,
                            X_OFFSET : X_OFFSET + CROP_WIDTH,
                        ]
                    out_mp = camera_output_mps[cam_idx].get_child(
                        f"{image_idx:02d}.png"
                    )
                    out_mp.save_local(img, upload=True)
        except Exception as e:
            print(f"Failed on {ROW_NUMBER}, {PLOT_NUMBER}")
            print(e)
            # print traceback
            traceback.print_exc()

            continue
