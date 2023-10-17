"""Prepares the data for paddle inference."""
import cv2
from S3MP.mirror_path import KeySegment, get_matching_s3_mirror_paths
from tqdm import tqdm

from ground_data_processing.utils.image_utils import crop_square_from_img_center
from ground_data_processing.utils.keymaps import (
    PROD_RAW_DATA_TO_UNFILTERED_INFERENCE_KEYMAP,
)
from ground_data_processing.utils.processing_utils import print_processing_info
from ground_data_processing.utils.s3_constants import CameraViews, InferenceMethods

CROP_SIZE = 1024
INFERENCE_METHOD = InferenceMethods.SQUARE_CROP


if __name__ == "__main__":
    print_processing_info()

    KEYMAP_TO_USE = PROD_RAW_DATA_TO_UNFILTERED_INFERENCE_KEYMAP
    DOWNSAMPLE = True
    DOWNSAMPLE_PCT = 50  # 50% of the original size (50% is 512 x 512)

    segments = [
        KeySegment(0, "Production Field 1 (Argo North-Home Minor)"),
        KeySegment(1, "Videos"),
        KeySegment(2, "6-27"),
        KeySegment(3, "Row 2"),
        KeySegment(7, incomplete_name="rogues", is_file=True),
    ]

    rogue_json_mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(rogue_json_mps)} plot folders.")

    for rogue_json_mp in tqdm(rogue_json_mps):
        plot_folder_mp = rogue_json_mp.get_sibling(f"{CameraViews.NADIR} Raw Images")
        print(f"Processing {plot_folder_mp.s3_key}...")

        base_output_folder_mp = KEYMAP_TO_USE.apply(plot_folder_mp)
        # TODO check if this mapping is correct

        print(f"Output folder: {base_output_folder_mp.s3_key}\n")
        for img_mp in tqdm(plot_folder_mp.get_children_on_s3()):
            img = img_mp.load_local(overwrite=True)
            if INFERENCE_METHOD == InferenceMethods.PADDLE_SLICE:
                preproc_img = img
            elif INFERENCE_METHOD == InferenceMethods.SQUARE_CROP:
                preproc_img = crop_square_from_img_center(img, CROP_SIZE)
            else:
                raise ValueError(f"Invalid inference method: {INFERENCE_METHOD}")

            if DOWNSAMPLE:
                preproc_img = cv2.resize(
                    preproc_img,
                    (0, 0),
                    fx=DOWNSAMPLE_PCT / 100,
                    fy=DOWNSAMPLE_PCT / 100,
                )

            output_img_mp = base_output_folder_mp.get_child(img_mp.local_path.name)
            output_img_mp.save_local(preproc_img, upload=True, overwrite=True)
