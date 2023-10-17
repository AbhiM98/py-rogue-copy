"""Pull rogue imagery into a local folder."""
from pathlib import Path

import cv2
from S3MP.mirror_path import get_matching_s3_mirror_paths
from tqdm import tqdm

from ground_data_processing.utils.absolute_segment_groups import (
    ProductionFieldSegments,
    ProductionFieldWithSplitPassSegments,
)
from ground_data_processing.utils.processing_utils import print_processing_info
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFiles,
    DataTypes,
    Fields,
)

if __name__ == "__main__":
    print_processing_info()
    local_save_dir = Path("C:/Users/Josh/Desktop/rogue_images/")
    local_save_dir.mkdir(exist_ok=True)

    segments = [
        ProductionFieldSegments.FIELD_DESIGNATION(Fields.FOUNDATION_FIELD_TWO),
        ProductionFieldSegments.DATA_TYPE(DataTypes.VIDEOS),
        ProductionFieldWithSplitPassSegments.DS_SPLIT_FILES(
            DataFiles.RogueLabelJSONs.UNFILTERED_ROGUES_JSON
        )
        # ProductionFieldSegments.DATA_TYPE(DataTypes.IMAGES),
        # ProductionFieldImageSegments.DS_SPLIT_IMAGE_DUMP(DataFolders.DS_SPLITS),
        # ProductionFieldImageSegments.DS_SPLIT_IMAGE_DATA_FILES(DataFiles.RogueLabelJSONs.ROGUES_JSON),
    ]

    rogues_json_mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(rogues_json_mps)} rogues jsons.")

    for rogues_json_mp in tqdm(rogues_json_mps):
        print(f"Processing {rogues_json_mp.s3_key}...")
        delay_sep_json_mp = rogues_json_mp.get_sibling(
            DataFiles.RogueLabelJSONs.ROGUES_W_DELAY_SEP_JSON
        )
        if delay_sep_json_mp.exists_on_s3():
            rogues_json_mp = delay_sep_json_mp

        rogue_labels = rogues_json_mp.load_local(overwrite=True)

        plot_designation_strs = [
            rogues_json_mp.get_key_segment(seg.depth)
            for seg in [
                ProductionFieldSegments.FIELD_DESIGNATION,
                ProductionFieldSegments.DATE,
                ProductionFieldSegments.ROW_DESIGNATION,
                ProductionFieldWithSplitPassSegments.ROW_SPLIT,
                ProductionFieldWithSplitPassSegments.DS_SPLIT_IDX
                # ProductionFieldImageSegments.DATE,
                # ProductionFieldImageSegments.ROW_DESIGNATION,
                # ProductionFieldImageSegments.DS_SPLIT_IMAGE_DUMP_IDX,
            ]
        ]

        plot_prefix: str = " ".join(plot_designation_strs)

        for img_key in sorted(rogue_labels.keys()):
            plant_class = rogue_labels[img_key]["plant_type"]
            if plant_class is None:
                print("bruh mode activated\n")
            # if plant_class in ["hybrid_rogue", "rogue", None]:
            if plant_class is None or "Rogue" in plant_class:
                local_folder = local_save_dir / plot_prefix
                local_folder.mkdir(exist_ok=True)

                for cam_view in CameraViews:
                    cam_folder_mp = rogues_json_mp.get_sibling(f"{cam_view} Raw Images")
                    rogue_image_mp = cam_folder_mp.get_child(f"{img_key}.png")
                    rogue_image = rogue_image_mp.load_local(overwrite=True)
                    img_name = f"{img_key}_{cam_view}.png"
                    cv2.imwrite(str(local_folder / img_name), rogue_image)
