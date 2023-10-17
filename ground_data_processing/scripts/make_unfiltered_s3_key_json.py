"""Make a JSON file with the S3 keys for the unfiltered data."""
import json

from S3MP.mirror_path import get_matching_s3_mirror_paths
from tqdm import tqdm

from ground_data_processing.utils.absolute_segment_groups import (
    ProdInferenceSegments,
    ProductionFieldSegments,
)
from ground_data_processing.utils.keymaps import PROD_INFERENCE_STATS_KEYMAP
from ground_data_processing.utils.processing_utils import (
    format_processing_info_as_json,
    print_processing_info,
)
from ground_data_processing.utils.s3_constants import DataFiles, Fields

if __name__ == "__main__":
    proc_info = print_processing_info(ret_info=True)
    proc_info = format_processing_info_as_json(*proc_info)
    CURRENT_KEYMAP = PROD_INFERENCE_STATS_KEYMAP

    segments = [
        ProductionFieldSegments.FIELD_DESIGNATION(Fields.FOUNDATION_FIELD_TWO),
        # ProductionFieldSegments.DATA_TYPE(DataTypes.MODEL_INFERENCE),
        # ProdInferenceSegments.MODEL_NAME(ModelNames.SOLO_V2_DEC_11_MODEL),
        # ProdInferenceSegments.DATE("7-05"),
        # ProdInferenceSegments.ROW_DESIGNATION("Row 2"),
        # ProdInferenceSegments.DS_SPLIT_ROOT_FOLDER("Pass B"),  # hack
        # ProdInferenceSegments.DS_SPLIT_CAMERA_VIEW(CameraViews.NADIR),
        ProdInferenceSegments.OUTPUT_SEGM_JSON_FILES(DataFiles.SEGM_JSON, depth=11),
    ]

    segm_json_mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(segm_json_mps)} plot folders.\n")
    s3_key_data = {
        "s3_keys_by_plot": [],
    }

    for segm_json_mp in tqdm(segm_json_mps):
        output_json_mp = segm_json_mp.get_sibling(DataFiles.MASK_AREAS_JSON)
        if not output_json_mp.exists_on_s3():
            continue
        print(f"\nProcessing {segm_json_mp.s3_key}...")
        segm_json_data = segm_json_mp.load_local()

        ds_split_base_folder_mp = CURRENT_KEYMAP.apply(segm_json_mp).get_parent()

        possible_rogue_label_mps = [
            ds_split_base_folder_mp.get_child(json_name)
            for json_name in DataFiles.RogueLabelJSONs
        ]
        if not any(mp.exists_on_s3() for mp in possible_rogue_label_mps):
            print("No rogue labels found.")
            rogue_label_s3_key = None
        else:
            for mp in possible_rogue_label_mps:
                if mp.exists_on_s3():
                    rogue_label_mp = mp
                    break
            rogue_label_s3_key = rogue_label_mp.s3_key
        s3_key_data["s3_keys_by_plot"].append(
            {
                "segm_s3_key": segm_json_mp.s3_key,
                "rogue_label_s3_key": rogue_label_s3_key,
                "leaf_area_s3_key": output_json_mp.s3_key,
            }
        )
    with open("C:/Users/Josh/Desktop/ff2_all_inferenced.json", "w") as f:
        json.dump(s3_key_data, f, indent=4)
