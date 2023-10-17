"""Calculate statistics on paddle inference and store as JSON."""
import numpy as np
import pycocotools.mask
from S3MP.keys import KeySegment
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
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFiles,
    DataFolders,
    DataTypes,
    Fields,
    ModelNames,
    VideoRez,
)

if __name__ == "__main__":
    proc_info = print_processing_info(ret_info=True)
    proc_info = format_processing_info_as_json(*proc_info)
    CURRENT_KEYMAP = PROD_INFERENCE_STATS_KEYMAP
    OVERWRITE = False

    # base_segments = rogues_prefix_segment_builder(
    #     planting_number=1, data_type=DataTypes.MODEL_INFERENCE
    # )
    # segments = [
    #     *base_segments,
    #     InferenceSegments.MODEL_NAME(ModelNames.SOLO_V2_DEC_11_MODEL),
    #     # InferenceSegments.ROW_AND_RANGE("Row07Col02"),
    #     InferenceSegments.CAMERA_VIEW(CameraViews.NADIR),
    #     InferenceSegments.OUTPUT_SEGM_JSON_FILES(DataFiles.SEGM_JSON)
    # ]
    segments = [
        ProductionFieldSegments.FIELD_DESIGNATION(Fields.PROD_FIELD_ONE),
        ProductionFieldSegments.DATA_TYPE(DataTypes.UNFILTERED_MODEL_INFERENCE),
        ProdInferenceSegments.MODEL_NAME(ModelNames.SOLO_V2_DEC_11_MODEL),
        ProdInferenceSegments.DATE("7-05"),
        ProdInferenceSegments.ROW_DESIGNATION("Row 2"),
        # ProdInferenceSegments.DS_SPLIT_CAMERA_VIEW(CameraViews.NADIR),
        # ProdInferenceSegments.OUTPUT_SEGM_JSON_FILES(DataFiles.SEGM_JSON, depth=10),
        ProdInferenceSegments.OUTPUT_SEGM_JSON_FILES(DataFiles.SEGM_JSON),
    ]

    segm_json_mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(segm_json_mps)} plot folders.\n")

    for segm_json_mp in tqdm(segm_json_mps):
        try:
            output_json_mp = segm_json_mp.get_sibling(DataFiles.MASK_AREAS_JSON)
            if output_json_mp.exists_on_s3() and not OVERWRITE:
                print(
                    f"\nSkipping {segm_json_mp.s3_key} because {output_json_mp.s3_key} already exists."
                )
                continue
            print(f"\nProcessing {segm_json_mp.s3_key}...")
            segm_json_data = segm_json_mp.load_local()
            mask_areas_by_image_id = {}
            for mask in segm_json_data:
                image_id = mask["image_id"]
                if image_id not in mask_areas_by_image_id:
                    mask_areas_by_image_id[image_id] = []
                single_mask = pycocotools.mask.area(mask["segmentation"])
                mask_score = mask["score"]
                # mask_areas_by_image_id[image_id].append(mask_area * mask_score)
                mask_areas_by_image_id[image_id].append(single_mask)

            pre_proc_img_folder_mp = segm_json_mp.get_sibling(
                DataFolders.PREPROCESSED_IMAGES
            )
            # Check if all images in folder have the same name length
            img_names = [
                img.local_path.name
                for img in pre_proc_img_folder_mp.get_children_on_s3()
            ]
            skip_name_fix = all(
                len(img_name) == len(img_names[0]) for img_name in img_names
            )

            ds_split_base_folder_mp = CURRENT_KEYMAP.apply(segm_json_mp)
            # insert resolution into mp hack
            ds_split_base_folder_mp.key_segments.insert(
                4, KeySegment(depth=4, name=str(VideoRez.r4k_120fps))
            )

            bot_img_folder_mp = ds_split_base_folder_mp.get_child(
                f"{CameraViews.BOTTOM} Raw Images"
            )
            if not bot_img_folder_mp.exists_on_s3():
                print(f"Could not find {bot_img_folder_mp.s3_key}. Skipping.")
                continue
            bot_img_mps = bot_img_folder_mp.get_children_on_s3()
            bot_img_names = [mp.local_path.stem for mp in bot_img_mps]
            bot_img_names = {name: {} for name in bot_img_names}

            if not skip_name_fix:
                print("Fixing image names...\n\n")
                # Fix cases where images go above 100
                for image_name in list(sorted(bot_img_names.keys())):
                    try:
                        num = int(image_name.split("_")[-1])
                    except ValueError:
                        num = int(image_name)
                    if num < 100:
                        new_key = f"{num:03d}"
                        bot_img_names[new_key] = bot_img_names.pop(image_name)
                    else:
                        # Move id to the end
                        start_id = 10
                        end_data = mask_areas_by_image_id[start_id]
                        # Shift left
                        for idx in range(start_id, num - 1):
                            mask_areas_by_image_id[idx] = mask_areas_by_image_id[
                                idx + 1
                            ]
                        # Add to end
                        mask_areas_by_image_id[num] = end_data

            output_json = {
                **proc_info,
                "images": {},
            }
            image_names = list(sorted(bot_img_names.keys()))
            mask_areas_by_image = [
                val
                for key, val in sorted(
                    mask_areas_by_image_id.items(), key=lambda item: item[0]
                )
            ]

            for image_name, single_image_mask_areas in zip(
                image_names, mask_areas_by_image
            ):
                single_image_mask_areas = np.array(single_image_mask_areas)
                top_10_pct_mask_areas = single_image_mask_areas[
                    np.argsort(single_image_mask_areas)[
                        -int(len(single_image_mask_areas) * 0.1) :
                    ]
                ]
                output_json["images"][image_name] = {
                    "all_mask_areas": [
                        int(mask_area) for mask_area in single_image_mask_areas
                    ],
                    "mean_mask_area": int(np.mean(single_image_mask_areas)),
                    "max_mask_area": int(np.max(single_image_mask_areas)),
                    "top_10_pct_mean_mask_area": int(np.mean(top_10_pct_mask_areas)),
                }

            output_json_mp.save_local(output_json, overwrite=OVERWRITE)
        except Exception as e:
            print(f"Error processing {segm_json_mp.s3_key}: {e}")
