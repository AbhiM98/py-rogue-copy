"""Calculate statistics on unfiltered paddle inference."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from S3MP.mirror_path import get_matching_s3_mirror_paths
from tqdm import tqdm

from ground_data_processing.measurements.mask_stats import (
    get_mask_areas_by_image_id,
    get_top_n_percent,
)
from ground_data_processing.utils.absolute_segment_groups import (
    ProdInferenceSegments,
    ProductionFieldSegments,
)
from ground_data_processing.utils.keymaps import PROD_INFERENCE_STATS_KEYMAP
from ground_data_processing.utils.processing_utils import print_processing_info
from ground_data_processing.utils.s3_constants import (
    DataFiles,
    DataTypes,
    Fields,
    ModelNames,
)

if __name__ == "__main__":
    print_processing_info()
    segments = [
        ProductionFieldSegments.FIELD_DESIGNATION(Fields.PROD_FIELD_ONE),
        ProductionFieldSegments.DATA_TYPE(DataTypes.UNFILTERED_MODEL_INFERENCE),
        ProdInferenceSegments.MODEL_NAME(ModelNames.SOLO_V2_DEC_11_MODEL),
        ProdInferenceSegments.DATE("7-05"),
        ProdInferenceSegments.OUTPUT_SEGM_JSON_FILES(DataFiles.SEGM_JSON),
    ]
    proc_keymap = PROD_INFERENCE_STATS_KEYMAP

    segm_json_mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(segm_json_mps)} plot folders.")

    for segm_json_mp in tqdm(segm_json_mps):
        print(f"Processing {segm_json_mp.s3_key}...")
        segm_json_data = segm_json_mp.load_local(overwrite=True)
        mask_areas_by_image_id = get_mask_areas_by_image_id(segm_json_data)

        for image_id, mask_areas in mask_areas_by_image_id.items():
            mask_areas_by_image_id[image_id] = get_top_n_percent(mask_areas, 0.1)

        cam_base_folder_mp = proc_keymap.apply(segm_json_mp)
        image_names = [
            image_mp.get_key_segment(-1).name
            for image_mp in cam_base_folder_mp.get_children_on_s3()
        ]
        image_names = {
            image_name: image_name.split("_")[-1] for image_name in image_names
        }

        # Fix cases where images go above 100
        for image_name in list(sorted(image_names.keys())):
            num = int(image_name.split("_")[-1]) + 1
            if num <= 100:
                new_key = f"plant_{num:03d}"
                image_names[new_key] = image_names.pop(image_name)
            else:
                # Move id to the end
                start_id = 10
                end_data = mask_areas_by_image_id[start_id]
                # Shift left
                for idx in range(start_id, num - 1):
                    mask_areas_by_image_id[idx] = mask_areas_by_image_id[idx + 1]
                # Add to end
                mask_areas_by_image_id[num] = end_data

        img_name = " ".join(proc_keymap.get_preserve_strs(segm_json_mp))
        field_name = segm_json_mp.get_key_segment(0)

        s3_name = DataFiles.PLOT_TOP_10PCT_MASK_AREA_PNG
        plt.title(f"Top 10% Mask Area By Image\n{field_name}\n{img_name}")

        # plot mask areas by image id
        # print(f"Mean mask area by image id: {mask_areas_by_image_id}.")
        mean_mask_areas = [
            np.mean(mask_areas) for mask_areas in mask_areas_by_image_id.values()
        ]
        std_mask_areas = [
            np.std(mask_areas) for mask_areas in mask_areas_by_image_id.values()
        ]
        plt.bar(mask_areas_by_image_id.keys(), mean_mask_areas, yerr=std_mask_areas)
        # plt.bar(mask_areas_by_image_id.keys(), mean_mask_areas)
        # Standardize plots
        plt.ylim(0, 100000)

        chart_mp = segm_json_mp.get_sibling(s3_name)
        plt.savefig(str(chart_mp.local_path))
        chart_mp.upload_from_mirror(overwrite=True)

        additional_save_dir = Path(
            "C:/Users/Josh/Desktop/unfiltered_plot_mask_area_charts/"
        )
        additional_save_dir.mkdir(exist_ok=True)
        additional_save_fn = f"{str(additional_save_dir / img_name)}.png"
        plt.savefig(str(additional_save_fn))
        plt.clf()
