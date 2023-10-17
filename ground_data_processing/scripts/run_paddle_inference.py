"""Run paddle inference on the images in the plot folder."""
import shutil
import subprocess

from S3MP.keys import KeySegment
from S3MP.mirror_path import MirrorPath, get_matching_s3_mirror_paths
from tqdm import tqdm

from ground_data_processing.utils.absolute_segment_groups import (
    ProdInferenceSegments,
    ProductionFieldSegments,
)
from ground_data_processing.utils.processing_utils import print_processing_info
from ground_data_processing.utils.s3_constants import (
    DataFolders,
    DataTypes,
    Fields,
    InferenceMethods,
    ModelNames,
)

if __name__ == "__main__":
    print_processing_info()

    config_path = "configs/solov2/solov2_r101_vd_fpn_3x_coco.yml"
    weights_path = "output/solov2_r101_vd_fpn_3x_coco/best_model.pdparams"
    draw_thresh = 0.25
    INFERENCE_METHOD = InferenceMethods.SQUARE_CROP
    OVERWRITE = True

    # base_segments = rogues_prefix_segment_builder(
    #     planting_number=1, data_type=DataFolders.MODEL_INFERENCE
    # )
    # segments = [
    #     *base_segments,
    #     InferenceSegments.MODEL_NAME(ModelNames.SOLO_V2_DEC_11_MODEL),
    #     InferenceSegments.PREPROC_METHOD(INFERENCE_METHOD),
    #     InferenceSegments.ROW_AND_RANGE(incomplete_name="Row05"),
    #     InferenceSegments.CAMERA_VIEW(CameraViews.NADIR),
    # ]
    segments = [
        ProductionFieldSegments.FIELD_DESIGNATION(Fields.FOUNDATION_FIELD_TWO),
        ProductionFieldSegments.DATA_TYPE(DataTypes.UNFILTERED_MODEL_INFERENCE),
        ProdInferenceSegments.MODEL_NAME(ModelNames.SOLO_V2_DEC_11_MODEL),
        ProdInferenceSegments.DATE("7-08"),
        ProdInferenceSegments.ROW_DESIGNATION("Row 8, 9"),
        KeySegment(7, "Pass B"),
    ]
    is_split_pass = any(Fields.FOUNDATION_FIELD_TWO in seg.name for seg in segments)
    full_depth_seg = KeySegment(incomplete_name=f"{DataFolders.RAW_IMAGES}", depth=8)
    if is_split_pass:
        full_depth_seg.depth = 10
    segments.append(full_depth_seg)
    print(segments)

    additional_args = []
    # Currently unused but could be used to change the inference method
    if INFERENCE_METHOD == InferenceMethods.PADDLE_SLICE:
        additional_args = [
            "--slice_infer",
            "--slice_size",
            "1024",
            "1024",
            "--overlap_ratio",
            "0.25",
            "0.25",
            "--match_threshold",
            "0.5",
        ]

    base_inference_folder_mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(base_inference_folder_mps)} plot folders.")

    if len(base_inference_folder_mps) == 0:
        print("No plot folders found.")
        exit()

    for base_inference_folder_mp in tqdm(base_inference_folder_mps):
        print(f"Processing {base_inference_folder_mp.s3_key}...")
        preproc_folder_mp = base_inference_folder_mp.get_child(
            DataFolders.PREPROCESSED_IMAGES
        )

        annotated_folder_mp = base_inference_folder_mp.get_child(
            DataFolders.ANNOTATED_IMAGES
        )
        if OVERWRITE and list(annotated_folder_mp.get_children_on_s3()):
            # Delete old images
            [mp.delete_all() for mp in annotated_folder_mp.get_children_on_s3()]
            annotated_folder_mp.local_path.mkdir(parents=True, exist_ok=True)

        inference_mask_json_mp = annotated_folder_mp.get_child("segm.json")
        if inference_mask_json_mp.exists_on_s3():
            print(f"Skipping {base_inference_folder_mp.s3_key} (already processed).")
            continue  # Skip if already processed

        for img_mp in preproc_folder_mp.get_children_on_s3():
            img_mp.download_to_mirror(overwrite=OVERWRITE)

        # Run inference and block until done
        subprocess.run(
            [
                "/usr/bin/python3",
                "-u",
                "tools/infer.py",
                "-c",
                config_path,
                "-o",
                f"weights={weights_path}",
                "--infer_dir",
                preproc_folder_mp.local_path,
                "--output_dir",
                annotated_folder_mp.local_path,
                "--draw_threshold",
                str(draw_thresh),
                "--save_results",
                "true",
                *additional_args,
            ]
        )

        # Upload the inference mask json
        camera = (
            "nadir" if "nadir" in str(inference_mask_json_mp.local_path) else "oblique"
        )
        new_inference_json_path = (
            inference_mask_json_mp.local_path.parent.parent / f"segm-{camera}.json"
        )
        shutil.move(
            inference_mask_json_mp.local_path,
            new_inference_json_path,
        )
        inference_mask_json_mp = MirrorPath.from_local_path(new_inference_json_path)
        inference_mask_json_mp.upload_from_mirror(overwrite=OVERWRITE)

        # Upload the annotated images
        for img_path in annotated_folder_mp.local_path.iterdir():
            img_mp = MirrorPath.from_local_path(img_path)
            img_mp.upload_from_mirror(overwrite=OVERWRITE)
