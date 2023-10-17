"""Download specified data to mirror."""
from S3MP.mirror_path import get_matching_s3_mirror_paths
from S3MP.transfer_configs import GB, MB, get_transfer_config

from ground_data_processing.utils.absolute_segment_groups import ProductionFieldSegments
from ground_data_processing.utils.processing_utils import print_processing_info
from ground_data_processing.utils.s3_constants import Fields, Resolutions

if __name__ == "__main__":
    print_processing_info()

    IN_RES = Resolutions.r4k
    OUT_RES = Resolutions.r720p

    get_transfer_config(
        n_threads=30, block_size=16 * MB, max_ram=8 * GB, set_global=True
    )

    # base_segments = rogues_prefix_segment_builder(
    #     planting_number=1, data_type=DataFolders.MODEL_INFERENCE
    # )
    # segments = [
    #     *base_segments,
    #     InferenceSegments.MODEL_NAME(ModelNames.SOLO_V2_DEC_11_MODEL),
    #     InferenceSegments.CAMERA_VIEW(CameraViews.NADIR),
    #     InferenceSegments.OUTPUT_IMAGE_FILES(incomplete_name=".png")
    # ]
    segments = [
        ProductionFieldSegments.FIELD_DESIGNATION(Fields.FOUNDATION_FIELD_ONE),
        ProductionFieldSegments.OG_VID_FILES(incomplete_name=".mp4"),
    ]

    print(segments)
    matching_mps = get_matching_s3_mirror_paths(segments)

    print(f"{len(matching_mps)} matching keys found.")
    print(matching_mps)
    # with FileSizeTQDMCallback(matching_mps, is_download=True) as tqdm_cb:
    for mp in matching_mps:
        mp.download_to_mirror()
    # with ProcessPoolExecutor(max_workers=30) as executor:
    #     for mp in matching_mps:
    #         executor.submit(mp.download_to_mirror, overwrite=True)
