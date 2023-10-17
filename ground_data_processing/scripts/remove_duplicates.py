"""Remove duplicates and crop images for processing."""
import concurrent.futures
from typing import List

import psutil
from S3MP.mirror_path import MirrorPath, get_matching_s3_mirror_paths
from tqdm import tqdm

from ground_data_processing.utils.absolute_segment_groups import (
    ImageSegments,
    RootSegments,
    VideoSegments,
)
from ground_data_processing.utils.plot_layout import get_rogue_and_base_type
from ground_data_processing.utils.rogues_key_utils import (
    plot_trial_prefix_segment_builder,
    rogues_key_video_plot_segment_builder,
)
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFiles,
    DataTypes,
    Dates,
    Framerates,
    QASMJsonClasses,
    Resolutions,
    get_week_idx_from_date_str,
)


# TODO handle crop?
def spread_normal_image(
    raw_img_key,
    image_idx,
    cam_input_mps: List[MirrorPath],
    cam_output_mps: List[MirrorPath],
):
    """Spread a normal image from the raw img directory to the output directory."""
    for cam_input_mp, cam_output_mp in zip(cam_input_mps, cam_output_mps):
        raw_img_mp = cam_input_mp.get_child(f"{raw_img_key}.png")
        img = raw_img_mp.load_local(download=True, overwrite=True)
        out_mp = cam_output_mp.get_child(f"plant_{image_idx:02d}.png")
        out_mp.save_local(img, upload=True, overwrite=True)


if __name__ == "__main__":
    N_THREADS = psutil.cpu_count(logical=False) // 2
    OVERWRITE = False
    IN_RES = Resolutions.r4k
    FRAMERATE = Framerates.fps120

    ROW_NUMBER = 7
    DATE = "6-28"
    base_segments = plot_trial_prefix_segment_builder(planting_number=1)
    segments, pass_reverse_flag = rogues_key_video_plot_segment_builder(
        date=DATE,
        row_number=ROW_NUMBER,
        plot_split_file=DataFiles.DUPLICATES_JSON,
        return_reverse_flag=True,
        existing_segments=base_segments,
    )
    # segments = [
    #     ProductionFieldSegments.FIELD_DESIGNATION(DataFolders.FOUNDATION_FIELD_TWO),
    #     ProductionFieldSegments.DATA_TYPE(DataFolders.VIDEOS),
    #     ProductionFieldSegments.DATE("7-08"),
    #     ProductionFieldSegments.ROW_DESIGNATION("Row 6, 11"),
    #     # ProductionFieldSegments.DS_SPLIT(DataFolders.DS_SPLITS),
    #     # ProductionFieldSegments.DS_SPLIT_FILES(DataFiles.DUPLICATES_JSON)
    #     ProductionFieldWithSplitPassSegments.ROW_SPLIT("Pass B"),
    #     ProductionFieldWithSplitPassSegments.DS_SPLIT(DataFolders.DS_SPLITS),
    #     ProductionFieldWithSplitPassSegments.DS_SPLIT_FILES(DataFiles.DUPLICATES_JSON)
    # ]

    matching_mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(matching_mps)} matching mirror paths.")

    for dup_json_mp in matching_mps:
        filter_data = dup_json_mp.load_local(download=True, overwrite=True)
        classes = [f_data["class"] for f_data in list(filter_data.values())]
        if QASMJsonClasses.MISSING in classes:
            print("Missing class found, searching for fixed JSON.")
            dup_fixed_json_mp = dup_json_mp.get_sibling(
                DataFiles.DUPLICATES_MISSING_FIXED_JSON
            )
            if dup_fixed_json_mp.exists_on_s3():
                print("Found fixed JSON, loading.")
                filter_data = dup_fixed_json_mp.load_local(
                    download=True, overwrite=True
                )
                classes = [f_data["class"] for f_data in list(filter_data.values())]
                dup_json_mp = dup_fixed_json_mp
            else:
                print("No fixed JSON found, skipping.")
                continue

        cam_input_folder_mps = [
            dup_json_mp.get_sibling(f"{camera} Raw Images") for camera in CameraViews
        ]

        plot_number = int(
            dup_json_mp.get_key_segment(VideoSegments.PLOT_SPLIT_IDX.depth)
        )
        plot_number = 21 - plot_number if pass_reverse_flag else plot_number
        plot_number += 1

        WEEK = get_week_idx_from_date_str(DATE, Dates.FC_2022) + 1
        ROGUE_TYPE, BASE_TYPE = get_rogue_and_base_type(ROW_NUMBER, plot_number)

        root_output_folder_mp = dup_json_mp.trim(4).replace_key_segments(
            [
                RootSegments.DATA_TYPE(DataTypes.IMAGES),
                ImageSegments.ROGUE_TYPE(ROGUE_TYPE),
                ImageSegments.BASE_TYPE(BASE_TYPE),
                ImageSegments.ROW_AND_RANGE(f"Row{ROW_NUMBER:02d}Col{plot_number:02d}"),
                ImageSegments.WEEK_NUMBER(f"Week {WEEK}"),
                ImageSegments.RESOLUTION(f"{IN_RES}"),
            ]
        )
        # ds_split_idx = dup_json_mp.get_key_segment(ProductionFieldWithSplitPassSegments.DS_SPLIT_IDX.depth)
        # ds_split_idx = dup_json_mp.get_key_segment(ProductionFieldSegments.DS_SPLIT_IDX.depth)

        # TODO cleaner trim
        # root_output_folder_mp = dup_json_mp.trim(4).replace_key_segments(
        #     [
        #         ProductionFieldImageSegments.DATA_TYPE(DataFolders.IMAGES),
        #         ProductionFieldImageSegments.ROW_DESIGNATION("Row 11"),
        #         ProductionFieldImageSegments.DS_SPLIT_IMAGE_DUMP(DataFolders.DS_SPLITS),
        #         ProductionFieldImageSegments.DS_SPLIT_IMAGE_DUMP_IDX(ds_split_idx),
        #     ],
        # )
        camera_output_mps = [
            root_output_folder_mp.get_child(cam_view) for cam_view in CameraViews
        ]

        if OVERWRITE:
            for cam_mp in camera_output_mps:
                cam_mp.delete_children_on_s3()
        else:
            child_lens = [
                len(cam_mp.get_children_on_s3()) for cam_mp in camera_output_mps
            ]
            if any(child_lens):
                print("Skipping as it's images already exist.")
                continue
        print(
            f"Filtering images from\n{dup_json_mp.s3_key}\nto\n{root_output_folder_mp.s3_key}"
        )

        image_idx = 1  # renaming for simplicity

        # Make list of filter data sorted by key
        filter_data = sorted(
            filter_data.items(), key=lambda x: x[0], reverse=pass_reverse_flag
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            for key, value in tqdm(filter_data):
                match value["class"]:
                    case QASMJsonClasses.NORMAL | QASMJsonClasses.MISSING_FIXED | QASMJsonClasses.PLANT:
                        executor.submit(
                            spread_normal_image,
                            key,
                            image_idx,
                            cam_input_folder_mps,
                            camera_output_mps,
                        )
                        image_idx += 1
                    case QASMJsonClasses.DUPLICATE:
                        continue
                    case QASMJsonClasses.MISSING:
                        break
                        # raise NotImplementedError
                    case _:
                        break
                        # raise ValueError("Invalid class.")
            else:
                executor.shutdown(wait=True)
                continue

            # We broke early, delete progress
            executor.shutdown(wait=True)
            print("Exited early, deleting progress.")
            for cam_mp in camera_output_mps:
                print(f"Deleting {cam_mp.s3_key}")
                cam_mp.delete_children_on_s3()

        exit()
