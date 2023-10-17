"""Script to scrape S3 for all existing fields and push them to ddb."""
from typing import List, Tuple

from S3MP.mirror_path import MirrorPath
from tqdm import tqdm

from ddb_tracking.grd_api import put_grd_row
from ddb_tracking.grd_constants import Field
from ddb_tracking.grd_structure import GRDPlantGroup, GRDRow, MultiViewMP
from ground_data_processing.utils.absolute_segment_groups import ImageSegments
from ground_data_processing.utils.plot_layout import (
    convert_plot_idx_to_plot_number,
    get_reverse_flag_from_pass_mp,
    get_rogue_and_base_type,
    get_row_number_from_pass_mp,
)
from ground_data_processing.utils.s3_constants import (
    DataFiles,
    DataTypes,
    Dates,
    VideoRez,
    get_week_idx_from_date_str,
)


def get_vid_split_helper(
    root_mp: MirrorPath, vid_suffix: str = ".mp4", image_dir_suffix: str = " Raw Images"
) -> List[Tuple[MultiViewMP, MultiViewMP]]:
    """Parse video splits."""
    vid_splits = []
    for split_mp in root_mp.get_children_on_s3():
        video_mvmp = MultiViewMP.from_root_mp(split_mp, suffix=vid_suffix)
        image_mvmp = MultiViewMP.from_root_mp(split_mp, suffix=image_dir_suffix)
        vid_splits.append((video_mvmp, image_mvmp))
    return vid_splits


def rogue_label_precedence_helper(
    root_mp: MirrorPath,
) -> MirrorPath:
    """Determine which rogue label to use."""
    for rogue_label_name in DataFiles.RogueLabelJSONs:
        rogue_label_mp = root_mp.get_child(rogue_label_name)
        if rogue_label_mp.exists_on_s3():
            return rogue_label_mp
    return None


def scrape_single_plot_trial_row(
    pass_root_mp: MirrorPath, grd_field_name: str, ordered_dates: List[str]
) -> GRDRow:
    """Scrape a single plot trial row."""
    pass_vid_mps = MultiViewMP.from_root_mp(pass_root_mp, suffix=".mp4")
    # Root MP setup
    plot_trial_root_mp = pass_root_mp.trim(3)
    filtered_images_root_mp = plot_trial_root_mp.get_child(DataTypes.IMAGES)
    unfiltered_images_root_mp = plot_trial_root_mp.get_child(
        DataTypes.UNFILTERED_IMAGES
    )

    pass_plots = get_vid_split_helper(
        pass_root_mp.get_child("Rel Plots"),
        vid_suffix=".mp4",
        image_dir_suffix=" Raw Images",
    )
    reverse_pass_flag = get_reverse_flag_from_pass_mp(pass_root_mp)
    row_number = get_row_number_from_pass_mp(pass_root_mp)
    date_str = pass_root_mp.key_segments[4].name

    week_number = get_week_idx_from_date_str(date_str, ordered_dates) + 1
    plant_groups = []
    for idx, (vid_mvmp, image_mvmp) in enumerate(pass_plots):
        # Construct raw image group
        raw_img_group = GRDPlantGroup(
            videos=vid_mvmp,
            image_directories=image_mvmp,
            is_plot=True,
            is_reverse_pass=reverse_pass_flag,
        )
        plant_groups.append(raw_img_group)
        # Search for filtered images.
        plot_number = convert_plot_idx_to_plot_number(idx, reverse_pass_flag)
        rogue_type, base_type = get_rogue_and_base_type(row_number, plot_number)
        filtered_key_segments = [
            ks.__copy__() for ks in filtered_images_root_mp.key_segments
        ]
        filtered_key_segments.extend(
            [
                ImageSegments.ROGUE_TYPE(rogue_type),
                ImageSegments.BASE_TYPE(base_type),
                ImageSegments.ROW_AND_RANGE(f"Row{row_number:02d}Col{plot_number:02d}"),
                ImageSegments.WEEK_NUMBER(f"Week {week_number}"),
                ImageSegments.RESOLUTION("4k"),
            ]
        )
        filtered_plot_folder_mp = MirrorPath(filtered_key_segments)
        if filtered_plot_folder_mp.exists_on_s3():
            filtered_vid_mvmp = MultiViewMP.from_root_mp(
                filtered_plot_folder_mp, suffix=".mp4"
            )
            filtered_images_mvmp = MultiViewMP.from_root_mp(
                filtered_plot_folder_mp, suffix=""
            )
            rogue_label_mp = rogue_label_precedence_helper(filtered_plot_folder_mp)
            filtered_img_group_mp = GRDPlantGroup(
                videos=filtered_vid_mvmp,
                image_directories=filtered_images_mvmp,
                rogue_labels=rogue_label_mp,
                is_plot=True,
                is_reverse_pass=reverse_pass_flag,
                is_filtered=True,
                is_reverse_order_corrected=True,
            )
            plant_groups.append(filtered_img_group_mp)

        # Search for unfiltered images.
        unfiltered_key_segments = [
            ks.__copy__() for ks in unfiltered_images_root_mp.key_segments
        ]
        unfiltered_key_segments.extend(
            [
                ImageSegments.ROGUE_TYPE(rogue_type),
                ImageSegments.BASE_TYPE(base_type),
                ImageSegments.ROW_AND_RANGE(f"Row{row_number:02d}Col{plot_number:02d}"),
                ImageSegments.WEEK_NUMBER(f"Week {week_number}"),
                ImageSegments.RESOLUTION("4k"),
            ]
        )
        unfiltered_plot_folder_mp = MirrorPath(unfiltered_key_segments)
        if unfiltered_plot_folder_mp.exists_on_s3():
            unfiltered_vid_mvmp = MultiViewMP.from_root_mp(
                unfiltered_plot_folder_mp, suffix=".mp4"
            )
            unfiltered_images_mvmp = MultiViewMP.from_root_mp(
                unfiltered_plot_folder_mp, suffix=""
            )
            rogue_label_mp = rogue_label_precedence_helper(unfiltered_plot_folder_mp)
            unfiltered_img_group_mp = GRDPlantGroup(
                videos=unfiltered_vid_mvmp,
                image_directories=unfiltered_images_mvmp,
                rogue_labels=rogue_label_mp,
                is_plot=True,
                is_filtered=False,
                is_reverse_pass=reverse_pass_flag,
                is_reverse_order_corrected=True,
            )
            plant_groups.append(unfiltered_img_group_mp)

    return GRDRow(
        field_name=grd_field_name,
        date=date_str,
        row_number=row_number,
        full_row_video_mps=pass_vid_mps,
        plant_groups=plant_groups,
    )


def scrape_plot_trial_planting(
    root_mp: MirrorPath, grd_field_name: Field, ordered_dates: List[str]
):
    """Scrape plot trial planting data."""
    # Loop through Videos
    vid_root_mp = root_mp.get_child(DataTypes.VIDEOS)
    date_mps = vid_root_mp.get_children_on_s3()
    for date_mp in tqdm(date_mps):
        # Loop through full videos (2 rows each)
        full_vid_mps = date_mp.get_children_on_s3()
        for full_vid_mp in tqdm(full_vid_mps):
            # Only deal with full res
            full_vid_mp = full_vid_mp.get_child(VideoRez.r4k_120fps.value)
            pass_a = full_vid_mp.get_child("Pass A")
            pass_a_grd_row = scrape_single_plot_trial_row(
                pass_a, grd_field_name, ordered_dates
            )
            put_grd_row(pass_a_grd_row)

            pass_b = full_vid_mp.get_child("Pass B")
            pass_b_grd_row = scrape_single_plot_trial_row(
                pass_b, grd_field_name, ordered_dates
            )
            put_grd_row(pass_b_grd_row)


if __name__ == "__main__":
    grd_field_name = Field.FARMER_CITY_2022_PLOT_TRIAL_PLANTING_ONE
    root_mp = MirrorPath.from_s3_key("Farmer City 2022/Small Plot/Planting 1")
    scrape_plot_trial_planting(
        root_mp,
        grd_field_name,
        ordered_dates=[d.value for d in Dates.FarmerCity.ALL_DATES],
    )
