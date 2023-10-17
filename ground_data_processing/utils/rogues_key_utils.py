"""Utils for rogues s3 keys."""
import string
from enum import StrEnum
from typing import List, Tuple, Union

from S3MP.keys import KeySegment, get_matching_s3_keys

from ground_data_processing.utils.relative_segment_groups import (
    FieldNameSegment,
    PlantingTrialRootSegments,
)
from ground_data_processing.utils.s3_constants import (
    DataTypes,
    Fields,
    Framerates,
    PlantingNumbers,
    Resolutions,
    TrialTypes,
)


def determine_pass_flag_and_pass_char(
    folder_name: str, row_number: int
) -> Tuple[bool, str]:
    """Determine if a folder name is a reverse pass and the pass char."""
    ints_in_folder_name = [int(i) for i in folder_name.split(", ")]
    pass_idx = ints_in_folder_name.index(row_number)
    pass_char = string.ascii_uppercase[pass_idx]

    return (pass_idx % 2 != 0), pass_char


def plot_trial_prefix_segment_builder(
    base_folder: str = Fields.FC_2022,
    trial_type: str = TrialTypes.SMALL_PLOT,
    planting_number: Union[str, int] = PlantingNumbers.PLANTING_ONE,
    data_type: StrEnum = DataTypes.VIDEOS,
) -> List[KeySegment]:
    """Segment builder for most common prefixes."""
    segments = []
    if base_folder:
        segments.append(FieldNameSegment.FIELD_NAME(base_folder))
    if trial_type:
        segments.append(PlantingTrialRootSegments.TRIAL_TYPE(trial_type))
    if planting_number:
        if isinstance(planting_number, int):
            planting_number = list(PlantingNumbers)[planting_number - 1]
        segments.append(PlantingTrialRootSegments.PLANTING_NUMBER(planting_number))
    if data_type:
        segments.append(KeySegment(depth=3, name=str(data_type)))

    return sorted(segments, key=lambda x: x.depth)


def rogues_key_video_plot_segment_builder(
    date: str | StrEnum = None,
    row_number: int = None,
    resolution: Resolutions = Resolutions.r4k,
    framerate: Framerates = Framerates.fps120,
    plot_idx: int = None,
    plot_split_file: str = None,
    return_reverse_flag: bool = False,
    existing_segments: List[KeySegment] = None,
) -> Tuple[List[KeySegment], bool]:
    """Filter builder using human indexing."""
    pass_reverse_flag = None

    segments = [] if existing_segments is None else existing_segments
    if date:
        if isinstance(date, StrEnum):
            date = date.value
        segments.append(KeySegment(depth=4, name=date))
    if row_number:
        """
        Row numbers can appear as tuples: (7, 6).
        We're searching for a single number, so we must find the tuple that contains the number.
        If it's unique, we also flag whether it's a reverse pass.
        e.g. for (7, 6) 7 is the forward pass and 6 is the reverse pass.
        """
        row_designation_seg = KeySegment(5)  # TODO not hardcode
        search_segs = segments + [row_designation_seg]
        possible_folders = get_matching_s3_keys(search_segs)
        folder_names = [folder.split("/")[-2] for folder in possible_folders]
        folder_names = [f_n for f_n in folder_names if f"{row_number}" in f_n]
        unique_folder_names = list(set(folder_names))
        if len(unique_folder_names) > 1:
            print("Multiple row number folders found, pass flag cannot be set.")
            segments.append(row_designation_seg)
        elif not unique_folder_names:
            print("No row number folders found.")
        else:
            selected_folder_name = unique_folder_names[0]
            pass_reverse_flag, pass_char = determine_pass_flag_and_pass_char(
                selected_folder_name, row_number
            )
            segments.append(row_designation_seg(selected_folder_name))
            segments.append(KeySegment(depth=6, name=f"Pass {pass_char}"))
    if resolution and framerate:
        segments.append(KeySegment(depth=7, name=f"{resolution}@{framerate}"))
    if plot_idx:
        if pass_reverse_flag is None:
            print(
                "Pass flag not set, cannot search for plot. Try passing in a row number."
            )
        else:
            adjusted_idx = (22 - plot_idx) if pass_reverse_flag else (plot_idx - 1)
            segments.append(KeySegment(depth=8, name=f"Plot {adjusted_idx:02d}"))
    if plot_split_file:
        segments.append(KeySegment(depth=9, name=plot_split_file))

    segments = sorted(segments, key=lambda x: x.depth)
    return (segments, pass_reverse_flag) if return_reverse_flag else segments
