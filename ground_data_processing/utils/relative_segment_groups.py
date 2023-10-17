"""Relative KeySegment Groups."""
from typing import Dict, List

from S3MP.keys import KeySegment


class SegmentGroupMetaclass(type):
    """Base class for KeySegment groups."""

    _segments_by_instance: Dict[str, Dict[str, KeySegment]] = {}

    def __iter__(cls):
        """Iterate over the KeySegments."""
        yield from cls._segments_by_instance[cls.__name__].items()

    def __call__(cls):
        """Return the KeySegments."""
        return cls.__iter__()

    def as_dict(cls, copy: bool = True):
        """Return the KeySegments as a dict."""
        if copy:
            return cls._segments_by_instance[cls.__name__].copy()
        return cls._segments_by_instance[cls.__name__]

    def __items__(cls):
        """Return the KeySegments."""
        return cls._segments_by_instance[cls.__name__].items()

    def items(cls):
        """Return the KeySegments."""
        return cls.__items__()

    def __values__(cls):
        """Return the KeySegments."""
        return cls._segments_by_instance[cls.__name__].values()

    def values(cls):
        """Return the KeySegments."""
        return [ks.__copy__() for ks in cls.__values__()]

    def values_as_list(cls) -> List[KeySegment]:
        """Return the KeySegments as a list."""
        return list(cls.__values__())

    def __keys__(cls):
        """Return the KeySegments."""
        return cls._segments_by_instance[cls.__name__].keys()

    def keys(cls):
        """Return the KeySegments."""
        return cls.__keys__()

    def keys_as_list(cls) -> List[str]:
        """Return the KeySegments as a list."""
        return list(cls.__keys__())

    def __getitem__(cls, key):
        """Return the KeySegment."""
        return cls._segments_by_instance[cls.__name__][key]

    def __contains__(cls, key: str | KeySegment):
        """Return if the KeySegment exists."""
        if isinstance(key, str):
            return key in cls._segments_by_instance[cls.__name__]
        elif isinstance(key, KeySegment):
            segment_ids = [
                id(segment)
                for segment in cls._segments_by_instance[cls.__name__].values()
            ]
            return id(key) in segment_ids

    def __new__(cls, clsname, bases, attrs):
        """Create a new instance of the class."""
        if clsname not in cls._segments_by_instance:
            cls._segments_by_instance[clsname] = {}
        for attr in attrs.keys():
            if not attr.startswith("__"):
                cls._segments_by_instance[clsname][attr] = attrs[attr]
        return super().__new__(cls, clsname, bases, attrs)

    def depths(cls):
        """Return the depths of the KeySegments."""
        return [
            segment.depth
            for segment in cls._segments_by_instance[cls.__name__].values()
        ]

    def unique_depths(cls):
        """Return the unique depths of the KeySegments."""
        return set(cls.depths())

    def unique_folder_depths(cls):
        """Return the unique folder depths of the KeySegments."""
        return {
            segment.depth
            for segment in cls._segments_by_instance[cls.__name__].values()
            if not segment.is_file
        }


class CommonSegments:
    """Common segments."""

    OG_VID_FILES = KeySegment(0, is_file=True, incomplete_name=".mp4")
    FIELD_NAME = KeySegment(0)


# Root Segments
class FieldNameSegment(metaclass=SegmentGroupMetaclass):
    """Root segments."""

    FIELD_NAME = CommonSegments.FIELD_NAME.__copy__()


class PlantingTrialRootSegments(metaclass=SegmentGroupMetaclass):
    """Root segments for planting trials."""

    # MATCHING_SEGMENTS = [Fields.FC_2022]

    FIELD_NAME = CommonSegments.FIELD_NAME.__copy__()
    TRIAL_TYPE = KeySegment(1)
    PLANTING_NUMBER = KeySegment(2)


class IsolatedVideoSegments(metaclass=SegmentGroupMetaclass):
    """Isolated video with no further need for splitting."""

    OG_VID_FILES = CommonSegments.OG_VID_FILES
    RAW_IMAGE_DUMP_CAMERA_VIEW = KeySegment(0)
    RAW_IMAGE_DUMP_FILES = KeySegment(1, is_file=True)


# Video Split Segments
class DSSplitSegments(metaclass=SegmentGroupMetaclass):
    """Video that has been split into 10s chunks."""

    OG_VID_FILES = CommonSegments.OG_VID_FILES
    DS_SPLIT_ROOT_FOLDER = KeySegment(0, "DS Splits")
    DS_SPLIT_IDX = KeySegment(1)
    DS_SPLIT_CAMERA_VIEW = KeySegment(2)
    DS_SPLIT_FILES = KeySegment(3, is_file=True)


class RowSplitVideoSegments(metaclass=SegmentGroupMetaclass):
    """Video that has been split into separate row passes."""

    OG_VID_FILES = CommonSegments.OG_VID_FILES
    ROW_DESIGNATION = KeySegment(0)
    ROW_SPLIT_FILES = KeySegment(1, is_file=True)


class PlotSplitVideoSegments(metaclass=SegmentGroupMetaclass):
    """Video that has been split into separate plot passes."""

    OG_VID_FILES = CommonSegments.OG_VID_FILES
    PLOT_SPLIT_ROOT_FOLDER = KeySegment(0)
    PLOT_SPLIT_IDX = KeySegment(1)
    PLOT_SPLIT_FILES = KeySegment(2, is_file=True)


# Data type segments


class DataTypeSegment(metaclass=SegmentGroupMetaclass):
    """Segment for data designation."""

    DATA_TYPE = KeySegment(0)


class DateAndRowSegments(metaclass=SegmentGroupMetaclass):
    """Segments for date and row designation."""

    DATE = KeySegment(0)
    ROW_DESIGNATION = KeySegment(1)


class ResolutionSegment(metaclass=SegmentGroupMetaclass):
    """Segment for resolution designation."""

    RESOLUTION = KeySegment(0)


class ModelInferenceSegments(metaclass=SegmentGroupMetaclass):
    """Segments for model inference."""

    MODEL_NAME = KeySegment(0)
    MODEL_FILES = KeySegment(1, is_file=True)
    PREPROC_METHOD = KeySegment(1)


class FCPlotDesignationSegments(metaclass=SegmentGroupMetaclass):
    """Segments for FC plot designation."""

    ROGUE_TYPE = KeySegment(0)
    BASE_TYPE = KeySegment(1)
    ROW_AND_PLOT_DESIGNATION = KeySegment(2)
    WEEK_NUMBER = KeySegment(3)


def align_segment_depths(
    segments: List[KeySegment],
    base_depth: int = 0,
) -> List[KeySegment]:
    """Offset and align the segments."""
    segments = [seg.__copy__() for seg in segments]
    for idx, seg in enumerate(segments):
        seg.depth = base_depth + idx
    return [seg.__copy__() for seg in segments]
