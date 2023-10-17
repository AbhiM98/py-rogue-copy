"""Segment definitions for the rogues project."""

from S3MP.keys import KeySegment


class RootSegments:
    """Root directory segments for S3."""

    PLOT_NAME_AND_YEAR = KeySegment(0)
    TRIAL_TYPE = KeySegment(1)
    PLANTING_NUMBER = KeySegment(2)
    DATA_TYPE = KeySegment(3)


class VideoSegments:
    """Video directory segments for S3."""

    DATE = KeySegment(4)
    ROW_DESIGNATION = KeySegment(5)
    RESOLUTION = KeySegment(6)
    OG_VID_FILES = KeySegment(7, is_file=True)
    ROW_SPLIT = KeySegment(7)
    ROW_SPLIT_FILES = KeySegment(8, is_file=True)
    PLOT_SPLIT = KeySegment(8)
    PLOT_SPLIT_IDX = KeySegment(9)
    PLOT_SPLIT_FILES = KeySegment(10, is_file=True)
    IMAGE_DUMP = KeySegment(10)
    IMAGE_DUMP_FILES = KeySegment(11, is_file=True)


class ImageSegments:
    """Image directory segments for S3."""

    ROGUE_TYPE = KeySegment(4)
    BASE_TYPE = KeySegment(5)
    ROW_AND_RANGE = KeySegment(6)
    WEEK_NUMBER = KeySegment(7)
    RESOLUTION = KeySegment(8)
    IMAGE_DATA_FILES = KeySegment(9, is_file=True)
    CAMERA_VIEW = KeySegment(9)
    IMAGE_DUMP = KeySegment(10)
    IMAGE_DUMP_FILES = KeySegment(11, is_file=True)


class InferenceSegments:
    """Inference segments for S3."""

    MODEL_NAME = KeySegment(4)
    MODEL_FILES = KeySegment(5, is_file=True)
    PREPROC_METHOD = KeySegment(5)
    ROGUE_TYPE = KeySegment(6)
    BASE_TYPE = KeySegment(7)
    ROW_AND_RANGE = KeySegment(8)
    WEEK_NUMBER = KeySegment(9)
    RESOLUTION = KeySegment(10)
    CAMERA_VIEW = KeySegment(11)
    OUTPUT_SEGM_JSON_FILES = KeySegment(12, is_file=True)
    DATA_DIR = KeySegment(12)
    INPUT_IMAGE_FILES = KeySegment(13, is_file=True)
    OUTPUT_IMAGE_FILES = KeySegment(13, is_file=True)
    OUTPUT_MASK_FILES = KeySegment(13, is_file=True)


class ProdInferenceSegments:
    """Production inference segments for S3."""

    MODEL_NAME = KeySegment(2)
    MODEL_FILES = KeySegment(3, is_file=True)
    PREPROC_METHOD = KeySegment(3)
    DATE = KeySegment(4)
    ROW_DESIGNATION = KeySegment(5)
    ROW_SPLIT_DESIGNATION = KeySegment(6)
    DS_SPLIT_ROOT_FOLDER = KeySegment(6)
    DS_SPLIT_IDX = KeySegment(7)
    DS_SPLIT_CAMERA_VIEW = KeySegment(8)
    DATA_DIR = KeySegment(9)
    OUTPUT_SEGM_JSON_FILES = KeySegment(9, is_file=True)
    INPUT_IMAGE_FILES = KeySegment(10, is_file=True)
    OUTPUT_IMAGE_FILES = KeySegment(10, is_file=True)


class ProductionFieldSegments:
    """Production field directory segments for S3."""

    FIELD_DESIGNATION = KeySegment(0)
    DATA_TYPE = KeySegment(1)
    DATE = KeySegment(2)
    ROW_DESIGNATION = KeySegment(3)
    RESOLUTION = KeySegment(4)
    OG_VID_FILES = KeySegment(5, is_file=True)
    IMAGE_DUMP = KeySegment(5)
    IMAGE_DUMP_FILES = KeySegment(6, is_file=True)
    DS_SPLIT = KeySegment(5)
    DS_SPLIT_IDX = KeySegment(6)
    DS_SPLIT_FILES = KeySegment(7, is_file=True)
    DS_SPLIT_IMG_DUMP = KeySegment(7)
    DS_SPLIT_IMG_DUMP_FILES = KeySegment(8, is_file=True)


class ProductionFieldWithSplitPassSegments:
    """Production field directory segments when there's multiple passes."""

    ROW_SPLIT = KeySegment(5)
    ROW_SPLIT_FILES = KeySegment(6, is_file=True)
    DS_SPLIT = KeySegment(6)
    DS_SPLIT_IDX = KeySegment(7)
    DS_SPLIT_FILES = KeySegment(8, is_file=True)
    DS_SPLIT_IMG_DUMP = KeySegment(8)
    DS_SPLIT_IMG_DUMP_FILES = KeySegment(9, is_file=True)


class ProductionFieldImageSegments:
    """Production field image directory segments for S3."""

    DATA_TYPE = KeySegment(1)
    DATE = KeySegment(2)
    ROW_DESIGNATION = KeySegment(3)
    IMAGE_DUMP = KeySegment(4)
    IMAGE_DUMP_FILES = KeySegment(5, is_file=True)
    GROUPED_IMAGE_DUMP = KeySegment(4)
    GROUPED_IMAGE_DUMP_ID = KeySegment(5)
    GROUPED_IMAGE_DUMP_FILES = KeySegment(6, is_file=True)
    DS_SPLIT_IMAGE_DUMP = KeySegment(4)
    DS_SPLIT_IMAGE_DUMP_IDX = KeySegment(5)
    DS_SPLIT_IMAGE_DATA_FILES = KeySegment(6, is_file=True)
    DS_SPLIT_CAMERA_VIEW = KeySegment(6)
    DS_SPLIT_IMAGES_DUMP_FILES = KeySegment(7, is_file=True)
