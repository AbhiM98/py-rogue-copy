"""Constants."""
from enum import StrEnum


class ConfigKeys:
    """Config keys.""" ""

    class BaseKeys(StrEnum):
        """Base keys."""

        RUN_TYPE = "RUN_TYPE"
        DATA = "DATA"
        AUGMENTATIONS = "AUGMENTATIONS"

    class RunType(StrEnum):
        """Run type keys."""

        AUGMENT = "augment"
        MERGE = "merge"

    class Data(StrEnum):
        """Data keys."""

        DATASET_S3_PATH = "dataset_s3_path"
        IMAGES_FOLDER_NAME = "images_folder_name"
        ANNOTATIONS_FOLDER_NAME = "annotations_folder_name"
        ANNOTATIONS_FILE_NAME = "annotations_file_name"
        DESTINATION_S3_PATH = "destination_s3_path"
        DEFAULT_HEIGHT = "default_height"
        DEFAULT_WIDTH = "default_width"
        OVERWRITE = "overwrite"
        # merge only
        NUMBER = "number"

    class Augmentations(StrEnum):
        """Augmentations keys."""

        RANDOM_CROP = "RANDOM_CROP"
        RANDOM_ZOOM = "RANDOM_ZOOM"

    class RandomCrop(StrEnum):
        """Random crop keys."""

        NUMBER = "number"
        HEIGHT = "crop_height"
        WIDTH = "crop_width"

    class RandomZoom(StrEnum):
        """Random zoom keys."""

        NUMBER = "number"
        HEIGHT = "crop_height"
        WIDTH = "crop_width"


class CocoKeys(StrEnum):
    """Coco keys."""

    INFO = "info"
    LICENSES = "licenses"
    IMAGES = "images"
    ANNOTATIONS = "annotations"
    CATEGORIES = "categories"


class CocoImageKeys(StrEnum):
    """Coco image keys."""

    ID = "id"
    FILE_NAME = "file_name"
    HEIGHT = "height"
    WIDTH = "width"


class CocoAnnotationKeys(StrEnum):
    """Coco annotation keys."""

    ID = "id"
    IMAGE_ID = "image_id"
    CATEGORY_ID = "category_id"
    BBOX = "bbox"
    AREA = "area"
    SEGMENTATION = "segmentation"
    IS_CROWD = "iscrowd"


class CocoCategoryKeys(StrEnum):
    """Coco category keys."""

    ID = "id"
    NAME = "name"


DEFAULT_IMAGE_HEIGHT = 1024
DEFAULT_IMAGE_WIDTH = 1024
AUGMENTATION_SUFFIXES = {
    ConfigKeys.Augmentations.RANDOM_CROP: "crop",
    ConfigKeys.Augmentations.RANDOM_ZOOM: "zoom",
}
