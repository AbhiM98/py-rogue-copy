"""S3 Constants."""
from __future__ import annotations

from enum import Enum, StrEnum
from typing import List

ROGUES_BUCKET_KEY = "sentera-rogues-data"
CROP_SIZE = 1024


class Resolutions:
    """Resolutions."""

    class _Resolution:
        """Resolution class helper."""

        def __init__(self, width, height):
            """Initialize resolution."""
            self.width = width
            self.height = height

        def __str__(self):
            """Return resolution as a string.""" ""
            return f"{self.width}x{self.height}"

        def as_tuple(self):
            """Return resolution as a tuple."""
            return (self.width, self.height)

    r640x480 = r480p = _Resolution(640, 480)
    r1280x720 = r720p = _Resolution(1280, 720)
    r1920x1080 = r1080p = _Resolution(1920, 1080)
    r3840x2160 = r2160p = r4k = _Resolution(3840, 2160)


class Framerates:
    """Framerates."""

    class _Framerate:
        """Framerate class helper."""

        def __init__(self, fps, display_fps=None):
            self.fps = fps
            self.display_fps = display_fps if display_fps is not None else fps

        def __str__(self):
            return f"{self.display_fps}fps"

    fps60 = _Framerate(fps=59.94, display_fps=60)
    fps120 = _Framerate(fps=119.88, display_fps=120)


class VideoRez(StrEnum):
    """Video resolutions and framerates."""

    r4k_120fps = f"{Resolutions.r4k}@{Framerates.fps120}"


class CameraViews(StrEnum):
    """Camera views."""

    NADIR = "nadir"
    BOTTOM = "bottom"
    OBLIQUE = "oblique"


class DataFolders:
    """Data folder names."""

    REL_PLOT_SPLIT = "Rel Plots"
    RAW_IMAGES = "Raw Images"
    THUMBNAILS = "Thumbnails"
    DS_SPLITS = "DS Splits"
    PREPROCESSED_IMAGES = "Preprocessed Images"
    ANNOTATED_IMAGES = "Annotated Images"
    INFERENCE_MASKS = "Inference Masks"


class Fields(StrEnum):
    """Field folder names."""

    PROD_FIELD_ONE = "Production Field 1 (Argo North-Home Minor)"
    FOUNDATION_FIELD_ONE = "Foundation Field 1"
    FOUNDATION_FIELD_TWO = "Foundation Field 2 (Dennis Zuber)"
    FC_2022 = "Farmer City 2022"


class DataTypes(StrEnum):
    """Data type folder names."""

    VIDEOS = "Videos"
    IMAGES = "Images"
    UNFILTERED_IMAGES = "Unfiltered Images"
    MODEL_INFERENCE = "Model Inference"
    UNFILTERED_MODEL_INFERENCE = "Unfiltered Model Inference"


class ProcessingSteps(StrEnum):
    """Processing step names."""

    GENERATE_GPS_OFFSETS = "Generate GPS Offsets"
    GENERATE_DS_SPLITS = "Generate DS Splits"
    EXTRACT_FRAMES = "Extract Frames"
    PREP_INFERENCE = "Prep Inference"
    PREP_AND_RUN_INFERENCE = "Prep and Run Inference"
    RUN_INFERENCE = "Run Inference"
    RUN_ALL = "Run All"


class TrialTypes(StrEnum):
    """Trial type folder names."""

    SMALL_PLOT = "Small Plot"
    STRIP_TRIAL = "Strip Trial"


class PlantingNumbers(StrEnum):
    """Planting number folder names."""

    PLANTING_ONE = "Planting 1"
    PLANTING_TWO = "Planting 2"


class SplitTypes(Enum):
    """Split type enum."""

    ROW_SPLIT_VIDEO = 0
    PLOT_SPLIT_VIDEO = 1
    DS_SPLIT_VIDEO = 2
    ISOLATED_VIDEO = 3


class ModelNames(StrEnum):
    """Model names."""

    SOLO_V2_DEC_11_MODEL = "Solo V2 Dec 11 Model (ResNet101-PaddleDetection)"
    SOLO_V2_SEPT_05_MODEL = "Solo V2 Sept 05 Model (ResNet101-PaddleDetection)"


class InferenceMethods(StrEnum):
    """Inference methods."""

    SQUARE_CROP = "Center Square Crop"
    PADDLE_SLICE = "Paddle Slice"


class PythonEntrypoints:
    """Python entrypoints for running inference."""

    DEFAULT = ["/usr/bin/python3"]
    PADDLE_VENV = ["conda", "run", "-n", "paddle-venv", "python"]


class LambdaFunctionNames(StrEnum):
    """Lambda function names."""

    SEND_SNS = "rogues-prod-send-sns-message-to-name"
    RUN_INFERENCE = "rogues-prod-run-paddle-inference"


class DataFiles:
    """Data filenames."""

    class Suffixes:
        """Suffixes."""

        CENTER_EXG_SLC_NPY = "_center_slice_20px.npy"
        CENTER_EXG_SLC_JSON = "_center_slice_20px.json"
        NORM_FRAME_DIFF_NPY = "_norm_frame_diff.npy"
        NORM_FRAME_DIFF_JSON = "_norm_frame_diff.json"
        SPREAD_COL_SUM_NPY = "_spread_col_sum.npy"
        SPREAD_COL_SUM_JSON = "_spread_col_sum.json"
        VERBOSE_SPREAD_COL_SUM = "_verbose_spread_col_sum.npy"
        FRAME_PHASH_64BIT = "_frame_phash_64bit.npy"
        FRAME_PHASH_8BIT = "_frame_phash_8bit.npy"

    STEM_WIDTHS_JSON = "stem_widths.json"
    OBLQ_EXG_TEST = "oblique_center_slice_20px.npy"
    EXG_SLC_20PX_NPY = "bottom_center_slice_20px.npy"
    EXG_SLC_20PX_JSON = "bottom_center_slice_20px.json"
    OFFSETS_JSON = "offsets.json"
    DUPLICATES_JSON = "duplicates.json"
    DUPLICATES_MISSING_FIXED_JSON = "duplicates_missing_fixed.json"
    SEGM_JSON = "segm.json"
    PLOT_MASK_AREA_PNG = "plot_mask_area.png"
    PLOT_MAX_MASK_AREA_PNG = "plot_max_mask_area.png"
    PLOT_TOP_10PCT_MASK_AREA_PNG = "plot_top_10pct_mask_area.png"
    MASK_AREAS_JSON = "mask_areas.json"

    class RogueLabelJSONs(StrEnum):
        """Rogue label JSON filenames."""

        ROGUES_W_DELAY_SEP_JSON = "rogues_w_delay_sep.json"
        ROGUES_JSON = "rogues.json"
        UNFILTERED_ROGUES_JSON = "unfiltered-rogues.json"


class CornTypes(StrEnum):
    """Corn types."""

    HYBRID_ONE_LOW = "1HL"
    HYBRID_TWO_LOW = "2HL"
    HYBRID_HIGH = "HH"
    MIX = "Mix"
    DELAY = "D3"
    FEMALE_ONE = "F1"
    FEMALE_TWO = "F2"
    FEMALE_THREE = "F3"
    MALE_ONE = "M1"
    BUFFER = "B"


class RawDates(StrEnum):
    """Raw dates helper."""

    JUNE_14 = "6-14"
    JUNE_15 = "6-15"
    JUNE_16 = "6-16"
    JUNE_21 = "6-21"
    JUNE_27 = "6-27"
    JUNE_28 = "6-28"
    JULY_05 = "7-05"
    JULY_06 = "7-06"
    JULY_08 = "7-08"
    JULY_12 = "7-12"


class Dates:
    """Dates."""

    # List makes StrEnum mad TODO fix
    class FarmerCity:
        """Farmer City dates."""

        WEEK_ONE = [RawDates.JUNE_14, RawDates.JUNE_15, RawDates.JUNE_16]
        # WEEK_ONE = RawDates.JUNE_16
        WEEK_TWO = RawDates.JUNE_21
        WEEK_THREE = RawDates.JUNE_28
        WEEK_FOUR = RawDates.JULY_06
        WEEK_FIVE = RawDates.JULY_12

        ALL_DATES = [WEEK_ONE[2], WEEK_TWO, WEEK_THREE, WEEK_FOUR, WEEK_FIVE]

    class ProductionFieldOneDates(StrEnum):
        """Dates for Production Field 1 (Argo North-Home Minor)."""

        WEEK_ONE = RawDates.JUNE_27
        WEEK_TWO = RawDates.JULY_05

    class FoundationFieldTwoDates(StrEnum):
        """Dates for Foundation Field 2 (Argo North-Home Major)."""

        WEEK_ONE = RawDates.JULY_08


class QASMJsonClasses(StrEnum):
    """QASM JSON Classes."""

    # Legacy
    DUPLICATE = "duplicate"
    MISSING = "missing"
    MISSING_FIXED = "missing_fixed"
    PLANT = "plant"
    ROGUE = "rogue"

    # Spring 2023
    NORMAL = "normal"
    DELAY_ROGUE = "delay_rogue"
    HYBRID_ROGUE = "hybrid_rogue"
    HYBRID_LOW_ROGUE = "hybrid_low_rogue"
    OTHER = "other"


class Directories(StrEnum):
    """Valid directories for the inferencing step."""

    APP = "app"
    PY_ROGUE_DETECTION = "py-rogue-detection"
    PADDLE_DETECTION = "PaddleDetection"


def get_week_idx_from_date_str(date_str: str, week_data: List) -> int:
    """Get week idx from date str."""
    return next(
        (
            idx
            for idx, week in enumerate(week_data)
            if isinstance(week, str)
            and week == date_str
            or not isinstance(week, str)
            and isinstance(week, list)
            and date_str in week
        ),
        None,
    )


def get_date_str_from_week_idx(week_idx: int, week_data: List) -> str:
    """Get date str from week idx."""
    date_str = week_data[week_idx]
    if isinstance(date_str, list):
        raise ValueError(f"Week idx {week_idx} has multiple dates.")
    else:
        return date_str


if __name__ == "__main__":
    print()
