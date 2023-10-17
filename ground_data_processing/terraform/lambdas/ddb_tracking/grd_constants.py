"""Constants for Ground Rogues Data."""
from enum import Enum

# For the docker container, we need to use backports.zoneinfo instead of zoneinfo bc python 3.8
try:
    from backports.zoneinfo import ZoneInfo
except ImportError:
    from zoneinfo import ZoneInfo

from typing import Callable, Tuple

GRD_S3_BUCKET_NAME = "sentera-rogues-data"
GRD_DDB_TABLE_NAME = "ground-rogues-data"
CAMERA_VIEWS = [
    "bottom",
    "nadir",
    "oblique",
]


# NOTE: These str Enums are NOT equivalent to StrEnum in Python 3.11+.
# We don't use them yet so i dont care
class StartDirection(str, Enum):
    """Start directions."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class ProcessFlags(str, Enum):
    """Str aliases for process flags in a GRDRow."""

    VIDEOS_ALIGNED = "videos_aligned"
    VIDEOS_SPLIT = "videos_split"
    FRAMES_EXTRACTED = "frames_extracted"
    FRAMES_PREPARED = "frames_prepared"
    FRAMES_INFERENCED = "frames_inferenced"


class TimeZones:
    """Time zone constants."""

    # Usage: datetime.now(tz=TimeZones.CENTRAL)
    CENTRAL = ZoneInfo("America/Chicago")
    UTC = ZoneInfo("UTC")


class DataFolders:
    """Data folder names."""

    REL_PLOT_SPLIT = "Rel Plots"
    RAW_IMAGES = "Raw Images"
    THUMBNAILS = "Thumbnails"
    DS_SPLITS = "DS Splits"
    PREPROCESSED_IMAGES = "Preprocessed Images"
    ANNOTATED_IMAGES = "Annotated Images"
    INFERENCE_MASKS = "Inference Masks"


# StrEnum is only available in Python 3.11+, so we can't use it here unfortunately.
RoguesS3PathKeys = {
    "FIELD_NAME": "field_name",
    "DATE": "date",  # YYYY-MM-DD format
    "ROW_NUMBER": "row_number",
    "TIMESTAMP": "timestamp",  # HHMMSS format
}

# Examples of different S3 path structures to the full raw camera files.
RoguesS3PathExamples = {
    "GROUND_ROGUES_DEV_2023": f"{RoguesS3PathKeys['FIELD_NAME']}/Videos/{RoguesS3PathKeys['DATE']}/Row {RoguesS3PathKeys['ROW_NUMBER']}/video_rez/bottom.mp4",
    "GROUND_ROGUES_SANDBOX": f"2023-dev-sandbox/{RoguesS3PathKeys['FIELD_NAME']}/{RoguesS3PathKeys['DATE']}/row-{RoguesS3PathKeys['ROW_NUMBER']}/{RoguesS3PathKeys['TIMESTAMP']}/bottom.mp4",
    "GROUND_ROGUES_SUMMER_2023": f"2023-field-data/{RoguesS3PathKeys['FIELD_NAME']}/{RoguesS3PathKeys['DATE']}/row-{RoguesS3PathKeys['ROW_NUMBER']}/{RoguesS3PathKeys['TIMESTAMP']}/bottom.mp4",
    "GROUND_ROGUES_RGBD_2023": f"2023-rgbd-data/{RoguesS3PathKeys['FIELD_NAME']}/{RoguesS3PathKeys['DATE']}/row-{RoguesS3PathKeys['ROW_NUMBER']}/{RoguesS3PathKeys['TIMESTAMP']}/bottom.mp4",
}


class RoguesS3PathStructure:
    """Expected structure of an S3 path for Rogues data."""

    def __init__(
        self,
        field_name_depth: int,
        date_depth: int,
        row_number_depth: int,
        timestamp_depth: int = None,  # optional
        field_name_parser: Callable = None,
        date_parser: Callable = None,
        row_number_parser: Callable = None,
        timestamp_parser: Callable = None,
    ):
        """Init."""
        self.field_name_depth = field_name_depth
        self.date_depth = date_depth
        self.row_number_depth = row_number_depth
        self.timestamp_depth = timestamp_depth
        self.field_name_parser = field_name_parser
        self.date_parser = date_parser
        self.row_number_parser = row_number_parser
        self.timestamp_parser = timestamp_parser

    def parse_s3_key(self, s3_key: str) -> Tuple[str, str, int, int, int]:
        """Get the field name, date, and row number from an S3 key."""
        folders = s3_key.split("/")
        # Use provided parser if available, otherwise use the raw folder name
        field_name = (
            self.field_name_parser(folders[self.field_name_depth])
            if self.field_name_parser
            else folders[self.field_name_depth]
        )
        date = (
            self.date_parser(folders[self.date_depth])
            if self.date_parser
            else folders[self.date_depth]
        )
        row_number = (
            self.row_number_parser(folders[self.row_number_depth])
            if self.row_number_parser
            else folders[self.row_number_depth]
        )

        # Timestamp is optional
        timestamp = None
        if self.timestamp_depth is not None:
            timestamp = (
                self.timestamp_parser(folders[self.timestamp_depth])
                if self.timestamp_parser
                else folders[self.timestamp_depth]
            )
            try:
                timestamp = int(timestamp)
            except ValueError:
                raise ValueError(
                    f"Timestamp parsed as '{timestamp}' could not be converted to an int, check the expected file structure."
                )

        # Check if row number is valid
        try:
            row_number = int(row_number)
        except ValueError:
            raise ValueError(
                f"Row number parsed as '{row_number}' could not be converted to an int, check the expected file structure."
            )

        return field_name, date, row_number, timestamp, self.timestamp_depth

    @staticmethod
    def from_s3_path_example(s3_path_example: RoguesS3PathExamples):
        """Construct S3PathStructure from RoguesS3PathExamples."""
        ret = RoguesS3PathStructure(0, 0, 0)  # init with dummy values
        folders = s3_path_example.split("/")

        # https://stackoverflow.com/questions/53086592/lazy-evaluation-when-use-lambda-and-list-comprehension
        # lambda is lazy evaluated, so we need to use a function to create the parser function
        # parser = lambda prefix, suffix: lambda s: s.removeprefix(prefix).removesuffix(
        #     suffix
        # )
        def parser(prefix, suffix):
            """Return a parser function."""
            return lambda s: s.removeprefix(prefix).removesuffix(suffix)

        for key in RoguesS3PathKeys.values():
            if key in folders:
                # If key alone is in folders, set the depth to the index of the key
                # and set the parser to None
                setattr(ret, f"{key}_depth", folders.index(key))  # set depth
                setattr(ret, f"{key}_parser", None)  # set parser
            else:
                try:
                    # If the key is a substring of a folder name, set the depth to the index
                    # and set the parser to the substring
                    depth = [idx for idx, s in enumerate(folders) if key in s][0]
                except IndexError:
                    if key == RoguesS3PathKeys["TIMESTAMP"]:
                        # Timestamp is optional
                        continue
                    raise ValueError(
                        f"Could not infer S3PathStructure from s3_path_example: key '{key}' not found in s3_path_example."
                    )
                setattr(ret, f"{key}_depth", depth)  # set depth

                # Determine the start and end of the key substring
                folder = folders[depth]
                substring_start = folder.index(key)
                substring_end = substring_start + len(key)

                # We'll assume that everything other than the substring key will
                # always be present, so we can parse by discarding the prefix and suffix of
                # of the folder.
                # eg. "Row 1" -> discard "Row " and return "1"
                prefix = folder[:substring_start]
                suffix = folder[substring_end:]
                setattr(ret, f"{key}_parser", parser(prefix, suffix))  # set parser

        return ret

    @staticmethod
    def infer_structure_and_parse_s3_key(s3_key: str):
        """Try and infer the S3PathStructure from an S3 key by trying to parse the key according to each RoguesS3PathExample."""
        for example in RoguesS3PathExamples.values():
            try:
                return RoguesS3PathStructure.from_s3_path_example(example).parse_s3_key(
                    s3_key
                )
            except ValueError:
                continue
        raise ValueError(
            f"Could not infer S3PathStructure from s3_key {s3_key}: no matching RoguesS3PathExample found."
        )
