"""Dataclasses for Ground Rogues Data structure."""
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from json import JSONEncoder
from typing import List

from S3MP.mirror_path import MirrorPath

from ddb_tracking.grd_constants import (
    CAMERA_VIEWS,
    ProcessFlags,
    RoguesS3PathStructure,
    TimeZones,
)

# CAMERA_VIEWS = [
#     "bottom",
#     "nadir",
#     "oblique",
# ]


class GRDJSONEncoder(JSONEncoder):
    """Encoder for Ground Rogues Data JSON."""

    def default(self, o):
        """Encode."""
        if isinstance(o, MirrorPath):
            return o.s3_key
        if isinstance(o, (MultiViewMP, GRDPlantGroup, GRDRow)):
            return o.as_json()
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


# TODO: Store prefix/suffix from RoguesS3PathStructure so we can parse an arbitrary timestamp
@dataclass
class VersionTimeStamp:
    """Dataclass for tracking a single version of a GRDRow."""

    timestamp: int
    folder_depth: int

    def as_json(self):
        """Convert to JSON."""
        return {
            "timestamp": self.timestamp,
            "folder_depth": self.folder_depth,
        }

    def __eq__(self, other) -> bool:
        """Check equality."""
        return self.timestamp == other.timestamp

    @property
    def timestamp_str(self) -> str:
        """Get the timestamp as a string."""
        timestamp_str = str(self.timestamp)
        # Pad with zeros if necessary to make it HHMMSS
        while len(timestamp_str) < 6:
            timestamp_str = f"0{timestamp_str}"
        return timestamp_str

    @staticmethod
    def from_json(json_dict):
        """Convert from JSON."""
        return VersionTimeStamp(
            timestamp=int(json_dict["timestamp"])
            if json_dict["timestamp"] is not None
            else None,
            folder_depth=int(json_dict["folder_depth"])
            if json_dict["folder_depth"] is not None
            else None,
        )


@dataclass
class VersionTracker:
    """Dataclass for tracking versions of a GRDRow."""

    versions: List[VersionTimeStamp] = field(default_factory=lambda: [])

    def __post_init__(self):
        """Sort the versions by timestamp."""
        self.versions.sort(key=lambda x: x.timestamp, reverse=True)

    def as_json(self):
        """Convert to JSON."""
        return {
            "versions": [version.as_json() for version in self.versions],
        }

    def get_latest_version(self) -> VersionTimeStamp:
        """Get the latest version."""
        return self.versions[0] if self.versions else None

    def add_version(self, version: VersionTimeStamp):
        """Add a version."""
        self.versions.append(version)
        self.versions.sort(key=lambda x: x.timestamp, reverse=True)

    def contains_version(self, version: VersionTimeStamp) -> bool:
        """Check if a version exists."""
        return any(v.timestamp == version.timestamp for v in self.versions)

    def contains_timestamp(self, timestamp: int) -> bool:
        """Check if a timestamp exists."""
        return any(v.timestamp == timestamp for v in self.versions)

    def set_version_in_s3_key(self, s3_key: str, timestamp: int = None) -> str:
        """Set the latest version in an s3 key."""
        if timestamp is None:
            # Get the latest version
            version = self.get_latest_version()
        else:
            # Get the version with the specified timestamp
            version = next(
                (
                    version
                    for version in self.versions
                    if version.timestamp == timestamp
                ),
                None,
            )
            if version is None:
                raise ValueError(f"Version with timestamp {timestamp} not found.")

        # Handle arbitrary folder depth
        folders = s3_key.split("/")
        depth = None
        if version.timestamp_str in folders:
            # Version is already in the s3 key
            return s3_key
        else:
            # Find the depth where any version timestamp is present in the current key
            for v in self.versions:
                if v.timestamp_str in folders:
                    depth = folders.index(v.timestamp_str)
                    break
            if depth is None:
                raise ValueError(
                    f"Folder depth of a previous version not found in s3 key {s3_key}."
                )

            # Replace the timestamp in the s3 key with the version timestamp
            folders[depth] = version.timestamp_str
            return "/".join(folders)

    @staticmethod
    def from_json(json_dict):
        """Convert from JSON."""
        return VersionTracker(
            versions=[
                VersionTimeStamp.from_json(version) for version in json_dict["versions"]
            ],
        )


@dataclass
class TimeStampedBool:
    """Dataclass for a boolean with a timestamp."""

    value: bool = False
    timestamp: str = datetime.now(tz=TimeZones.CENTRAL).strftime("%Y-%m-%d %H:%M:%S")

    def as_json(self):
        """Convert to JSON."""
        return {
            "value": self.value,
            "timestamp": self.timestamp,
        }

    def __bool__(self):
        """Return the value."""
        return self.value

    def __eq__(self, other) -> bool:
        """Check equality."""
        return self.value == other.value

    def set_value(self, value: bool):
        """Set the value."""
        self.value = value
        self.timestamp = datetime.now(tz=TimeZones.CENTRAL).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    @staticmethod
    def from_json(json_dict):
        """Convert from JSON."""
        return TimeStampedBool(
            value=json_dict["value"],
            timestamp=json_dict["timestamp"],
        )


@dataclass
class MultiViewMP:
    """Dataclass for multiple view MirrorPaths."""

    bottom_mp: MirrorPath
    nadir_mp: MirrorPath
    oblique_mp: MirrorPath

    def as_json(self):
        """Convert to JSON."""
        return {
            "bottom_mp": self.bottom_mp.s3_key,
            "nadir_mp": self.nadir_mp.s3_key,
            "oblique_mp": self.oblique_mp.s3_key,
        }

    def as_dict(self):
        """Convert to dict."""
        return {
            "bottom": self.bottom_mp,
            "nadir": self.nadir_mp,
            "oblique": self.oblique_mp,
        }

    def as_list(self):
        """Convert to list."""
        return [
            self.bottom_mp,
            self.nadir_mp,
            self.oblique_mp,
        ]

    def all_exists(self):
        """Check if all MirrorPaths exist."""
        return all(
            [
                self.bottom_mp.exists_on_s3(),
                self.nadir_mp.exists_on_s3(),
                self.oblique_mp.exists_on_s3(),
            ]
        )

    def set_version_from_version_tracker(
        self, version_tracker: VersionTracker, timestamp: int = None
    ):
        """Set the version of the MultiViewMP by updating all of its s3 keys."""
        if timestamp is None:
            # Get the latest version
            version = version_tracker.get_latest_version()
            timestamp = version.timestamp

        self.bottom_mp = MirrorPath.from_s3_key(
            version_tracker.set_version_in_s3_key(
                self.bottom_mp.s3_key, timestamp=timestamp
            )
        )
        self.nadir_mp = MirrorPath.from_s3_key(
            version_tracker.set_version_in_s3_key(
                self.nadir_mp.s3_key, timestamp=timestamp
            )
        )
        self.oblique_mp = MirrorPath.from_s3_key(
            version_tracker.set_version_in_s3_key(
                self.oblique_mp.s3_key, timestamp=timestamp
            )
        )

    def get_root_mp(self):
        """Get the root MirrorPath."""
        # Find the depth where the str "bottom" is present
        # This handles cases where the root is more than one level above the paths
        # Eg: "DS 001/bottom Raw Images/Preprocessed Images/" will return "DS 001/"
        depth = next(
            (
                key_segment.depth
                for key_segment in self.bottom_mp.key_segments
                if CAMERA_VIEWS[0] in key_segment.name
            ),
            self.bottom_mp.key_segments[-1].depth,
        )
        # depth+1 gets the mp of up to the folder containing "bottom"
        return self.bottom_mp.trim(depth + 1).get_parent()

    @staticmethod
    def from_json(json_dict):
        """Convert from JSON."""
        return MultiViewMP(
            bottom_mp=MirrorPath.from_s3_key(json_dict["bottom_mp"]),
            nadir_mp=MirrorPath.from_s3_key(json_dict["nadir_mp"]),
            oblique_mp=MirrorPath.from_s3_key(json_dict["oblique_mp"]),
        )

    @staticmethod
    def from_root_mp(root_mp: MirrorPath, suffix: str = ".mp4"):
        """Make a MultiViewMP from a root MirrorPath and a suffix."""
        return MultiViewMP(
            bottom_mp=root_mp.get_child(f"{CAMERA_VIEWS[0]}{suffix}"),
            nadir_mp=root_mp.get_child(f"{CAMERA_VIEWS[1]}{suffix}"),
            oblique_mp=root_mp.get_child(f"{CAMERA_VIEWS[2]}{suffix}"),
        )


@dataclass
class GRDPlantGroup:
    """Dataclass for Ground Rogues Data plant group."""

    videos: MultiViewMP
    image_directories: MultiViewMP
    preprocessed_images: MultiViewMP = None
    inferenced_images: MultiViewMP = None
    rogue_labels: MirrorPath = None
    n_images: int = None
    ds_split_number: int = None

    def set_version_from_version_tracker(
        self, version_tracker: VersionTracker, timestamp: int = None
    ):
        """Set the version of the plant group by updating all of its s3 keys."""
        if timestamp is None:
            # Get the latest version
            version = version_tracker.get_latest_version()
            timestamp = version.timestamp

        mps = [
            self.videos,
            self.image_directories,
            self.preprocessed_images,
            self.inferenced_images,
        ]
        for mp in mps:
            if mp:
                mp.set_version_from_version_tracker(
                    version_tracker, timestamp=timestamp
                )

    def as_json(self):
        """Convert to JSON."""
        return {
            "videos": self.videos.as_json(),
            "image_directories": self.image_directories.as_json(),
            "preprocessed_images": self.preprocessed_images.as_json()
            if self.preprocessed_images
            else None,
            "inferenced_images": self.inferenced_images.as_json()
            if self.inferenced_images
            else None,
            "rogue_labels": self.rogue_labels.s3_key if self.rogue_labels else None,
            "n_images": self.n_images,
            "ds_split_number": self.ds_split_number,
        }

    @staticmethod
    def from_json(json_dict):
        """Convert from JSON."""
        # Add any missing keys
        new_keys = ["n_images", "preprocessed_images", "inferenced_images"]
        for key in new_keys:
            if key not in json_dict:
                json_dict[key] = None

        return GRDPlantGroup(
            videos=MultiViewMP.from_json(json_dict["videos"]),
            image_directories=MultiViewMP.from_json(json_dict["image_directories"]),
            preprocessed_images=MultiViewMP.from_json(json_dict["preprocessed_images"])
            if json_dict["preprocessed_images"]
            else None,
            inferenced_images=MultiViewMP.from_json(json_dict["inferenced_images"])
            if json_dict["inferenced_images"]
            else None,
            rogue_labels=MirrorPath.from_s3_key(json_dict["rogue_labels"])
            if json_dict["rogue_labels"]
            else None,
            n_images=int(json_dict["n_images"]) if json_dict["n_images"] else None,
            ds_split_number=int(json_dict["ds_split_number"])
            if json_dict["ds_split_number"] is not None
            else None,
        )


@dataclass
class GRDRow:
    """Dataclass for Ground Rogues Data row."""

    field_name: str
    date: str  # YYYY-MM-DD
    row_number: int
    full_row_video_mps: MultiViewMP
    n_ds_splits: int = 0  # Number of DS splits, same as len(plant_groups)
    plant_groups: List[GRDPlantGroup] = field(default_factory=lambda: [])
    date_row_number: str = None  # f"{date}#{row_number}"
    start_direction: str = None  # North, South, East, West

    # Version tracker: Rows with the same date and row number but different timestamps are different versions
    version_tracker: VersionTracker = field(
        default_factory=lambda: VersionTracker(versions=[])
    )

    # Process Flags (default as False and with current timestamp)
    videos_aligned: TimeStampedBool = field(
        default_factory=lambda: TimeStampedBool()
    )  # Align Videos with GPS timestamps
    videos_split: TimeStampedBool = field(
        default_factory=lambda: TimeStampedBool()
    )  # Split videos into frames
    frames_extracted: TimeStampedBool = field(
        default_factory=lambda: TimeStampedBool()
    )  # Extract frames from plot videos
    frames_prepared: TimeStampedBool = field(
        default_factory=lambda: TimeStampedBool()
    )  # Prepare frames for inferencing
    frames_inferenced: TimeStampedBool = field(
        default_factory=lambda: TimeStampedBool()
    )  # Inference images

    def __post_init__(self):
        """Post init. This way we can set fields using class methods."""
        if not self.date_row_number:
            self.date_row_number = self.get_date_row_number(self.date, self.row_number)

    def init_plant_groups(self, ds_split_mps: List[MirrorPath] = None):
        """Initialize the plant groups that are not yet initialized."""
        if len(self.plant_groups) != self.n_ds_splits:
            assert self.n_ds_splits == len(
                ds_split_mps
            ), f"self.n_ds_splits ({self.n_ds_splits}) must match len(ds_split_mps) ({len(ds_split_mps)})."
            print(
                f"Found {len(self.plant_groups)} GRDPlantGroups for {self.field_name} {self.date}, expected {self.n_ds_splits}."
            )
            existing_ds_splits = [pg.ds_split_number for pg in self.plant_groups]

            # Create GRDRow for rows not in the database
            for ds_split_number in range(0, self.n_ds_splits):
                if ds_split_number in existing_ds_splits:
                    continue
                pg = GRDPlantGroup(
                    videos=MultiViewMP.from_root_mp(ds_split_mps[ds_split_number]),
                    image_directories=MultiViewMP.from_root_mp(
                        ds_split_mps[ds_split_number], suffix=" Raw Images"
                    ),
                    ds_split_number=ds_split_number,
                )
                self.plant_groups.append(pg)

            # Sort by ds_split_number
            self.plant_groups.sort(key=lambda pg: pg.ds_split_number)

    def set_version(self, timestamp: int = None):
        """Set the version of the row by updating all of its s3 keys and reseting all process flags."""
        # Set the version of the full row videos
        self.full_row_video_mps.set_version_from_version_tracker(
            self.version_tracker, timestamp=timestamp
        )
        # Set the version of the plant group videos
        for pg in self.plant_groups:
            pg.set_version_from_version_tracker(
                self.version_tracker, timestamp=timestamp
            )

        # Reset all process flags
        self.set_all_process_flags(False)

    def add_and_set_version(self, version: VersionTimeStamp, use_latest: bool = True):
        """Add a version and set it or the latest version."""
        self.version_tracker.add_version(version)
        if use_latest:
            # Will default to the latest
            self.set_version()
        else:
            # Will use the version provided
            self.set_version(timestamp=version.timestamp)

    def set_videos_aligned(self, value: bool):
        """Set videos_aligned."""
        self.videos_aligned.set_value(value)

    def set_videos_split(self, value: bool):
        """Set videos_split."""
        self.videos_split.set_value(value)

    def set_frames_extracted(self, value: bool):
        """Set frames_extracted."""
        self.frames_extracted.set_value(value)

    def set_frames_prepared(self, value: bool):
        """Set frames_prepared."""
        self.frames_prepared.set_value(value)

    def set_frames_inferenced(self, value: bool):
        """Set frames_inferenced."""
        self.frames_inferenced.set_value(value)

    def set_all_process_flags(self, value: bool):
        """Set all process flags."""
        self.set_videos_aligned(value)
        self.set_videos_split(value)
        self.set_frames_extracted(value)
        self.set_frames_prepared(value)
        self.set_frames_inferenced(value)

    def get_plant_group_by_ds_split_number(self, ds_split_number: int) -> GRDPlantGroup:
        """Get a plant group by ds_split_number."""
        return next(
            (pg for pg in self.plant_groups if pg.ds_split_number == ds_split_number),
            None,
        )

    def reset_inference_paths_for_ds_split_number(self, ds_split_number: int):
        """Reset the inferenced_images paths for a plant group with the specified ds_split_number."""
        pg = self.get_plant_group_by_ds_split_number(ds_split_number)
        pg.inferenced_images = None

    def reset_all_inference_paths(self):
        """Reset the inferenced_images paths for all plant groups."""
        for pg in self.plant_groups:
            pg.inferenced_images = None

    def as_json(self):
        """Convert to JSON."""
        if self.date_row_number is None:
            self.date_row_number = self.get_date_row_number(self.date, self.row_number)

        return {
            "field_name": self.field_name,
            "date#row_number": self.date_row_number,  # purely for nice sorting
            "date": self.date,
            "row_number": self.row_number,
            "version_tracker": self.version_tracker.as_json()
            if self.version_tracker
            else None,
            "full_row_video_mps": self.full_row_video_mps.as_json(),
            "n_ds_splits": self.n_ds_splits,
            "plant_groups": [pg.as_json() for pg in self.plant_groups],
            "start_direction": self.start_direction if self.start_direction else None,
            "videos_aligned": self.videos_aligned.as_json(),
            "videos_split": self.videos_split.as_json(),
            "frames_extracted": self.frames_extracted.as_json(),
            "frames_prepared": self.frames_prepared.as_json(),
            "frames_inferenced": self.frames_inferenced.as_json(),
        }

    @staticmethod
    def from_json(json_dict):
        """Convert from JSON."""
        # Add any missing process flags, initialize to False
        for key in ProcessFlags:
            if key not in json_dict:
                json_dict[key] = TimeStampedBool().as_json()

        return GRDRow(
            field_name=json_dict["field_name"],
            date=json_dict["date"],
            row_number=int(json_dict["row_number"]),
            date_row_number=json_dict["date#row_number"]
            if "date#row_number" in json_dict
            else None,
            version_tracker=VersionTracker.from_json(json_dict["version_tracker"])
            if ("version_tracker" in json_dict and json_dict["version_tracker"])
            else VersionTracker(),
            full_row_video_mps=MultiViewMP.from_json(json_dict["full_row_video_mps"]),
            n_ds_splits=int(json_dict["n_ds_splits"])
            if ("n_ds_splits" in json_dict and json_dict["n_ds_splits"] is not None)
            else len(json_dict["plant_groups"]),
            plant_groups=[
                GRDPlantGroup.from_json(pg) for pg in json_dict["plant_groups"]
            ],
            start_direction=json_dict["start_direction"]
            if "start_direction" in json_dict
            else None,
            videos_aligned=TimeStampedBool.from_json(json_dict["videos_aligned"]),
            videos_split=TimeStampedBool.from_json(json_dict["videos_split"]),
            frames_extracted=TimeStampedBool.from_json(json_dict["frames_extracted"]),
            frames_prepared=TimeStampedBool.from_json(json_dict["frames_prepared"]),
            frames_inferenced=TimeStampedBool.from_json(json_dict["frames_inferenced"]),
        )

    @staticmethod
    def from_s3_key(s3_key: str):
        """Parse an s3 key and attempt to match it to known folder structures."""
        (
            field_name,
            date,
            row_number,
            timestamp,
            timestamp_depth,
        ) = RoguesS3PathStructure.infer_structure_and_parse_s3_key(s3_key)
        version = (
            VersionTimeStamp(timestamp=timestamp, folder_depth=timestamp_depth)
            if timestamp
            else None
        )
        return GRDRow(
            field_name=field_name,
            date=date,
            row_number=row_number,
            version_tracker=VersionTracker(versions=[version])
            if version
            else None,  # When no version is present, keep as unversioned
            # Get the parent of the s3_key, which is the folder containing the full row videos
            full_row_video_mps=MultiViewMP.from_root_mp(
                MirrorPath.from_s3_key(s3_key).get_parent()
            ),
        )

    @staticmethod
    def get_date_row_number(date: str, row_number: int) -> str:
        """Get the date#row_number format from a date, row_number and year, using the current year if none is provided."""
        return f"{date}#{row_number:02d}"
