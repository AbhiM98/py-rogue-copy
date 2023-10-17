"""Utility for storing parameters for ground_data_processing. Modeled after mfstand."""
from typing import List

import boto3
from S3MP.mirror_path import MirrorPath

from ddb_tracking.grd_api import get_grd_row, key_query_grd_rows, put_grd_row
from ddb_tracking.grd_constants import ProcessFlags
from ddb_tracking.utils.s3_utils import get_n_folders_in_folder
from ground_data_processing.utils.s3_constants import ROGUES_BUCKET_KEY, ProcessingSteps

PROCESS_FLAG_MAPPINGS = {
    ProcessFlags.VIDEOS_ALIGNED: ProcessingSteps.GENERATE_GPS_OFFSETS,
    ProcessFlags.VIDEOS_SPLIT: ProcessingSteps.GENERATE_DS_SPLITS,
    ProcessFlags.FRAMES_EXTRACTED: ProcessingSteps.EXTRACT_FRAMES,
    ProcessFlags.FRAMES_PREPARED: ProcessingSteps.PREP_INFERENCE,
    ProcessFlags.FRAMES_INFERENCED: ProcessingSteps.RUN_INFERENCE,
}


class FieldParams:
    """Params for processing a whole field."""

    def __init__(
        self,
        field_name: str,
        date: str,
        processing_step: ProcessingSteps = None,
        nadir_crop_height: int = None,
        nadir_crop_width: int = None,
        overwrite: bool = False,
        rerun: bool = False,
    ):
        """Init FieldParams."""
        self.field_name = field_name
        self.date = date
        self.processing_step = processing_step
        self.nadir_crop_height = nadir_crop_height
        self.nadir_crop_width = nadir_crop_width
        self.overwrite = overwrite
        self.rerun = rerun

        # Setup ddb first so we know the folder structure
        self._setup_ddb()
        self._setup_s3()
        self._setup_row_params()

    def _setup_ddb(self):
        """Load ddb data."""
        self.ddb_resource = boto3.resource("dynamodb", region_name="us-east-1")
        self.grdrows = key_query_grd_rows(
            {"field_name": self.field_name, "date": self.date}, self.ddb_resource
        )

        if len(self.grdrows) == 0:
            # No rows in the database
            raise ValueError(
                f"No rows for {self.field_name} {self.date} found in database."
            )

        # To find the folder holding the rows, we find the root of the videos
        # and then go up one level
        self.row_holder_mp = (
            self.grdrows[0].full_row_video_mps.get_root_mp().get_parent()
        )

        # Sort grdrows by row_number
        self.grdrows.sort(key=lambda grdrow: grdrow.row_number)

    def _setup_s3(self):
        """Construct S3 paths."""
        self.s3_client = boto3.client("s3")
        self.s3_bucket = ROGUES_BUCKET_KEY
        self.n_rows = get_n_folders_in_folder(
            self.row_holder_mp.s3_key, client=self.s3_client
        )

    def _setup_row_params(self):
        """Perform setup of row params."""
        self.row_params = [
            RowParams.from_field_params(self, row_number)
            for row_number in range(1, self.n_rows + 1)
        ]


class RowParams:
    """Params for processing a single row."""

    def __init__(
        self,
        field_name: str,
        date: str,
        row_number: int,
        processing_step: ProcessingSteps = None,
        ds_split_numbers: List[int] = None,
        nadir_crop_height: int = None,
        nadir_crop_width: int = None,
        overwrite: bool = False,
        rerun: bool = False,
        s3_client=None,
        ddb_resource=None,
    ):
        """Init RowParams."""
        self.field_name = field_name
        self.date = date
        self.row_number = row_number
        self.processing_step = processing_step
        self.ds_split_numbers = ds_split_numbers
        self.nadir_crop_height = nadir_crop_height
        self.nadir_crop_width = nadir_crop_width
        self.overwrite = overwrite
        self.rerun = rerun
        self.s3_client = s3_client
        self.ddb_resource = ddb_resource

        # Setup ddb first as S3 relies on the grdrow
        self._setup_ddb()
        self._setup_s3()

    def _setup_ddb(self):
        """Load ddb data."""
        if self.ddb_resource is None:
            self.ddb_resource = boto3.resource("dynamodb", region_name="us-east-1")
        self.grdrow = get_grd_row(
            self.field_name, self.row_number, self.date, self.ddb_resource
        )
        if self.grdrow is None:
            # Row isn't in the database
            raise ValueError(
                f"Row {self.row_number} for {self.field_name} {self.date} not found in database."
            )
        if self.rerun:
            match self.processing_step:
                case ProcessingSteps.RUN_ALL:
                    # Reset all flags when rerunning the whole row
                    # Set all process flags to False
                    self.grdrow.set_all_process_flags(False)
                    # Remove all existing plant groups
                    self.grdrow.plant_groups = []
                    # Set n_ds_splits to 0
                    self.grdrow.n_ds_splits = 0
                case ProcessingSteps.PREP_AND_RUN_INFERENCE:
                    # Set prep and run inference flags to False
                    self.grdrow.set_frames_prepared(False)
                    self.grdrow.set_frames_inferenced(False)
                    if self.ds_split_numbers:
                        # Remove the inference paths for the specified DS splits
                        [
                            self.grdrow.reset_inference_paths_for_ds_split_number(
                                ds_split_number
                            )
                            for ds_split_number in self.ds_split_numbers
                        ]
                    else:
                        # Remove all inference paths
                        self.grdrow.reset_all_inference_paths()
                case ProcessingSteps.RUN_INFERENCE:
                    # Set inference flag to False
                    self.grdrow.set_frames_inferenced(False)
                    if self.ds_split_numbers:
                        # Remove the inference paths for the specified DS splits
                        [
                            self.grdrow.reset_inference_paths_for_ds_split_number(
                                ds_split_number
                            )
                            for ds_split_number in self.ds_split_numbers
                        ]
                    else:
                        # Remove all inference paths
                        self.grdrow.reset_all_inference_paths()
                case _:
                    # Set the flag for the specified process to False
                    for process_flag in ProcessFlags:
                        if PROCESS_FLAG_MAPPINGS[process_flag] == self.processing_step:
                            match process_flag:
                                case ProcessFlags.VIDEOS_ALIGNED:
                                    self.grdrow.set_videos_aligned(False)
                                case ProcessFlags.VIDEOS_SPLIT:
                                    self.grdrow.set_videos_split(False)
                                case ProcessFlags.FRAMES_EXTRACTED:
                                    self.grdrow.set_frames_extracted(False)
                                case ProcessFlags.FRAMES_PREPARED:
                                    self.grdrow.set_frames_prepared(False)
                                case ProcessFlags.FRAMES_INFERENCED:
                                    self.grdrow.set_frames_inferenced(False)
                            break

        # Track which version of the row we're working with
        self.version = self.grdrow.version_tracker.get_latest_version()

    def _setup_s3(self):
        """Construct S3 paths."""
        if self.s3_client is None:
            self.s3_client = boto3.client("s3")
        self.s3_bucket = ROGUES_BUCKET_KEY
        self.row_mp = self.grdrow.full_row_video_mps.get_root_mp()
        # If no splits are present, this will be an empty folder
        self.ds_splits_mp = self.row_mp.get_child("DS Splits")
        # if no splits are present, plant_groups will be None and ds_split_mps will be []
        self.ds_split_mps = []
        self.ds_split_numbers_mps = []
        if self.grdrow.plant_groups:
            self.ds_split_mps = [
                self.ds_splits_mp.get_child(f"DS {plant_group.ds_split_number:03d}")
                for plant_group in self.grdrow.plant_groups
            ]

            if self.ds_split_numbers:
                # Filter to only the specified DS split indices
                self.ds_split_numbers_mps = [
                    self.ds_splits_mp.get_child(f"DS {plant_group.ds_split_number:03d}")
                    for plant_group in self.grdrow.plant_groups
                    if plant_group.ds_split_number in self.ds_split_numbers
                ]

    def push_to_ddb(self):
        """Push to ddb."""
        # Guard against a row's version changing during processing
        current_grdrow = get_grd_row(
            self.field_name, self.row_number, self.date, self.ddb_resource
        )
        current_version = current_grdrow.version_tracker.get_latest_version()
        if current_version != self.version:
            raise ValueError(
                f"Row {self.row_number} for {self.field_name} {self.date} changed during processing: {self.version.timestamp} -> {current_version.timestamp}."
            )
        put_grd_row(self.grdrow, self.ddb_resource)

    def set_n_ds_splits(self, n_ds_splits: int):
        """Set n_ds_splits, remake S3 paths, create GRDPlantGroups."""
        self.grdrow.n_ds_splits = n_ds_splits
        self.ds_split_mps = [
            self.ds_splits_mp.get_child(f"DS {ds_split_number:03d}")
            for ds_split_number in range(0, self.grdrow.n_ds_splits)
        ]
        self.grdrow.init_plant_groups(self.ds_split_mps)

    def get_grd_plant_group_from_ds_split_mp(self, ds_split_mp: MirrorPath):
        """Get GRDPlantGroup from ds_split_mp."""
        # Get 'DS {split_number:03d}' from s3_key and cast to int
        ds_split_number = int(ds_split_mp.key_segments[-1].name.split(" ")[-1])

        # First check if the plant_groups contain the index, and if the plant group at this index has a ds_split_number that matches
        if (
            len(self.grdrow.plant_groups) > ds_split_number
            and self.grdrow.plant_groups[ds_split_number].ds_split_number
            == ds_split_number
        ):
            return self.grdrow.plant_groups[ds_split_number]

        # Otherwise try and get a plant group whose ds_split_number matches
        return next(
            (
                plant_group
                for plant_group in self.grdrow.plant_groups
                if plant_group.ds_split_number == ds_split_number
            ),
            None,
        )

    def update_process_flag_and_push_to_ddb(
        self, process_flag: ProcessFlags, value: bool = None
    ):
        """Update process flag and push to ddb."""
        if value is None:
            print(
                f"No flag value provided, skip setting {process_flag} of {self.field_name} {self.date} Row {self.row_number}."
            )
        match process_flag:
            case ProcessFlags.VIDEOS_ALIGNED:
                self.grdrow.set_videos_aligned(value)
            case ProcessFlags.VIDEOS_SPLIT:
                self.grdrow.set_videos_split(value)
            case ProcessFlags.FRAMES_EXTRACTED:
                self.grdrow.set_frames_extracted(value)
            case ProcessFlags.FRAMES_PREPARED:
                self.grdrow.set_frames_prepared(value)
            case ProcessFlags.FRAMES_INFERENCED:
                self.grdrow.set_frames_inferenced(value)
            case _:
                raise ValueError(f"Invalid process flag: {process_flag}")

        self.push_to_ddb()

    @property
    def all_proccesses_done(self) -> bool:
        """Return True if all processes are done, False otherwise."""
        return all(
            [
                self.grdrow.videos_aligned,
                self.grdrow.videos_split,
                self.grdrow.frames_extracted,
                self.grdrow.frames_prepared,
                self.grdrow.frames_inferenced,
            ]
        )

    @property
    def all_incomplete_processes(self) -> List[ProcessingSteps]:
        """Return a list of all incomplete processing steps."""
        return [
            PROCESS_FLAG_MAPPINGS[process_flag]
            for process_flag in ProcessFlags
            if not getattr(self.grdrow, process_flag.value)
        ]

    @property
    def next_incomplete_process(self) -> ProcessingSteps:
        """Return the next incomplete processing step (Note that this returns a ProcessingStep, not a ProcessFlag)."""
        if not self.grdrow.videos_aligned:
            return ProcessingSteps.GENERATE_GPS_OFFSETS
        elif not self.grdrow.videos_split:
            return ProcessingSteps.GENERATE_DS_SPLITS
        elif not self.grdrow.frames_extracted:
            return ProcessingSteps.EXTRACT_FRAMES
        elif not self.grdrow.frames_prepared:
            return ProcessingSteps.PREP_INFERENCE
        elif not self.grdrow.frames_inferenced:
            return ProcessingSteps.RUN_INFERENCE
        else:
            return None

    @staticmethod
    def from_field_params(field_params: FieldParams, row_number: int):
        """Construct RowParams from FieldParams."""
        return RowParams(
            field_name=field_params.field_name,
            date=field_params.date,
            row_number=row_number,
            processing_step=field_params.processing_step,
            overwrite=field_params.overwrite,
            rerun=field_params.rerun,
            s3_client=field_params.s3_client,
            ddb_resource=field_params.ddb_resource,
        )
