# Ground Rogues Database (GRD) 
Utilities to aid in interaction with the Ground Rogues Database (GRD). Also includes the Terraform files that define the AWS resources for the database.

IMPORTANT: The core database functionality is currently duplicated in `ground_data_processing/terraform/lambdas/ddb_tracking/` in order to be used in the AWS lambda functions used in automated processing. Any changes to the database functionality should be made in both locations, and care should be taken to avoid introducing features not backwards compatible with python 3.8 and the default dependencies available in AWS lambda.

## Primary Classes 
### GRDRow
A dataclass for a single row of ground rogues data. Each row is stored as an item in ddb, with the following information:

| Field | Description |
| ----------- | ----------- |
| `field_name`: str | Field name |
| `date`: str | Date when data was collected. YYYY-MM-DD format |
| `row_number`: int | Row number |
| `date_row_number`: str | DDB sort key, `{YYYY-MM-DD}#{row_number:02d}`, eg. `2022-06-16#05`|
| `version_tracker`: [VersionTracker](#versiontracker) | Tracks different versions of the same row. |
| `full_row_video_mps`: [MultiViewMP](#multiviewmp) | Paths to the full, uncut videos of each of the three camera views (nadir, oblique, bottom) |
| `n_ds_splits`: int | Number of DS Splits. Equivalent to `len(plant_groups)`.|
| `plant_groups`: List[[GRDPlantGroup](#grdplantgroup)] | A list of each subdivision within the row, each stored as a [GRDPlantGroup](#grdplantgroup). |
| `start_direction`: str | Direction of the video collection. `"north"`, `"south"`, `"east"`, or `"west"`.|
| `videos_aligned`: [TimeStampedBool](#timestampedbool) | Whether videos have been aligned with GPS timestamps. |
| `videos_split`: [TimeStampedBool](#timestampedbool) | Whether videos have been split into frames. |
| `frames_extracted`: [TimeStampedBool](#timestampedbool) | Whether frames have been extracted. |
| `frames_prepared`: [TimeStampedBool](#timestampedbool) | Whether frames have been prepared for inferencing. |
| `frames_inferenced`: [TimeStampedBool](#timestampedbool) | Whether an inference has been performed on the frames. |

### GRDPlantGroup
A dataclass for a single plant group. A single row may contain many plant groups, and each will have the following information:

| Field | Description |
| ----------- | ----------- |
| `videos`: [MultiViewMP](#multiviewmp) | Paths to the cut videos containing the plant group. One for each camera view. |
| `image_directories`: [MultiViewMP](#multiviewmp) | Paths to the image directories containing the plant group. One for each camera view. |
| `n_images`: int | Number of images (files w/extention `.png`) in each `image_directory`. |
| `rogue_labels`: MirrorPath | Path to the file containing the labels for the plant group. |
| `ds_split_number`: int | ds split number if the plant group is a ds split. |

### MultiViewMP
Dataclass that contains paths to all three camera views (nadir, oblique, and bottom).

### TimeStampedBool
A boolean flag with an associated timestamp. Useful for storing the status of a processing step as well as when it was last updated.
`__bool__` and `__eq__`are overridden to perform like a normal boolean value.

| Field | Description |
| ----------- | ----------- |
| `value`: boolean | Boolean value. |
| `timestamp`: str | Timestamp. By default it is the time at instance creation. |

### VersionTimeStamp
A version of the row, which is comprised of the timestamp and the folder depth of that timestamp
in the s3 paths to the raw images. 

| Field | Description |
| ----------- | ----------- |
| `timestamp`: int | Timestamp. HHMMDD, 24 format, thus a simple numerical comparison will provide the greater of two times. (150000 > 110000) |
| `folder_depth`: int | The depth of the folder in the s3 path to the raw imagery. |

### VersionTracker
A dataclass that tracks the version of the row through a list of [VersionTimeStamp](#versiontimestamp) objects.

| Field | Description |
| ----------- | ----------- |
| `versions`: List[[VersionTimeStamp](#versiontimestamp)] | List of [VersionTimeStamp](#versiontimestamp) objects. |

## Primary Functions
### `key_query_grd_rows()`
Provide a `dict` of what key/values to query to get all matching [GRDRows](#grdrow) from the database. Providing no input will return all the [GRDRows](#grdrow) currently in the database. Providing any combination of [GRDRow](#grdrow) key/value pairs will return all [GRDRows](#grdrow) that match all of the provided arguments.

Example:

```python
query = {
    "field_name": "Farmer City 2022 Plot Trial Planting 1",
    "date": "6-16",
}

# Will return the grdrows with the matching `field_name` and `date`
grdrows = key_query_grd_rows(query)
print(len(grdrows))

>> 4

```

### `get_grd_row()`
Provide a `field_name`, `row_number`, and `date` to get a single [GRDRow](#grdrow) from the database.

### `get_grd_row_from_full_video_s3_key()`
Provide a `s3_key` string to get a single [GRDRow](#grdrow) from the database. This function will try and match the given `s3_key` to the set of expected file structures define in the `RoguesS3PathExamples`in `grd_constants.py`. Additional file structures must be added manually to the examples class.

### `put_grd_row()`
Save a [GRDRow](#grdrow) to the database.