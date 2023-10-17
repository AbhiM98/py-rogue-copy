# ground_data_processing
A python module for processing ground data, most commonly video imagery collected on a hagie rig with 3 camera views.
## High-Level Breakdown
### Data Collection 
Data was collected by a detassler rig with 3 cameras driving up and down rows (small technicallity is that the detassler had two rigs and 6 cameras, but no concept of that needs to be reflected in the current repo). 
| Camera View | Description |
| ----------- | ----------- |
| Nadir | Placed directory above the corn row. |
| Bottom | Placed with view parallel to the ground, aimed at the bottom stems. As low to the ground as the rig allowed. |
| Oblique | Placed above the bottom camera, looking down at an angle on the corn row. Camera was rotated 90 degrees to allow more plant height to be in frame. |

Cameras would be started by hand (read: not synchronized), the detassler would drive up and down a number of rows (read: 1-n, with every other backwards), and then save the video files to Box file storage (read: not S3).
### Farmer City Trials 

The 2022 Farmer City plot trials (originally the main focus of data collection) had 12 rows. Each row had a base type, from 3 female types and 1 male type, in a rolling pattern (F1, F2, F3, M1, F1...).
Each row had 22 plots (with approximately 80 plants each), from the following types: 
| Type      | Description |
| ----------- | ----------- |
| B | Base Type |
| 1HL | Hybrid Type 1 |
| 2HL | Hybrid Type 2 |
| HH | Hybrid High Density |
| Mix | Mix of all seeds |
| D3 | Delayed Emergence (3 plants) |

These types add a number of different plants into the base type, to achieve a high variation of plot environments across the whole trial. 
A programmatic map of the trial can be found in `utils/plot_layout.py`.

Data was collected at 4 dates at two identical trials, resulting in napkin math of:
```
4 Dates * 2 Trials * 12 Rows * 22 Plots * 80 Plants = 168960 Plants 
```
Not to mention each plant was collected from 3 views, and via video, so a rolling view of the plant could be extracted.
This is a lot of data.

### Opportunistic Data Collection
The detassler rig was also used to collect data from other fields, including two foundation fields and one production field. These fields are examples of what will actually be seen in production: long, homogenous rows with no prior knowledge of rogue density. 


### Data Breakdown Approach
There are a few major tasks that need to occur to isolate individual plant images and sort them by trial spec:

1. Video Alignment - Synchronize the 3 views 
2. Video Division - A combination of the following (this will vary based on trial type):
    * Row division - Divide a raw video into each single 'Row Pass' 
    * Plot division - Divide a row video into each individual plot
    * DS Split division - Longer, homogenous rows get divided into 10 second (decasecond -> DS) segments.
    * Image divison - Divide each video into ~80-100 plants
3. Mapping Raw Divisions into a Human-Readable structure
4. Inserting human QA processes to verify the workflow is working 

The goal of the division and breakdown is to end up with video segments that contain around 100 plants. This is done to reduce the amount of data that needs to be processed at once, and to allow for more efficient parallelization of the processing.
Depending on the field and the data collected, any combination of the video divison methods in (2) might be used. 

Using the `S3MP` module, the segments for this project are defined in `utils/rogue_segments.py`.
For the Farmer City Plot trials, the raw division structure is detailed in `VideoSegments`, and the human-readable structure is detailed in `ImageSegments`.
For the opportunistic collections, the structure is detailed in `ProductionFieldSegments`, `ProductionFieldWithSplitPassSegments`, and `ProductionFieldImageSegments`.
For machine learning inference, the structure is detailed in `InferenceSegments` and `ProdInferenceSegments`.

These segments do allow for nice key matching with `S3MP`, but are definitely WIP and subject to changes.


### Data Breakdown Process - Farmer City Plot Trials
Most videos cover two plots, so the process is as follows (the programmatic tasks are roughly identical to what is found in `scripts/run_full_vid_division.py`):
1. Have user annotate video offsets (a few methods of automating this are in the works)
2. Generate a ExG measurement for a center slice of each frame from the bottom video
3. Use the ExG data to split video into `Pass A` and `Pass B` - `Pass B` is known to be "backwards"
4. Split passes into plots `00-21` with ExG data using a FFT-based peak finding algorithm (`scripts/identify_and_clip_plots.py`).
5. Split plots into images from the 3 videos with spread column sum data using a peak-tracing method (`scripts/clip_images_col_spread.py`).
6. Have a user annotate duplicate plants
7. Migrate filtered data (with backwards/reverse passes in mind) to a more human-friendly file structure

2-5 are done in `scripts/run_full_vid_division.py` (although they individually exist in their own scripts) and 7 is done by `scripts/remove_duplicates.py`.

## Code Files By Directory
### `ground_data_processing/data_processors`
Files that process data.
| File | Description |
| ----------- | ----------- |
| `clip_images.py` | Clip images from peak finding on framewise data file. | 
| `generate_exg_npy.py` | Generated ExG measurement npy for a video. Multiprocessing variants available. | 
| `stem_peak_tracker.py` | Processing objects for tracking stems across many video frames. |
| `thresh_split_video.py` | Split a video into desired number of segments using peak finding on a framewise data file. | 

### `ground_data_processing/measurements`
Measurements on data.
| File | Description |
| ----------- | ----------- |
| `frame_measurements` | Measurements on video frames. | 
| `line_segments.py` | Line segment measurements. |
| `mask_stats.py` | Mask statistics on ML inference. |

### `ground_data_processing/old_scripts`
I claim no responsibility for what's in this directory. 

### `ground_data_processing/scripts`
Scripts to be adjusted to run processing on data.
 | File | Description |
| ----------- | ----------- |
| `box_to_s3.py` | After a file is _manually_ downloaded from Box, this will upload it to S3 in the correct location. | 
| `calc_inference_stats.py` | Calculate statistics on ML inference. |
| `calc_stem_width.py` | Calculate stem width on bottom camera imagery. | 
| `clip_images.py` | Clip images from a video. | 
| `clip_images_col_spread.py` | Clip images from a video using column spread peak tracking. This is the best clipping method currently. |
| `clip_images_hp_sum.py` | WIP script on image clipping based on a paper I read. The paper is a good read. Houses some useful utilies that the column spread logic usess. | 
| `ds_splits.py` | Split a video into decaseconds. |
| `generate_framewise_data.py` | Generate a value for each frame in a video, and save it. | 
| `handle_missing.py` | Ad-hoc GUI for handling missing plants. |
| `identify_and_clip_passes.py` | Clip video into row passes. | 
| `identify_and_clip_plots.py` | Clip pass video into individual plots. | 
| `make_manual_offset_json.py` | Make an offset JSON from human inputs. | 
| `manual_offset_leading_edge.py` | Generate an offset by identifiying the leading edge of some framewise measurement in each video. GUI and usability are good, effective framewise measurement is still WIP. | 
| `npy_plot_sync.py` | Plot viewer with syncronized plots. |
| `plot_trial_crop_to_images_unfiltered.py` | Crop unfiltered images from plot trials. | 
| `prep_paddle_inference.py` | Prepare data for inference with PaddlePaddle. |
| `remove_duplicates.py` | Migrate image files from the raw directory to the human-friendly one while removing images marked as duplicates, with optional crop. | 
| `run_full_vid_division.py` | Take an original video and run all steps to get raw images. | 
| `run_paddle_inference.py` | Run inference on a video with PaddlePaddle. Must be run from within the `PaddleDetection` repo. |
| `s3_download.py` | Download files to local mirror. | 
| `s3_downsize.py` | Downsize video or image files. | 
| `s3_rename.py` | Rename files on S3 (given how S3 works, this is really a copy to a new key and a delete of the old key). | 
| `validate_offset.py` | Displays the first images from a specified video to verify if the offset is correct. |
| `view_npy_plot_split.py` | View a plot of npy data with annotated peak finding. | 


### `ground_data_processing/utils`
| File | Description |
| ----------- | ----------- |
| `absolute_segment_groups.py` | `S3MP` file structure definitions, with depths defined as absolute values. |
| `exg_utils.py` | Video iterators with conditional operations based on ExG data. | 
| `ffmpeg_utils.py` | Utilities for FFmpeg multiprocessing. | 
| `hough.py` | Hough transform utilities (custom implementation that returns the accumulator array). |
| `image_utils.py` | Image operations and thresholds. | 
| `iter_utils.py` | Iteration utilities. | 
| `keymaps.py` | `Keymap` object for nontrivial mappings between different S3 directories. |
| `multiprocessing_utils.py` | Generalized global process management. | 
| `peak_finding_utils.py` | Peak finding and search-based peak finding wrappers. | 
| `plot_layout.py` | Plot trial layout in a programmatic, usable form. | 
| `plot_utils.py` | `pyplot` utilities. | 
| `processing_utils.py` | Utilities for logging versioning and other metadata for a given processing task. |
| `relative_segment_groups.py` | `S3MP` file structure definitions, with depths defined as relative values for modularity. | 
| `rogue_key_utils.py` | Utilities for getting common groups of S3 key segments. | 
| `s3_constants.py` | Commonly used S3 key segment names, groups, and functions. | 
| `video_utils.py` | Utilities for working with video files. | 

# Ground Rogues Spring 2023
A refactor of the ground rogues processing pipeline designed to be more modular and extensible.

### `ground_data_processing/params.py`
Home to the [FieldParams](#fieldparams) and [RowParams](#rowparams) classes, which manage all the S3, ddb, etc setup needed to run the [processing steps](#ground_data_processingprocessing_steps).

## FieldParams
Parameters for processing a single field.

| Field | Description |
| ----------- | ----------- |
| `field_name`: str | Field name. |
| `date`: str | Date when data was collected. YYYY-MM-DD format. |
| `overwrite`: bool | Whether files on S3 should be overwritten if present already. |
| `setup_s3`: bool | Whether to initialize the S3 paths/resources. Used automatically when creating [RowParams](#rowparams) from [FieldParams](#fieldparams) to avoid unnecessary duplication of labor. |
| `setup_ddb`: bool | Whether to initialize the ddb resources. Used automatically when creating [RowParams](#rowparams) from [FieldParams](#fieldparams) to avoid unnecessary duplication of labor. |
| `s3_client`: boto3 client object | boto3 client for S3 calls. |
| `s3_bucket`: str | Name of S3 bucket. |
| `row_holder_mp`: MirrorPath | Path to the folder that holds the Row folders. |
| `n_rows`: int | Number of rows present in the row holder folder. |
| `ddb_resource`: boto3 resource object | boto3 resource for ddb calls. |
| `grdrows`: List[GRDRow] | List of GRDRow objects from the Ground Rogues Database, one per row. See the README in `ddb_tracking` for details. |
| `row_params`: List[[RowParams](#rowparams)] | List of [RowParams](#rowparams) objects, one per row. Access a row using `row_params[row_number - 1]` |

## RowParams
Parameters for processing a single row. Most processing steps run on a by-row level.

| Field | Description |
| ----------- | ----------- |
| `field_name`: str | Field name. |
| `date`: str | Date when data was collected. YYYY-MM-DD format. |
| `row_number`: int | Row number. |
| `overwrite`: bool | Whether files on S3 should be overwritten if present already. |
| `setup_s3`: bool | Whether to initialize the S3 paths/resources. Used automatically when creating [RowParams](#rowparams) from [FieldParams](#fieldparams) to avoid unnecessary duplication of labor. |
| `setup_ddb`: bool | Whether to initialize the ddb resources. Used automatically when creating [RowParams](#rowparams) from [FieldParams](#fieldparams) to avoid unnecessary duplication of labor. |
| `s3_client`: boto3 client object | boto3 client for S3 calls. |
| `s3_bucket`: str | Name of S3 bucket. |
| `row_mp`: MirrorPath | Path to the `Row {row_number}` folder. |
| `ds_splits_mp`: MirrorPath | Path to the `DS Splits` folder. |
| `ddb_resource`: boto3 resource object | boto3 resource for ddb calls. |
| `grdrow`: GRDRow | GRDRow object from the Ground Rogues Database. See the README in `ddb_tracking` for details. |

### `ground_data_processing/processing_steps`
A central location for the function calls that make up the main processing steps.
| File | Description | Args |
| ----------- | ----------- | ----------- |
| `generate_offset_from_gps_timestamp.py` | Main function `generate_offset_from_gps_timestamp()` | RowParams
| `ds_splits.py` | Main function `generate_ds_splits()` | RowParams
| `clip_images_col_spread.py` | Main function `extract_frames_from_row()` | RowParams
| `prep_unfiltered_inference.py` | Main function `prep_unfiltered_inference()` | RowParams
| `run_paddle_inference.py` | Main function `run_paddle_inference()` | RowParams

### `ground_data_processing/processing_entrypoints`
Scripts to run the [processing steps](#ground_data_processingprocessing_steps) from the command line.
| File | Description | Args |
| ----------- | ----------- | ----------- |
| `base.py` | Function `base_entrypoint()` wraps a processing step in the logic to parse command line arguments. All other entrypoints will use the arguments parsed here.  | `--field_name` <string> name of field. `--date` <string> date (YYYY-MM-DD). `--row_number` <int> row number to process. If not present, all rows in a field will be processed. `--ds_split_numbers` <string> Comma separated list of DS Split numbers (ints), ie "1, 2, 3". `--nadir_crop_height` <string> Crop height used when preparing images for inference, ie "2000". `--nadir_crop_width` <string> Crop width used when preparing images for inference, ie "1000". `--overwrite` <string> "True" will overwrite existing files on S3. By default, "False". `--rerun` <string> "True" will rerun processing steps, even if they are already marked as done. By default, "False".
| `generate_gps_offsets.py` | Script to determine the frame offsets of the three camera views using the GPS timestamps in said videos.  | See `base.py`.
| `generate_ds_splits.py` | Script to generate ds splits from the `offset.json` and the raw videos. | See `base.py`.
| `extract_frames.py` | Script to split the videos for each DS Split into images centered on a single plant. | See `base.py`.
| `prep_inference.py` | Script to crop and resize the images so that they are ready to be inferenced. | See `base.py`.
| `run_inference.py` | Script to run the Paddle inference. Depends on having the `PaddleDetection` repo setup with the SoloV2 model config/params files present. Can be run from `py-rogue-detection` or from within `PaddleDetection`. | See `base.py`.
| `prep_and_run_paddle_inference.py` | Script to run just the frame preperation, and then invoke ecs tasks to run the inference. | See `base.py`.
| `run_all.py` | Script to run the five processing steps in sequence, `generate_offset_from_gps_timestamp()`, `generate_ds_splits()`, `extract_frames_from_row()`, `prep_unfiltered_inference()`, and then a lambda function is invoked to run the paddle inference on ECS. | See `base.py`.

### `ground_data_processing/paddle_files`
Files for running inference with PaddlePaddle. Instead of including the whole PaddleDetection repo, only the files needed for inference are included here. They must be placed in the `PaddleDetection` repo to work. These files are also included in `ground_data_processing/terraform/paddle_files` for use in building the docker image used for running just the paddle inference.

| File | Description |
| ----------- | ----------- |
| `coco_instance.yml` | Replaces `PaddleDetection/configs/datasets/coco_instance.yml` |
| `solov2_r101_vd_fpn_3x_coco.yml` | Replaces `PaddleDetection/configs/solov2/solov2_r101_vd_fpn_3x_coco.yml` |
| `infer.py` | Replaces `PaddleDetection/tools/infer.py` |
| `best_model.pdparams` | File is too big to include on git, so this needs to be added manually. Then it needs to be placed in a new directory `PaddleDetection/output/solov2_r101_vd_fpn_3x_coco/best_model.pdparams`  |

## Rogues Docker Images
Two docker images are used in the automation pipeline, `rogues-prod-ecs-img` and `rogues-prod-inference-img`. Initially these were combined into one, however ECS was timing out when pulling the image from ECR because it was rather large. Splitting them into two allows for less bloat in the individual images.

### `rogues-prod-ecs-img`
This is the image used for the core image processing steps (gps offsets, ds splits, frame extraction, and inference prep.). It is defined in the Dockerfile found in the root of the `py-rogue-detection` repo, along with a `.dockerignore`. It can be build and pushed manually using the following commands:

    >> cd py-rogue-detection
    >> docker build . --tag "475283710372.dkr.ecr.us-east-1.amazonaws.com/rogues-processing:rogues-prod-ecs-img" --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_ecdsa)"
    >> aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "475283710372.dkr.ecr.us-east-1.amazonaws.com"
    >> docker push "475283710372.dkr.ecr.us-east-1.amazonaws.com/rogues-processing:rogues-prod-ecs-img"

Note that the private ssh key used in the `docker build` command is taken from the local `.ssh/id_ecdsa` file. If you wish to use a different file, be sure to change all references to `id_ecdsa` in the Dockerfile.

### `rogues-prod-inference-img`
This is the image used for the running the paddle inference. It lives in the `ground_data_processing/terraform/` directory becuase it runs in python 3.8.10, and the lambda dir already had some reworked dependencies like `ddb_tracking` and `S3MP` that are retrofitted to work in python 3.8. 

    >> cd py-rogue-detection/ground_data_processing/terraform
    >> docker build . --tag "475283710372.dkr.ecr.us-east-1.amazonaws.com/rogues-processing:rogues-prod-inference-img" --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_ecdsa)"
    >> aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "475283710372.dkr.ecr.us-east-1.amazonaws.com"
    >> docker push "475283710372.dkr.ecr.us-east-1.amazonaws.com/rogues-processing:rogues-prod-inference-img"

Note once again that the private ssh key used in the `docker build` command is taken from the local `.ssh/id_ecdsa` file. If you wish to use a different file, be sure to change all references to `id_ecdsa` in the Dockerfile.

Also note that the two key differences in these commands (when compared to `rogues-prod-ecs-img`) are
- the location of the Dockerfile
- the tag name


## Launching Processing Tasks
The processing steps are run as ECS tasks, which can be launched from `terraform/ecs_procs.py`. The two main tasks that can be run manually are `batch_run_processing_step` and `batch_run_prep_and_run_paddle_inference`. The former will run the processing step you specify (one of `run-all`, `generate-ds-splits`, `extract-frames`, `run-paddle-inference`), and the latter will run just the inference prep followed by the paddle inference. The arguments for both functions are mostly the same, and are detailed below. They are passed into their respective functions as the `event` field (dict). To run, simply call the function with the desired arguments. For example, to run `batch_run_prep_and_run_paddle_inference`, you could do the following:
```
    event = {
        "field_name": "2023-field-data",
        "date": "2023-07-06",
        "rows": {
            "1": [1,2,3], # row 1, splits 1 through 3
            "2": [1,2,3], # row 2, splits 1 through 3
        },
        "nadir_crop_height": "1600",
        "nadir_crop_width": "1000",
        "overwrite": "True",
        "rerun": "True",
    }
    batch_run_prep_and_run_paddle_inference(event, None)
```
Note that the `event` field is passed in as the first argument, and the second argument is `None`. This is because the `batch_run_prep_and_run_paddle_inference` function is designed to be called from a lambda function, which passes in the `event` and `context` fields. The `event` field is passed in as a dict, and the arguments are accessed as `event["field_name"]`, `event["date"]`, etc. The arguments are detailed below.

| Arg | Description | Required |
| ----------- | ----------- | ----------- |
| `field_name`: str | Field name. | Yes |
| `date`: str | Date when data was collected. YYYY-MM-DD format. | Yes |
| `processing_step`: str | Process to run. One of `run-all`, `generate-ds-splits`, `extract-frames`, `run-paddle-inference` | Only for `batch_run_processing_step`.  |
| `row_numbers`: List[int] | Row numbers to process. Eg: [1,2,4] | Only for `batch_run_processing_step`. |
| `rows`: Dict[str, List[int]] | Dictionary of row numbers to DS Split numbers. Eg: {"1": [1,2,3], "2": [1,2,3]} for rows 1 and 2, splits 1 through 3. An empty list will run all splits in a row. Eg: {"1": []} will run all splits in row 1. | Only for `batch_run_prep_and_run_paddle_inference`. |
| `nadir_crop_height`: str | Pixel height used when cropping nadir images for inference. Eg: "2000". | No. If not present, will crop normally. |
| `nadir_crop_width`: str | Pixel width used when cropping nadir images for inference. Eg: "2000". | No. Currently only valid if `nadir_crop_height` is also specified. |
| `overwrite`: str | Whether to overwrite existing files on S3. Eg: "True". | No. If not present, will not overwrite. |
| `rerun`: str | Whether to rerun processing steps, even if they are already marked as done. Eg: "True". | No. If not present, will skip completed steps. |
| `delay`: str | Number of seconds to wait in between launching each task. Eg: "60" for 5 seconds. | No. If not present, will not delay. |
| `batch_size`: str | Number of tasks to launch at once before applying the `delay`. Eg: "5" for 5 tasks. | No. If not present, will launch in batches of 1. Only valid if `delay` is specified. |

## Notes
- IMPORTANT: since the ECS Cluster is designed to gradually spin up as rows are uploaded, running a large batch of jobs cold turkey can cause issues. For this reason, the `delay` field exists. When starting tasks, check the number of active instances [here](https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#AutoScalingGroupDetails:id=prod-rogues-ecs-cluster-one-20230530192028803000000008;view=details). For running a full field, if there is currently a desired capacity of ~10 or 15, things will be fine. If there is currently a desired capacity of 0 or 1, you should probably add a delay of 30 seconds or so with a batch size of 1. This will allow the cluster to spin up a few instances before launching more tasks. You can also manually change the desired capacity right before launching tasks if you want to get things going sooner. The autoscaling group will automatically spin things down when tasks start to finish, so don't worry about micromanaging the desired capacity.
- To see what tasks are actively running, you can go to the [ECS console](https://us-east-1.console.aws.amazon.com/ecs/v2/clusters/prod-rogues-ecs-cluster/tasks?region=us-east-1) and see a list of tasks. Note that tasks can and will wait in a queue for up to 10 minutes before starting, so don't worry if tasks don't show up in the ECS console right away after launching. You can also see the logs for each task by clicking on the task ID, and then clicking on the `logs` tab. The logs are also stored on [CloudWatch](https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group/prod-rogues-ecs-api-log-group).  
- `rerun` will reset process flags and allow you to rerun a process
- `overwrite` will overwrite existing files if "True", otherwise will skip replacing existing files
- When both `rerun` and `overwrite` are "True", the process will be rerun and overwrite existing files
- When `rerun` alone is "True", the process will be rerun, but will skip replacing existing files. This is useful if a process failed part-way through, and you want to rerun it without overwriting the files that were already processed.
- When neither `rerun` nor `overwrite` are "True", the process will be skipped if all files are already present
- To generate an overview of what rows have been processed, run `terraform/grd_reports.py`, which will query the database and dump a csv in `sentera-rogues-data/2023-field-data` on S3.
- To stop running tasks, you can use the same console link above, and select the tasks you wish to stop using the checkboxes on the left. Then click on the `Stop` button, and select `Stop Selected`.