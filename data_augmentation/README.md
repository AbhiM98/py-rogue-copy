# Data Augmentation Module
This module provides configurable execution of basic data augmentation of MSCOCO datasets. It provides a framework that can be easily modified to include new data augmentation types, as well as a simple interface for configuring the augmentation process. Additionally, it enables the merging of several MSCOCO datasets into a single dataset.

## Usage
The main entrypoint for the module is the `run_from_config.py` script. This script takes a configuration file as input, and executes the data augmentation process as specified in the configuration file. The script is run as follows:
    
    >> python run_from_config.py --config <path_to_config_file>


## Configuration
The configuration file is a yaml file that specifies the parameters for the data augmentation process. Additionally, in order to change the S3 bucket used by S3MP, change the `S3MPConfig.default_bucket_key` value in `__init__.py`.


The configuration file is divided into several sections, each of which is described below.

### RUN_TYPE
This section specifies the type of data augmentation to be performed. The options are:
* `merge`: Merge several MSCOCO datasets into a single dataset.
* `augment`: Perform data augmentation on a single MSCOCO dataset.

### DATA
This section specifies the input and output paths for the data augmentation process. The options are:
* `dataset_s3_path`: The S3 path to the dataset to be augmented. This path should point to the root of the dataset.
* `images_folder_name`: The name of the folder containing the images in the dataset. For example, if the images are located at `<dataset_s3_path>/images`, then `images` should be specified here.
* `annotations_folder_name`: The name of the folder containing the annotations in the dataset. For example, if the annotations are located at `<dataset_s3_path>/labels`, then `labels` should be specified here.
* `annotations_file_name`: The name of the annotations file in the dataset. For example, if the annotations file is located at `<dataset_s3_path>/annotations/dataset.json`, then `dataset.json` should be specified here.
* `destination_s3_path`: The S3 path to which the augmented dataset should be written. This path should point to the root of the dataset. The augmented dataset will be written to the same folder structure as the original dataset.
* `default_height`: The default height (in pixels) to which all images will be resized during augmentation. If this value is not specified, then the original height of the image will be used.
* `default_width`: The default width (in pixels) to which all images will be resized during augmentation. If this value is not specified, then the original width of the image will be used.
* `overwrite`: Whether to overwrite the *source* dataset stored in the local mirror. If this value is not specified, then the default value of `False` will be used to save time and avoid spurrious downloads. Note that the *destination* dataset will always be overwritten if it exists.

#### `merge` run type only
When the `merge` run type is specified, the `DATA` section should contain a list of datasets to be merged. Each dataset must have at minimum a `dataset_s3_path`, and any other parameters not specified will be taken from the first dataset in the list. As an exception, the `destination_s3_path` can *only* be specified for the first dataset in the list, and will be used for the merged dataset. Additionally, there is an additional `DATA` parameter that can be specified:
* `number`: The number of images to be sampled from the dataset. If this value is not specified, then *all* images in the dataset will be used (ie, the value will NOT be copied from the first dataset in the list).

See `lis_2023_merge.yaml` for an example of a `merge` run type configuration file.

### AUGMENTATION
This section specifies the parameters for the data augmentation process. The following options are present for all augmentation types:
* `number`: The number of times to perform the augmentation. 

#### RANDOM_CROP
This augmentation type performs a random crop of the image to the specified crop size, and then pads the image to the `default_height` and `default_width`. The following options are present:
* `crop_height`: The height (in pixels) of the crop.
* `crop_width`: The width (in pixels) of the crop.

#### RANDOM_ZOOM
This augmentation type performs a random zoom of the image by cropping to specified crop size, and then resizing the image to the `default_height` and `default_width`. The following options are present:
* `crop_height`: The height (in pixels) of the crop.
* `crop_width`: The width (in pixels) of the crop.


## Development
The module contains two main classes that manage the data augmentation/dataset merging processes: `CocoManager` and `AugmentationManager`, located in `coco_manager.py` and `augmentation_manager.py`, respectively. The `CocoManager` class is responsible for loading/saving the dataset to and from S3, and the `AugmentationManager` class is responsible for performing the data augmentations. The `run_from_config.py` script is responsible for parsing the configuration file and calling the appropriate methods on the `CocoManager` and `AugmentationManager` classes.