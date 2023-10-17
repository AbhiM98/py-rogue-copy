"""Main entrypoint for running data augmentation from a config yaml file."""

import argparse
import os

import yaml

from data_augmentation.augmentation_manager import AugmentationManager
from data_augmentation.coco_manager import CocoManager
from data_augmentation.constants import ConfigKeys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run data augmentation from a config yaml file."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config yaml file.",
        required=True,
    )
    args = parser.parse_args()

    # Get the config file
    config_path: str = args.config
    if not os.path.isfile(config_path):
        raise ValueError(f"Config file {config_path} does not exist.")

    # Load the config file
    with open(config_path, "r") as f:
        config: dict = yaml.safe_load(f)

    run_type = config[ConfigKeys.BaseKeys.RUN_TYPE]

    # Ensure run type is valid
    if run_type not in [x for x in ConfigKeys.RunType]:
        raise ValueError(
            f"Invalid run type {run_type}. Chose from {[run_type.value for run_type in ConfigKeys.RunType]}."
        )

    match run_type:
        case ConfigKeys.RunType.MERGE:
            # Import here or else things break
            from S3MP.mirror_path import MirrorPath

            data = config[ConfigKeys.BaseKeys.DATA]
            # Ensure data is a list of dicts
            if not isinstance(data, list):
                raise ValueError(
                    f"To merge, {ConfigKeys.BaseKeys.DATA.value} must be a list of dicts, received {type(data)}"
                )
            # Warn user that the destination_s3_path will only be taken from the first dict
            if any(ConfigKeys.Data.DESTINATION_S3_PATH in d for d in data[1:]):
                print(
                    f"[WARNING]: {ConfigKeys.Data.DESTINATION_S3_PATH.value} will only be taken from the first dict, using: {data[0][ConfigKeys.Data.DESTINATION_S3_PATH]}."
                )

            # Construct the coco managers
            coco_managers = CocoManager.from_configs(data)
            print(f"[INFO]: Loaded {len(coco_managers)} coco managers.")

            # Get the desired number of images per dataset
            n_images_per_coco_manager = [
                d[ConfigKeys.Data.NUMBER] if ConfigKeys.Data.NUMBER in d else None
                for d in data
            ]

            # Get desired destination coco mp and dest images folder from the first dict
            dest_coco_json_mp = (
                MirrorPath.from_s3_key(data[0][ConfigKeys.Data.DESTINATION_S3_PATH])
                .get_child(data[0][ConfigKeys.Data.ANNOTATIONS_FOLDER_NAME])
                .get_child(data[0][ConfigKeys.Data.ANNOTATIONS_FILE_NAME])
            )
            dest_images_dir_mp = MirrorPath.from_s3_key(
                data[0][ConfigKeys.Data.DESTINATION_S3_PATH]
            ).get_child(data[0][ConfigKeys.Data.IMAGES_FOLDER_NAME])

            # Merge the coco managers
            CocoManager.merge_coco_managers(
                coco_managers=coco_managers,
                n_images_per_coco_manager=n_images_per_coco_manager,
                dest_coco_json_mp=dest_coco_json_mp,
                dest_images_dir_mp=dest_images_dir_mp,
            )
        case ConfigKeys.RunType.AUGMENT:
            # Run the data augmentation
            AugmentationManager(config).run()
