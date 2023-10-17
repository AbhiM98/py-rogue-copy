"""Class for managing COCO dataset."""
from __future__ import annotations

import copy
import json
import os
import random
import time

import cv2
from S3MP.mirror_path import MirrorPath
from tqdm import tqdm

from data_augmentation.constants import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    CocoAnnotationKeys,
    CocoCategoryKeys,
    CocoImageKeys,
    CocoKeys,
    ConfigKeys,
)
from data_augmentation.types import segmentation
from data_augmentation.utils import (
    add_suffix_to_image_name,
    area_and_bbox_from_segmentation,
    area_from_segmentation,
    bbox_from_segmentation,
    delete_mp,
    get_image_name_from_mp,
    mp_copy_s3_only,
)


class CocoManager:
    """Class for managing COCO dataset."""

    def __init__(
        self,
        coco_json_mp: MirrorPath = None,
        coco_json: dict = None,
        images_dir_mp: MirrorPath = None,
        default_height: int = DEFAULT_IMAGE_HEIGHT,
        default_width: int = DEFAULT_IMAGE_WIDTH,
        overwrite: bool = False,
    ):
        """Initialize a CocoManager."""
        # Get coco json from mirror path if provided
        if coco_json_mp is not None:
            if not coco_json_mp.exists_on_s3():
                print(
                    f"[INFO]: MirrorPath {coco_json_mp.s3_key} is not an existing json file on S3, initializing with empty dataset."
                )
                coco_json = None
            else:
                print(f"[INFO]: Loading coco json from {coco_json_mp.s3_key}...")
                coco_json_mp.download_to_mirror(overwrite=overwrite)
                with open(coco_json_mp.local_path, "r") as f:
                    coco_json = json.load(f)
        # Initialize with empty dataset if coco json not provided
        if coco_json is None:
            coco_json = {
                # initialize with empty dataset
                CocoKeys.IMAGES: {
                    "date_created": time.strftime("%Y-%m-%d"),
                },
                CocoKeys.LICENSES: [],
                CocoKeys.IMAGES: [],
                CocoKeys.ANNOTATIONS: [],
                CocoKeys.CATEGORIES: [],
            }
        # Check that coco json is a dict
        elif not isinstance(coco_json, dict):
            raise ValueError(f"Coco json must be a dict, not {type(coco_json)}.")

        self.coco_json_mp: MirrorPath = coco_json_mp
        self.coco_json: dict = coco_json
        self.images_dir_mp: MirrorPath = images_dir_mp
        self._default_height: int = default_height
        self._default_width: int = default_width
        self.overwrite: bool = overwrite

        # Useful properties
        self.images = self.coco_json[CocoKeys.IMAGES]
        self.annotations = self.coco_json[CocoKeys.ANNOTATIONS]
        self.image_names = [image[CocoImageKeys.FILE_NAME] for image in self.images]
        # init annotations_by_image_name
        self.annotations_by_image_name: dict = {}
        for annotation in self.annotations:
            image_name: str = self.image_names[annotation[CocoAnnotationKeys.IMAGE_ID]]
            if image_name not in self.annotations_by_image_name:
                self.annotations_by_image_name[image_name]: list = []
            self.annotations_by_image_name[image_name].append(annotation)

    @property
    def categories(self) -> list[dict]:
        """Return the categories field of the coco json."""
        return self.coco_json[CocoKeys.CATEGORIES]

    def set_categories(self, categories: list[dict]):
        """Set the categories field of the coco json."""
        self.coco_json[CocoKeys.CATEGORIES] = categories

    def add_image(self, image: dict):
        """Add an image to the coco json."""
        # Ensure all keys are present
        if any(key not in image for key in CocoImageKeys):
            missing_keys = [key for key in CocoImageKeys if key not in image]
            raise ValueError(f"Coco image missing keys: {missing_keys}.")
        self.coco_json[CocoKeys.IMAGES].append(image)

    def collision_avoid_add_image(self, image: dict) -> tuple[str, int]:
        """Add an image to the coco json, avoiding name collisions, and return the image name and image id."""
        image_name = image[CocoImageKeys.FILE_NAME]
        # Check for collisions
        ux_idx = 0
        while image_name in self.image_names:
            ux_idx += 1
            new_image_name = add_suffix_to_image_name(image_name, f"dup_{ux_idx}")
            print(
                f"[INFO]: Duplicate image name {image_name}, renaming to {new_image_name}"
            )
            image_name = new_image_name

        # Add image
        image_id = len(self.images)
        self.add_image(
            {
                CocoImageKeys.ID: image_id,
                CocoImageKeys.FILE_NAME: image_name,
                CocoImageKeys.HEIGHT: image[CocoImageKeys.HEIGHT],
                CocoImageKeys.WIDTH: image[CocoImageKeys.WIDTH],
            }
        )
        return image_name, image_id

    def collision_avoid_add_image_from_mp(
        self, image_mp: MirrorPath
    ) -> tuple[MirrorPath, int]:
        """Add an image to the coco json, avoiding name collisions, and return the image mp and image id."""
        image_name = get_image_name_from_mp(image_mp)

        # Check that image exists
        if image_mp.exists_in_mirror():
            # Get image height and width
            image = cv2.imread(str(image_mp.local_path))
            height, width = image.shape[:2]
        else:
            # If the image doens't exist yet, use default height and width
            height, width = self._default_height, self._default_width

        dummy_image = {
            CocoImageKeys.FILE_NAME: image_name,
            CocoImageKeys.HEIGHT: height,
            CocoImageKeys.WIDTH: width,
        }

        # Add image
        image_name, image_id = self.collision_avoid_add_image(dummy_image)

        new_image_mp = image_mp.get_parent().get_child(image_name)
        return new_image_mp, image_id

    def add_annotation(self, annotation: dict):
        """Add an annotation to the coco json."""
        # Ensure all keys are present
        if any(key not in annotation for key in CocoAnnotationKeys):
            missing_keys = [key for key in CocoAnnotationKeys if key not in annotation]
            raise ValueError(f"Coco annotation missing keys: {missing_keys}.")
        self.coco_json[CocoKeys.ANNOTATIONS].append(annotation)

    def add_annotation_from_segmentation(
        self,
        seg: segmentation,
        bbox: list[float] = None,
        area: float = None,
        image_id: int = None,
        category_id: int = None,
        is_crowd: int = 0,
    ):
        """Add an annotation to the coco json from a segmentation."""
        # Check inputs
        if seg is None:
            raise ValueError("Segmentation must be provided for annotation.")
        if image_id is None:
            raise ValueError("Image id must be provided for annotation.")
        if category_id is None:
            # If there is only one category, use it
            if len(self.coco_json[CocoKeys.CATEGORIES]) == 1:
                category_id = self.coco_json[CocoKeys.CATEGORIES][0][
                    CocoCategoryKeys.ID
                ]
            else:
                raise ValueError("Category id must be provided for annotation.")
        if bbox is None and area is None:
            area, bbox = area_and_bbox_from_segmentation(seg)
        if bbox is None:
            bbox = bbox_from_segmentation(seg)
        if area is None:
            area = area_from_segmentation(seg)

        # Add annotation
        annotation_id = len(self.annotations)
        self.add_annotation(
            {
                CocoAnnotationKeys.ID: annotation_id,
                CocoAnnotationKeys.IMAGE_ID: image_id,
                CocoAnnotationKeys.CATEGORY_ID: category_id,
                CocoAnnotationKeys.BBOX: bbox,
                CocoAnnotationKeys.AREA: area,
                CocoAnnotationKeys.SEGMENTATION: seg,
                CocoAnnotationKeys.IS_CROWD: is_crowd,
            }
        )

    def save(self, coco_json_mp: MirrorPath = None):
        """Save the coco json to a mirror path, both locally and on s3."""
        if coco_json_mp is None:
            coco_json_mp: MirrorPath = self.coco_json_mp
        # Ensure directory exists
        os.makedirs(coco_json_mp.local_path.parent, exist_ok=True)
        with open(coco_json_mp.local_path, "w") as f:
            json.dump(self.coco_json, f)
        coco_json_mp.upload_from_mirror(overwrite=self.overwrite)

    @staticmethod
    def from_config(config: dict) -> CocoManager:
        """Construct a CocoManager from a config dict."""
        src_root_mp = MirrorPath.from_s3_key(config[ConfigKeys.Data.DATASET_S3_PATH])
        images_dir_mp = src_root_mp.get_child(
            config[ConfigKeys.Data.IMAGES_FOLDER_NAME]
        )
        coco_json_mp = src_root_mp.get_child(
            config[ConfigKeys.Data.ANNOTATIONS_FOLDER_NAME]
        ).get_child(config[ConfigKeys.Data.ANNOTATIONS_FILE_NAME])
        return CocoManager(
            coco_json_mp=coco_json_mp,
            images_dir_mp=images_dir_mp,
            default_height=config[ConfigKeys.Data.DEFAULT_HEIGHT],
            default_width=config[ConfigKeys.Data.DEFAULT_WIDTH],
            overwrite=config[ConfigKeys.Data.OVERWRITE]
            if ConfigKeys.Data.OVERWRITE in config
            else False,
        )

    @staticmethod
    def from_configs(configs: list[dict]) -> list[CocoManager]:
        """Construct a list of CocoManagers from a list of config dicts."""
        # Construct coco managers, and if any is missing a field, default to the first one
        for config in configs:
            for config_key in ConfigKeys.Data:
                match config_key:
                    case ConfigKeys.Data.OVERWRITE | ConfigKeys.Data.NUMBER:
                        continue
                    case ConfigKeys.Data.DATASET_S3_PATH:
                        if config_key not in config:
                            raise ValueError(
                                f"Config {config} is missing {config_key}."
                            )
                    case _:
                        # For all other keys, if they're not provided, default to the first config
                        if config_key not in config:
                            config[config_key] = configs[0][config_key]

        return [CocoManager.from_config(config) for config in configs]

    @staticmethod
    def merge_coco_managers(
        coco_managers: list[CocoManager],
        n_images_per_coco_manager: list[int],
        dest_coco_json_mp: MirrorPath,
        dest_images_dir_mp: MirrorPath,
    ) -> CocoManager:
        """Merge a list of coco managers into a single coco manager, and upload images + json to s3."""
        # If any of the coco managers doesn't have the images_dir_mp field,
        # raise an error
        if any(coco_manager.images_dir_mp is None for coco_manager in coco_managers):
            raise ValueError(
                "All coco managers must have the images_dir_mp field set to merge them."
            )

        # If the coco managers don't all have the same catergories, raise an error
        if any(
            coco_manager.categories != coco_managers[0].categories
            for coco_manager in coco_managers
        ):
            raise ValueError(
                "All coco managers must have the same categories to merge them."
            )

        # If the length of the n_images_per_coco_manager list doesn't match the length of the coco_managers list, raise an error
        if len(n_images_per_coco_manager) != len(coco_managers):
            raise ValueError(
                f"n_images_per_coco_manager (len: {len(n_images_per_coco_manager)}) list must match coco_managers list (len: {len(coco_managers)})."
            )

        # Create new coco manager
        # If the destination coco json already exists, remove it
        delete_mp(dest_coco_json_mp)
        coco_manager = CocoManager(
            coco_json_mp=dest_coco_json_mp,
            images_dir_mp=dest_images_dir_mp,
        )

        # Add categories
        coco_manager.set_categories(coco_managers[0].categories)

        # Add images and annotations
        for idx, coco_manager_ in enumerate(coco_managers):
            # If n_images_per_coco_manager is None, add all images
            if n_images_per_coco_manager[idx] is None:
                n_images_per_coco_manager[idx] = len(coco_manager_.images)
            # Make random selection of images
            images = coco_manager_.images
            if len(images) > n_images_per_coco_manager[idx]:
                images = random.sample(images, n_images_per_coco_manager[idx])
            print(f"[INFO]: Adding {len(images)} images from coco manager {idx+1}...")
            for image in tqdm(images):
                # Ensure all images have unique names
                image_mp = coco_manager_.images_dir_mp.get_child(
                    image[CocoImageKeys.FILE_NAME]
                )
                new_image_mp, image_id = coco_manager.collision_avoid_add_image_from_mp(
                    image_mp
                )
                # Add annotations for the image
                for annotation in coco_manager_.annotations_by_image_name[
                    image[CocoImageKeys.FILE_NAME]
                ]:
                    annotation = copy.deepcopy(annotation)
                    annotation[CocoAnnotationKeys.IMAGE_ID] = image_id
                    try:
                        coco_manager.add_annotation(annotation)
                    except ValueError:
                        # If annotation is missing keys, fill them in
                        coco_manager.add_annotation_from_segmentation(
                            annotation[CocoAnnotationKeys.SEGMENTATION],
                            bbox=annotation[CocoAnnotationKeys.BBOX]
                            if CocoAnnotationKeys.BBOX in annotation
                            else None,
                            area=annotation[CocoAnnotationKeys.AREA]
                            if CocoAnnotationKeys.AREA in annotation
                            else None,
                            image_id=image_id,
                            category_id=annotation[CocoAnnotationKeys.CATEGORY_ID]
                            if CocoAnnotationKeys.CATEGORY_ID in annotation
                            else None,
                            is_crowd=annotation[CocoAnnotationKeys.IS_CROWD]
                            if CocoAnnotationKeys.IS_CROWD in annotation
                            else 0,
                        )
                # Move image to new location (S3 only for speed)
                new_image_name = get_image_name_from_mp(new_image_mp)
                dest_mp = dest_images_dir_mp.get_child(new_image_name)
                mp_copy_s3_only(image_mp, dest_mp)

        # Save
        coco_manager.save()
        return coco_manager
