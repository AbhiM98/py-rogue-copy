"""Class for managing performing data augmentations on a COCO dataset."""
import concurrent.futures
import os

import psutil
from S3MP.global_config import S3MPConfig
from S3MP.mirror_path import MirrorPath
from tqdm import tqdm

from data_augmentation._version import __version__
from data_augmentation.augmentations import random_crop, random_zoom
from data_augmentation.coco_manager import CocoManager
from data_augmentation.constants import AUGMENTATION_SUFFIXES, ConfigKeys
from data_augmentation.types import segmentation
from data_augmentation.utils import (
    add_suffix_to_image_name,
    delete_mp,
    get_image_name_from_mp,
    multithread_download_mps_to_mirror,
)


class AugmentationManager:
    """Class for managing performing data augmentations on a COCO dataset."""

    def __init__(self, config: dict):
        """Initialize an AugmentationManager."""
        print("[INFO]: Initializing AugmentationManager...")
        self.config: dict = config

        self.overwrite: bool = (
            config[ConfigKeys.Data.OVERWRITE]
            if ConfigKeys.Data.OVERWRITE in config
            else False
        )

        # Set the source and destination paths
        self.src_root_mp = MirrorPath.from_s3_key(
            config[ConfigKeys.BaseKeys.DATA][ConfigKeys.Data.DATASET_S3_PATH]
        )
        self.images_dir_mp = self.src_root_mp.get_child(
            config[ConfigKeys.BaseKeys.DATA][ConfigKeys.Data.IMAGES_FOLDER_NAME]
        )
        self.labels_dir_mp = self.src_root_mp.get_child(
            config[ConfigKeys.BaseKeys.DATA][ConfigKeys.Data.ANNOTATIONS_FOLDER_NAME]
        )
        dest_root_mp = MirrorPath.from_s3_key(
            config[ConfigKeys.BaseKeys.DATA][ConfigKeys.Data.DESTINATION_S3_PATH]
        )
        self.dest_images_dir_mp = dest_root_mp.get_child(
            config[ConfigKeys.BaseKeys.DATA][ConfigKeys.Data.IMAGES_FOLDER_NAME]
        )
        self.dest_labels_dir_mp = dest_root_mp.get_child(
            config[ConfigKeys.BaseKeys.DATA][ConfigKeys.Data.ANNOTATIONS_FOLDER_NAME]
        )

        # Create the destination paths
        os.makedirs(self.dest_images_dir_mp.local_path, exist_ok=True)
        os.makedirs(self.dest_labels_dir_mp.local_path, exist_ok=True)

        # Get the coco json
        self.src_coco_json_mp = self.labels_dir_mp.get_child(
            config[ConfigKeys.BaseKeys.DATA][ConfigKeys.Data.ANNOTATIONS_FILE_NAME]
        )
        self.src_coco_manager = CocoManager(
            coco_json_mp=self.src_coco_json_mp,
            overwrite=self.overwrite,
        )
        self.dest_coco_json_mp = self.dest_labels_dir_mp.get_child(
            config[ConfigKeys.BaseKeys.DATA][ConfigKeys.Data.ANNOTATIONS_FILE_NAME]
        )
        # Delete the destination coco json if it exists to avoid collisions
        delete_mp(self.dest_coco_json_mp)
        self.dest_coco_manager = CocoManager(
            coco_json_mp=self.dest_coco_json_mp,
            default_height=config[ConfigKeys.BaseKeys.DATA][
                ConfigKeys.Data.DEFAULT_HEIGHT
            ],
            default_width=config[ConfigKeys.BaseKeys.DATA][
                ConfigKeys.Data.DEFAULT_WIDTH
            ],
        )

        # Use same categories as source coco json
        self.dest_coco_manager.set_categories(self.src_coco_manager.categories)

        if not self.src_root_mp.exists_on_s3():
            raise ValueError(
                f"Source root {self.src_root_mp.s3_key} does not exist on S3 in bucket {S3MPConfig.default_bucket_key}."
            )

    @property
    def n_augments_per_image(self) -> int:
        """Get the total number of augmentations to perform per image."""
        n_augments_per_image = 0
        for augment in self.config[ConfigKeys.BaseKeys.AUGMENTATIONS]:
            # All augmentations have a number
            n_augments_per_image += self.config[ConfigKeys.BaseKeys.AUGMENTATIONS][
                augment
            ][ConfigKeys.RandomZoom.NUMBER]
        return n_augments_per_image

    def run(self):
        """Run the data augmentations."""
        n_procs = psutil.cpu_count(logical=False)
        proc_executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_procs)
        all_proc_futures: list[concurrent.futures.Future] = []

        print(f"[INFO]: Running data augmentations version {__version__}")
        print(f"[INFO]: Source dataset: {self.images_dir_mp.s3_key}")
        print(f"[INFO]: Running on {n_procs} processes...")
        # Total number of images to augment = number of images * number of augmentations per image
        image_mps = self.images_dir_mp.get_children_on_s3()
        # Download all images to mirror
        multithread_download_mps_to_mirror(image_mps, overwrite=self.overwrite)
        pbar = tqdm(
            total=len(image_mps) * self.n_augments_per_image,
            desc="Augmentation Progress",
        )  # Init pbar
        for image_mp in image_mps:
            # Run the data augmentations
            augment_fn = None
            for augment in self.config[ConfigKeys.BaseKeys.AUGMENTATIONS]:
                match augment:
                    case ConfigKeys.Augmentations.RANDOM_CROP:
                        augment_fn = self._random_crop
                    case ConfigKeys.Augmentations.RANDOM_ZOOM:
                        augment_fn = self._random_zoom
                    case _:
                        raise ValueError(f"Augmentation {augment} not implemented.")
                pf = proc_executor.submit(
                    augment_fn,
                    image_mp,
                )
                all_proc_futures.append(pf)

        # Increment pbar as processes finish
        for _ in concurrent.futures.as_completed(all_proc_futures):
            pbar.update(n=1)

        all_proc_futures_no_except = [
            pf for pf in all_proc_futures if not pf.exception()
        ]
        all_proc_futures_except = [pf for pf in all_proc_futures if pf.exception()]
        # Get the results and add them to the coco manager
        print("[INFO]: Adding augmented images to coco dataset...")
        for pf in tqdm(all_proc_futures_no_except):
            # results will contain tuples of (output_mp, segmentations)
            output_mp, segs = pf.result()
            # Add the image to the coco manager
            _, image_id = self.dest_coco_manager.collision_avoid_add_image_from_mp(
                image_mp=output_mp,
            )
            # Add the segmentations to the coco manager
            for seg in segs:
                self.dest_coco_manager.add_annotation_from_segmentation(
                    seg=seg,
                    image_id=image_id,
                )
        for pf in all_proc_futures_except:
            raise pf.exception()

        proc_executor.shutdown(wait=True)

        print("[INFO]: Saving new coco dataset...")
        self.dest_coco_manager.save()
        print("[INFO]: Done!")

    def _augmentation_fn_wrapper(
        self,
        augmentation_key: str,
        image_mp: MirrorPath,
        augmentation_fn: callable,
        fn_kwargs: dict,
    ):
        """Wrap an augmentation function with general housekeeping tasks."""
        # Get the random crop config
        config: dict = self.config[ConfigKeys.BaseKeys.AUGMENTATIONS][augmentation_key]
        # All augmentations have a number
        number: int = config[ConfigKeys.RandomCrop.NUMBER]
        image_name = get_image_name_from_mp(image_mp)

        for i in range(number):
            suffix = f"{AUGMENTATION_SUFFIXES[augmentation_key]}_{i}"
            new_image_name = add_suffix_to_image_name(image_name, suffix)
            output_mp = self.dest_images_dir_mp.get_child(new_image_name)

            # Run random crop
            segs: list[segmentation] = augmentation_fn(
                image_mp=image_mp,
                annotations=self.src_coco_manager.annotations_by_image_name[image_name],
                output_mp=output_mp,
                **fn_kwargs,
            )

            return output_mp, segs

    def _random_crop(
        self, image_mp: MirrorPath
    ) -> tuple[MirrorPath, list[segmentation]]:
        """Run random crop."""
        config = self.config[ConfigKeys.BaseKeys.AUGMENTATIONS][
            ConfigKeys.Augmentations.RANDOM_CROP
        ]
        return self._augmentation_fn_wrapper(
            augmentation_key=ConfigKeys.Augmentations.RANDOM_CROP,
            image_mp=image_mp,
            augmentation_fn=random_crop,
            fn_kwargs={
                "crop_height": config[ConfigKeys.RandomCrop.HEIGHT],
                "crop_width": config[ConfigKeys.RandomCrop.WIDTH],
            },
        )

    def _random_zoom(
        self, image_mp: MirrorPath
    ) -> tuple[MirrorPath, list[segmentation]]:
        """Run random zoom."""
        config = self.config[ConfigKeys.BaseKeys.AUGMENTATIONS][
            ConfigKeys.Augmentations.RANDOM_ZOOM
        ]
        return self._augmentation_fn_wrapper(
            augmentation_key=ConfigKeys.Augmentations.RANDOM_ZOOM,
            image_mp=image_mp,
            augmentation_fn=random_zoom,
            fn_kwargs={
                "crop_height": config[ConfigKeys.RandomZoom.HEIGHT],
                "crop_width": config[ConfigKeys.RandomZoom.WIDTH],
            },
        )
