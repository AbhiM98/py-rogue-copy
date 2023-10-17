"""Core augmentation logic."""
import copy

import cv2
import numpy as np
from S3MP.mirror_path import MirrorPath

from data_augmentation.bbox import BBox
from data_augmentation.types import segmentation


def random_crop(
    image_mp: MirrorPath,
    annotations: list[dict],
    output_mp: MirrorPath,
    crop_height: int,
    crop_width: int,
) -> list[segmentation]:
    """Randomly crop an image and its annotations, and return a list of new segmentations."""
    # Load the image
    image = cv2.imread(str(image_mp.local_path))
    # Get the original height and width
    height, width = image.shape[:2]

    # Get the crop coordinates
    x = np.random.randint(0, width - crop_width)
    y = np.random.randint(0, height - crop_height)
    crop_bbox = BBox(x, y, crop_width, crop_height)
    # Crop the image
    cropped_image = image[y : y + crop_height, x : x + crop_width]
    # Pad image back to original size
    padded_image = np.zeros((height, width, 3), dtype=np.uint8)
    padded_image[y : y + crop_height, x : x + crop_width] = cropped_image
    # Save the image
    cv2.imwrite(str(output_mp.local_path), padded_image)
    # New crop should always overwrite any old crops to stay consistent with the labels
    output_mp.upload_from_mirror(overwrite=True)

    # Get the segmentations that are in the crop
    return get_segmentations_in_crop(annotations, crop_bbox)


def random_zoom(
    image_mp: MirrorPath,
    annotations: list[dict],
    output_mp: MirrorPath,
    crop_height: int,
    crop_width: int,
) -> list[segmentation]:
    """Randomly zoom an image and its annotations, and return a list of new segmentations."""
    # Load the image
    image = cv2.imread(str(image_mp.local_path))
    # Get the original height and width
    height, width = image.shape[:2]

    # Get the zoom coordinates
    x = np.random.randint(0, width - crop_width)
    y = np.random.randint(0, height - crop_height)
    crop_bbox = BBox(x, y, crop_width, crop_height)
    # Crop the image
    cropped_image = image[y : y + crop_height, x : x + crop_width]
    # Resize/Zoom the image
    resized_image = cv2.resize(cropped_image, (width, height))
    # Save the image
    cv2.imwrite(str(output_mp.local_path), resized_image)
    # New crop should always overwrite any old crops to stay consistent with the labels
    output_mp.upload_from_mirror(overwrite=True)

    # Get the segmentations that are in the crop
    segmentations = get_segmentations_in_crop(annotations, crop_bbox)
    # Resize the segmentations
    for seg in segmentations:
        for point in seg:
            point[0] = point[0] - x  # change of origin
            point[1] = point[1] - y  # change of origin
            point[0] = int(point[0] * width / crop_width)  # change of scale
            point[1] = int(point[1] * height / crop_height)  # change of scale

    return segmentations


def get_segmentations_in_crop(
    annotations: list[dict], crop_bbox: BBox
) -> list[segmentation]:
    """Get the segmentations that are completely or partially in the crop."""
    # Get the annotations for this crop
    cropped_segmentations: list[segmentation] = []
    for anno in annotations:
        # Get the segmentation
        seg = copy.deepcopy(anno["segmentation"])
        # Get the bbox
        anno_bbox = BBox.from_list(anno["bbox"])

        # Check if the bbox is completely in the crop
        if crop_bbox.contains_bbox(anno_bbox):
            # Add the segmentation to the list
            cropped_segmentations.append(seg)

        # Check if part of the bbox is in the crop
        elif crop_bbox.partially_contains_bbox(anno_bbox):
            # Get the new segmentation
            new_segmentation: segmentation = []
            for point in seg:
                # coordinates need be within the crop, ie greater than x,y and less than x_max, y_max
                # first catch coords below the minimumns (x,y)
                new_point = [
                    point[0] if point[0] >= crop_bbox.x else crop_bbox.x,
                    point[1] if point[1] >= crop_bbox.y else crop_bbox.y,
                ]
                # then catch coords above the maximums (x_max, y_max)
                new_point[0] = (
                    new_point[0]
                    if new_point[0] < crop_bbox.x_max
                    else crop_bbox.x_max - 1
                )
                new_point[1] = (
                    new_point[1]
                    if new_point[1] < crop_bbox.y_max
                    else crop_bbox.y_max - 1
                )
                # add the new point to the new segmentation
                new_segmentation.append(new_point)

            cropped_segmentations.append(new_segmentation)

    return cropped_segmentations
