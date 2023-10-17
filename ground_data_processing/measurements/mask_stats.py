"""Segmentation mask statistics."""
from typing import Dict, List

import numpy as np
import pycocotools.mask
from numpy.typing import ArrayLike

# Clarity
RawMask = Dict


def get_masks_by_image_id(segm_json_data: List[RawMask]) -> Dict[str, List[RawMask]]:
    """Get masks by image id."""
    masks_by_image_id = {}
    for mask in segm_json_data:
        image_id = mask["image_id"]
        if image_id not in masks_by_image_id:
            masks_by_image_id[image_id] = []
        masks_by_image_id[image_id].append(mask)
    return masks_by_image_id


def get_mask_areas(masks: List[RawMask]) -> List[float]:
    """Get mask areas."""
    mask_areas = []
    for mask in masks:
        mask_area = pycocotools.mask.area(mask["segmentation"])
        mask_areas.append(mask_area)
    return mask_areas


def get_mask_areas_by_image_id(segm_json_data: List[RawMask]) -> Dict[str, List[float]]:
    """Get mask areas by image id."""
    masks_by_image_id = get_masks_by_image_id(segm_json_data)
    mask_areas_by_image_id = {}
    for image_id, masks in masks_by_image_id.items():
        mask_areas = get_mask_areas(masks)
        mask_areas_by_image_id[image_id] = mask_areas
    return mask_areas_by_image_id


def get_top_n_percent(data: ArrayLike, n: float) -> ArrayLike:
    """Get top n percent of data."""
    data = np.array(data)
    return data[np.argsort(data)[-int(len(data) * n) :]]
