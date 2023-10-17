"""Image utilities."""
import functools
import os

import cv2
import imagehash
import numpy as np
from numpy.typing import ArrayLike
from PIL import Image
from skimage import color


class ColorRanges:
    """Color ranges."""

    CV2_RGB = (0, 255)
    CV2_HUE = (0, 179)
    CV2_SAT = (0, 255)
    CV2_VAL = (0, 255)
    PDN_HUE = (0, 360)
    PDN_SAT = (0, 100)
    PDN_VAL = (0, 100)


def convert_int_range(conv_range, in_range=(0, 255), out_range=(0, 360)):
    """Convert range of integers."""
    conv_factor = (out_range[1] - out_range[0]) / (in_range[1] - in_range[0])
    scaled_range = (np.array(conv_range) - in_range[0]) * conv_factor + out_range[0]
    return (round(scaled_range[0]), round(scaled_range[1]))


def move_color_channel_to_last(img, expected_channel_size=3):
    """Move color channel to last dimension."""
    current_channel_idx = img.shape.index(expected_channel_size)
    if current_channel_idx != -1:
        img = np.moveaxis(img, current_channel_idx, -1)
    return img


# Shape = (H, W, C)
# HSV filters
def hsv_hue_range(img, hue_range=(25, 102)):
    """Excess green based on HSV space.

    Default hue range is (25, 102), which is the range of green.
    """
    img = move_color_channel_to_last(img)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.where(
        (hsv_img[:, :, 0] >= hue_range[0]) & (hsv_img[:, :, 0] <= hue_range[1]), 255, 0
    ).astype(np.uint8)


def hsv_sat_range(img, sat_range=(50, 80)):
    """Excess green based on HSV space.

    Default sat range is (50, 80)
    """
    img = move_color_channel_to_last(img)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.where(
        (hsv_img[:, :, 1] >= sat_range[0]) & (hsv_img[:, :, 1] <= sat_range[1]), 255, 0
    ).astype(np.uint8)


# RGB filters
def excess_green(img):
    """Excess green, G > B and G > R."""
    img = move_color_channel_to_last(img)
    return np.where(
        (img[:, :, 1] > img[:, :, 0]) & (img[:, :, 1] > img[:, :, 2]), 255, 0
    ).astype(np.uint8)


def harsh_exg(img):
    """G > B + R."""
    img = move_color_channel_to_last(img)
    img = img.astype(np.float32)
    return np.where((img[:, :, 1] > img[:, :, 0] + img[:, :, 2]), 255, 0).astype(
        np.uint8
    )


def excess_luminance(img):
    """Excess luminance, L > 50."""
    img = move_color_channel_to_last(img)
    img = color.rgb2lab(img)
    return np.where((img[:, :, 0] > 50), 255, 0).astype(np.uint8)


def build_depth_map(img, img2):
    """
    Use cv2 to build a stereo depth map of two images.

    Args:
    img: np.ndarray, shape (H, W, C)
    img2: np.ndarray, shape (H, W, C)

    Returns:
    depth_map: np.ndarray with shape (H, W)
    """
    # convert to grayscale

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # show images
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.imshow("img", img2)
    # cv2.waitKey(0)

    # downsample images
    downsample = 0.2
    img = cv2.resize(img, (0, 0), fx=downsample, fy=downsample)
    img2 = cv2.resize(img2, (0, 0), fx=downsample, fy=downsample)

    # stereo depth map
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=25)
    disparity2 = stereo.compute(img, img2)  # plants move right to left
    # check if we need to swap images
    if np.sum(disparity2) < 0:
        disparity2 = stereo.compute(img2, img)  # plants move left to right
    disparity2 = cv2.normalize(
        disparity2,
        disparity2,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    # rescale to 0 - 1 - 0
    # disparity2 = np.where(disparity2 > 0.25, 1 - disparity2, disparity2)
    # disparity2 = disparity2 * 2

    # cut out values above 2*255/3 and below 255/3
    # MIN_THRESH = 80
    # MAX_THRESH = 256
    # disparity2 = np.where((disparity2 > MIN_THRESH) & (disparity2 < MAX_THRESH), disparity2, 0)
    disparity2 = cv2.resize(disparity2, (0, 0), fx=1 / downsample, fy=1 / downsample)

    # plt.imshow(cv2.normalize(disparity2, disparity2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), 'gray')

    return disparity2


def harsh_exg_col_sum(img, img2=None):
    """
    Get the harsh excess green filter, then sum across rows and columns.

    If the second image is passed a depth estimation will be attempted.

    Args:
    img: np.ndarray, shape (H, W, C)
    img2 (optional): np.ndarray, shape (H, W, C)

    Returns:
    row_sum: np.ndarray with shape (H, 1)
    np.ndarray with shape (W, 1)
    """
    depth_map = None
    if img2 is not None:
        # build depth map
        excess_green_mask = excess_green(img)
        excess_green_mask2 = excess_green(img2)
        # pass masked images to depth map

        depth_map = build_depth_map(
            cv2.bitwise_and(img, img, mask=excess_green_mask),
            cv2.bitwise_and(img2, img2, mask=excess_green_mask2)
            # img, img2
        )

    # cv2.imshow("img", cv2.bitwise_and(img, img, mask=depth_map))
    # cv2.waitKey(0)

    img = excess_green(img)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # img = cv2.distanceTransform(img, cv2.DIST_L2, 3)

    # cut out small areas
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            cv2.drawContours(img, [contour], -1, 0, -1)

    # cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # cv2.threshold(img, 0.15, 1, cv2.THRESH_BINARY, img)

    center_x = img.shape[0] // 2
    # apply depth map as weights
    # img = img * depth_map
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # depth_map_viz = cv2.normalize(depth_map, depth_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imshow("img", depth_map_viz)
    # cv2.waitKey(0)
    if depth_map is not None:
        return np.sum(
            np.multiply(
                img[center_x - 100 : center_x + 100, :],
                depth_map[center_x - 100 : center_x + 100],
            ),
            axis=1,
        ).reshape(-1, 1), np.sum(img, axis=0).reshape(-1, 1)
    return np.sum(img[center_x - 100 : center_x + 100, :], axis=1).reshape(
        -1, 1
    ), np.sum(img, axis=0).reshape(-1, 1)


def min_val_exg(img, min_val=127):
    """G > B and G > R and G > min_val."""
    img = move_color_channel_to_last(img)
    return np.where(
        (img[:, :, 1] > img[:, :, 0])
        & (img[:, :, 1] > img[:, :, 2])
        & (img[:, :, 1] > min_val),
        255,
        0,
    ).astype(np.uint8)


def rel_prop_exg(img, prop=0.7):
    """G > (B + R) * prop."""
    img = move_color_channel_to_last(img)
    img = img.astype(np.float32)
    return np.where(
        (img[:, :, 1] > (img[:, :, 0] + img[:, :, 2]) * prop), 255, 0
    ).astype(np.uint8)


def slice_center_segment(img, width, midpoint=None):
    """Slice center segment of image."""
    if midpoint is None:
        midpoint = img.shape[1] // 2
    return img[:, midpoint - width : midpoint + width, :]


def slice_and_flatten_center_segment(img, width, midpoint=None):
    """Slice and flatten center segment of image."""
    return slice_center_segment(img, width, midpoint).reshape(1, -1)


def prop_nonzero(arr):
    """Proportion of nonzero pixels."""
    return np.count_nonzero(arr) / arr.shape[0]


# Green weighting logic
@functools.lru_cache(maxsize=None)
def inv_distance_to_center(shape):
    """Inverse distance to center."""
    h, w = shape
    y, x = np.mgrid[:h, :w]
    dist_to_center = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)
    dist_to_center[dist_to_center == 0] = 1
    return 1 / dist_to_center


def exp_weight_with_shift(img, shift=2):
    """Exponential weight with shift."""
    return np.exp(img + 1)


def weight_mask_against_dist_to_center(img):
    """Weight mask against distance to center."""
    img = img.astype(np.float32)
    return img * inv_distance_to_center(img.shape)


def img_resize(in_path, resized_path, width, overwrite: bool):
    """Resize an image."""
    img = cv2.imread(in_path)
    # Check if image already exists
    if os.path.exists(resized_path) and not overwrite:
        return
    img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    cv2.imwrite(resized_path, img)


def get_img_phash(img: ArrayLike, hash_size: int = 64):
    """Get hash of image."""
    return imagehash.phash(Image.fromarray(img), hash_size=hash_size)


def get_img_phash_flat_arr(img: ArrayLike, hash_size: int = 64):
    """Get hash of image as array."""
    return np.array(get_img_phash(img, hash_size).hash).reshape(1, -1)


# TODO use dill or smthn idk
def get_img_phash_flat_64(img: ArrayLike):
    """Get hash of image as array."""
    return get_img_phash_flat_arr(img, 64)


def get_img_phash_flat_8(img: ArrayLike):
    """Get hash of image as array."""
    return get_img_phash_flat_arr(img, 8)


def crop_square_from_img_center(img, resize_size):
    """Crops a square from the center of an image."""
    height, width = img.shape[:2]
    if width == height:
        return cv2.resize(img, (resize_size, resize_size))
    elif width > height:
        left = (width - height) / 2
        right = left + height
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top = (height - width) / 2
        bottom = top + width
    img = img[int(top) : int(bottom), int(left) : int(right)]
    return cv2.resize(img, (resize_size, resize_size))
