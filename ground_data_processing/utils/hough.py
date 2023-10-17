"""Numba-accelerated hough transform implementation."""

from typing import List, Tuple

import numba as nb
import numpy as np


@nb.njit(fastmath=True, cache=True)
def nb_get_rho_relations(img, rho_rez):
    """Get rho relations for an image."""
    dim_y = img.shape[0]
    dim_x = img.shape[1]
    # This is the maximum rho value, approximately
    max_rho = dim_x + dim_y
    n_rho = (2 * max_rho) + 1
    rho_scale = rho_rez / n_rho

    return max_rho, rho_scale


@nb.njit(fastmath=True, cache=True)
def nb_map_rho_to_rho_idx(rho, max_rho, rho_scale):
    """Map rho to it's index."""
    return int((rho + max_rho) * rho_scale)


@nb.njit(fastmath=True, cache=True)
def nb_map_rho_idx_to_rho(rho_idx, max_rho, rho_scale):
    """Map rho index to rho."""
    return (rho_idx / rho_scale) - max_rho


@nb.njit(parallel=True, fastmath=True, cache=True)
def nb_hough(
    img, theta_resolution, rho_resolution, min_theta_idx=0, max_theta_idx=np.inf
):
    """Potential for better parallelization."""
    dim_y = img.shape[0]
    dim_x = img.shape[1]
    # We take this as the maximum/minimum rho value.
    max_rho, rho_scale = nb_get_rho_relations(img, rho_resolution)
    theta_vals = np.linspace(0, np.pi, theta_resolution)
    accum_array = np.zeros((rho_resolution, theta_resolution))
    x_range = np.arange(dim_x)

    # Optional constraints for processing optimization.
    theta_min = int(max([min_theta_idx, 0]))
    theta_max = int(min([max_theta_idx, theta_resolution]))
    for theta_idx in nb.prange(theta_min, theta_max):
        theta_val = theta_vals[theta_idx]
        sin_val = np.sin(theta_val)
        x_cos_vals = x_range * np.cos(theta_val)
        for y_idx in nb.prange(dim_y):
            y_sin_val = y_idx * sin_val
            img_vals = img[y_idx, :]
            for x_idx in np.nonzero(img_vals)[0]:
                rho = x_cos_vals[x_idx] + y_sin_val
                rho_idx = nb_map_rho_to_rho_idx(rho, max_rho, rho_scale)
                accum_array[rho_idx][theta_idx] += img_vals[x_idx]

    return accum_array


@nb.njit(fastmath=True, cache=True)
def nb_map_y_to_image(y, dim_y):
    """Map y to image index, returns True if already within."""
    if y >= 0 and y < dim_y:
        return y, True
    return (0, False) if y < 0 else (dim_y - 1, False)


@nb.njit(fastmath=True, cache=True)
def nb_map_hough_line_to_image(dim_y, dim_x, sin_val, cos_val, rho):
    """Map hough value to boundary."""
    if sin_val == 0:
        y1 = 0
        y2 = dim_y - 1
        x1 = int(rho)
        x2 = int(rho)
    else:
        b = rho / sin_val
        slope = cos_val / sin_val
        x1 = 0
        x2 = dim_x - 1
        y1 = slope * x1 + b
        y2 = slope * x2 + b
    image_within = y1 < 0 and y2 >= dim_y or y1 >= dim_y and y2 < 0
    y1, y1_within = nb_map_y_to_image(y1, dim_y)
    y2, y2_within = nb_map_y_to_image(y2, dim_y)
    if image_within or y1_within or y2_within:
        if not y1_within:
            x1 = (y1 - b) / slope
        if not y2_within:
            x2 = (y2 - b) / slope
        return (x1, y1), (x2, y2)
    return None


@nb.njit(parallel=True, fastmath=True, cache=True)
def nb_hough_scaler(img, theta_resolution, rho_resolution):
    """Scale the hough array by line length through an image."""
    dim_y = img.shape[0]
    dim_x = img.shape[1]
    scale_array = np.zeros((rho_resolution, theta_resolution))
    theta_vals = np.linspace(0, np.pi, theta_resolution)
    rho_idxs = np.arange(rho_resolution)
    max_rho, rho_scale = nb_get_rho_relations(img, rho_resolution)
    rhos = nb_map_rho_idx_to_rho(rho_idxs, max_rho, rho_scale)
    for theta_idx in nb.prange(theta_resolution):
        theta = theta_vals[theta_idx]
        sin_val = np.sin(theta)
        cos_val = np.cos(theta)
        for rho_idx in nb.prange(rho_resolution):
            rho = rhos[rho_idx]
            mapped = nb_map_hough_line_to_image(dim_y, dim_x, sin_val, cos_val, rho)
            if mapped is not None:
                (x1, y1), (x2, y2) = mapped
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                scale_array[rho_idx][theta_idx] = length
    return scale_array


def get_image_bound_polar_lines(image_shape) -> List[Tuple[float, float]]:
    """Get the polar lines for the image bounds."""
    image_height, image_width = image_shape
    return [
        (0, np.pi / 2),  # top
        (image_height, np.pi / 2),  # bottom
        (0, 0),  # left
        (image_width, 0),  # right
    ]


def trim_point_within_bounds(point, image_shape):
    """Trim a point to be on image bounds."""
    image_height, image_width = image_shape
    point = [int(point[0]), int(point[1])]
    if not point:
        return None
    if point[0] < 0 or point[0] > image_width:
        return None
    return None if point[1] < 0 or point[1] > image_height else point


def polar_line_intersection(line1_rho, line1_theta, line2_rho, line2_theta):
    """Find the intersection of two lines in polar coordinates."""
    try:
        with np.errstate(divide="raise", invalid="raise"):
            a = np.array(
                [
                    [np.cos(line1_theta), np.sin(line1_theta)],
                    [np.cos(line2_theta), np.sin(line2_theta)],
                ]
            )
            b = np.array([line1_rho, line2_rho])
            x0, y0 = np.linalg.solve(a, b)
            return (x0, y0)
    except (ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
        return None


def convert_polar_line_to_row(line_rho, line_theta, image_shape):
    """Convert a line in polar coordinates to a row in image coordinates."""
    if len(image_shape) > 2:
        image_shape = image_shape[:2]
    image_bound_polar_lines = get_image_bound_polar_lines(image_shape)
    intersections = [
        polar_line_intersection(line_rho, line_theta, *bound_line)
        for bound_line in image_bound_polar_lines
    ]
    # return intersections
    intersections = [
        trim_point_within_bounds(intersection, image_shape)
        for intersection in intersections
    ]
    intersections = [intersection for intersection in intersections if intersection]
    return None if len(intersections) != 2 else intersections


def polar_line_indices_to_rows(polar_line_indices, theta_vals, rho_relations, rgb):
    """Convert polar line indices to rows."""
    polar_lines = [
        [nb_map_rho_idx_to_rho(rho_idx, *rho_relations), theta_vals[theta_idx]]
        for rho_idx, theta_idx in polar_line_indices
    ]
    rows = [
        convert_polar_line_to_row(*polar_line, image_shape=rgb.shape)
        for polar_line in polar_lines
    ]
    return [row for row in rows if row is not None]
