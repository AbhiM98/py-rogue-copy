"""Functions for working with line segments."""
from typing import List

import cv2
import numpy as np
from numpy.typing import ArrayLike
from skimage.draw import line as line_iter

# TODO use py-plant-stats for some of this
# although pps uses python 3.7 which is annoying


def get_hough_p_line_segments_with_theta_tol(
    img: ArrayLike, desired_theta: float, theta_tol: float
) -> List[
    List[int]
]:  # For these type hints, they might be nested one deeper than specified
    """Get line segments from HoughP transform."""
    lines = cv2.HoughLinesP(
        img, rho=1, theta=np.pi / 180, threshold=100, minLineLength=25, maxLineGap=50
    )
    ret_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        theta = np.arctan2(y2 - y1, x2 - x1)
        if desired_theta - np.abs(theta) < theta_tol:
            ret_lines.append(line)
    return ret_lines


def get_hough_lines(
    img: ArrayLike, desired_theta: float, theta_tol: float
) -> List[List[float]]:
    """Get line segment from Hough transform."""
    lines = cv2.HoughLines(img, rho=1, theta=np.pi / 180, threshold=150)
    ret_lines = []
    for line in lines:
        __, theta = line[0]
        if np.abs(desired_theta - theta) < theta_tol:
            ret_lines.append(line)
    return ret_lines


def draw_hough_lines(
    img: ArrayLike, lines: ArrayLike, color: tuple = (0, 0, 255), thickness: int = 2
):
    """Draw hough lines on image. Uses polar coordinates."""
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 5000 * (-b))
        y1 = int(y0 + 5000 * (a))
        x2 = int(x0 - 5000 * (-b))
        y2 = int(y0 - 5000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def filter_lines_to_img_width_center(
    lines: ArrayLike, img: ArrayLike, center_tol: float
):
    """Filter lines to those center_tol% of image width from center."""
    ret_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_avg = (x1 + x2) / 2
        img_x_center = img.shape[1] / 2
        if np.abs(x_avg - img_x_center) < img.shape[1] * center_tol:
            ret_lines.append(line)
    return ret_lines


def get_horizontal_overlap_bounds(line_a: ArrayLike, line_b: ArrayLike):
    """Get the bounds of line_a -> line_b along each row of pixels."""
    bounds = {}
    l1_x1, l1_y1, l1_x2, l1_y2 = line_a[0]
    l2_x1, l2_y1, l2_x2, l2_y2 = line_b[0]

    l1_min_y = min(l1_y1, l1_y2)
    l1_max_y = max(l1_y1, l1_y2)
    l2_min_y = min(l2_y1, l2_y2)
    l2_max_y = max(l2_y1, l2_y2)
    if l1_min_y > l2_max_y or l2_min_y > l1_max_y:
        return bounds
    overlapping_y_vals = np.intersect1d(
        np.arange(l1_min_y, l1_max_y), np.arange(l2_min_y, l2_max_y)
    )

    line_a_iter = line_iter(l1_y1, l1_x1, l1_y2, l1_x2)
    line_b_iter = line_iter(l2_y1, l2_x1, l2_y2, l2_x2)
    line_a_points = np.array(list(zip(*line_a_iter)))
    line_b_points = np.array(list(zip(*line_b_iter)))
    for y_val in overlapping_y_vals:
        line_a_x_vals = line_a_points[line_a_points[:, 0] == y_val][:, 1]
        line_b_x_vals = line_b_points[line_b_points[:, 0] == y_val][:, 1]
        bounds[y_val] = (np.min(line_a_x_vals), np.max(line_b_x_vals))
    return bounds


# TODO these are unused and are untested
def get_min_line_segment_between_line_segments(line_a: ArrayLike, line_b: ArrayLike):
    """Get the minimum line segment between two line segments."""
    x1, y1, x2, y2 = line_a[0]
    x3, y3, x4, y4 = line_b[0]
    endpoint_lines = [
        np.array([x1, y1, x3, y3]),
        np.array([x1, y1, x4, y4]),
        np.array([x2, y2, x3, y3]),
        np.array([x2, y2, x4, y4]),
    ]
    lengths = [
        np.linalg.norm(np.array([_x1, _y1]) - np.array([_x2, _y2]))
        for _x1, _y1, _x2, _y2 in endpoint_lines
    ]
    return endpoint_lines[np.argmin(lengths)]


def get_min_dist_between_line_segments(line_a: ArrayLike, line_b: ArrayLike):
    """Get the minimum distance between two line segments."""
    x1, y1, x2, y2 = line_a[0]
    x3, y3, x4, y4 = line_b[0]

    return min(
        np.linalg.norm(np.array([x1, y1]) - np.array([x3, y3])),
        np.linalg.norm(np.array([x1, y1]) - np.array([x4, y4])),
        np.linalg.norm(np.array([x2, y2]) - np.array([x3, y3])),
        np.linalg.norm(np.array([x2, y2]) - np.array([x4, y4])),
    )
