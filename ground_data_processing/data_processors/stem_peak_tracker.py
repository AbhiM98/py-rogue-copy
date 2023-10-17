"""Stem peaking tracking across frames. WIP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import matplotlib.colors as plt_colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from numpy.typing import ArrayLike


@dataclass
class PeakData:
    """Data for a single peak."""

    frame_index: int
    column_index: int
    prominence_val: float
    spread_val: float
    width: float

    def __repr__(self):
        """Return repr."""
        return f"PeakData(frame_index={self.frame_index}, column_index={self.column_index}, prominence_val={self.prominence_val}, spread_val={self.spread_val}, width={self.width})"

    def as_point(self):
        """Return as a point."""
        return np.array([self.frame_index, self.column_index], dtype=np.int32)


class NPPeakSet:
    """Group of peaks with set operations, but with an np array under the hood."""

    peak_data: ArrayLike

    def __init__(self, peaks: ArrayLike):
        """Initialize."""
        self.peak_data = peaks

    def __repr__(self):
        """Return repr."""
        return f"NPPeakSet(peaks={self.peak_data})"

    def __sub__(self, other: NPPeakSet) -> NPPeakSet:
        """Subtract two peak sets."""
        return NPPeakSet(np.setdiff1d(self.peak_data, other.peak_data, axis=0))

    def __add__(self, other: NPPeakSet) -> NPPeakSet:
        """Add two peak sets."""
        return NPPeakSet(np.concatenate((self.peak_data, other.peak_data), axis=0))

    def __or__(self, other: NPPeakSet) -> NPPeakSet:
        """Union two peak sets."""
        return NPPeakSet(
            np.unique(np.concatenate((self.peak_data, other.peak_data), axis=0), axis=0)
        )

    def __and__(self, other: NPPeakSet) -> NPPeakSet:
        """Intersect two peak sets."""
        # https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
        _, ncols = self.peak_data.shape
        dtype = {
            "names": [f"f{i}" for i in range(ncols)],
            "formats": ncols * [self.peak_data.dtype],
        }
        c = np.intersect1d(self.peak_data.view(dtype), other.peak_data.view(dtype))
        return NPPeakSet(c.view(self.peak_data.dtype).reshape(-1, ncols))

    def __eq__(self, other: NPPeakSet) -> bool:
        """Check if two peak sets are equal."""
        return np.array_equal(self.peak_data, other.peak_data)

    def __len__(self) -> int:
        """Return the number of peaks."""
        return np.shape(self.peak_data)[0]

    def intersects(self, other: NPPeakSet) -> bool:
        """Check if two peak sets intersect."""
        return np.shape((self & other).peak_data)[0] > 0

    def frame_idx_subset(self, min_frame_idx: int, max_frame_idx: int) -> NPPeakSet:
        """Return a subset of the peak set based on frame indices."""
        return NPPeakSet(
            self.peak_data[
                (min_frame_idx <= self.peak_data[:, 0])
                & (self.peak_data[:, 0] <= max_frame_idx)
            ]
        )

    def column_idx_subset(self, min_column_idx: int, max_column_idx: int) -> NPPeakSet:
        """Return a subset of the peak set based on column indexes."""
        return NPPeakSet(
            self.peak_data[
                (min_column_idx <= self.peak_data[:, 1])
                & (self.peak_data[:, 1] <= max_column_idx)
            ]
        )

    def scale_dimension_on_other(
        self, dim_to_scale: int, ref_dim: int, scale_factor: float = 1.0
    ) -> NPPeakSet:
        """Scale a dimension of the peak set based on another dimension."""
        # Normalize the scale dim
        dim_to_scale_max = np.max(self.peak_data[:, dim_to_scale])
        ref_max = np.max(self.peak_data[:, ref_dim])
        scale_factor *= dim_to_scale_max / ref_max
        # Scale the dimension
        new_peak_data = self.peak_data.copy()
        new_peak_data[:, dim_to_scale] = new_peak_data[:, dim_to_scale] * scale_factor
        return NPPeakSet(new_peak_data)

    def scale_dimension_to_range(
        self, dim_to_scale: int, max_val: float = 1.0
    ) -> NPPeakSet:
        """Scale a dimension of the peak set to a range."""
        # Normalize the scale dim
        dim_to_scale_max = np.max(self.peak_data[:, dim_to_scale])
        scale_factor = max_val / dim_to_scale_max
        # Scale the dimension
        new_peak_data = self.peak_data.copy()
        new_peak_data[:, dim_to_scale] = new_peak_data[:, dim_to_scale] * scale_factor
        return NPPeakSet(new_peak_data)

    def plot(self, color: str = "black", **kwargs):
        """Plot the peak set."""
        plt.plot(self.peak_data[:, 0], self.peak_data[:, 1], color=color, **kwargs)

    def trace_plot(self, **kwargs):
        """Plot the trace of the peak set."""
        self.plot(linestyle="-", marker="x", **kwargs)

    def scatter(self, color: str = "black", **kwargs):
        """Scatter plot the peak set."""
        plt.scatter(self.peak_data[:, 0], self.peak_data[:, 1], color=color, **kwargs)

    def scatter_with_colored_dimension(
        self, dim: int, log_color: bool = False, **kwargs
    ):
        """Scatter plot the peak set, with a dimension colored."""
        plt.scatter(
            self.peak_data[:, 0],
            self.peak_data[:, 1],
            c=self.peak_data[:, dim],
            **kwargs,
        )
        if log_color:
            pcm = plt.cm.ScalarMappable(
                norm=plt_colors.LogNorm(
                    vmin=np.min(self.peak_data[:, dim]),
                    vmax=np.max(self.peak_data[:, dim]),
                )
            )
            plt.colorbar(pcm, ax=plt.gca(), label=f"Dimension {dim}")
        else:
            plt.colorbar()


@dataclass
class KDDimSpec:
    """Specification for a kd-tree dimension."""

    dim_idx: int
    min_bound: float
    max_bound: float

    def flip(self) -> KDDimSpec:
        """Flip the dimension."""
        return KDDimSpec(self.dim_idx, -self.max_bound, -self.min_bound)


def kd_trace(
    tree: scipy.spatial.cKDTree,
    tree_data: ArrayLike,
    start_point: ArrayLike,
    dim_specs: List[KDDimSpec],
    search_k: int = 50,
):
    """Trace through a kd-tree."""
    max_dist = max(abs(d_s.max_bound) for d_s in dim_specs)

    current_point = start_point
    homies = []

    data_clip = np.array([d_s.dim_idx for d_s in dim_specs])
    while True:
        distances, neighbor_indices = tree.query(
            current_point[data_clip], k=search_k, distance_upper_bound=max_dist
        )
        neighbor_indices = neighbor_indices[neighbor_indices != tree.n]
        neighbors = tree_data[neighbor_indices]
        if len(neighbors) <= 1:
            break
        for d_s in dim_specs:
            neighbors = neighbors[
                (
                    neighbors[:, d_s.dim_idx]
                    >= d_s.min_bound + current_point[d_s.dim_idx]
                )
                & (
                    neighbors[:, d_s.dim_idx]
                    <= d_s.max_bound + current_point[d_s.dim_idx]
                )
            ]
        # Check if constraints removed self, if not, remove self
        if len(neighbors) != 0 and np.all(
            neighbors[0] == current_point
        ):  # TODO verify this works
            neighbors = neighbors[1:]
        if len(neighbors) == 0:
            break
        current_point = neighbors[0]
        homies.append(current_point)
    return homies


# def get_ideal_theta_hough_accum(
#     peak_set: NPPeakSet,
#     search_rez: int = 360,
#     final_rez: int = 2500,
#     straddle_width: int = 50,
# ) -> Tuple[ArrayLike, int]:
#     """Get hough line segments from a peak set."""
#     max_x = np.max(peak_set.peak_data[:, 0])
#     max_y = np.max(peak_set.peak_data[:, 1])
#     img = np.zeros((max_x + 1, max_y + 1), dtype=np.uint8)
#     point_kernel = np.ones((3, 3), dtype=np.uint8)
#     for point in peak_set.peak_data[:, :2]:
#         point = point.astype(int)
#         with contextlib.suppress(Exception):
#             img[point[1] - 1 : point[1] + 2, point[0] - 1 : point[0] + 2] = point_kernel

#     search_accum = nb_hough(img, search_rez, search_rez)
#     ideal_theta_idx = np.argmax(np.max(search_accum, axis=0))
#     ideal_theta_idx = ideal_theta_idx * (final_rez / search_rez)
#     accum = nb_hough(
#         img,
#         final_rez,
#         final_rez,
#         min_theta_idx=ideal_theta_idx - straddle_width,
#         max_theta_idx=ideal_theta_idx + straddle_width,
#     )
#     ideal_theta_idx = int(ideal_theta_idx)

#     return accum, ideal_theta_idx
