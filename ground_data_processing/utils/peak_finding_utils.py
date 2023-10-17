"""Peak finding utilities."""
import itertools
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks, peak_widths


def binary_search_peaks(data: ArrayLike, desired_n_peaks, desired_tol=4, e_tol=10):
    """Binary search peakfinding."""
    current_prominence = 1.0
    incr = current_prominence / 2
    while True:
        peaks, props = find_peaks(data, prominence=current_prominence)
        peak_diff = len(peaks) - desired_n_peaks
        if abs(peak_diff) < desired_tol:
            break
        elif peak_diff > 0:
            current_prominence += incr
        else:
            current_prominence -= incr
        incr /= 2
        if incr < 0.001:
            if abs(peak_diff) < e_tol:  # TODO one-way tolerances
                print(f"Found peaks with tolerance of {e_tol}")
                break
            print(f"Peak diff: {peak_diff}")
            raise ValueError("Could not find desired number of peaks.")

    return peaks


def binary_search_threshold_min_size(
    data: ArrayLike, max_min_size, desired_n_thresholds, max_val=0.1
) -> List[List[int]]:
    """Binary search until n thresholds are found."""
    bounds = []
    incr = max_min_size / 2
    min_size = max_min_size
    n_thresholds_found = []
    min_sizes = []
    while True:
        min_sizes.append(min_size)
        bounds = get_threshold_bounds(data, min_size, max_val)
        n_thresholds_found.append(len(bounds))
        if len(bounds) == desired_n_thresholds:
            break
        if len(bounds) > desired_n_thresholds:
            min_size += round(incr)
        else:
            min_size -= round(incr)
        incr /= 2
        incr = max(incr, 1)
        if min_size in min_sizes[:-1]:
            print("N thresholds found at each step:")
            print(n_thresholds_found)
            print(min_sizes)
            print(incr)
            print(bounds)
            print([bound[1] - bound[0] for bound in bounds])
            raise ValueError("Could not find desired number of thresholds.")
    return bounds


def get_threshold_bounds(data: ArrayLike, min_size=20, max_val=0.1) -> List[List[int]]:
    """Get the bounds of data that falls within the threshold."""
    bounds = []
    if np.all(data[:min_size] < max_val):
        current_bound_start = 0
        thresh_flag = True
    else:
        thresh_flag = False
    for idx, val in enumerate(data):
        if val < max_val:
            if not thresh_flag:
                thresh_flag = True
                current_bound_start = idx
        elif thresh_flag:  # Above threshold, with flag true
            if idx - current_bound_start > min_size:
                bounds.append([current_bound_start, idx])
            thresh_flag = False
    if thresh_flag:
        bounds.append([current_bound_start, len(data)])
    return bounds


def create_bound_around_peak(data, peak_idx, min_size=20, min_val=0.8) -> List[int]:
    """Find the bound around a peak."""
    bot_idx = peak_idx - min_size
    while bot_idx > 0 and data[bot_idx] > min_val:
        bot_idx -= 1
    top_idx = peak_idx + min_size
    while top_idx < len(data) and data[top_idx] > min_val:
        top_idx += 1
    return [bot_idx, top_idx]


def filter_to_evenly_spaced_peaks(data, peaks, props, desired_n_peaks, search_n=6):
    """Filter to n evenly spaced peaks."""
    leftmost_peak = peaks[0]
    rightmost_peak = peaks[-1]

    widths = peak_widths(data, peaks, rel_height=0.1)[0]

    prominences = props["prominences"]
    heights = props["peak_heights"]
    # prominence_sorted_idx = np.argsort(prominences)
    # prominences = prominences[prominence_sorted_idx]
    # peaks_by_prominence = peaks[prominence_sorted_idx]

    desired_search_indexes = np.linspace(leftmost_peak, rightmost_peak, desired_n_peaks)
    scaled_area = heights * widths * prominences
    filtered_peaks = []
    for desired_idx in desired_search_indexes:
        distances = np.abs(peaks - desired_idx)
        n_closest = np.argpartition(distances, search_n)[:search_n]
        # Scale on height. Peak width would be next step.
        biggest = np.argmax(scaled_area[n_closest])
        filtered_peaks.append(peaks[n_closest[biggest]])

    return filtered_peaks


def peak_based_threshold_bounds(
    data: ArrayLike,
    desired_n_bounds: int,
    inverse: bool = True,
    search_height: float = 0.95,
) -> List[List[int]]:
    """Get the bounds of data that falls within the threshold."""
    if inverse:
        data = 1 - data
    # data_len = len(data)
    peaks, props = find_peaks(
        data,
        height=search_height,
        prominence=0.5,
    )
    peaks = filter_to_evenly_spaced_peaks(data, peaks, props, desired_n_bounds + 1)
    return list(itertools.pairwise(peaks))


@dataclass
class FreqComponent:
    """Dataclass for a frequency component of FFT data."""

    freq: int
    phase: float
    amplitude: float
    data_len: int

    def get_plot_data(self, use_amplitude: bool = False) -> Tuple[ArrayLike, ArrayLike]:
        """Get the data for plotting."""
        x = np.linspace(0, self.data_len, self.data_len)
        amp = self.amplitude if use_amplitude else 1.0
        y = amp * np.sin(2 * np.pi * self.freq * x + self.phase)
        return x, y

    def get_phase_splits(self):
        """Get divisons of one phase of the wave."""
        _, freq_y = self.get_plot_data(False)
        all_crossings = np.where(np.diff(np.sign(freq_y)))[0]
        gradient_at_crossings = np.gradient(freq_y)[all_crossings]
        return all_crossings[gradient_at_crossings > 0]


def find_closest_dominant_fft_frequency(
    data: ArrayLike, desired_freq: int, n_search_freqs: int
) -> FreqComponent:
    """
    Find the closest dominant frequency in the FFT to a desired frequency.

    Only searches the first n_search_freqs frequencies in the FFT.
    Returns the frequency, it's phase offset, and it's amplitude.
    """
    fft_data = np.fft.fft(data)

    peak_freqs, peak_data = find_peaks(np.abs(fft_data), height=0.1)
    peak_freq_heights = peak_data["peak_heights"]
    sorted_indices = np.argsort(peak_freq_heights)[::-1]
    sorted_peak_freqs = peak_freqs[sorted_indices]
    sorted_fft_data = fft_data[sorted_peak_freqs]

    closest_freq_idx = np.argmin(
        np.abs(sorted_peak_freqs[:n_search_freqs] - desired_freq)
    )
    closest_freq = sorted_peak_freqs[closest_freq_idx]
    closest_freq_phase = np.angle(sorted_fft_data[closest_freq_idx])
    closest_freq_amplitude = np.abs(sorted_fft_data[closest_freq_idx])

    return FreqComponent(
        freq=closest_freq,
        phase=closest_freq_phase,
        amplitude=closest_freq_amplitude,
        data_len=len(data),
    )


def correct_splits(
    splits: ArrayLike, data_arr: ArrayLike, n_top_peaks: int = 30
) -> ArrayLike:
    """Correct splits by snapping them to the closest peak.

    Uses n_top_peaks to determine the number of peaks to search for when snapping.
    Generally setting this to about 1.5x the number of splits works well.
    """
    data_len = len(data_arr)
    peak_sep = data_len // n_top_peaks

    peaks, peak_data = find_peaks(data_arr, prominence=0.1, distance=peak_sep)
    prominences = peak_data["prominences"]

    sorted_indices = np.argsort(prominences)[::-1]
    top_peaks = peaks[sorted_indices]
    # top_peak_proms = prominences[sorted_indices]

    # Find the closest peak to each split
    top_peaks = list(top_peaks)
    corrected_splits = []
    for split in splits:
        closest_peak_idx = np.argmin(np.abs(top_peaks - split))
        corrected_splits.append(top_peaks.pop(closest_peak_idx))
        if not top_peaks:
            break
    return np.array(corrected_splits)


def get_n_best_split_bounds(
    data_arr: ArrayLike, splits: ArrayLike, n_splits: int = 22
) -> List[Tuple]:
    """Get the n best split boundaries from the splits.

    This takes the split boundaries and finds the n with the highest average value.
    """
    splits_bounded = np.concatenate([[0], splits, [len(data_arr)]])
    splits_bounded = np.sort(splits_bounded)
    split_bounds = list(itertools.pairwise(splits_bounded))
    avg_split_vals = [np.mean(data_arr[split[0] : split[1]]) for split in split_bounds]

    sorted_indices = np.argsort(avg_split_vals)[::-1]
    best_split_bounds = [split_bounds[idx] for idx in sorted_indices[:n_splits]]
    best_split_bounds.sort(key=lambda x: x[0])
    return best_split_bounds
