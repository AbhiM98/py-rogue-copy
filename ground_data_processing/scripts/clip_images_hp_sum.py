"""Clip images.

Clip according to a processing technique defined in
CORN PLANT LOCATION, SPACING AND STALK DIAMETER MEASUREMENTS USING OPTICAL SENSING TECHNOLOGIES By YEYIN SHI
The (modified) process is as follows:
- Exg Segmentation
- Filter small contours on a threshold
- Close small holes on a threshold
- Sum along the columns of the image
- Peak find

A meta-analysis across frames would be cool.
"""

import matplotlib.pyplot as plt
import numpy as np
from S3MP.keys import get_matching_s3_keys
from S3MP.mirror_path import MirrorPath
from scipy.signal import find_peaks, peak_widths

from ground_data_processing.utils.absolute_segment_groups import ProductionFieldSegments
from ground_data_processing.utils.image_utils import excess_green, harsh_exg
from ground_data_processing.utils.s3_constants import (
    CameraViews,
    DataFiles,
    DataTypes,
    Fields,
)
from ground_data_processing.utils.video_utils import get_ffmpeg_reader_trimmed


def hp_sum_frame_proc(frame):
    """Frame process."""
    exg = excess_green(frame)
    # TODO actual preprocessing
    column_sums = np.sum(exg.astype(np.float32), axis=0)
    col_max = np.max(column_sums)
    column_sums /= col_max

    mean = np.mean(column_sums)
    # min_height = 0.6
    # peaks = binary_search_peaks(column_sums, 8, 4, 10)
    peaks, proms = find_peaks(column_sums, prominence=0.2)
    print()
    print(mean)
    # print(peaks)
    # print(proms)
    plt.plot(column_sums)
    plt.plot(np.ones_like(column_sums) * mean, "--", color="black")
    plt.scatter(peaks, column_sums[peaks], marker="x")
    plt.show()
    return peaks


def spread_col_sum_frame_proc(frame, min_col_sum=255 * 10):
    """Get a score of each column based on the spread of the column sum."""
    exg = harsh_exg(frame)

    col_spread_vals = np.zeros(exg.shape[1])
    col_sums = np.sum(exg.astype(np.float32), axis=0)
    valid_col_idxs = np.where(col_sums > min_col_sum)[0]

    for col_idx in valid_col_idxs:
        col = exg[:, col_idx]
        col_sum = col_sums[col_idx]
        col_idxs = np.nonzero(col)
        dists = np.diff(col_idxs)
        col_spread_vals[col_idx] = col_sum / np.mean(dists)

    norm_col_spread_vals = col_spread_vals / np.max(col_spread_vals)
    peaks, props = find_peaks(norm_col_spread_vals, prominence=0.01)
    widths_res = peak_widths(norm_col_spread_vals, peaks, rel_height=0.9)
    return peaks, props["prominences"], col_spread_vals[peaks], widths_res[0]


def collect_all_spread_col_data(frame, min_col_sum=255 * 10):
    """Collect all relevant spread column data.

    Data:
        - Raw column sums
        - Locally normalized column sums
        - Raw column spread vals
        - Locally normalized column spread vals
    """
    exg = excess_green(frame)

    col_spread_vals = np.zeros(exg.shape[1])
    col_sums = np.sum(exg.astype(np.float32), axis=0)
    norm_col_sums = col_sums / np.max(col_sums)
    valid_col_idxs = np.where(col_sums > min_col_sum)[0]

    for col_idx in valid_col_idxs:
        col = exg[:, col_idx]
        col_sum = col_sums[col_idx]
        col_idxs = np.nonzero(col)
        dists = np.diff(col_idxs)
        col_spread_vals[col_idx] = col_sum / np.mean(dists)

    norm_col_spread_vals = col_spread_vals / np.max(col_spread_vals)
    return np.array([col_sums, norm_col_sums, col_spread_vals, norm_col_spread_vals])


def apply_peakfinding_to_data(data, min_prominence=0.2):
    """Apply peak finding to data."""
    peaks, props = find_peaks(data, prominence=min_prominence)
    widths_res = peak_widths(data, peaks, rel_height=0.9)
    return peaks, props["prominences"], widths_res[0]


def visualize_peakfinding_data(ax, data, title, min_prom=0.2):
    """Visualize peak finding data."""
    peaks, _, _ = apply_peakfinding_to_data(data, min_prom)
    ax.plot(data)
    ax.scatter(peaks, data[peaks], marker="x")
    ax.set_title(title)


def visualize_all_peak_data_per_frame(frame):
    """Visualize all peak data per frame."""
    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    (
        col_sums,
        norm_col_sums,
        col_spread_vals,
        norm_col_spread_vals,
    ) = collect_all_spread_col_data(frame)
    visualize_peakfinding_data(ax[0, 0], col_sums, "Column Sums")
    visualize_peakfinding_data(ax[0, 1], norm_col_sums, "Normalized Column Sums")
    visualize_peakfinding_data(ax[0, 2], col_spread_vals, "Column Spread Values")
    visualize_peakfinding_data(
        ax[1, 0], norm_col_spread_vals, "Normalized Column Spread Values"
    )
    ax[1, 1].imshow(frame)
    ax[1, 2].imshow(excess_green(frame))
    plt.show()


if __name__ == "__main__":
    CAMERA_VIEW = CameraViews.BOTTOM
    OVERWRITE = True

    segments = [
        ProductionFieldSegments.FIELD_DESIGNATION(Fields.PROD_FIELD_ONE),
        ProductionFieldSegments.DATA_TYPE(DataTypes.VIDEOS),
        ProductionFieldSegments.DATE("6-27"),
        ProductionFieldSegments.ROW_DESIGNATION("Row 1"),
        ProductionFieldSegments.OG_VID_FILES(incomplete_name=".mp4"),
    ]

    matching_keys = get_matching_s3_keys(segments)
    bottom_vids = [key for key in matching_keys if CAMERA_VIEW in key]
    bottom_vid_mp = MirrorPath.from_s3_key(bottom_vids[0])

    bottom_vid_mp.download_to_mirror()
    # TODO make this a util
    npy_mp = bottom_vid_mp.get_sibling(DataFiles.EXG_SLC_20PX_NPY)
    npy_data = npy_mp.load_local(download=True)
    data_mean = np.mean(npy_data)
    peaks, _ = find_peaks(npy_data, height=data_mean)
    bot_vid_start = peaks[0]
    vid_iter = get_ffmpeg_reader_trimmed(
        bottom_vid_mp.local_path,
        start_frame=bot_vid_start,
        end_frame=len(npy_data),
        use_tqdm=True,
    )
    peaks_by_frame = []

    fig, axs = plt.subplots(2, 3)
    col_sum_max = 255 * 2160
    prom_cutoff = 0.4
    relative_prom = prom_cutoff * col_sum_max
    plt.show(block=False)
    for idx, frame in enumerate(vid_iter):
        spread_col_vals = collect_all_spread_col_data(frame)
        spread_col_vals = np.array(spread_col_vals)

        # axs[0, 0].plot(raw_col_sum)
        # peaks, props, widths = apply_peakfinding_to_data(raw_col_sum, min_prominence=relative_prom)
        # axs[0, 0].scatter(peaks, raw_col_sum[peaks], marker="x")
        # axs[0, 0].set_title("Raw Col Sum")
        # axs[0, 1].plot(norm_col_sum)
        # peaks, props, widths = apply_peakfinding_to_data(norm_col_sum, prom_cutoff)
        # axs[0, 1].scatter(peaks, norm_col_sum[peaks], marker="x")
        # axs[0, 1].set_title("Norm Col Sum")
        # axs[0, 2].plot(raw_col_spread)
        # peaks, props, widths = apply_peakfinding_to_data(raw_col_spread, min_prominence=relative_prom)
        # axs[0, 2].scatter(peaks, raw_col_spread[peaks], marker="x")
        # axs[0, 2].set_title("Raw Col Spread")
        # axs[1, 0].plot(norm_col_spread)
        # peaks, props, widths = apply_peakfinding_to_data(norm_col_spread, prom_cutoff)
        # axs[1, 0].scatter(peaks, norm_col_spread[peaks], marker="x")
        # axs[1, 0].set_title("Norm Col Spread")
        # axs[1, 1].imshow(frame)
        # axs[1, 2].imshow(excess_green(frame))
        # # wait for button press
        # plt.draw()
        # plt.waitforbuttonpress()
        # for ax in axs.flatten():
        #     ax.clear()

        # peaks = frame_proc(frame)
        # peaks = spread_col_sum_frame_proc(frame)
        # peaks = hp_sum_frame_proc(frame)
        # peaks_by_frame.append(peaks)
        # anno = frame.copy()
        # for peak in peaks:
        #     anno[:, peak - 5 : peak + 5] = (0, 0, 255)
        # # print()
        # # print(len(peaks))
        # img_mp = root_dir.get_child(f"{idx}.png")
        # anno = cv2.cvtColor(anno, cv2.COLOR_BGR2RGB)
        # img_mp.save_local(anno, upload=False)
        if idx > 500:
            break

    # # print(len(vid_iter))
    # root_dir = bottom_vid_mp.get_sibling("bot clip test")
    # root_dir.local_path.mkdir(exist_ok=True)

    # line_width = 5
    # peaks_by_frame = []

    # for idx, frame in enumerate(vid_iter):
    #     # peaks = frame_proc(frame)
    #     peaks = spread_col_sum_frame_proc(frame)
    #     peaks_by_frame.append(peaks)
    #     # anno = frame.copy()
    #     # for peak in peaks:
    #     #     anno[:, peak - 5 : peak + 5] = (0, 0, 255)
    #     # # print()
    #     # # print(len(peaks))
    #     # img_mp = root_dir.get_child(f"{idx}.png")
    #     # anno = cv2.cvtColor(anno, cv2.COLOR_BGR2RGB)
    #     # img_mp.save_local(anno, upload=False)
    #     if idx > 500:
    #         break
    # points = []
    # for idx, peaks in enumerate(peaks_by_frame):
    #     points.extend([idx, peak] for peak in peaks)

    # points = np.array(points).reshape(-1, 1, 2).astype(np.float32)

    # lines = cv2.HoughLinesPointSet(
    #     points,
    #     lines_max=100,
    #     threshold=2,
    #     min_rho=0,
    #     max_rho=2000,
    #     rho_step=0.5,
    #     min_theta=0,
    #     max_theta=2 * np.pi,
    #     theta_step=np.pi / 260,
    # )
    # # plot lines
    # plt.scatter(points[:, 0, 0], points[:, 0, 1])
    # x0 = 0
    # x1 = 50
    # for line in lines:
    #     votes, rho, theta = line[0]
    #     y0 = (rho - x0 * np.cos(theta)) / np.sin(theta)
    #     y1 = (rho - x1 * np.cos(theta)) / np.sin(theta)
    #     plt.plot([x0, x1], [y0, y1], color="red")
    # plt.show()

    # meanshift = MeanShift(bandwidth=5)
    # meanshift.fit(np.array([x_s, y_s]).T)
    # labels = meanshift.labels_
    # cluster_centers = meanshift.cluster_centers_
    # n_clusters_ = len(np.unique(labels))
    # print("number of estimated clusters : %d" % n_clusters_)
    # plt.scatter(x_s, y_s, c=labels, s=50, cmap='viridis')
    # plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=200, alpha=0.5)
    # plt.show()
    # ransac = RANSACRegressor()
    # ransac.fit(np.array(x_s).reshape(-1, 1), np.array(y_s).reshape(-1, 1))
    # inlier_mask = ransac.inlier_mask_
    # outlier_mask = np.logical_not(inlier_mask)
    # line_X = np.arange(0, 50, 1)
    # line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    # plt.scatter(x_s, y_s, color="red", marker=".", label="Inliers")
    # plt.scatter(
    #     np.array(x_s)[outlier_mask],
    #     np.array(y_s)[outlier_mask],
    #     color="blue",
    #     marker=".",
    #     label="Outliers",
    # )
    # plt.show()
    # Multiple linear regression
    # LR = LinearRegression()
    # LR.fit(np.array(x_s).reshape(-1, 1), np.array(y_s).reshape(-1, 1))
    # print(LR.coef_)
    # print(LR.intercept_)
    # lines = np.array(x_s) * LR.coef_ + LR.intercept_
    # plot lines

    # plt.scatter(x_s, y_s)
    # plt.show()
    # mixture = GaussianMixture(n_components=4, init_params="k-means++")
    # mixture.fit(np.array([x_s, y_s]).T)
    # labels = mixture.predict(np.array([x_s, y_s]).T)
    # # db = DBSCAN(eps=5, min_samples=3).fit(np.array([x_s, y_s]).T)
    # # labels = db.labels_
    # colors = plt.cm.Spectral(np.linspace(0, 1, len(set(labels))))
    # for k, col in zip(set(labels), colors):
    #     if k == -1:
    #         col = "k"
    #     class_member_mask = labels == k
    #     xy = np.array([x_s, y_s]).T[class_member_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=col, markeredgecolor="k", markersize=14)
    # plt.show()
    # plt.scatter(x_s, y_s)
    # plt.show()
