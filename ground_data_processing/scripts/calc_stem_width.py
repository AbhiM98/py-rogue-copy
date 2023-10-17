"""Calculate stem width for centered images."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from S3MP.mirror_path import get_matching_s3_mirror_paths
from tqdm import tqdm

from ground_data_processing.measurements.line_segments import (
    filter_lines_to_img_width_center,
    get_horizontal_overlap_bounds,
    get_hough_p_line_segments_with_theta_tol,
)
from ground_data_processing.utils.absolute_segment_groups import ImageSegments
from ground_data_processing.utils.image_utils import excess_green
from ground_data_processing.utils.processing_utils import (
    format_processing_info_as_json,
    print_processing_info,
)
from ground_data_processing.utils.rogues_key_utils import (
    plot_trial_prefix_segment_builder,
)
from ground_data_processing.utils.s3_constants import DataFiles, DataTypes

if __name__ == "__main__":
    processing_info = format_processing_info_as_json(*print_processing_info())

    # raise numpy exceptions
    np.seterr(all="raise")

    base_segments = plot_trial_prefix_segment_builder(
        planting_number=1, data_type=DataTypes.IMAGES
    )
    segments = [
        *base_segments,
        ImageSegments.ROW_AND_RANGE(incomplete_name="Row07"),
        ImageSegments.WEEK_NUMBER("Week 3"),
        ImageSegments.RESOLUTION("4k"),
    ]

    root_ds_split_folder_mps = get_matching_s3_mirror_paths(segments)
    print(f"Found {len(root_ds_split_folder_mps)} valid MirrorPaths.")

    CENTER_TOL = 0.03
    EXG_PROP_THRESH = 0.6
    MAX_WIDTH_THRESH_PX = 200

    ADJ_CENTER_TOL = 0.1

    for root_ds_split_folder_mp in tqdm(root_ds_split_folder_mps):
        print(f"\nProcessing {root_ds_split_folder_mp.s3_key}...")
        bottom_folder_mp = root_ds_split_folder_mp.get_child("bottom")
        anno_folder_mp = root_ds_split_folder_mp.get_child("util_anno")
        if not anno_folder_mp.local_path.exists():
            anno_folder_mp.local_path.mkdir(parents=True, exist_ok=True)
        else:
            for f in anno_folder_mp.local_path.iterdir():
                f.unlink()
        stem_width_json_data = {
            **processing_info,
            "stem_widths": {},
        }

        mean_by_img = []
        idx = 0
        for bottom_mp in tqdm(bottom_folder_mp.get_children_on_s3()):
            idx += 1
            if idx < 48:
                continue
            # img_name = bottom_mp.local_path.name
            # if img_name != "plant_58.png":
            #     continue
            bottom_img = bottom_mp.load_local(download=True)

            # Get the EXG image
            exg_img = excess_green(bottom_img)

            # Get center slice
            center_idx = int(exg_img.shape[1] / 2)
            slice_width = int(exg_img.shape[1] * ADJ_CENTER_TOL)
            center_slice = exg_img[
                :, center_idx - slice_width : center_idx + slice_width
            ]
            mean_by_img.append(np.mean(center_slice))
            # Get column means
            col_means = np.mean(center_slice, axis=0)
            # Save Plot
            plt.figure()
            plt.plot(col_means)
            plt.savefig(anno_folder_mp.local_path / f"col_means_{idx}.png")
            plt.close()
            # Get the row averages
            # row_averages = np.mean(exg_img, axis=1)
            # # Save plot of row averages
            # plt.plot(row_averages)
            # plt.savefig(anno_folder_mp.local_path / f"row_averages_{idx}.png")

            # canny_img = cv2.Canny(bottom_img, 0, 50)
            # sobel_img = cv2.Scharr(bottom_img, cv2.CV_64F, dx=1, dy=0)
            hsv_img = cv2.cvtColor(bottom_img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_img)
            hsv_stacked = np.vstack([h, s, v])
            hue_canny_img = cv2.Canny(h, 0, 50)

            hue_masked = np.ma.masked_where(exg_img == 0, h)
            hue_masked = np.ma.filled(hue_masked, 0)

            hue_masked_sobel = cv2.Sobel(hue_masked, cv2.CV_64F, dx=1, dy=0, ksize=5)

            sobel_thresh = 1.5 * np.std(hue_masked_sobel)
            # print(f"Sobel Threshold: {sobel_thresh}")
            hue_masked_sobel_leading_edges = np.where(
                hue_masked_sobel > sobel_thresh, 255, 0
            ).astype(np.float64)
            hue_masked_sobel_falling_edges = np.where(
                hue_masked_sobel < -sobel_thresh, -255, 0
            ).astype(np.float64)
            hue_masked_sobel /= np.max(hue_masked_sobel)
            hue_masked_sobel_leading_edges *= hue_masked_sobel
            hue_masked_sobel_falling_edges *= hue_masked_sobel
            hue_masked_sobel_leading_edges = hue_masked_sobel_leading_edges.astype(
                np.uint8
            )
            hue_masked_sobel_falling_edges = hue_masked_sobel_falling_edges.astype(
                np.uint8
            )

            IMAGE_X_CENTER = bottom_img.shape[1] / 2
            THETA_TOL = np.pi / 16
            leading_edge_line_segments = get_hough_p_line_segments_with_theta_tol(
                hue_masked_sobel_leading_edges,
                desired_theta=np.pi / 2,
                theta_tol=THETA_TOL,
            )
            leading_edge_line_segments = filter_lines_to_img_width_center(
                leading_edge_line_segments, hue_masked_sobel_leading_edges, CENTER_TOL
            )

            falling_edge_line_segments = get_hough_p_line_segments_with_theta_tol(
                hue_masked_sobel_falling_edges,
                desired_theta=np.pi / 2,
                theta_tol=THETA_TOL,
            )
            falling_edge_line_segments = filter_lines_to_img_width_center(
                falling_edge_line_segments, hue_masked_sobel_falling_edges, CENTER_TOL
            )
            # plt.figure(0)
            # plt.imshow(bottom_img)
            # plt.figure(1)
            # plt.imshow(hue_masked_sobel)
            # plt.show()
            utilized_leading_segs = []
            utilized_falling_segs = []

            distances = []
            if not leading_edge_line_segments or not falling_edge_line_segments:
                print("\nNot enough lines found.\n")
                stem_width = 0
            else:
                for line_a in leading_edge_line_segments:
                    for line_b in falling_edge_line_segments:
                        overlapping_bounds = get_horizontal_overlap_bounds(
                            line_a, line_b
                        )
                        if not overlapping_bounds:
                            continue
                        # Remove bounds where line b < line a
                        rm_keys = [
                            key
                            for key, bound in overlapping_bounds.items()
                            if bound[0] >= bound[1]
                        ]
                        for key in rm_keys:
                            del overlapping_bounds[key]
                        if not overlapping_bounds:
                            continue
                        for y_val, bound in overlapping_bounds.items():
                            exg_pxs = exg_img[y_val, bound[0] : bound[1]]
                            exg_prop = np.sum(exg_pxs / 255) / len(exg_pxs)
                            if exg_prop > EXG_PROP_THRESH:
                                distances.append(bound[1] - bound[0])
                                utilized_leading_segs.append(line_a)
                                utilized_falling_segs.append(line_b)

                if not distances:
                    print("\nNo distances found.\n")
                    stem_width = 0
                else:
                    stem_width = np.mean(distances)

            for line in utilized_leading_segs:
                x1, y1, x2, y2 = line[0]
                cv2.line(
                    hue_masked_sobel_leading_edges, (x1, y1), (x2, y2), (255, 255, 0), 2
                )
                cv2.line(bottom_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            for line in utilized_falling_segs:
                x1, y1, x2, y2 = line[0]
                cv2.line(
                    hue_masked_sobel_falling_edges, (x1, y1), (x2, y2), (255, 255, 0), 2
                )
                cv2.line(bottom_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            anno_out_mp = anno_folder_mp.get_child(
                f"{bottom_mp.local_path.stem}_{stem_width:02f}.png"
            )
            anno_out_mp.save_local(bottom_img, upload=False)

            stem_width_json_data["stem_widths"][bottom_mp.local_path.stem] = stem_width

        stem_width_json_mp = root_ds_split_folder_mp.get_child(
            DataFiles.STEM_WIDTHS_JSON
        )
        stem_width_json_mp.save_local(stem_width_json_data, upload=True, overwrite=True)

        full_plot_mp = anno_folder_mp.get_sibling("mean_plot.png")
        plt.figure()
        plt.plot(mean_by_img)
        plt.savefig(anno_folder_mp.get_sibling("mean_plot.png").local_path)
        plt.close()
