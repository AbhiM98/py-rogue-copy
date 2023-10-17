"""Tools used in analysis."""
import os
from pathlib import Path

import cv2
import numpy as np
from aws_utils.s3_utils import download_urls_to_files_multithread
from pycocotools.mask import area, decode
from S3MP.mirror_path import KeySegment, get_matching_s3_mirror_paths
from scipy import ndimage, stats
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
from tqdm import tqdm

from analysis.utils.analysis_params import AnalysisParameters, S3Paths
from analysis.utils.peak_detector import PeakDetector


class Stats:
    """Class to aggregate statistical information extracted from images."""

    def __init__(self, name: str, vals: list) -> None:
        """Init."""
        self.name = name
        self.x = vals
        if len(self.x) == 0:
            self.mean = 0
            # self.median = 0
            # self.mode = 0
            self.skew = 0
            self.kurtosis = 0
            self.stddev = 0
            self.max = 0
            self.sum_top_10_pct = 0
            self.is_empty = True
            return
        self.mean = np.mean(self.x)
        self.skew = stats.skew(self.x)
        self.kurtosis = stats.kurtosis(self.x)
        self.stddev = np.std(self.x)
        self.max = max(self.x)
        temp = self.x.copy()
        temp.sort()
        self.sum_top_10_pct = sum(temp[-(len(temp) // 10) :])
        self.is_empty = False

    def to_dict(self):
        """Return a dictionary representation of the stats."""
        return {
            "name": self.name,
            "mean": self.mean,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "stddev": self.stddev,
            "max": self.max,
            "sum_top_10_pct": self.sum_top_10_pct,
        }

    def __repr__(self, verbose=False) -> str:
        """Return a string representation of the stats."""
        if verbose:
            print(f"name: {self.name}")
            print(f"has {len(self.x)} entries")
            print(f"mean: {self.mean}")
            # print(f"median: {self.median}")
            # print(f"mode: {self.mode}")
            print(f"skew: {self.skew}")
            print(f"kurtosis: {self.kurtosis}")
            print(f"stddev: {self.stddev}")
            print(f"max: {self.max}")
            print(f"sum_top_10_pct: {self.sum_top_10_pct}")
        return f"Stats: {self.name}"


def get_id(plant_id, rogues_json=None):
    """
    Return an id appropriate to the rogues json.

    This function exists because of inconsistencies in labeling
    -----------------------------------------------------------
    Args:
    plant_id (str): the id of the plant
    rogues_json: the rogues json
    -----------------------------------------------------------
    Returns:
    the id of the plant (str)
    """
    if plant_id in ["datetime", "name", "version"]:
        return "junk"

    if rogues_json is not None:
        keys = list(rogues_json.keys())
        keys.sort()
        if "plant" in keys[0]:
            AnalysisParameters.PLANT_TYPE = "class"
        return keys[int(plant_id.split("_")[-1])]

    return plant_id.split("_")[-1]


def get_skeleton(img_mask):
    """
    Skeletonize an image and return the number of pixels in the skeleton.

    Args:
    img_mask (np.array): the image to skeletonize

    Returns:
    the number of pixels in the skeleton (int)
    """
    x = np.ascontiguousarray(img_mask)
    skeleton = skeletonize(x)
    return len(np.where(skeleton)[0])


def get_width(img_mask):
    """
    Create contours, rotates the image, finds width.

    Args:
    img_mask (np.array): the image to find the width of

    Returns:
    the width of the image (int)
    """
    _, img = cv2.threshold(img_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # requires cv2.CHAIN_APPROX_NONE to work properly, might be able to get away with Simple at a later time
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # draw contours to image
    cont_canvas = np.zeros_like(img)
    cv2.drawContours(cont_canvas, contours, -1, color=255, thickness=1)

    # get the pca decomp vectors to rotate image
    pca = PCA(n_components=2)
    cols, rows = np.where(cont_canvas == 255)
    arr = np.array([cols, rows]).T
    try:
        pca.fit_transform(arr)
    except Exception:
        print("[ERROR] could not fit pca")
        return 0

    # angle of rotation
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    angle = np.degrees(angle)

    # rotate image and threshold again due to scaling
    rotated = ndimage.rotate(cont_canvas, -angle)
    _, rotated = cv2.threshold(rotated, 50, 255, cv2.THRESH_BINARY)

    # find all places where contour has only 2 active pixels and measure width
    widths = []
    # rand_sample = np.random.randint(0, len(rotated))
    cols, rows = np.where(rotated > 0)
    for x in np.unique(cols):
        if len(np.where(cols == x)[0]) == 2:
            vals = np.where(rotated[x] > 0)[0]
            widths.append(np.amax(vals) - np.amin(vals))

    return np.mean(widths) if len(widths) > 0 else 0


def get_centroid(mask):
    """
    Return the centroid of a mask.

    Args:
    mask (np.array): the mask to find the centroid of

    Returns:
    the centroid of the mask (tuple)
    """
    _, img = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # requires cv2.CHAIN_APPROX_NONE to work properly, might be able to get away with Simple at a later time
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # draw contours to image
    cont_canvas = np.zeros_like(img)
    cv2.drawContours(cont_canvas, contours, -1, color=255, thickness=1)

    cols, rows = np.where(cont_canvas > 0)
    return (np.average(rows), np.average(cols))


def get_measurements(masks):
    """
    Take in a set of masks for a single image and return some statistics on the lengths and widths.

    Args:
    masks ([np.array, np.array]): the masks to get measurements from ([nadir masks, oblique masks])

    Returns:
    Stats: the lengths of the masks
    Stats: the widths of the masks
    Stats: the areas of the masks
    list: the centroids of the masks
    """
    values = {
        "nadir-lengths": [],
        "nadir-widths": [],
        "nadir-areas": [],
        "nadir-centroids": [],
        "nadir-overlap": None,
        "oblique-lengths": [],
        "oblique-widths": [],
        "oblique-areas": [],
        "oblique-centroids": [],
        "oblique-mask-pct-in-upper-thirty": [],
        "oblique-mask-tallest": [],
        "oblique-mask-cumulative-dist": [],
        "oblique-overlap": None,
    }
    # nadir loop
    for x in tqdm(masks[0]):
        decoded_mask = decode(x)
        c = get_centroid(decoded_mask)
        # threshold to cut leaves at the left and right of the image
        if (
            c[0] < AnalysisParameters.CENTROID_THRESHOLD[0]
            or c[0] > AnalysisParameters.CENTROID_THRESHOLD[1]
        ):
            continue
        values["nadir-lengths"].append(get_skeleton(decoded_mask))
        values["nadir-widths"].append(get_width(decoded_mask * 255))
        values["nadir-areas"].append(area(x))
        values["nadir-centroids"].append((c, values["nadir-areas"][-1]))
        if values["nadir-overlap"] is None:
            values["nadir-overlap"] = decoded_mask
        else:
            values["nadir-overlap"] = np.add(values["nadir-overlap"], decoded_mask)

    # plot nadir-overlap as heatmap in plt
    # if len(np.where(values['nadir-overlap'] > 1)[0]) > 0:
    #     fig, ax = plt.subplots(1,1)
    #     img = ax.imshow(values['nadir-overlap'])
    #     fig.colorbar(img, ax=ax)
    #     plt.show()

    # oblique loop
    for x in tqdm(masks[1]):
        decoded_mask = decode(x)
        c = get_centroid(decoded_mask)
        values["oblique-lengths"].append(get_skeleton(decoded_mask))
        values["oblique-widths"].append(get_width(decoded_mask * 255))
        values["oblique-areas"].append(area(x))
        # calculate the percent of the mask in the upper 30% of the image
        values["oblique-mask-pct-in-upper-thirty"].append(
            np.sum(decoded_mask[: int(decoded_mask.shape[0] * 0.3)])
        )
        # calculate the minimum y value of the mask
        values["oblique-mask-tallest"].append(
            max(decoded_mask.shape) - np.min(np.where(decoded_mask > 0)[0])
            if len(np.where(decoded_mask > 0)[0]) > 0
            else 0
        )
        # add sum of masked pixels in each row to cumulative dist
        if len(values["oblique-mask-cumulative-dist"]) == 0:
            values["oblique-mask-cumulative-dist"] = [0] * decoded_mask.shape[1]
        values["oblique-mask-cumulative-dist"] = [
            x + y
            for x, y in zip(
                values["oblique-mask-cumulative-dist"], np.sum(decoded_mask, axis=1)
            )
        ]
        values["oblique-centroids"].append((c, values["oblique-areas"][-1]))

    return (
        Stats("nadir-lengths", values["nadir-lengths"]),
        Stats("nadir-widths", values["nadir-widths"]),
        Stats("nadir-areas", values["nadir-areas"]),
        values["nadir-centroids"],
        Stats("oblique-lengths", values["oblique-lengths"]),
        Stats("oblique-widths", values["oblique-widths"]),
        Stats("oblique-areas", values["oblique-areas"]),
        Stats(
            "oblique-mask-pct-in-upper-thirty",
            values["oblique-mask-pct-in-upper-thirty"],
        ),
        Stats("oblique-mask-tallest", values["oblique-mask-tallest"]),
        np.cumsum(values["oblique-mask-cumulative-dist"])
        / np.sum(values["oblique-mask-cumulative-dist"]),
        values["oblique-centroids"],
    )


def get_influenced_signal_with_mean(data, window, sigma, influence):
    """
    Get the RealTimePeakDetection with window, sigma, and influence.

    Return signals, influenced mean, num_stddevs.

    Args:
    data (np.array): the data to get the influenced signal from
    window (int): the window size for the RealTimePeakDetection
    sigma (float): the sigma value for the RealTimePeakDetection
    influence (float): the influence value for the RealTimePeakDetection

    Returns:
    np.array: the signal
    np.array: the influenced mean
    np.array: the number of standard deviations
    """
    rtpd = PeakDetector(data[:window], window, sigma, influence)
    signal_with_influenced_mean = [
        rtpd.thresholding_algo(data[i]) if i >= window else (0, 0, 0)
        for i in range(len(data))
    ]
    return (
        np.array([x[0] for x in signal_with_influenced_mean]),
        np.array([x[1] for x in signal_with_influenced_mean]),
        np.array([x[2] for x in signal_with_influenced_mean]),
    )


def get_local_path(s3path):
    """
    Build a local path from an s3 path.

    Args:
    s3path (str): path to file on s3

    Returns:
    local path (str)
    """
    lp = AnalysisParameters.LOCAL_PATH_ROOT
    s3path = s3path.split("/")
    if "segm" in s3path[-1]:
        return f"{lp}/{s3path[-3]}/{s3path[-1]}"

    return f"{lp}/{s3path[-2]}/{s3path[-1]}"


def pull_s3_to_local(s3path, local):
    """
    Take in a list of paths to files on s3 and a list of local paths and pull the s3 file to local.

    Args:
    s3path (list): list of paths to files on s3
    local (list): list of local paths

    Returns:
    None
    """
    # create any directories that don't exist
    for p in local:
        if not os.path.exists(p):
            os.makedirs(Path(p).parent.absolute(), exist_ok=True)

    download_urls_to_files_multithread(s3path, local)


def pull_false_pos_images(directory, images, is_rogue=False):
    """
    Pull false flagged rogues from s3 to local.

    TODO: this function currently doesn't work, needs to be updated.

    Args:
    directory (str): directory to pull images from
    images (list): list of images to pull
    is_rogue (bool): whether or not the images are rogues

    Returns:
    None
    """
    images, img_ids = [x[0] for x in images], [x[1] for x in images]
    print(images, img_ids)

    base_path_annotated = "nadir Raw Images/Annotated Images"
    base_path_preproc = "nadir Raw Images/Preprocessed Images"
    base_path_bottom = "video images/bottom"
    base_path_nadir = "video images/nadir"
    base_path_oblique = "video images/oblique"

    # TODO: fix this bit, it's broken.
    folder = directory.split("/")[-1].split("_")
    print(folder)
    field, date, row = folder[0], folder[1], folder[2]
    subfield, planting = None, None
    if len(folder) > 3:
        subfield = folder[1]
        planting = folder[2]
        date = folder[3]
        row = folder[4]

    row_tag = str(
        int(row.split("Row")[-1])
    )  # this is a silly way to remove the leading 0
    row_name = None
    if subfield:
        row_name = [
            x
            for x in S3Paths.fields[f"{field}_{subfield}_{planting}"]["row_names"]
            if row_tag in x
        ][0]
    else:
        row_name = [x for x in S3Paths.fields[field]["row_names"] if row_tag in x][0]

    for i, img in enumerate(images):
        processed_img_segments = build_segments(
            field if not subfield else f"{field}_{subfield}_{planting}",
            subfield,
            planting,
            date,
            row_name,
            "A" if row_tag in row_name.split(",")[0] else "B",
            ds_split=" ".join(img.split("_")[:2]) if "DS" in img else img.split("_")[0],
            name=img_ids[i],
            inference_type="inference img",
        )
        video_img_segments = build_segments(
            field if not subfield else f"{field}_{subfield}_{planting}",
            subfield,
            planting,
            date,
            row_name,
            "A" if row_tag in row_name.split(",")[0] else "B",
            ds_split=" ".join(img.split("_")[:2]) if "DS" in img else img.split("_")[0],
            name=img_ids[i],
            inference_type="video img",
        )

        s3_paths_preproc = [
            f"{AnalysisParameters.S3_PATH_ROOT}/{x.s3_key}"
            for x in get_matching_s3_mirror_paths(processed_img_segments)
        ]
        s3_paths_video = [
            f"{AnalysisParameters.S3_PATH_ROOT}/{x.s3_key}"
            for x in get_matching_s3_mirror_paths(video_img_segments)
        ]
        s3_paths = s3_paths_preproc + s3_paths_video

        if is_rogue:
            img = f"{img}_rogue"
        local_paths = [
            f"{directory}/{'_'.join(img.split('_')[:2])}/{base_path_annotated}/{img}.png",
            f"{directory}/{'_'.join(img.split('_')[:2])}/{base_path_preproc}/{img}.png",
            f"{directory}/{'_'.join(img.split('_')[:2])}/{base_path_bottom}/{img}.png",
            f"{directory}/{'_'.join(img.split('_')[:2])}/{base_path_nadir}/{img}.png",
            f"{directory}/{'_'.join(img.split('_')[:2])}/{base_path_oblique}/{img}.png",
        ]

        for x in local_paths:
            if not os.path.exists(Path(x).parent):
                os.makedirs(Path(x).parent.absolute(), exist_ok=True)

        pull_s3_to_local(s3_paths, local_paths)


def get_ds_split(img: str) -> str:
    """
    Get the decasecond split from an image name.

    Args:
    img (str): image name

    Returns:
    decasecond split (str)
    """
    return " ".join(img.split("_")[:2])


def get_img_num(img: str) -> str:
    """
    Get the image number from an image name.

    Args:
    img (str): image name

    Returns:
    image number (str)
    """
    return img.split("_")[-1]


def build_segments(
    field: str,
    subfield: str,
    planting: str,
    date: str,
    row: str,
    row_pass: str,
    ds_split: str = None,
    name: str = None,
    inference_type: str = None,
    try_next_depth: bool = False,
) -> list:
    """
    Build the S3 KeySegments for the inference paths.

    Args:
    field (str): field name
    subfield (str): subfield name
    planting (str): planting name
    date (str): date name
    row (str): row name
    row_pass (str): row pass name
    ds_split (str): decasecond split name
    name (str): image name
    inference_type (str): type of inference
    try_next_depth (bool): whether or not to try the next depth

    Returns:
    segments (list): list of KeySegments
    """
    # TODO: this function is a mess, needs to be cleaned up
    # this massive block of if statements isn't particularly elegant, but it works
    depth_type = None
    if inference_type is not None:
        if inference_type not in ["inference", "video", "inference img", "video img"]:
            raise RuntimeError(
                f"[ERROR][build_segments] type {inference_type} not configured."
            )
        depth_type = (
            "inference_depths" if "inference" in inference_type else "video_depths"
        )

    segments = []
    depth = S3Paths.fields[field][depth_type]["field"]
    segments.append(KeySegment(depth, S3Paths.fields[field]["name"]))

    if subfield is not None:
        depth = S3Paths.fields[field][depth_type]["subfield"]
        segments.append(KeySegment(depth, S3Paths.subfields[subfield]))

    if planting is not None:
        depth = S3Paths.fields[field][depth_type]["planting"]
        segments.append(KeySegment(depth, S3Paths.planting[planting]))

    depth = S3Paths.fields[field][depth_type][
        "inference" if "inference" in inference_type else "video"
    ]
    segments.append(
        KeySegment(
            depth,
            "Unfiltered Model Inference" if "inference" in inference_type else "Videos",
        )
    )

    if date is not None:
        depth = S3Paths.fields[field][depth_type]["date"]
        segments.append(KeySegment(depth, date))

    if row is not None:
        depth = S3Paths.fields[field][depth_type]["row"]
        segments.append(KeySegment(depth, row))

    if row_pass is not None:
        depth = S3Paths.fields[field][depth_type]["row_pass"]
        segments.append(KeySegment(depth, S3Paths.passes[row_pass]))

    if name is None:
        depth = S3Paths.fields[field][depth_type][
            "segm" if inference_type == "inference" else "rogues"
        ]
        segments.append(
            KeySegment(
                depth + int(try_next_depth),
                incomplete_name="segm" if inference_type == "inference" else "s.json",
                is_file=True,
            )
        )
    else:
        depth = S3Paths.fields[field][depth_type]["ds_id"]
        segments.append(KeySegment(depth, name=ds_split))
        depth = S3Paths.fields[field][depth_type]["img"]
        segments.append(KeySegment(depth, name=f"{name}.png", is_file=True))

    return segments


def nms(
    masks: np.array,
    scores: np.array,
    iou_threshold: float = 0.05,
    decoded: bool = False,
) -> list:
    """
    Run Non-Maximal Supression on a set of masks.

    Args:
    masks (np.array): masks to run nms on
    scores (np.array): scores for each mask
    iou_threshold (float): iou threshold
    decoded (bool): whether or not the masks are decoded

    Returns:
    selected_masks (list): masks after nms
    """
    decoded_masks = None
    if not decoded:
        decoded_masks = np.array([decode(x) for x in masks])
    else:
        decoded_masks = masks

    full_image = np.zeros_like(decoded_masks[0])
    for x in decoded_masks:
        full_image = np.add(full_image, x)

    selected_masks = []
    selected_decoded_masks = []
    masks = [(x, y, z) for x, y, z in zip(decoded_masks, scores, masks)]
    masks.sort(key=lambda x: x[1], reverse=True)

    img = np.zeros_like(masks[0][0])
    while len(masks) > 0:
        m, score, enc_m = masks.pop(0)
        if len(selected_masks) == 0:
            selected_masks.append(enc_m)
            selected_decoded_masks.append(m)
            continue
        ious = [
            np.sum(np.logical_and(m, x)) / np.sum(np.logical_or(m, x))
            for x in selected_decoded_masks
        ]
        area = np.sum(m)
        if max(ious) < iou_threshold and area > 100:
            selected_masks.append(enc_m)
            selected_decoded_masks.append(m)
            img = np.add(img, m)

    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    plt.imshow(full_image, cmap="inferno")
    plt.colorbar()
    plt.show()

    plt.imshow(img, cmap="inferno")
    plt.colorbar()
    plt.show()

    return selected_masks
