"""Utility functions for data augmentation."""
import concurrent.futures

import numpy as np
import psutil
from S3MP.global_config import S3MPConfig
from S3MP.mirror_path import MirrorPath
from tqdm import tqdm

from data_augmentation.types import segmentation


def poly_area(x: list[float], y: list[float]):
    """Calculate area of polygon."""
    # source: https://github.com/opencv/cvat/issues/2074
    x = np.array(x, dtype=np.float_)
    y = np.array(y, dtype=np.float_)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def area_from_segmentation(seg: segmentation) -> float:
    """Calculate area from segmentation."""
    x_points = [p[0] for p in seg]
    y_points = [p[1] for p in seg]
    return poly_area(x_points, y_points)


def bbox_from_segmentation(seg: segmentation) -> list[float]:
    """Calculate bounding box from segmentation."""
    x_coords = [p[0] for p in seg]
    y_coords = [p[1] for p in seg]
    new_bbox = [
        min(x_coords),
        min(y_coords),
        max(x_coords) - min(x_coords),
        max(y_coords) - min(y_coords),
    ]
    return new_bbox


def area_and_bbox_from_segmentation(
    seg: segmentation,
) -> tuple[float, list[float]]:
    """Calculate area and bounding box from segmentation."""
    x_points = [p[0] for p in seg]
    y_points = [p[1] for p in seg]
    area = poly_area(x_points, y_points)
    new_bbox = [
        min(x_points),
        min(y_points),
        max(x_points) - min(x_points),
        max(y_points) - min(y_points),
    ]
    return area, new_bbox


def get_image_name_from_mp(image_mp: MirrorPath) -> str:
    """Get the image name from a MirrorPath."""
    return image_mp.key_segments[-1].name


def add_suffix_to_image_name(image_name: str, suffix: str) -> str:
    """Add a suffix to an image name."""
    ext = image_name.split(".")[-1]
    return f"{image_name[:-(len(ext)+1)]}_{suffix}.{ext}"


def mp_copy_s3_only(src_mp: MirrorPath, dest_mp: MirrorPath):
    """Copy a file from S3 to S3."""
    S3MPConfig.s3_client.copy_object(
        CopySource={"Bucket": S3MPConfig.default_bucket_key, "Key": src_mp.s3_key},
        Bucket=S3MPConfig.default_bucket_key,
        Key=dest_mp.s3_key,
    )


def multithread_download_mps_to_mirror(
    mps: list[MirrorPath], overwrite: bool = False
) -> None:
    """Download a list of MirrorPaths to the local mirror."""
    n_procs = psutil.cpu_count(logical=False)
    proc_executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_procs)
    all_proc_futures: list[concurrent.futures.Future] = []
    pbar = tqdm(total=len(mps), desc="Downloading to mirror")  # Init pbar
    for mp in mps:
        pf = proc_executor.submit(mp.download_to_mirror, overwrite=overwrite)
        all_proc_futures.append(pf)

    # Increment pbar as processes finish
    for _ in concurrent.futures.as_completed(all_proc_futures):
        pbar.update(n=1)

    all_proc_futures_except = [pf for pf in all_proc_futures if pf.exception()]
    for pf in all_proc_futures_except:
        raise pf.exception()

    proc_executor.shutdown(wait=True)


def delete_mp(mp: MirrorPath) -> None:
    """Delete a MirrorPath both locally and on S3 if it exists."""
    if mp.exists_on_s3():
        print(f"[INFO]: Deleting {mp.s3_key} from S3.")
        mp.delete_s3()
    if mp.exists_in_mirror():
        print(f"[INFO]: Deleting {mp.local_path} from local mirror.")
        mp.delete_local()
