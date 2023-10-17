"""Script to move box files to s3."""
import traceback
import zipfile
from pathlib import Path
from typing import List

from S3MP.callbacks import FileSizeTQDMCallback
from S3MP.global_config import S3MPConfig
from S3MP.mirror_path import MirrorPath
from S3MP.transfer_configs import GB, MB, get_transfer_config

from ground_data_processing.utils.absolute_segment_groups import (
    RootSegments,
    VideoSegments,
)
from ground_data_processing.utils.s3_constants import CameraViews, Fields, VideoRez


def find_subfolder_that_exists(root: Path, possible_subfolders: List[str]) -> Path:
    """Find a subfolder that exists."""
    for subfolder in possible_subfolders:
        subfolder_path = root / subfolder
        if subfolder_path.exists():
            return subfolder_path
    raise ValueError(f"Could not find any of {possible_subfolders} in {root}.")


def standard_row_tuple(first_int, second_int):
    """Return standard naming tuple for two rows."""
    return (f"{first_int}, {second_int}", f"{first_int}, {second_int}")


def trim_on_max_file_size(
    files: List[str], ratio=0.1, check_possible_split: bool = False
) -> List[str]:
    """Trim file list on max size in list, remove all less than proportion of max size."""
    file_sizes = [Path(file).stat().st_size for file in files]
    if check_possible_split:
        # raise error if any file is ~3.7GB
        epsilon = 50 * MB
        if any(abs(size - 3.7 * GB) < epsilon for size in file_sizes):
            raise ValueError("Found file that is ~3.7GB, which might be a split video.")

    max_size = max(file_sizes)
    return [file for file, size in zip(files, file_sizes) if size > max_size * ratio]


# sourcery skip: list-comprehension
if __name__ == "__main__":
    OVERWRITE = True

    base_dir = Path("/mnt/c/Users/nschroeder/Downloads/box_files/")
    box_dir = base_dir / "row 5"

    async_transfer_config = get_transfer_config(
        n_threads=30, block_size=8 * MB, max_ram=4 * GB
    )
    S3MPConfig.transfer_config = async_transfer_config

    upload_list = []
    n_rows = 8
    midpoint = n_rows // 2
    upload_list = [
        # ("9, 4", "Row 9,4")
        # ("Row 5b, 2a", "Row 5B, 2A"),
        # ("Row 4b, 3a", "Row 4B, 3A"),
        # ("Row 6b, 1a", "Row 6B, 1A"),
        # ("Row 3b, 4a", "Row 3B, 4A"),
        # ("Row 1b, 6a", "Row 1B, 6A"),
        # ("Row 2b, 5a", "Row 2B, 5A"),
        ("Row 5", "")
    ]
    # for idx in range(1, midpoint+1):
    #     upload_list.append(
    #         (f"Row {idx}, {n_rows + 1-idx}", f"Row {idx}, {n_rows + 1-idx}")
    #     )
    # print(upload_list)

    possible_folder_names_per_camera = {
        CameraViews.BOTTOM: ["bottom", "Bottom", "Ground"],
        CameraViews.NADIR: ["nadir", "Nadir", "top", "Top"],
        CameraViews.OBLIQUE: ["oblique", "Oblique", "middle", "Middle"],
    }

    for row_pass, zip_name in upload_list:
        s3_segments = [
            RootSegments.PLOT_NAME_AND_YEAR(Fields.PROD_FIELD_ONE),
            RootSegments.DATA_TYPE("Videos"),
            VideoSegments.DATE("7-05"),
            VideoSegments.ROW_DESIGNATION(row_pass),
            VideoSegments.RESOLUTION(VideoRez.r4k_120fps),
        ]
        s3_key = "/".join([x.name for x in s3_segments]) + "/"
        if s3_key[-1] == "/":
            s3_key = s3_key[:-1]

        try:
            root_box = box_dir / zip_name
            skip_unzip = bool(root_box.exists())
            # skip_unzip = True
            current_zip_fn = f"{zip_name}.zip"
            current_zip_path = box_dir / current_zip_fn
            if not skip_unzip:
                with zipfile.ZipFile(current_zip_path, "r") as zip_ref:
                    zip_ref.extractall(box_dir)

            upload_mps: List[MirrorPath] = []
            for (
                camera_view,
                possible_folder_names,
            ) in possible_folder_names_per_camera.items():
                cam_folder = find_subfolder_that_exists(root_box, possible_folder_names)
                if (cam_folder / "DCIM" / "100GOPRO").exists():
                    cam_folder = cam_folder / "DCIM" / "100GOPRO"
                elif (cam_folder / "100GOPRO").exists():
                    cam_folder = cam_folder / "100GOPRO"

                vids = list(cam_folder.glob("*.mp4"))
                print(vids)
                if not vids:
                    raise ValueError(f"Found no videos in {cam_folder}")

                for vid in vids:
                    if vid.name == "merged.mp4":
                        print("found merged.mp4, using this as video")
                        vids = [vid]
                        break
                else:
                    print("trimming")
                    vids = trim_on_max_file_size(vids, 0.4)

                if len(vids) > 1:
                    raise ValueError(f"Found multiple videos in {cam_folder}")

                u_mp = MirrorPath.from_s3_key(f"{s3_key}/{camera_view}.mp4")
                u_mp.override_local_path(Path(vids[0]))
                upload_mps.append(u_mp)
                # u_mp.local_path = Path(vids[0])

            if not OVERWRITE:
                upload_mps = [
                    upload_mp
                    for upload_mp in upload_mps
                    if not upload_mp.exists_on_s3()
                ]
                if not upload_mps:
                    print(f"Skipping {row_pass} as files already exist on S3.")
                    continue  # skip if all files already exist

            with FileSizeTQDMCallback(upload_mps, is_download=False) as tqdm_cb:
                # upload_mp.upload_from_mirror()
                for upload_mp in upload_mps:
                    # resume_multipart_upload(upload_mp)
                    upload_mp.upload_from_mirror(overwrite=OVERWRITE)
                    # if upload_mp.exists_on_s3():
                    #     upload_mp.delete_local()
                    # res = multipart_upload_from_mirror(upload_mp)
                # [upload_mp.upload_from_mirror_if_not_present() for upload_mp in upload_mps]
                # upload_threads = [
                #     upload_from_mirror_thread(upload_mp)
                #     for upload_mp in upload_mps
                #     if not upload_mp.exists_on_s3()
                # ]

                # sync_gather_threads(upload_threads)

            print(f"Uploaded {zip_name} to {s3_key}.")
        except Exception:
            print(traceback.format_exc())
            print(f"Failed to upload {zip_name}")
            continue
