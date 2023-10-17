"""Grab images."""
import os

from S3MP.global_config import S3MPConfig

user = os.environ["USER"]
S3MPConfig.mirror_root = f"/home/{user}/s3_mirror"
if not os.path.exists(S3MPConfig.mirror_root):
    os.makedirs(S3MPConfig.mirror_root)
S3MPConfig.default_bucket_key = "sentera-rogues-data"


if __name__ == "__main__":
    from S3MP.mirror_path import KeySegment, get_matching_s3_mirror_paths

    # define the bucket keys that correspond to the imgaes on s3
    base_segments = [
        KeySegment(0, "2023-field-data"),
        KeySegment(1, "Waterman_Strip_Trial"),
        KeySegment(2, "2023-07-10"),
    ]
    label_segments = [KeySegment(7, incomplete_name="staked_rogues.json", is_file=True)]
    img_segments = [
        KeySegment(7, incomplete_name="Raw"),
        KeySegment(8, incomplete_name="png", is_file=True),
    ]

    label_mps = get_matching_s3_mirror_paths(base_segments + label_segments)
    print(f"found {len(label_mps)} label files to download to mirror")

    for mp in label_mps:
        print(
            f"[INFO] Downloading label file /{mp.s3_key} and corresponding images to mirror ..."
        )
        print(f"[INFO] these files will be downloaded to {mp.local_path.parent}")
        mp.download_to_mirror()
        segments = [
            KeySegment(i, x) for i, x in enumerate(mp.s3_key.split("/")[0:-1])
        ] + img_segments
        print(segments)
        img_mps = get_matching_s3_mirror_paths(segments)
        print(f"[INFO] found {len(img_mps)} images to download to mirror")
        for img_mp in img_mps:
            img_mp.download_to_mirror()
        break
