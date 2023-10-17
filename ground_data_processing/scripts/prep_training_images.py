"""View an npy plot."""
import os

import cv2
from S3MP.global_config import S3MPConfig
from tqdm import tqdm

if __name__ == "__main__":
    S3MPConfig.default_bucket_key = "sentera-rogues-data"
    S3MPConfig.mirror_root = "/home/ec2-user/s3_mirror/"
    from S3MP.mirror_path import KeySegment, get_matching_s3_mirror_paths

    input_segments = [
        KeySegment(0, "2023-dev-sandbox"),
        KeySegment(1, "training_data_lightly"),
        KeySegment(3, incomplete_name="png", is_file=True),
    ]

    mps = get_matching_s3_mirror_paths(input_segments)
    print(f"Found {len(mps)} matching paths.")

    from ground_data_processing.scripts.prep_unfiltered_inference import (
        crop_square_from_img_center,
    )

    for mp in tqdm(mps):
        if not os.path.exists(mp.local_path):
            mp.download_to_mirror()
        # print(mp.local_path)
        img = cv2.imread(str(mp.local_path))
        img = crop_square_from_img_center(img, 1024, False, False)
        cv2.imwrite(str(mp.local_path), img)
        mp.upload_from_mirror(overwrite=True)
