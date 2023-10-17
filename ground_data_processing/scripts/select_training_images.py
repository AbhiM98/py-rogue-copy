"""View an npy plot."""
import os
import shutil
import time
from pathlib import Path

import numpy as np
from S3MP.global_config import S3MPConfig

if __name__ == "__main__":
    S3MPConfig.default_bucket_key = "sentera-rogues-data"
    S3MPConfig.mirror_root = "/home/ec2-user/s3_mirror/"
    from S3MP.mirror_path import KeySegment, MirrorPath, get_matching_s3_mirror_paths

    input_segments = [
        KeySegment(0, "2023-field-data"),
        KeySegment(1, "Williamsburg_Strip_Trial"),
        KeySegment(2, "2023-07-06"),
        # KeySegment(3, incomplete_name="row-3"),
        KeySegment(7, incomplete_name="Thumbnail"),
    ]

    mps = get_matching_s3_mirror_paths(input_segments)
    print(f"Found {len(mps)} matching paths.")

    for mp in mps:
        print(mp.s3_key)
        # each mp is a MirrorPath object
        # that points to a thumbnail folder
        # for a particular camera view
        images = np.random.randint(0, len(mp.get_children_on_s3()), 5)
        images = np.unique(images)
        for child in [
            x
            for x in mp.get_children_on_s3()
            if int(x.key_segments[-1].name.split(".")[0]) in images
        ]:
            # each child is a MirrorPath object
            # that points to a thumbnail image
            cam = child.key_segments[-2].name.split(" ")[0]

            # make a new image path
            child.download_to_mirror(overwrite=True)
            new_local = f"/home/ec2-user/s3_mirror/2023-dev-sandbox/training_data_lightly/{cam}/{str(time.time()).replace('.','')}.png"
            if not os.path.exists(os.path.dirname(new_local)):
                os.makedirs(os.path.dirname(new_local))
            shutil.move(child.local_path, new_local)

            new_mp = MirrorPath.from_local_path(Path(new_local))
            new_mp.upload_from_mirror()
