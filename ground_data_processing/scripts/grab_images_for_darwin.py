"""Grab images for Darwin."""
import shutil

from S3MP.global_config import S3MPConfig

S3MPConfig.mirror_root = "/home/nschroed/s3_mirror"
S3MPConfig.default_bucket_key = "sentera-rogues-data"

if __name__ == "__main__":
    from S3MP.mirror_path import KeySegment, get_matching_s3_mirror_paths

    # open the text file with the list of files
    file_list = "/mnt/c/Users/nschroeder/Downloads/filenames-rogues-leaf-segmentation-coreset-samples_300-0297-1690313578667.txt"
    with open(file_list) as f:
        files = f.readlines()
    files = [x.strip() for x in files]
    segments = [
        KeySegment(0, "2023-dev-sandbox"),
        KeySegment(1, "training_data_lightly"),
        KeySegment(2, "nadir"),
    ]

    for i, file in enumerate(files):
        mp = get_matching_s3_mirror_paths(
            segments + [KeySegment(3, file, is_file=True)]
        )[0]
        print(mp.s3_key)
        print(mp.local_path)
        print(mp.exists_on_s3())
        mp.download_to_mirror()
        shutil.move(
            str(mp.local_path),
            f"/mnt/c/Users/nschroeder/Downloads/rogues-leaf-segmentation-coreset-samples_300/{str(i).zfill(3)}.png",
        )
