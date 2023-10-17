"""Check the number of images in each DS Split in a row."""
from S3MP.mirror_path import KeySegment, get_matching_s3_mirror_paths

if __name__ == "__main__":
    # segments
    segments = [
        KeySegment(0, "2023-field-data"),
        KeySegment(1, "Waterman_Strip_Trial"),
        KeySegment(2, "2023-07-10"),
        KeySegment(3, "row-09"),
        KeySegment(6, incomplete_name="DS "),
    ]

    # get matching keys
    mps = get_matching_s3_mirror_paths(segments)

    print(f"found {len(mps)} matching keys")

    # regroup by row
    mps_by_row = {}
    for mp in mps:
        row = mp.s3_key.split("/")[-5]
        if row not in mps_by_row:
            mps_by_row[row] = []
        mps_by_row[row].append(mp)

    for row in mps_by_row:
        print(
            f"row {row} has {len(mps_by_row[row])} keys: {'/'.join(mps_by_row[row][0].s3_key.split('/')[:-2])}"
        )
        total = 0
        for mp in mps_by_row[row]:
            # get the number of images produced in the bottom thumbnails
            bottom_imgs = mp.get_child("bottom Thumbnails").get_children_on_s3()
            total += len(bottom_imgs)
            warning = "" if len(bottom_imgs) > 50 else "WARNING: "
            if (
                mps_by_row[row].index(mp) == len(mps_by_row[row]) - 1
                or mps_by_row[row].index(mp) == 0
            ):
                # first and last splits often have fewer images
                warning = "" if len(bottom_imgs) > 25 else "WARNING: "
            # if warning != "":
            print(
                f"{warning}{mp.s3_key.split('/')[-5]} {mp.s3_key.split('/')[-2]} has {len(bottom_imgs)} images, total {total}"
            )
