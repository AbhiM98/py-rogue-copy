"""Rename a file on s3."""

import boto3
from S3MP.global_config import S3MPConfig
from S3MP.keys import get_matching_s3_keys

from ground_data_processing.utils.absolute_segment_groups import VideoSegments
from ground_data_processing.utils.rogues_key_utils import (
    plot_trial_prefix_segment_builder,
)


def plot_int_swap(og_int_a, og_int_b):
    """Swap plot order."""
    return (f"{og_int_a}, {og_int_b}", f"{og_int_b}, {og_int_a}")


if __name__ == "__main__":
    SWAP_PATTERN = plot_int_swap(6, 7)

    segments = [
        *plot_trial_prefix_segment_builder(planting_number=2),
        VideoSegments.DATE("7-12"),
        VideoSegments.ROW_DESIGNATION(SWAP_PATTERN[0]),
    ]

    matching_keys = get_matching_s3_keys(segments)
    if len(matching_keys) > 1:
        exit()
    base_key = matching_keys[0]
    print(base_key)

    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(S3MPConfig.default_bucket_key)
    child_keys = bucket.objects.filter(Prefix=base_key)
    child_keys = [key.key for key in child_keys]
    print(f"Base Key: {base_key}")
    print(
        f"Replacing {SWAP_PATTERN[0]} with {SWAP_PATTERN[1]} in {len(child_keys)} keys."
    )
    user_entry = input("Are you sure you want to do this? (y/n) > ")
    if user_entry.lower() != "y":
        exit()

    new_keys = []
    for child_key in child_keys:
        new_key = child_key.replace(*SWAP_PATTERN)
        new_keys.append(new_key)

    for old_key, new_key in zip(child_keys, new_keys):
        print(old_key, new_key)
        bucket.Object(new_key).copy_from(
            CopySource=f"{S3MPConfig.default_bucket_key}/{old_key}"
        )
        bucket.Object(old_key).delete()
