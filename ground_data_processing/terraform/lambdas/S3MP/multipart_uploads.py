"""S3MP multipart uploads."""
import concurrent.futures
import math

from S3MP.global_config import S3MPConfig
from S3MP.mirror_path import MirrorPath
from S3MP.transfer_configs import MB


# TODO prefix optimization
def get_mpu(mirror_path: MirrorPath):
    """Check if a multipart upload has started."""
    bucket = S3MPConfig.bucket
    mpus = bucket.multipart_uploads.all()
    for mpu in mpus:
        if mpu.key == mirror_path.s3_key:
            if list(mpu.parts.all()):
                return mpu
            mpu.abort()  # Abort empty uploads


def resume_multipart_upload(
    mirror_path: MirrorPath,
    max_threads: int = 30,
):
    """Start or resume a multipart upload from a mirror path."""
    mpu = get_mpu(mirror_path)
    if not mpu:
        print("\nMultipart upload not found, starting new one.")
        return mirror_path.upload_from_mirror_if_not_present()

    mpu_parts = list(mpu.parts.all())
    mpu_parts.sort(key=lambda part: part.part_number)

    # get size bytes
    total_size_bytes = mirror_path.local_path.stat().st_size

    n_uploaded_parts = len(mpu_parts)
    part_size = max(part.size for part in mpu_parts)
    n_total_parts = math.ceil(total_size_bytes / part_size)

    mpu_dict = {
        "Parts": [
            {"ETag": part.e_tag, "PartNumber": part.part_number} for part in mpu_parts
        ]
    }
    print()
    print(f"Resuming multipart upload with {n_uploaded_parts}/{n_total_parts} parts.")

    with open(mirror_path.local_path, "rb") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Verify existing parts.
            assert all(part.size == part_size for part in mpu_parts[:-1])

            f.seek(part_size * n_uploaded_parts)
            if S3MPConfig.callback:
                S3MPConfig.callback(part_size * n_uploaded_parts)
            thread_futures = []
            uploaded_parts = []
            for part_number in range(n_uploaded_parts + 1, n_total_parts + 1):
                current_data = f.read(part_size)
                part = mpu.Part(part_number)
                uploaded_parts.append(part)
                thread_futures.append(executor.submit(part.upload, Body=current_data))
            for u_part, thread_future in zip(uploaded_parts, thread_futures):
                mpu_dict["Parts"].append(
                    {
                        "ETag": thread_future.result()["ETag"],
                        "PartNumber": u_part.part_number,
                    }
                )
                if S3MPConfig.callback:
                    S3MPConfig.callback(part_size)

    obj = mpu.complete(MultipartUpload=mpu_dict)
    if abs(total_size_bytes - obj.content_length) > MB:
        print()
        print(
            f"Uploaded size {obj.content_length} does not match local size {total_size_bytes}"
        )
        obj.delete()
        print("Deleted object, restarting upload.")
        return resume_multipart_upload(mirror_path, max_threads=max_threads)
