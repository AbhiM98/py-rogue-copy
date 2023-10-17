"""Asynchronous transfer utilities."""
import asyncio
from typing import Coroutine, List

import aioboto3
from S3MP.global_config import S3MPConfig
from S3MP.mirror_path import MirrorPath


async def async_upload_from_mirror(mirror_path: MirrorPath):
    """Asynchronously upload a file from a MirrorPath."""
    session = aioboto3.Session()
    async with session.resource("s3") as s3_resource:
        bucket = await s3_resource.Bucket(mirror_path.s3_bucket_key)
        await bucket.upload_file(str(mirror_path.local_path), mirror_path.s3_key)


def upload_from_mirror_thread(
    mirror_path: MirrorPath,
) -> Coroutine:
    """Upload from mirror on a separate thread."""
    bucket = S3MPConfig.get_bucket(mirror_path.s3_bucket_key)
    return asyncio.to_thread(
        bucket.upload_file,
        str(mirror_path.local_path),
        mirror_path.s3_key,
        Callback=S3MPConfig.callback,
        Config=S3MPConfig.transfer_config,
    )


async def _async_gather_threads(coroutines: List[Coroutine]) -> List[Coroutine]:
    """Gather threads."""
    await asyncio.gather(*coroutines)


def sync_gather_threads(coroutines: List[Coroutine]) -> List[Coroutine]:
    """Gather threads."""
    return asyncio.run(_async_gather_threads(coroutines))
