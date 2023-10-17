"""Transfer configurations and utilities."""
from S3MP.global_config import S3MPConfig
from S3MP.types import S3TransferConfig
from s3transfer.constants import GB, KB, MB


def get_transfer_config(
    n_threads: int,
    block_size: int = 8 * MB,
    max_ram: int = 4 * GB,
    io_queue_size: int = 10e4,
    io_chunk_size: int = 256 * KB,
    set_global: bool = True,
) -> S3TransferConfig:
    """Get transfer config."""
    max_in_mem_upload_chunks = (max_ram - (n_threads * block_size)) // block_size
    max_in_mem_download_chunks = max_ram // block_size

    config = S3TransferConfig(
        multipart_threshold=block_size,
        multipart_chunksize=block_size,
        max_request_concurrency=n_threads,
        max_submission_concurrency=n_threads,
        max_in_memory_upload_chunks=max_in_mem_upload_chunks,
        max_in_memory_download_chunks=max_in_mem_download_chunks,
        max_io_queue_size=io_queue_size,
        io_chunksize=io_chunk_size,
    )
    config.use_threads = n_threads > 1
    if set_global:
        S3MPConfig.transfer_config = config
    return config
