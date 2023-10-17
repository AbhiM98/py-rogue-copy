"""Utilities for ffmpeg."""
import time

import ffmpeg
from S3MP.mirror_path import MirrorPath


class FFmpegProcessS3Upload:
    """FFmpeg process, followed by an S3 upload."""

    def __init__(self, process: ffmpeg.Stream, s3_key: str):
        """Init."""
        self.process = process
        self.s3_key = s3_key
        self.mp = MirrorPath.from_s3_key(s3_key)

    def start_process_async(self):
        """Start process."""
        self.process = self.process.run_async()
        return self  # for chaining

    def is_finished(self):
        """Check if finished, upload if so."""
        if self.process.poll() is not None:
            self.mp.upload_from_mirror(True)
            return True
        return False


# Static
class FFmpegProcessManager:
    """Manage ffmpeg processes."""

    max_processes = 10
    processes = []

    @staticmethod
    def process_check():
        """Check through processes and remove finished ones to trigger uploads."""
        for process in FFmpegProcessManager.processes:
            if process.is_finished():
                FFmpegProcessManager.processes.remove(process)

    @staticmethod
    def wait_for_one_process_to_finish():
        """Wait for one process to finish."""
        while len(FFmpegProcessManager.processes) > 0:
            time.sleep(0.1)
            for process in FFmpegProcessManager.processes:
                if process.is_finished():
                    FFmpegProcessManager.processes.remove(process)
                    return

    @staticmethod
    def wait_for_all_processes_to_finish():
        """Wait for all processes to finish."""
        while len(FFmpegProcessManager.processes) > 0:
            time.sleep(0.1)
            for process in FFmpegProcessManager.processes:
                if process.is_finished():
                    FFmpegProcessManager.processes.remove(process)

    @staticmethod
    def add_process(ffmpeg_proc: FFmpegProcessS3Upload):
        """Add a process to the list of processes."""
        if len(FFmpegProcessManager.processes) >= FFmpegProcessManager.max_processes:
            FFmpegProcessManager.wait_for_one_process_to_finish()
        FFmpegProcessManager.processes.append(ffmpeg_proc.start_process_async())

    @staticmethod
    def set_max_processes(max_processes: int):
        """Set the maximum number of processes."""
        FFmpegProcessManager.max_processes = max_processes


# Static
class FFmpegFlags:
    """FFmpeg flags."""

    input_flags = {}
    output_flags = {}

    @staticmethod
    def set_input_flag(flag_name: str, flag_value: str):
        """Set an input flag."""
        FFmpegFlags.input_flags[flag_name] = flag_value

    @staticmethod
    def set_output_flag(flag_name: str, flag_value: str):
        """Set an output flag."""
        FFmpegFlags.output_flags[flag_name] = flag_value
