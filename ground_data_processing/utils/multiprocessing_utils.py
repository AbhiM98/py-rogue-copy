"""Multiprocessing utilities."""
import time
from multiprocessing import Process, Queue
from typing import List


class QueuedProcess:
    """Process with a queue."""

    def __init__(self, process: Process, queue: Queue):
        """Init."""
        self.process = process
        self.pid = process.pid
        self.queue = queue
        self.finished = False
        self.ret_val = None

    def is_finished(self):
        """Check if finished."""
        return self.finished

    def wait_for_finish(self, timeout=0.5):
        """Check if finished."""
        self.process.join(timeout)
        if not self.process.is_alive():
            self.finished = True
            self.ret_val = self.queue.get()

    def get_ret_val(self):
        """Get return value."""
        return self.ret_val


# Static
class MultiprocessingManager:
    """Multiprocessing manager."""

    @staticmethod
    def wait_for_all_queued_processes_to_finish(
        processes: List[QueuedProcess], print_freq=None
    ):
        """Wait for all queued processes to finish."""
        last_timestamp = time.time()
        while any(not p.is_finished() for p in processes):
            now = time.time()
            if print_freq and now - last_timestamp > print_freq:
                print(f"\n{len(processes)} processes remaining.")
                last_timestamp = now
            for proc in processes:
                if not proc.is_finished():
                    proc.wait_for_finish()
            time.sleep(5)

    @staticmethod
    def wait_for_all_processes_to_finish(process: List[Process]):
        """Wait for all processes to finish."""
        while any(p.is_alive() for p in process):
            for p in process:
                if p.is_alive():
                    p.join(0.1)

    @staticmethod
    def get_ret_vals(processes: List[QueuedProcess]):
        """Get return values."""
        return [proc.get_ret_val() for proc in processes]


def queue_wrapper(fn, queue, *args, **kwargs):
    """Wrap a function in a queue."""
    return queue.put(fn(*args, **kwargs))


def queue_wrap_process(fn, *args, **kwargs) -> QueuedProcess:
    """Wrap a function in a queue and start a process."""
    queue = Queue()
    fn_args = [fn, queue] + list(args)
    p = Process(target=queue_wrapper, args=fn_args, kwargs=kwargs)
    p.start()
    return QueuedProcess(p, queue)
