"""Iteration utilities."""
import collections
import contextlib
from typing import Callable, List

from tqdm import tqdm

from ground_data_processing.utils.plot_utils import constant_plot_thread


def tail(n, iterable):
    """Itertools recipe for getting the last n elements of an iterable."""
    return iter(collections.deque(iterable, maxlen=int(n)))


def get_value_per_frame_generalized(
    iter_callable: List[Callable] | Callable,
    proc_fns: List[Callable] | Callable,
    iter_wrappers: List[Callable] | Callable = None,
    length: int = None,
    display_freq=None,
):
    """Get a value for each frame in an iterable."""
    if not isinstance(iter_callable, list):
        iter_callable = [iter_callable]

    iter_callable = [x() for x in iter_callable]

    if iter_wrappers:
        if not isinstance(iter_wrappers, list):
            iter_wrappers = [iter_wrappers]
        for wrapper in iter_wrappers:
            iter_callable = [wrapper(x) for x in iter_callable]

    if not isinstance(proc_fns, list):
        proc_fns = [proc_fns]

    if display_freq:
        plot_thread_queue = constant_plot_thread()
    data = []
    if length:
        print(f"Iterating over {length} frames.")
    for idx, iter_data in enumerate(tqdm(zip(*iter_callable), total=length)):
        for proc_fn in proc_fns:
            iter_data = proc_fn(
                *iter_data
            )  # unpack the tuple straight into the function
        data.append(iter_data)
        if display_freq and idx % display_freq == 0:
            plot_thread_queue.put(data)
        if length and idx >= length:
            with contextlib.suppress(Exception):
                iter_callable.close()
            break
    return data
