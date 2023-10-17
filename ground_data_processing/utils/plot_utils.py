"""Persistent thread plotting."""
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from numpy.typing import ArrayLike


def plot_1d_data_with_markers(
    data: ArrayLike, marker_idxs: ArrayLike, marker: str = "x", show: bool = False
):
    """Plot data with markers."""
    x_count, y_data = zip(*enumerate(data))
    plt.plot(x_count, y_data)
    if type(marker_idxs) is not list:
        plt.plot(marker_idxs, [data[m] for m in marker_idxs], marker)
    else:
        for marker_idx, _mark in zip(marker_idxs, marker):
            if marker_idx < len(data):
                plt.plot(marker_idx, data[marker_idx], _mark)
    if show:
        plt.show()


def constant_2d_elim_plot(queue, initial_points):
    """Plot a constant function."""
    while True:
        if not queue.empty():
            x = queue.get()
            x = np.array(x)
            plt.cla()
            plt.clf()
            plt.scatter(
                initial_points[:, 0], initial_points[:, 1], c=initial_points[:, 2]
            )
            plt.scatter(x[:, 0], x[:, 1], marker="x", c="r")
            plt.colorbar()
        plt.pause(0.1)


def constant_2d_elim_plot_thread(initial_points):
    """Plot a constant function."""
    queue = Queue()
    p = Process(target=constant_2d_elim_plot, args=(queue, initial_points))
    p.start()
    return queue


def constant_2d_colored_plot(queue):
    """Plot a constant function."""
    while True:
        if not queue.empty():
            x = queue.get()
            x = np.array(x)
            plt.cla()
            plt.scatter(x[:, 0], x[:, 1], c=x[:, 2])
        plt.pause(0.1)


def constant_2d_colored_plot_thread():
    """Plot a constant function."""
    queue = Queue()
    p = Process(target=constant_2d_colored_plot, args=(queue,))
    p.start()
    return queue


def constant_plot(queue):
    """Plot a constant function."""
    while True:
        if not queue.empty():
            x = queue.get()
            plt.cla()
            plt.plot(list(range(len(x))), x)
        plt.pause(0.1)


def constant_plot_thread():
    """Plot a constant function."""
    queue = Queue()
    p = Process(target=constant_plot, args=(queue,))
    p.start()
    return queue


def constant_imshow(queue):
    """Plot a constant function."""
    while True:
        if not queue.empty():
            x = queue.get()
            plt.cla()
            plt.imshow(x)
        plt.pause(0.1)


def constant_imshow_thread():
    """Plot a constant function."""
    queue = Queue()
    p = Process(target=constant_imshow, args=(queue,))
    p.start()
    return queue


def multi_image_viewer(images, titles=None, cmap=None):
    """View multiple images."""
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(10, 10))
    for idx, image in enumerate(images):
        axes[idx].imshow(image, cmap=cmap)
        if titles:
            axes[idx].set_title(titles[idx])
    plt.show()


def multi_image_viewer_with_nav_buttons(filenames, frames=None, initial_idx=0):
    """View multiple images."""
    ret_idxs = []
    frame_dict = {}
    if len(frames) == len(filenames):
        # We have frames already loaded.
        for frame, fn in zip(frames, filenames):
            frame_dict[fn] = frame
    fig, ax = plt.subplots()
    fig.set_dpi(100)
    idx = initial_idx

    def update_image():
        """Update image."""
        nonlocal idx
        img_name = filenames[idx]
        if img_name in frame_dict:
            img = frame_dict[img_name]
        else:
            img = cv2.imread(img_name)
            # resize to 720p
            img = cv2.resize(img, (1280, 720))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            frame_dict[img_name] = img

        ax.imshow(img, interpolation="none")
        ax.set_title(img_name.split(r"\\\\")[-1])
        fig.canvas.draw()

    def idx_mod_curry(diff):
        """Produce an idx-modifying function."""

        def _callback(event):
            """Update image after callback."""
            nonlocal idx
            idx = (
                max(0, idx + diff) if diff < 0 else min(len(filenames) - 1, idx + diff)
            )
            update_image()

        return _callback

    def mark_and_continue_callback(event):
        """Mark and continue."""
        nonlocal ret_idxs
        ret_idxs.append(int(Path(filenames[idx]).stem))
        idx_mod_curry(1)(event)

    def mark_callback(event):
        """Mark current idx as value."""
        nonlocal ret_idxs
        ret_idxs.append(int(Path(filenames[idx]).stem))
        plt.close("all")

    def reject_callback(event):
        """Mark current idx as value."""
        nonlocal ret_idxs
        plt.close("all")

    btn_width = 0.05
    btn_margin = 0.01
    btn_bot_pos = 0.00
    btn_offset = btn_width + btn_margin

    # toggle_amounts = [20, 10, 5, 3, 1]
    toggle_amounts = [100, 50, 20, 10, 1]
    diffs = []
    names = []
    for t_amnt in toggle_amounts:
        diffs.extend([-t_amnt, t_amnt])
        names.extend([f"<<{t_amnt}", f"{t_amnt}>>"])

    bt_axs = []
    btns = []
    for btn_idx, (diff, name) in enumerate(zip(diffs, names)):
        bt_axs.append(
            plt.axes([0.05 + btn_idx * btn_offset, btn_bot_pos, btn_width, 0.075])
        )
        btns.append(Button(bt_axs[-1], name))
        btns[-1].on_clicked(idx_mod_curry(diff))
    final_offset = 0.05 + btn_offset * len(diffs)

    markbt = plt.axes([final_offset, btn_bot_pos, btn_width, 0.075])
    bmark = Button(markbt, "Mark")
    bmark.on_clicked(mark_callback)

    rejectbt = plt.axes([final_offset + btn_offset, btn_bot_pos, btn_width, 0.075])
    breject = Button(rejectbt, "Reject")
    breject.on_clicked(reject_callback)

    mark_and_continuebt = plt.axes(
        [final_offset + 2 * btn_offset, btn_bot_pos, btn_width, 0.075]
    )
    bmark_and_continue = Button(mark_and_continuebt, "Mark & Continue")
    bmark_and_continue.on_clicked(mark_and_continue_callback)

    update_image()
    plt.show()
    return ret_idxs


def plot_classifier(
    img: ArrayLike,
    classes: List[str],
):
    """Plot classifier for a single image."""
    ret_class = None

    def _select_class_mod_curry(selected_class: str):
        """Curry for class selection."""

        def _event_cb(event):
            """Event callback."""
            nonlocal ret_class
            ret_class = selected_class
            plt.close("all")

        return _event_cb

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Select class")

    btn_bot_pos = 0.02
    btn_width = 0.2
    btn_spacing = 0.04
    btns = []
    for idx, class_name in enumerate(classes):
        bt_ax = plt.axes(
            [0.05 + idx * (btn_width + btn_spacing), btn_bot_pos, btn_width, 0.075]
        )
        btn = Button(bt_ax, class_name)
        btn.on_clicked(_select_class_mod_curry(class_name))
        btns.append(btn)

    # Maximize window
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.show()
    return ret_class
