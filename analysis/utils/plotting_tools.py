"""Tools for plotting measurements."""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.utils.analysis_params import AnalysisParameters, PlottingParameters

# please use PlottingParameters to change defaults for these functions


def plot_n_lines(
    lines: tuple,
    labels: tuple = (""),
    title: str = "",
    output_name: str = "",
    vlines: list = None,
    x_labels: list = None,
    _kinternal: bool = True,
    log_y: bool = False,
):
    """
    Plot n lines with a few options.

    Args:
    lines (tuple): a tuple of y values to plot (index = 1)
    labels (tuple): a tuple of labels for each line
    title (str): title of the plot
    output_name (str): name of the output file
    vlines (list): a list of x values to plot vertical lines at
    x_labels (list): a list of x values to label the x axis with
    _kinternal (bool): whether or not to add the Sentera annotation

    Returns:
    None
    """
    # TODO redo with seaborn instead of matplotlib

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))

    for i, x in enumerate(lines):
        axs.plot(
            x,
            label=labels[i],
            color=PlottingParameters.COLORS[i],
            linewidth=PlottingParameters.LW_THIN
            if i == 0
            else PlottingParameters.LW_THICK,
        )
    if vlines is not None:
        for i, row in vlines.iterrows():
            solid = AnalysisParameters.ANALYSIS_FEATURES[-1] in vlines.columns and (
                "White" in row[AnalysisParameters.ANALYSIS_FEATURES[-1]]
                or row[AnalysisParameters.ANALYSIS_FEATURES[-1]]
                == AnalysisParameters.HYBRID
            )
            axs.axvline(
                i,
                color=PlottingParameters.COLOR_VLINE,
                linestyle=PlottingParameters.LSTYLE_SOLID
                if solid
                else PlottingParameters.LSTYLE_DASHED,
                linewidth=PlottingParameters.LW_VLINE,
            )

    axs.legend(loc=PlottingParameters.LOC_LEGEND)
    axs.set_title(title)

    plt.subplots_adjust(
        bottom=PlottingParameters.ADJUST_BOTTOM,
        left=PlottingParameters.ADJUST_LEFT,
        right=PlottingParameters.ADJUST_RIGHT,
        top=PlottingParameters.ADJUST_TOP,
    )

    # Sentera annotation
    axs.annotate(
        PlottingParameters.ANNOT_SENTERA,
        xy=(0.05, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        color=PlottingParameters.COLOR_LIST["Green"],
        fontweight="bold",
    )
    if _kinternal:
        axs.annotate(
            PlottingParameters.ANNOT_INTERNAL,
            xy=(0.005, 0.99),
            xycoords="axes fraction",
            ha="left",
            va="top",
            color=PlottingParameters.COLOR_LIST["Gray"],
            fontweight="bold",
            fontsize=20,
            fontstyle="italic",
        )

    output_name = output_name.replace(" ", "_")

    pickle.dump(fig, open(f"{output_name}.pickle", "wb"))

    if x_labels is not None:
        plt.xticks(x_labels.index, x_labels.values, rotation=45, ha="right")

        def format_xticks(tick_val, tick_pos):
            if x_labels.values[tick_val].split("_")[-1] == "000":
                return "_".join(x_labels.values[tick_val].split("_")[:1])
            else:
                return ""

        axs.xaxis.set_major_formatter(format_xticks)
    if log_y:
        axs.set_yscale("log")

    fig.savefig(f"{output_name}.png")
    # plt.show()

    plt.close("all")


def plot_split(
    lines: tuple,
    labels: tuple = (""),
    title: str = "",
    output_name: str = "",
    vlines: list = None,
    x_labels: list = None,
):
    """
    Plot n lines for a ds split with a few options.

    Args:
    lines (tuple): a tuple of y values to plot (index = 1)
    labels (tuple): a tuple of labels for each line
    title (str): title of the plot
    output_name (str): name to write the file as
    vlines (list): list of x values to draw a vertical line
    x_labels (list): labels (DS Split + Image Num: DS_XXX_XXX)

    Returns:
    None
    """
    # TODO redo with seaborn instead of matplotlib

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))

    for i, x in enumerate(lines):
        axs.plot(
            x,
            label=labels[i],
            color=PlottingParameters.COLORS[i],
            linewidth=PlottingParameters.LW_THIN
            if i == 0
            else PlottingParameters.LW_THICK,
        )
    if vlines is not None:
        for i, row in vlines.iterrows():
            solid = AnalysisParameters.ANALYSIS_FEATURES[-1] in vlines.columns and (
                "White" in row[AnalysisParameters.ANALYSIS_FEATURES[-1]]
                or row[AnalysisParameters.ANALYSIS_FEATURES[-1]]
                == AnalysisParameters.HYBRID
            )
            axs.axvline(
                int(row["name"].split("_")[-1]),
                color=PlottingParameters.COLOR_VLINE,
                linestyle=PlottingParameters.LSTYLE_SOLID
                if solid
                else PlottingParameters.LSTYLE_DASHED,
                linewidth=PlottingParameters.LW_VLINE,
            )
    # if 'Leaf Area' in title:
    #     axs.set_ylim([3000, 130000])
    axs.legend(loc=PlottingParameters.LOC_LEGEND)
    axs.set_title(title)

    plt.subplots_adjust(
        bottom=PlottingParameters.ADJUST_BOTTOM_SPLIT,
        left=PlottingParameters.ADJUST_LEFT,
        right=PlottingParameters.ADJUST_RIGHT,
        top=PlottingParameters.ADJUST_TOP,
    )

    # Sentera annotation
    axs.annotate(
        PlottingParameters.ANNOT_SENTERA,
        xy=(0, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        color=PlottingParameters.COLOR_LIST["Green"],
        fontweight="bold",
    )

    axs.annotate(
        PlottingParameters.ANNOT_INTERNAL,
        xy=(0.005, 0.99),
        xycoords="axes fraction",
        ha="left",
        va="top",
        color=PlottingParameters.COLOR_LIST["Gray"],
        fontweight="bold",
        fontsize=20,
        fontstyle="italic",
    )

    output_name = output_name.replace(" ", "_")

    if x_labels is not None:
        plt.xticks(
            [i for i in range(len(x_labels.values))],
            x_labels.values,
            rotation="vertical",
        )

    fig.savefig(f"{output_name}.png")
    # plt.show()

    plt.close("all")


def plot_centroids(
    centroids: list, path: str, scatter: bool = False, rogues: pd.Series.index = None
):
    """
    Make histograms for visual inspection to define thresholds for cutting out leaves from neighboring rows.

    Args:
    centroids: list of lists of centroids
    path: path to save the plots

    Returns:
    None
    """
    if not scatter:
        c_list = []
        c_weights = []
        for c in centroids:
            c_list = c_list + [x[0] for x in c]
            c_weights = c_weights + [x[1] for x in c]

        fig, axs = plt.subplots(2, 2)
        axs[1, 0].hist2d(
            [x[0] for x in c_list],
            [-1 * x[1] for x in c_list],
            bins=(64, 64),
            weights=c_weights,
            cmap="viridis",
        )
        axs[0, 0].hist([x[0] for x in c_list], bins=64, weights=c_weights)
        axs[1, 1].hist(
            [-1 * x[1] for x in c_list],
            bins=64,
            weights=c_weights,
            orientation="horizontal",
        )

        title = centroids.name
        fig.savefig(f"{path}/{title}_hist.png")
        plt.close("all")

    else:
        c_list = []
        for c in centroids:
            x, y = np.average(
                [x[0][0] for x in c], axis=0, weights=[x[1] for x in c]
            ), np.average([x[0][1] for x in c], axis=0, weights=[x[1] for x in c])
            c_list.append((x, 1024 - y))

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
        axs.plot(
            [x[1] for x in c_list],
            color="black",
            alpha=0.5,
            marker=".",
            linestyle="None",
        )
        if rogues is not None:
            rogue_centroids = [c_list[x] for x in rogues.index]
            axs.plot(
                rogues.index,
                [x[1] for x in rogue_centroids],
                color="red",
                marker="o",
                linestyle="None",
            )

        fig.savefig(f"{path}/{centroids.name}_scatter.png")


def plot_2d_mask_cumulative(data: list, rogues: pd.Series.index, path: str) -> None:
    """
    Plot the cumulative distribution of the y values of the masks.

    Args:
    data: list of lists of centroids
    rogues: list of rogue indices
    path: path to save the plots

    Returns:
    None
    """
    # transpose data
    d = []
    for x in data.values:
        d.append(np.asarray(x) if len(x) > 0 else np.asarray([0 for _ in range(1024)]))

    d = np.asarray(d)
    d_transpose = np.transpose(d)

    # first point below 30%
    d_30 = []
    for x in d:
        d_30.append(1024 - np.where(x < 0.33)[0][-1] if len(x) > 0 else 0)

    # first point below 50%
    d_10 = []
    for x in d:
        d_10.append(1024 - np.where(x < 0.1)[0][-1] if len(x) > 0 else 0)

    # mean of each column
    d_mean = []
    for x in d:
        d_mean.append(np.mean(x) if len(x) > 0 else 0)

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(16, 12))
    # seaborn heatmap
    sns.heatmap(d_transpose, ax=axs[0], cmap="viridis", cbar=False)
    axs[0].set_title("Mask Heatmap with Mean, 10%, and 30% Lines")
    # 2d plot
    axs[1].plot(d_mean, color="black", alpha=0.5, marker=".", linestyle="None")
    axs[1].set_xlim([0, len(d_mean)])
    axs[1].set_ylabel("Mean")
    axs[2].plot(d_10, color="black", alpha=0.5, marker=".", linestyle="None")
    axs[2].set_xlim([0, len(d_10)])
    axs[2].set_ylabel("first point under 10%")
    axs[3].plot(d_30, color="black", alpha=0.5, marker=".", linestyle="None")
    axs[3].set_xlim([0, len(d_30)])
    axs[3].set_ylabel("first point under 30%")
    if rogues is not None:
        axs[1].plot(
            rogues.index,
            [d_mean[x] for x in rogues.index],
            color="red",
            marker="o",
            linestyle="None",
        )
        axs[2].plot(
            rogues.index,
            [d_10[x] for x in rogues.index],
            color="red",
            marker="o",
            linestyle="None",
        )
        axs[3].plot(
            rogues.index,
            [d_30[x] for x in rogues.index],
            color="red",
            marker="o",
            linestyle="None",
        )

    fig.savefig(f"{path}/cumulative.png")
    plt.close("all")

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    example_index = 157
    cum_dist_at_index = d[example_index]
    print(cum_dist_at_index)

    # make histogram from cumulative distribution
    hist_at_index = []
    for i in range(len(cum_dist_at_index)):
        if i == 0:
            hist_at_index.append(cum_dist_at_index[i])
        else:
            hist_at_index.append(cum_dist_at_index[i] - cum_dist_at_index[i - 1])

    hist_at_index = np.asarray(hist_at_index) * 800 / max(hist_at_index)

    axs.plot(hist_at_index, color="black", alpha=0.9, marker=".", linestyle="None")

    fig.savefig(f"{path}/cumulative_example_hist.png")
    plt.close("all")

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    axs.scatter(
        [i for i in range(len(cum_dist_at_index) - 1)],
        cum_dist_at_index[1:] * sum(hist_at_index),
        c=cum_dist_at_index[1:],
        cmap="viridis",
        alpha=0.9,
        marker=".",
        linestyle="None",
    )
    axs.scatter(
        [i for i in range(len(cum_dist_at_index) - 1)],
        hist_at_index[1:],
        c="black",
        alpha=0.9,
        marker=".",
        linestyle="None",
    )
    # set log y
    axs.set_yscale("log")
    fig.savefig(f"{path}/cumulative_example_cum_dist.png")
    plt.close("all")
