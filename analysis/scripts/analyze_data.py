"""Script for analyzing measurements, building rogue detection models, and plotting."""
import json
import os

import click

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis.utils.analysis_tools as tools
from analysis.utils.analysis_params import AnalysisParameters
from analysis.utils.models import Model


@click.command()
@click.option(
    "--dataset", "-d", default=None, help="path to dataset to run on", multiple=True
)
@click.option(
    "--mode",
    "-m",
    default="build",
    help="mode to run in ['plot', 'model', 'all', 'build', 'dynamic']",
)
@click.option("--model", default=None, help="path to model to load for analysis")
@click.option(
    "--output_dir",
    default=None,
    help="output directory for plots and things, only need to specify if more than one dataset is provided",
)
@click.option(
    "--public",
    "-p",
    is_flag=True,
    show_default=True,
    default=False,
    help="option to produce public facing plots",
)
@click.option(
    "--splits",
    is_flag=True,
    show_default=True,
    default=False,
    help="option to produce plots by ds split",
)
@click.option(
    "--no_plot",
    is_flag=True,
    show_default=True,
    default=False,
    help="do not make plots",
)
@click.option(
    "--pull_false_pos",
    is_flag=True,
    show_default=True,
    default=False,
    help="pull false positive images from s3 for inspection",
)
@click.option(
    "--threshold",
    is_flag=True,
    show_default=True,
    default=False,
    help="makes the threshold plot and exits",
)
@click.option(
    "--unstaked",
    is_flag=True,
    show_default=True,
    default=False,
    help="indicates the row is truly unstaked, changes output metrics",
)
@click.option(
    "--reverse",
    is_flag=True,
    show_default=True,
    default=False,
    help="processes the data in reverse order",
)
@click.option(
    "--no_oblique",
    is_flag=True,
    show_default=True,
    default=False,
    help="processes the data without oblique images",
)
@click.option(
    "--log", is_flag=True, show_default=True, default=False, help="turns on logging"
)
def analyze_data(
    dataset,
    mode,
    model,
    output_dir,
    public,
    splits,
    no_plot,
    pull_false_pos,
    threshold,
    unstaked,
    reverse,
    no_oblique,
    log,
):
    """
    Script to make triggers for, and plots of, metrics in rogues data.

    Usage: python scripts/analyze_data.py -d path/to/dataset/measurements_df.gz -m mode

    Options:
    --dataset (-d): path to dataset to run on
    --mode (-m): mode to run in. Available modes are:
    'plot': gives a first look at data, plots the features in stats
    'model': creates a model for the data, creates plots for the model
    'all': plots the stat features, and creates a model with plots
    'build': builds a model, not included when running with 'all'
    --model: model to use when processing data
    --output_dir: output directory to use when more than one dataset is specified
    --public (-p): removes the "internal" label from plots, changes output directory to public plots directory
    --splits: makes all normal plots by DS Split for easy identification
    --no_plot: turns off plotting for faster accuracy results and model tuning
    --pull_false_pos: turns on pulling images of false positives from s3 for inspection
    --threshold: turns on the threshold determination/plotting
    --unstaked: indicates that the row is truly unstaked, there are no associated rogues.json files
    --reverse: processes the data in reverse order, useful if a rogue shows up in the first 30 plants
    --no_oblique: do not use oblique imagery to build the model
    --log: turns on logging
    """
    # guard against bad input
    if not dataset:
        raise RuntimeError(
            "[--dataset] not specified, please provide a path to a dataset"
        )
    if not all([os.path.exists(x) for x in dataset]):
        raise FileNotFoundError(f"[--dataset] provided file {dataset} does not exist")
    if len(dataset) > 1 and not output_dir:
        raise RuntimeError("[--output_dir] please specify and output directory")
    if mode not in ["plot", "model", "all", "build", "dynamic"]:
        raise RuntimeError(
            f"[--mode] {mode} not configured, please choose from ['plot', 'model', 'all']"
        )

    # load the dataset
    dataframes = [
        pd.read_pickle(d, compression=AnalysisParameters.COMPRESSION_DICT)
        for d in dataset
    ]
    for i, df in enumerate(dataframes):
        df.sort_values("name", ascending=not reverse, inplace=True)
        df["dataset"] = [dataset[i].split("/")[-3] for x in range(len(df))]
        df[AnalysisParameters.ANALYSIS_FEATURES[-1]] = df[
            AnalysisParameters.ANALYSIS_FEATURES[-1]
        ].astype(str)

    df = pd.concat(dataframes, ignore_index=True)
    # sort the dataframe by ds split
    df.sort_values(by=["name"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # set up plotting paths
    plot_path = None
    if output_dir is None:
        output_dir = os.path.dirname(dataset[0])
    else:
        output_dir = f"s3_mirror/{output_dir}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    plot_path = f"{output_dir}/plots_internal"
    if not no_plot:
        print(f"[INFO] plots will be saved to: {plot_path}")
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

    # set up model
    model = mode if model is None else json.load(open(model, "r"))
    model_name = dataset[0].split("/")[-2] if output_dir is None else output_dir
    m = Model(
        model_name,
        df,
        model=model,
        unstaked=unstaked,
        plot_path=plot_path,
        plot_features=not no_plot,
        plot_splits=splits,
        public=public,
        log=log,
    )

    if threshold:
        m.make_threshold_plot()
        print(f"[INFO] centroids saved to {plot_path}/centroid_hist.png")
        print("[INFO] please review to determine threshold")
        return  # exit after plotting centroids

    if mode == "build":
        print("[INFO] building model")
        m.build_static_model(no_oblique=no_oblique)
        m.evaluate_static_model()
        if not no_plot:
            m.plot_model_features()
        return  # exit after building and evaluating the model

    elif mode == "dynamic":
        print("[INFO] running dynamic model")
        m.run_dynamic_model()
        return  # exit after running dynamic model

    elif mode in ["model", "all"]:
        print("[INFO] evaluating model")
        # false negatives
        positive_rate = m.evaluate_static_model()
        m.save_model()
        if not no_plot:
            print(f"[INFO] plotting model features to: {plot_path}")
            m.plot_model_features()

        if positive_rate != 1:
            false_negative_imgs = np.array(
                m.false_negatives.loc[
                    :, AnalysisParameters.ANALYSIS_FEATURES[:2]
                ].values
            )
            print(
                f"[INFO] pulling missed rogues: {[x[0] for x in false_negative_imgs]}"
            )
            tools.pull_false_pos_images(output_dir, false_negative_imgs, is_rogue=True)

        if pull_false_pos:
            false_pos_images = np.array(
                m.false_positives.loc[
                    :, AnalysisParameters.ANALYSIS_FEATURES[:2]
                ].values
            )  # flatten this monstrosity
            print(f"[INFO] pulling {[x[0] for x in false_pos_images]}")
            print(f"[INFO] writing to {output_dir}")
            tools.pull_false_pos_images(output_dir, false_pos_images)

    else:
        raise RuntimeError(
            f"[--mode] {mode} not configured, please choose from ['plot', 'model', 'all', 'build']"
        )


if __name__ == "__main__":
    analyze_data()
