"""Extracts measurements from leaf masks, extracts location of rogues, writes a dataframe to a gzip file."""
import concurrent.futures as cf
import json
import os
from collections import OrderedDict
from multiprocessing import cpu_count
from pathlib import Path

import click
import pandas as pd
from S3MP.global_config import S3MPConfig
from tqdm import tqdm

import analysis.utils.analysis_tools as tools

# a lot of things needed to be hard coded for this, so if you have a question
# go checkout AnalysisParameters first
from analysis.utils.analysis_params import AnalysisParameters

# Set global configuration
ROGUES_BUCKET_KEY = "sentera-rogues-data"
S3MPConfig.default_bucket_key = ROGUES_BUCKET_KEY
MIRROR_ROOT = "/".join(os.getcwd().split("/")[:2])  # /home/user/
MIRROR_ROOT = f"{MIRROR_ROOT}/s3_mirror/"
S3MPConfig.mirror_root = MIRROR_ROOT


@click.command()
@click.option(
    "-i",
    "--input_file",
    default=None,
    help="text file containing the paths produced by pull_inferenced_data.py",
)
@click.option(
    "--output_dir",
    default=None,
    help="User specified output directory, if not provided the program will attempt to determine the location.",
)
@click.option(
    "--threshold",
    default=[None],
    help="centroid threshold for removing leaves from neighboring rows",
    multiple=True,
)
@click.option(
    "--update_rogues",
    is_flag=True,
    default=False,
    show_default=True,
    help="update the measurements file with the new rogue labels",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="overwrite the rogues/semg files if they already exists",
)
@click.option(
    "--no_s3",
    is_flag=True,
    default=False,
    show_default=True,
    help="Allows user to prevent pushing to s3",
)
def run_analysis(
    input_file,
    output_dir,
    threshold,
    update_rogues,
    overwrite,
    no_s3,
):
    """
    Run the leaf measurements analysis.

    Options:
        --input_file: input JSON (usually made using scripts/build_s3_keys_json.py) that specifies which files to use.
        --output_dir: user specified output directory
        --threshold: centroid threshold for removing leaves from neighboring rows (left and right)
        --update_rogues: update the measurements file with the new rogue labels
        --overwrite: overwrites local files with new files from s3
        --no_s3: flag to write measurement files to s3
    """
    from S3MP.mirror_path import KeySegment, MirrorPath

    if not os.path.exists(input_file):
        raise RuntimeError(f"[ERROR] file does not exist: {input_file}")
    else:
        print(f"[INFO] input file: {input_file}")
    if not os.path.exists(output_dir):
        raise RuntimeError(f"[ERROR] directory does not exist: {output_dir}")
    else:
        print(f"[INFO] output directory: {output_dir}")
    try:
        if len(threshold) > 1:
            AnalysisParameters.CENTROID_THRESHOLD = [int(x) for x in threshold]
        elif len(threshold) == 1 and threshold[0] is not None:
            AnalysisParameters.CENTROID_THRESHOLD = [0, int(threshold[0])]
        elif len(threshold) == 1 and threshold[0] is None:
            print("[WARNING] no threshold provided, using default value")
        else:
            print("[ERROR] something went very wrong with the threshold")

        print(f"[INFO] centroid threshold: {AnalysisParameters.CENTROID_THRESHOLD}")

    except Exception:
        raise RuntimeError("[--threshold] Provided value could not be cast as int")

    if not overwrite:
        print("[WARNING] not overwriting local files with new files from s3")

    if no_s3:
        print("[WARNING] measurement files will not be pushed to s3")

    # grab all the files we're going to analyze
    files = json.load(open(input_file, "r"), object_pairs_hook=OrderedDict)
    # print(files)
    files = files[
        AnalysisParameters.JSON_PRIMARY_KEY
    ]  # files is now a list of dictionaries
    print(files)

    # check if the files are on s3 and download to mirror if they are
    for split in files:
        for key in split.keys():
            # check if the file exists
            if split[key] is None:
                continue
            split[key] = MirrorPath(
                [KeySegment(i, x) for i, x in enumerate(split[key].split("/"))],
            )
            if split[key].exists_on_s3():
                split[key].download_to_mirror(overwrite=overwrite)

    # check if update_rogues is set, if yes update rogues and exit
    if update_rogues:
        # check if measurements_df.gz exists
        if not os.path.exists(f"{dir}/measurements_df.gz"):
            raise RuntimeError("[--update_rogues] measurements_df.gz does not exist")
        # load measurements_df.gz
        df = pd.read_pickle(
            f"{dir}/measurements_df.gz", compression=AnalysisParameters.COMPRESSION_DICT
        )

        # loop over rogue jsons in files
        new_labels = []
        for split in files:
            if split[AnalysisParameters.JSON_ROGUE_KEY] is None:
                continue
            json_dict = json.load(
                open(split[AnalysisParameters.JSON_ROGUE_KEY].local_path, "rb")
            )
            for area_id in json_dict:
                if area_id in ["name", "datetime", "version"]:
                    continue
                rogue = (
                    json_dict[area_id][AnalysisParameters.PLANT_TYPE]
                    if AnalysisParameters.STAKE_COLOR not in json_dict[area_id]
                    else f"{json_dict[area_id][AnalysisParameters.PLANT_TYPE]} {json_dict[area_id][AnalysisParameters.STAKE_COLOR]}"
                )
                new_labels.append(
                    [
                        split[AnalysisParameters.JSON_ROGUE_KEY].local_path,
                        area_id,
                        rogue,
                    ]
                )

        for key in files:
            for f in files[key]:
                if AnalysisParameters.ROGUE in f:
                    # this is a special file type
                    json_dict = json.load(open(f"{key}/{f}", "rb"))
                    for area_id in json_dict:
                        if area_id in ["name", "datetime", "version"]:
                            continue
                        rogue = (
                            json_dict[area_id][AnalysisParameters.PLANT_TYPE]
                            if AnalysisParameters.STAKE_COLOR not in json_dict[area_id]
                            else f"{json_dict[area_id][AnalysisParameters.PLANT_TYPE]} {json_dict[area_id][AnalysisParameters.STAKE_COLOR]}"
                        )
                        new_labels.append([key, area_id, rogue])

        # sort new_labels by key, then by area_id
        new_labels = sorted(new_labels, key=lambda x: (x[0], int(x[1])))

        # set rogue labels in df using key and area id
        for i in range(len(new_labels)):
            img_name = (
                f'{new_labels[i][0].split("/")[-1]}_{new_labels[i][1]}'  # DS XXX_XXX
            )
            df.loc[df["name"] == img_name, AnalysisParameters.ROGUE] = new_labels[i][2]

        # write df to measurements_df.gz
        df.to_pickle(
            f"{dir}/measurements_df.gz", compression=AnalysisParameters.COMPRESSION_DICT
        )

        return

    rows = []
    masks_by_image_id = {}
    segm_key = None
    rogue_file_names = [
        split[AnalysisParameters.JSON_ROGUE_KEY].local_path
        for split in files
        if split[AnalysisParameters.JSON_ROGUE_KEY] is not None
    ]
    print("[INFO] preprocessing ...")
    for split in files:
        json_dict = {}
        for f in split.keys():  # loop over jsons in sub-directory
            # this is a segmentation file, must be opened in binary mode
            if split[f] is None:
                continue
            json_dict[f] = json.load(
                open(split[f].local_path, "rb" if AnalysisParameters.SEGM in f else "r")
            )

        # create ids and sort
        segm_key = (
            AnalysisParameters.JSON_SEGM_KEY
            if AnalysisParameters.JSON_SEGM_KEY in json_dict
            else AnalysisParameters.JSON_SEGM_NADIR_KEY
        )

        assume_ds_000 = False
        for m in json_dict[segm_key]:
            if m[AnalysisParameters.SCORE] < AnalysisParameters.SCORE_THRESHOLD:
                continue
            # id_m format is DS_XXX_XXX % (SplitNum, ImageNum)
            try:
                ds_num = [
                    x for x in str(split[segm_key].local_path).split("/") if "DS" in x
                ][1]
            except Exception as e:
                if not assume_ds_000:
                    assume_ds_000 = True
                    print(f"[WARNING] caught exception: {e}")
                    json_local_path = split[segm_key].local_path
                    print(f"[WARNING] {json_local_path} not in DS format")
                    print("[WARNING] continuing with assumed DS 000")
                ds_num = "DS 000"

            img_id = str(m[AnalysisParameters.IMG_ID]).zfill(3)
            id_m = f"{ds_num}_{img_id}"
            if id_m not in masks_by_image_id:
                area_id = tools.get_id(
                    id_m,
                    json_dict[AnalysisParameters.JSON_ROGUE_KEY]
                    if AnalysisParameters.JSON_ROGUE_KEY in json_dict
                    else None,
                )
                try:
                    if "-" not in area_id:
                        int(area_id)
                except Exception:
                    # caught an entry that isn't in the 000 format
                    # probably part of the creation tag
                    print(f"[WARNING] {area_id} not a valid area ID for {id_m}")
                    continue
                masks_by_image_id[id_m] = {
                    AnalysisParameters.MASKS: [
                        [],
                        [],
                    ],  # [0] is the nadir masks, [1] is the oblique masks
                    AnalysisParameters.SCORES: [
                        [],
                        [],
                    ],  # [0] is the nadir scores, [1] is the oblique scores
                    AnalysisParameters.ROGUE: json_dict[
                        AnalysisParameters.JSON_ROGUE_KEY
                    ][area_id][AnalysisParameters.PLANT_TYPE]
                    if AnalysisParameters.JSON_ROGUE_KEY in json_dict
                    else AnalysisParameters.NOT_LABELED,
                    AnalysisParameters.IMG_ID: area_id,
                }
                # check if stake color available
            masks_by_image_id[id_m][AnalysisParameters.MASKS][0].append(
                m[AnalysisParameters.SEGMENTATION]
            )
            masks_by_image_id[id_m][AnalysisParameters.SCORES][0].append(
                m[AnalysisParameters.SCORE]
            )

        if AnalysisParameters.JSON_SEGM_OBLIQUE_KEY in json_dict:
            for m in json_dict[AnalysisParameters.JSON_SEGM_OBLIQUE_KEY]:
                if m[AnalysisParameters.SCORE] < AnalysisParameters.SCORE_THRESHOLD:
                    continue
                ds_num = [
                    x
                    for x in str(
                        split[AnalysisParameters.JSON_SEGM_OBLIQUE_KEY].local_path
                    ).split("/")
                    if "DS" in x
                ][1]
                id_m = f"{ds_num}_{str(m[AnalysisParameters.IMG_ID]).zfill(3)}"
                if id_m not in masks_by_image_id:
                    print(f"[WARNING] {id_m} not in masks_by_image_id")
                    continue
                masks_by_image_id[id_m][AnalysisParameters.MASKS][1].append(
                    m[AnalysisParameters.SEGMENTATION]
                )
                masks_by_image_id[id_m][AnalysisParameters.SCORES][1].append(
                    m[AnalysisParameters.SCORE]
                )

    # run nms on the nadir and oblique masks
    print("[INFO] running nms ...")

    tools.nms(
        masks_by_image_id["DS 000_096"][AnalysisParameters.MASKS][0],
        masks_by_image_id["DS 000_096"][AnalysisParameters.SCORES][0],
    )
    tools.nms(
        masks_by_image_id["DS 000_097"][AnalysisParameters.MASKS][0],
        masks_by_image_id["DS 000_097"][AnalysisParameters.SCORES][0],
    )

    futures_nadir = {}
    futures_oblique = {}
    with cf.ProcessPoolExecutor() as executor:
        for idx in tqdm(masks_by_image_id):
            futures_nadir[idx] = executor.submit(
                tools.nms,
                masks_by_image_id[idx][AnalysisParameters.MASKS][0],
                masks_by_image_id[idx][AnalysisParameters.SCORES][0],
            )
            futures_oblique[idx] = executor.submit(
                tools.nms,
                masks_by_image_id[idx][AnalysisParameters.MASKS][1],
                masks_by_image_id[idx][AnalysisParameters.SCORES][1],
            )

    for idx in tqdm(masks_by_image_id):
        masks_by_image_id[idx][AnalysisParameters.MASKS][0] = futures_nadir[
            idx
        ].result()
        masks_by_image_id[idx][AnalysisParameters.MASKS][1] = futures_oblique[
            idx
        ].result()

    if len(threshold) == 0:
        # drop all images not in DS 000
        masks_by_image_id = {
            x: masks_by_image_id[x] for x in masks_by_image_id if "DS 000" in x
        }

    print(f"[INFO] found {len(masks_by_image_id)} images to analyze ...")
    print(f"[INFO] found {len(rogue_file_names)} rogue files ...")

    # analyze images by id, in parallel
    time_to_finish_in_seconds = (1 / 1.48) * sum(
        [len(masks_by_image_id[x][AnalysisParameters.MASKS]) for x in masks_by_image_id]
    )
    time_to_finish_in_seconds /= cpu_count()
    if not threshold[0]:
        print("[INFO] processing only the first DS split for thresholding")
        print(
            "[INFO] to determine the threshold run scripts/analyze_data.py with the --threshold option"
        )
        print("[INFO] ...now on to the show ...")
    print("[INFO] queueing jobs ...")
    print("[INFO] plan on up to 3 minutes per DS split ...")
    print(
        f"[INFO] with {len(files)} split(s), execution time is at most ~{3*len(files):.0f} minutes ..."
    )
    with cf.ProcessPoolExecutor() as executor:  # run all the sub-directories in parallel
        for idx in masks_by_image_id:
            pf = executor.submit(
                tools.get_measurements, masks_by_image_id[idx][AnalysisParameters.MASKS]
            )
            # collect the futures
            rows.append(
                [
                    idx,  # name
                    masks_by_image_id[idx][
                        AnalysisParameters.IMG_ID
                    ],  # for pulling images later on
                    pf,  # future: pf.result() contains lengths, widths, areas
                    masks_by_image_id[idx][AnalysisParameters.ROGUE],  # rogue
                ]
            )

    print()
    # take all the images we just analyzed and put them in a dataframe, 1 row per image

    if output_dir is None:
        folders = str(files[0][segm_key].local_path).split("/")
        base_local_path = "/".join(folders[: folders.index("DS Splits")])
    else:
        base_local_path = output_dir

    print("[INFO] jobs done ... building dataframes")
    df_out_unstaked = (
        pd.DataFrame(columns=AnalysisParameters.ANALYSIS_FEATURES)
        if segm_key == AnalysisParameters.SEGM
        else pd.DataFrame(columns=AnalysisParameters.ANALYSIS_FEATURES_OBLIQUE)
    )
    df_out_staked = (
        pd.DataFrame(columns=AnalysisParameters.ANALYSIS_FEATURES)
        if segm_key == AnalysisParameters.SEGM
        else pd.DataFrame(columns=AnalysisParameters.ANALYSIS_FEATURES_OBLIQUE)
    )

    while rows:
        r = rows.pop(0)
        future = r[2]
        if future.exception() is None:
            new_row = [r[0], r[1], *future.result(), r[3]]
            try:
                # check if the image is in a staked split
                rogue_file_name = [
                    x for x in rogue_file_names if r[0].split("_")[0] in str(x)
                ]
                if rogue_file_name and "staked" in str(rogue_file_name[0]):
                    # this image is in a staked split
                    df_out_staked.loc[len(df_out_staked.index)] = new_row
                else:
                    # this image is in an unstaked split
                    df_out_unstaked.loc[len(df_out_unstaked.index)] = new_row
            except Exception as e:
                print(f"Image {r[0]} failed for some reason:")
                print(r[0])
                print(r[1])
                print(*future.result())
                print(r[3])
                print(e)
                exit()
        else:
            print(f"Image {r[0]} failed with exception:")
            future.result()

    # write the DataFrame to pickle, gzip for compression
    print(f"[INFO] writing dataframes to {base_local_path}")
    df_out_staked.to_pickle(
        f"{base_local_path}/staked_measurements_df.gz",
        compression=AnalysisParameters.COMPRESSION_DICT,
    )
    df_out_unstaked.to_pickle(
        f"{base_local_path}/unstaked_measurements_df.gz",
        compression=AnalysisParameters.COMPRESSION_DICT,
    )

    if not no_s3:  # I dislike double negatives, but it had to be done
        mp = MirrorPath.from_local_path(
            Path(f"{base_local_path}/staked_measurements_df.gz")
        )
        mp.upload_from_mirror(overwrite=True)
        mp = MirrorPath.from_local_path(
            Path(f"{base_local_path}/unstaked_measurements_df.gz")
        )
        mp.upload_from_mirror(overwrite=True)


if __name__ == "__main__":
    run_analysis()
