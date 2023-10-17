"""Builds json file with list of paths to segmentation and rogues files on S3."""
import json
import os

import click
from S3MP.global_config import S3MPConfig
from S3MP.mirror_path import KeySegment, get_matching_s3_mirror_paths

from analysis.utils.analysis_params import S3Paths
from analysis.utils.analysis_tools import build_segments


@click.command()
@click.option(
    "--field", default=None, required=True, help="field [PF1, FF1, FF2, FC22]"
)
@click.option(
    "--subfield", default=None, help="subfield (small plot, strip trial) [SP, ST]"
)
@click.option("--planting", default=None, help="planting 1 or planting 2 [P1, P2]")
@click.option("--date", default=None, help="date in format M-DD")
@click.option("--row", default=None, help="row")
@click.option("--row_pass", default=None, help="pass [A, B]")
@click.option("--output", default=None, required=True, help="string to name json file")
@click.option("--year", default=None, help="year [22, 23]")
def build_s3_keys_json(
    field,
    subfield,
    planting,
    date,
    row,
    row_pass,
    output,
    year,
):
    """
    Take in some information from the user like field, date, and row in order to build the s3 keys json used to pull the inferenced data from s3.

    Options:
    field: field [PF1, FF1, FF2, FC22],
    subfield: subfield (small plots or strip trial) [SP, ST]
    planting: 1 or 2
    date: M-DD
    row: the row you'd like to specify
    row_pass: A or B
    """
    # field = field.upper() if field is not None else field
    row_pass = row_pass.upper() if row_pass is not None else row_pass

    # option handling
    if field not in S3Paths.fields:
        raise RuntimeError(
            f"[ERROR] field {field} not configured\nAvailable fields are {[x for x in S3Paths.fields.keys()]}"
        )

    if subfield is not None and subfield not in S3Paths.subfields:
        raise RuntimeError(
            f"[ERROR] subfield {subfield} not configured\nAvailable fields are {S3Paths.subfields}"
        )

    if planting is not None and planting not in S3Paths.planting:
        raise RuntimeError(
            f"[ERROR] planting {planting} not configured\nAvailable plantings are {S3Paths.planting}"
        )

    if date is not None and "-" not in date:
        raise RuntimeError(f"[ERROR] date {date} not in M-DD format")

    if row_pass is not None and row_pass not in S3Paths.passes:
        raise RuntimeError(
            f"[ERROR] pass {row_pass} not configured\nAvailable passes are {S3Paths.passes}"
        )

    if year is not None and year not in S3Paths.years:
        raise RuntimeError(
            f"[ERROR] year {year} not configured\nAvailable years are {S3Paths.years}"
        )

    # build inference and video segments
    if year != "23":
        inference_segments = build_segments(
            field, subfield, planting, date, row, row_pass, inference_type="inference"
        )
        inference_mps = get_matching_s3_mirror_paths(inference_segments)
        print(f"[INFO] found {len(inference_mps)} matching inference paths")
        if len(inference_mps) == 0:
            print("[INFO] no inference paths found, trying again")
            inference_segments = build_segments(
                field,
                subfield,
                planting,
                date,
                row,
                row_pass,
                inference_type="inference",
                try_next_depth=True,
            )
            inference_mps = get_matching_s3_mirror_paths(inference_segments)
            print(f"[INFO] found {len(inference_mps)} matching inference paths")
            if len(inference_mps) == 0:
                print("[ERROR] no inference paths found, exiting")
                return

        inference_mps_nadir = []
        inference_mps_oblique = []
        if any("nadir" in str(x.s3_key) for x in inference_mps):
            inference_mps_nadir = [
                x for x in inference_mps if "segm-nadir" in str(x.s3_key)
            ]
            inference_mps_oblique = [
                x for x in inference_mps if "oblique" in str(x.s3_key)
            ]

        if len(inference_mps_nadir) > 0:
            print(f"[INFO] found {len(inference_mps_nadir)} nadir inference paths")
            print(f"[INFO] found {len(inference_mps_oblique)} oblique inference paths")

        video_segments = build_segments(
            field, subfield, planting, date, row, row_pass, inference_type="video"
        )
        video_mps = get_matching_s3_mirror_paths(video_segments)
        duplicates_loc = [
            i for i, x in enumerate(video_mps) if "duplicates" in x.s3_key
        ]
        if len(duplicates_loc) > 0:
            print(f"[INFO] removing {len(duplicates_loc)} duplicate video paths")
            for i in duplicates_loc[::-1]:
                del video_mps[i]

        print(f"[INFO] found {len(video_mps)} matching video paths")
        if len(video_mps) == 0:
            print(video_segments)

        if len(inference_mps) != len(video_mps):
            # align the entries
            # this will almost always be an issue where video_mps < inference_mps
            if len(inference_mps_oblique) > 0:
                inference_mps_nadir, inference_mps_oblique = align_mps(
                    inference_mps_nadir, inference_mps_oblique, both_inf=True
                )
                _, video_mps = align_mps(inference_mps_nadir, video_mps)
                _, video_mps = align_mps(inference_mps_oblique, video_mps)
            else:
                inference_mps, video_mps = align_mps(inference_mps, video_mps)

        # load it all into a json and write it to file
        json_dict = {"s3_keys_by_plot": []}
        if len(inference_mps_oblique) > 0:
            for x, y, z in zip(inference_mps_nadir, inference_mps_oblique, video_mps):
                json_dict["s3_keys_by_plot"].append(
                    {
                        "segm_s3_key_nadir": x.s3_key if x is not None else x,
                        "segm_s3_key_oblique": y.s3_key if y is not None else y,
                        "rogue_label_s3_key": z.s3_key if z is not None else z,
                    }
                )
        else:
            for x, y in zip(inference_mps, video_mps):
                json_dict["s3_keys_by_plot"].append(
                    {
                        "segm_s3_key": x.s3_key if x is not None else x,
                        "rogue_label_s3_key": y.s3_key if y is not None else y,
                    }
                )

        if not os.path.exists("jsons"):
            os.makedirs("jsons")
        with open(f"jsons/{output}_s3_keys.json", "w") as f:
            json.dump(json_dict, f, indent=4)

        print(f"[INFO] keys written to jsons/{output}_s3_keys.json")
    else:
        # 2023 data has a much more consistent structure
        # TODO: use grd database (ddb_tracking)
        inference_segments = [
            KeySegment(0, "2023-field-data"),
            KeySegment(1, field),
            KeySegment(2, date),
            KeySegment(3, row),
            KeySegment(11, incomplete_name="json", is_file=True),
        ]

        inference_mps = get_matching_s3_mirror_paths(inference_segments)
        print(f"[INFO] found {len(inference_mps)} matching inference paths")

        video_segments = [
            KeySegment(0, "2023-field-data"),
            KeySegment(1, field),
            KeySegment(2, date),
            KeySegment(3, row),
            KeySegment(7, incomplete_name="rogues.json", is_file=True),
        ]

        video_mps = get_matching_s3_mirror_paths(video_segments)
        print(f"[INFO] found {len(video_mps)} matching video paths")
        print(video_mps)

        max_timestamp = max(
            [int(x.s3_key.split("/")[4].split("_")[0]) for x in inference_mps]
        )
        inference_mps = [x for x in inference_mps if str(max_timestamp) in x.s3_key]
        video_mps = [x for x in video_mps if str(max_timestamp) in x.s3_key]

        # there will always be both oblique and nadir inference paths
        inference_mps_nadir = [x for x in inference_mps if "nadir" in str(x.s3_key)]
        inference_mps_oblique = [x for x in inference_mps if "oblique" in str(x.s3_key)]

        if len(inference_mps_nadir) > 0:
            print(f"[INFO] found {len(inference_mps_nadir)} nadir inference paths")
            print(f"[INFO] found {len(inference_mps_oblique)} oblique inference paths")

        if len(inference_mps) != len(video_mps):
            # align the entries
            # this will almost always be an issue where video_mps < inference_mps
            inference_mps_nadir, inference_mps_oblique = align_mps_23(
                inference_mps_nadir, inference_mps_oblique, both_inf=True
            )
            _, video_mps = align_mps_23(inference_mps_nadir, video_mps)
            _, video_mps = align_mps_23(inference_mps_oblique, video_mps)

        print(video_mps)
        # load it all into a json and write it to file
        json_dict = {"s3_keys_by_plot": []}
        for x, y, z in zip(inference_mps_nadir, inference_mps_oblique, video_mps):
            json_dict["s3_keys_by_plot"].append(
                {
                    "segm_s3_key_nadir": x.s3_key if x is not None else x,
                    "segm_s3_key_oblique": y.s3_key if y is not None else y,
                    "rogue_label_s3_key": z.s3_key if z is not None else z,
                }
            )

        if not os.path.exists("jsons"):
            os.makedirs("jsons")
        with open(f"jsons/{output}_s3_keys.json", "w") as f:
            json.dump(json_dict, f, indent=4)

        print(f"[INFO] keys written to jsons/{output}_s3_keys.json")


def align_mps_23(inf_mps, vid_mps, both_inf=False):
    """Match keys and extend shorter list to match longer list."""
    inf_key = -6
    vid_key = inf_key if both_inf else -2

    inf_mp_ds = [x.s3_key.split("/")[inf_key] if x else None for x in inf_mps]
    vid_mp_ds = [x.s3_key.split("/")[vid_key] if x else None for x in vid_mps]

    inf_mp_ds = [int(x.split(" ")[1]) if x else None for x in inf_mp_ds]
    vid_mp_ds = [int(x.split(" ")[1]) if x else None for x in vid_mp_ds]

    max_val = max(
        max([x for x in inf_mp_ds if x is not None]),
        max([x for x in vid_mp_ds if x is not None]),
    )
    min_val = min(
        min([x for x in inf_mp_ds if x is not None]),
        min([x for x in vid_mp_ds if x is not None]),
    )

    ret_inf_mps = [None for _ in range(min_val, max_val + 1)]
    ret_vid_mps = [None for _ in range(min_val, max_val + 1)]
    for i in range(min_val, max_val + 1):
        if i in inf_mp_ds:
            ret_inf_mps[i - min_val] = inf_mps[inf_mp_ds.index(i)]
        if i in vid_mp_ds:
            ret_vid_mps[i - min_val] = vid_mps[vid_mp_ds.index(i)]

    return ret_inf_mps, ret_vid_mps


def align_mps(inf_mps, vid_mps, both_inf=False):
    """Match keys and extend shorter list to match longer list."""
    inf_key = -3
    vid_key = -3 if both_inf else -2
    is_small_plot = any(
        [
            x.s3_key.split("/")[inf_key - 1] in ["Rel Plots", "Hybrid Plots"]
            for x in inf_mps
            if x is not None
        ]
        + [
            x.s3_key.split("/")[inf_key - 1] in ["Rel Plots", "Hybrid Plots"]
            for x in vid_mps
            if x is not None
        ]
    )
    if is_small_plot and not any([x is None for x in vid_mps]):
        # match folder to Hybrid or Rel
        vid_mps = [
            x for x in vid_mps if x.s3_key.split("/")[vid_key - 1] in inf_mps[0].s3_key
        ]

    inf_mp_ds = [x.s3_key.split("/")[inf_key] for x in inf_mps if x is not None]
    vid_mp_ds = [x.s3_key.split("/")[vid_key] for x in vid_mps if x is not None]
    if is_small_plot:
        inf_mp_ds = [int(x) for x in inf_mp_ds]
        vid_mp_ds = [int(x) for x in vid_mp_ds]
    else:
        inf_mp_ds = [int(x.split(" ")[1]) for x in inf_mp_ds]
        vid_mp_ds = [int(x.split(" ")[1]) for x in vid_mp_ds]

    max_val = max(max(inf_mp_ds), max(vid_mp_ds))
    min_val = min(min(inf_mp_ds), min(vid_mp_ds))
    for i in range(min_val, max_val + 1):
        if i not in inf_mp_ds:
            inf_mps.insert(i, None)
        if i not in vid_mp_ds:
            vid_mps.insert(i, None)

    return inf_mps, vid_mps


if __name__ == "__main__":
    S3MPConfig.default_bucket_key = "sentera-rogues-data"
    build_s3_keys_json()
