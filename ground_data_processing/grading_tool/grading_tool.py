"""Tool to trade analysts labels against 'true' labels."""
import json

import click
import pandas as pd
from S3MP.mirror_path import KeySegment, MirrorPath, get_matching_s3_mirror_paths


def apply_tolerance(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate through the dataframe and check the analysts labels.

    Allowing a tolerance of 1 plant in either direction. Also check for unusual labels
    like delays or others and don't hold incorrect answers against the
    analyst.
    """
    for i in range(len(dataframe)):
        if dataframe.loc[i, "pass"]:
            continue
        # check for unusual labels
        if dataframe.loc[i, "key"] in ["Delay Rogue", "Other"]:
            dataframe.loc[i, "pass"] = True
            continue

        if dataframe.loc[i, "label"] in ["Hybrid Rogue", "Hybrid Low Rogue"]:
            # look one ahead and behind in analyst labels

            if i > 0 and i < len(dataframe) - 1:
                if dataframe.loc[i - 1, "key"] in ["Hybrid Rogue", "Hybrid Low Rogue"]:
                    dataframe.loc[i, "pass"] = True
                    dataframe.loc[i - 1, "pass"] = True
                    continue
                if dataframe.loc[i + 1, "key"] in ["Hybrid Rogue", "Hybrid Low Rogue"]:
                    dataframe.loc[i, "pass"] = True
                    dataframe.loc[i + 1, "pass"] = True
                    continue

    # drop all rows where both "label" and "key" are both "Normal"
    dataframe = dataframe.drop(
        dataframe[
            (dataframe["label"] == "Normal") & (dataframe["key"] == "Normal")
        ].index
    )

    # drop all rows were "pass" is True
    dataframe = dataframe.drop(dataframe[dataframe["pass"] is True].index)

    return dataframe


@click.command()
@click.option(
    "--name", default="test", help="Name of the person whose labels are being graded."
)
def grade_labels(
    name: str,
) -> None:
    """
    Take in the analysts name and create a csv file comparing the analyst's labels against the 'true' labels.

    Args:
    name: Name of the person whose labels are being graded.

    Returns:
    None
    """
    # get the 'true' labels
    true_segments = [
        KeySegment(0, "analyst-training-data"),
        KeySegment(1, "labelled-data"),
        KeySegment(3, incomplete_name=".json", is_file=True),
    ]
    true_labels = get_matching_s3_mirror_paths(true_segments)

    # get the analysts labels
    analyst_segments = [
        KeySegment(0, "analyst-training-data"),
        KeySegment(1, f"{name}-labels"),
        KeySegment(3, incomplete_name=".json", is_file=True),
    ]
    analyst_labels = get_matching_s3_mirror_paths(analyst_segments)

    [x.download_to_mirror() for x in true_labels]
    [x.download_to_mirror() for x in analyst_labels]

    # check that there are labels to grade
    if len(analyst_labels) == 0:
        print(f"No labels found for {name}.")
        return

    # check that all analyst labels are present
    if len(analyst_labels) != len(true_labels):
        print(f"Missing labels for {name}.")
        print("Only found the following labels:")
        print(*[x.s3_key for x in analyst_labels], sep="\n")
        return

    dataframe_cols = ["split", "image", "key", "label", "pass"]
    dataframe = pd.DataFrame(columns=dataframe_cols)

    # iterate through the labels and compare them
    for true_label, analyst_label in zip(true_labels, analyst_labels):
        # load the jsons into dictionaries
        true_dict = {}
        analyst_dict = {}
        with open(true_label.local_path, "r") as f:
            true_dict = json.load(f)
        with open(analyst_label.local_path, "r") as f:
            analyst_dict = json.load(f)

        ds_split = true_label.s3_key.split("/")[-2]
        image_names = [
            x for x in true_dict.keys() if x not in ["name", "version", "datetime"]
        ]

        for image_name in image_names:
            dataframe.loc[len(dataframe)] = [
                ds_split,
                image_name,
                true_dict[image_name]["plant_type"],
                analyst_dict[image_name]["plant_type"],
                true_dict[image_name]["plant_type"]
                == analyst_dict[image_name]["plant_type"],
            ]

    # clean the dataframe a bit
    dataframe = apply_tolerance(dataframe)

    save_loc = str(analyst_labels[0].local_path).split("/")[:-2]
    save_loc = "/".join(save_loc) + "/" + name + "_grading.csv"
    dataframe.to_csv(save_loc, index=False)

    # put the csv in the analysts s3 bucket
    csv_mp = MirrorPath(
        [
            KeySegment(0, "analyst-training-data"),
            KeySegment(1, f"{name}-labels"),
            KeySegment(2, f"{name}_grading.csv", is_file=True),
        ]
    )
    csv_mp.upload_from_mirror()
    print(f"Grading for {name} complete.")
    print(f"Grading csv saved to {csv_mp.s3_key}")


if __name__ == "__main__":
    grade_labels()
