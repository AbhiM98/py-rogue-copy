"""Lambda handlers for ECS processes."""
import time

from utils.ecs_utils import ecs_lambda_handler_block
from utils.lambda_utils import invoke_lambda

N_DS_SPLITS_PER_LAMBDA = 4


def generate_ds_splits(event, context):
    """Extract frames ECS lambda handler."""
    return ecs_lambda_handler_block(event, "GENERATE DS SPLITS")


def extract_frames(event, context):
    """Extract frames ECS lambda handler."""
    return ecs_lambda_handler_block(event, "EXTRACT FRAMES")


def run_all(event, context):
    """Extract frames ECS lambda handler."""
    return ecs_lambda_handler_block(event, "RUN ALL")


def run_paddle_inference(event, context):
    """Extract frames ECS lambda handler."""
    return ecs_lambda_handler_block(event, "RUN PADDLE INFERENCE")


def prep_and_run_paddle_inference(event, context):
    """Extract frames ECS lambda handler."""
    return ecs_lambda_handler_block(event, "PREP AND RUN PADDLE INFERENCE")


def batch_run_processing_step(event, context):
    """Extract frames ECS lambda handler.

    event["row_numbers"] = ["1", "2", ...]
    """
    processing_step = event["processing_step"]
    row_numbers = event["row_numbers"]
    n_invokes_since_last_sleep = 0
    for row_number in row_numbers:
        params = event
        params["row_number"] = str(row_number)
        print(
            f"Invoking {processing_step} for {params['field_name']} row {row_number} on {params['date']}"
        )
        invoke_lambda(processing_step, params, run_async=True)
        if "delay" in event:
            if "batch_size" in event:
                n_invokes_since_last_sleep += 1
                if n_invokes_since_last_sleep >= int(event["batch_size"]):
                    n_invokes_since_last_sleep = 0
                    # Sleep for the specified delay
                    time.sleep(int(event["delay"]))
            else:
                # Sleep for the specified delay
                time.sleep(int(event["delay"]))


def batch_run_prep_and_run_paddle_inference(event, context):
    """Extract frames ECS lambda handler.

    event["rows"] = {row_number: [ds_split_number1, ds_split_number2, ...]}, ie {"1": [1, 2, 3], "2": [1, 2, 3], ...}
    """
    rows = event["rows"]
    n_invokes_since_last_sleep = 0
    for row_number, ds_split_numbers in rows.items():
        params = event
        params["row_number"] = row_number
        params["ds_split_numbers"] = ",".join(
            [str(ds_split_number) for ds_split_number in ds_split_numbers]
        )
        print(
            f"Invoking paddle prep and inference lambda for row {row_number}, DS Splits {ds_split_numbers}"
        )
        invoke_lambda("prep-and-run-paddle-inference", params, run_async=True)
        # Sleep for the specified delay
        if "delay" in event:
            if "batch_size" in event:
                n_invokes_since_last_sleep += 1
                if n_invokes_since_last_sleep >= int(event["batch_size"]):
                    n_invokes_since_last_sleep = 0
                    # Sleep for the specified delay
                    time.sleep(int(event["delay"]))
            else:
                # Sleep for the specified delay
                time.sleep(int(event["delay"]))


if __name__ == "__main__":
    # NOTE: `rerun` will reset process flags and allow you to rerun a process
    # NOTE: `overwrite` will overwrite existing files if "True", otherwise will skip

    # WATERMAN
    event = {
        "field_name": "Waterman_Strip_Trial",
        "date": "2023-06-20",
        # "processing_step": "extract-frames",
        # "row_numbers": [1,3,9],
        "rows": {
            "1": [],
            # "2": [],
            "3": [],
            # "4": [],
            # "5": [],
            # "6": [],
            # "7": [],
            # "8": [],
            "9": [],
            # "10": [],
            # "11": [],
            # "12": [],
            # "13": [],
            # "14": [],
            # "15": [],
            # "16": [],
            # "17": [],
            # "18": [],
            # "19": [],
            # "20": [],
            # "21": [],
            # "22": [],
            # "23": [],
            # "24": [],
            # "25": [],
            # "26": [],
            # "27": [],
            # "28": [],
            # "29": [],
            # "30": [],
        },
        "nadir_crop_height": "1600",
        "nadir_crop_width": "1000",
        "overwrite": "True",
        "rerun": "True",
        # "delay": "1",
    }

    # Williamsburg
    # event = {
    #     "field_name": "Williamsburg_Strip_Trial",
    #     "date": "2023-06-21",
    #     # "processing_step": "extract-frames",
    #     # "row_numbers": [1,3,9],
    #     "rows": {
    #         "1": [],
    #         # "2": [],
    #         "3": [],
    #         # "4": [],
    #         # "5": [],
    #         # "6": [],
    #         # "7": [],
    #         # "8": [],
    #         "9": [],
    #         # "10": [],
    #         # "11": [],
    #         # "12": [],
    #         # "13": [],
    #         # "14": [],
    #         # "15": [],
    #         # "16": [],
    #         # "17": [],
    #         # "18": [],
    #         # "19": [],
    #         # "20": [],
    #         # "21": [],
    #         # "22": [],
    #         # "23": [],
    #         # "24": [],
    #         # "25": [],
    #         # "26": [],
    #         # "27": [],
    #         # "28": [],
    #         # "29": [],
    #         # "30": [],
    #     },
    #     "nadir_crop_height": "1600",
    #     "nadir_crop_width": "1000",
    #     "overwrite": "True",
    #     "rerun": "True",
    #     # "delay": "30",
    # }

    # batch_run_processing_step(event, None)
    batch_run_prep_and_run_paddle_inference(event, None)
