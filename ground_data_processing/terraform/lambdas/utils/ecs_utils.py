"""Utils for ECS lambdas."""
import json
import os
import time
from typing import List

import boto3
from utils.lambda_utils import get_return_block_with_cors
from utils.sns_utils import ROGUES_LAUNCH_FAILURE, publish_message_to_name

AUTOSCALING_GROUP_NAME = "prod-rogues-ecs-cluster-one-20230530192028803000000008"
AUTOSCALING_STARTING_DESIRED_CAPACITY = 2  # When starting a new task when there are no tasks currently running, set the desired capacity to this
AUTOSCALING_MAX_CAPACITY = 40
CLUSTER_NAME = "prod-rogues-ecs-cluster"
RUN_ALL_JOBS_PER_INSTANCE = 2

ParamAppenders = {
    "field_name": {
        "name": "FIELD_NAME",
        "value": None,
    },
    "date": {
        "name": "DATE",
        "value": None,
    },
    "row_number": {
        "name": "ROW_NUMBER",
        "value": None,
    },
    "overwrite": {
        "name": "OVERWRITE",
        "value": None,
    },
    "rerun": {
        "name": "RERUN",
        "value": None,
    },
    "ds_split_numbers": {
        "name": "DS_SPLIT_NUMBERS",
        "value": None,
    },
    "nadir_crop_height": {
        "name": "NADIR_CROP_HEIGHT",
        "value": None,
    },
    "nadir_crop_width": {
        "name": "NADIR_CROP_WIDTH",
        "value": None,
    },
}


def ecs_lambda_handler_block(event, task_str):
    """Block for all ECS lambdas."""
    # When a ecs lambda is called, ensure that the autoscaling group has a non-zero max capacity
    check_and_update_autoscaling_capacity()

    try:
        body = json.loads(event["body"])
    except Exception:
        body = event

    print(body)
    task_params = get_rogues_ecs_cluster_task_params()
    print(task_params)

    task_params = add_additional_task_params(task_params, body)

    ecs_client = boto3.client("ecs")
    ec2_fail = False
    n_retries = 0
    max_retries = 5
    timeout_per_retry = 120  # Seconds

    try:
        response = ecs_client.run_task(**task_params)
        print(response)
    except Exception as e:
        print(e)
        ec2_fail = True

    if ec2_fail or len(response["failures"]) > 0:
        print("EC2 Task failed to launch, retrying...")
        ec2_fail = (
            True  # Mark fail flag for the case when len(response["failures"]) > 0
        )
        n_retries = 1
        while ec2_fail and n_retries <= max_retries:
            # Sleep for two minutes to give autoscaling some time
            # Max timeout for lambda is 15 minutes, so we can retry a few times
            time.sleep(timeout_per_retry)
            print(f"Retry {n_retries}/{max_retries}...")
            try:
                response = ecs_client.run_task(**task_params)
                print(response)

                if len(response["failures"]) > 0:
                    n_retries += 1
                else:
                    # If successful, break out of loop
                    ec2_fail = False
            except Exception as e:
                print(e)
                n_retries += 1

        # If still failing, publish message to SNS
        if ec2_fail:
            # Launch on FARGATE
            print("EC2 Task failed to launch, launching on FARGATE...")
            publish_message_to_name(
                ROGUES_LAUNCH_FAILURE,
                f"ROGUES {task_str} EC2 LAUNCH FAILED, LAUNCHING ON FARGATE: {json.dumps(body)}",
            )
            task_params = get_rogues_ecs_cluster_task_params(launch_type="FARGATE")
            task_params = add_additional_task_params(task_params, body)
            response = ecs_client.run_task(**task_params)
            print(response)

            if len(response["failures"]) > 0:
                publish_message_to_name(
                    ROGUES_LAUNCH_FAILURE,
                    f"ROGUES {task_str} TASK FAILED TO LAUNCH: {json.dumps(body)}",
                )
            return

    # If successful, protect from autoscaling termination while task is pending
    # Be careful not to exceed 15 minute lambda function timeout
    # To be safe, we will suspend autoscaling termination for 10 minutes max
    timeout = 10 * 60 - (n_retries * timeout_per_retry)
    suspend_and_resume_autoscaling_termination(timeout)

    return get_return_block_with_cors(
        body=f"ROGUES {task_str} TASK STARTED", needs_encoding=False
    )


def build_default_ecs_entrypoint(script_name):
    """Build default ecs entrypoint."""
    return f"python ground_data_processing/processing_entrypoints/{script_name} --field_name $FIELD_NAME"


def add_additional_task_params(task_params: dict, add_params: dict):
    """Append necessary components."""
    for key in add_params.keys():
        if key in ParamAppenders.keys():
            appender = ParamAppenders[key]
            appender["value"] = add_params[key]
            task_params["overrides"]["containerOverrides"][0]["environment"].append(
                appender
            )
    return task_params


def get_rogues_ecs_cluster_task_params(
    launch_type=None,
):
    """Get ecs cluster task params."""
    if launch_type is None:
        launch_type = f"{os.environ['launchType']}"
    ret = {
        "taskDefinition": f"{os.environ['task_def_id']}:{os.environ['task_revision']}",
        "launchType": launch_type,
        "cluster": os.environ["ecs_cluster_name"],
        "count": 1,
        "networkConfiguration": {
            "awsvpcConfiguration": {
                "subnets": json.loads(os.environ["subnets"])["subnets"],
                "securityGroups": json.loads(os.environ["securityGroups"])[
                    "securityGroups"
                ],
            }
        },
        "overrides": {
            "containerOverrides": [
                {
                    "name": os.environ["task_name"],
                    "environment": [],
                }
            ]
        },
    }
    if launch_type == "FARGATE":
        ret["platformVersion"] = "LATEST"
    return ret


def suspend_and_resume_autoscaling_termination(timeout: int = 300):
    """Suspend and resume autoscaling termination."""
    autoscaling_client = boto3.client("autoscaling", region_name="us-east-1")
    autoscaling_client.suspend_processes(
        AutoScalingGroupName=AUTOSCALING_GROUP_NAME,
        ScalingProcesses=["Terminate"],
    )
    print(f"Suspended autoscaling termination for {timeout} seconds...")
    time.sleep(timeout)  # Seconds

    # Check if there are any pending tasks: if so, we can assume that another lambda
    # will take care of resuming autoscaling termination

    # Get the number of pending tasks
    n_pending = get_n_tasks_pending()
    if n_pending > 0:
        print(
            f"There are still {n_pending} pending tasks, not resuming autoscaling termination..."
        )
    else:
        autoscaling_client.resume_processes(
            AutoScalingGroupName=AUTOSCALING_GROUP_NAME,
            ScalingProcesses=["Terminate"],
        )
        print("No pending tasks, resumed autoscaling termination...")


def protect_tasks_from_scale_in(task_arns: List[str]):
    """Protect ecs task(s) from scale in."""
    # Cast to list if not already
    if not isinstance(task_arns, list):
        task_arns = [task_arns]
    ecs_client = boto3.client("ecs", region_name="us-east-1")
    res = ecs_client.update_task_protection(
        cluster=os.environ["ecs_cluster_name"],
        tasks=task_arns,
        protectionEnabled=True,
        expiresInMinutes=2880,  # 48 hours, which is the max
    )
    print(res)
    if "failures" in res.keys() and len(res["failures"]) > 0:
        print(f"Failed to protect task(s) {task_arns} from scale in...")
    else:
        print(f"Protected task(s) {task_arns} from scale in...")


def parse_response_and_get_task_arn(response: dict):
    """Parse response and get task arn."""
    task_arn = None
    if "tasks" in response.keys() and len(response["tasks"]) > 0:
        task_arn = response["tasks"][0]["containers"][0]["taskArn"]
    return task_arn


def get_n_tasks_running_or_pending(cluster_name: str = CLUSTER_NAME):
    """Get number of ECS tasks running or pending."""
    ecs_client = boto3.client("ecs", region_name="us-east-1")
    res = ecs_client.list_tasks(
        cluster=cluster_name,
        desiredStatus="RUNNING",
    )
    n_running = len(res["taskArns"])
    res = ecs_client.list_tasks(
        cluster=cluster_name,
        desiredStatus="PENDING",
    )
    n_pending = len(res["taskArns"])
    return n_running + n_pending


def get_n_tasks_pending(cluster_name: str = CLUSTER_NAME):
    """Get number of pending ECS tasks."""
    ecs_client = boto3.client("ecs", region_name="us-east-1")
    res = ecs_client.list_tasks(
        cluster=cluster_name,
        desiredStatus="PENDING",
    )
    return len(res["taskArns"])


def safely_set_scale_in_protection(
    protect_from_scale_in: bool,
    cluster_name: str = CLUSTER_NAME,
    auto_scaling_group_name: str = AUTOSCALING_GROUP_NAME,
):
    """Set the scale in protection on the host instance, taking care not to disable it if other tasks are running or pending."""
    try:
        # https://docs.aws.amazon.com/AmazonECS/latest/developerguide/container-metadata.html
        with open(os.environ["ECS_CONTAINER_METADATA_FILE"]) as f:
            ecs_metadata = json.load(f)
        # Get the container instance ARN
        container_instance_arn = ecs_metadata["ContainerInstanceARN"]

        # Get the Instance ID using the ARN
        ecs_client = boto3.client("ecs", region_name="us-east-1")
        res = ecs_client.describe_container_instances(
            cluster=cluster_name,
            containerInstances=[container_instance_arn],
        )
        print(res)
        instance_id = res["containerInstances"][0]["ec2InstanceId"]
    except Exception as e:
        print(
            f"Error getting instance ID from ECS metadata: {e}, scale-in protection not set."
        )
        return

    # Check if there are any running or pending tasks on the instance
    if not protect_from_scale_in:
        running_tasks = res["containerInstances"][0]["runningTasksCount"]
        pending_tasks = res["containerInstances"][0]["pendingTasksCount"]
        total_tasks = running_tasks + pending_tasks
        if total_tasks > 1:
            print(
                f"Instance {instance_id} has {running_tasks} running tasks and {pending_tasks} pending tasks, not disabling scale in protection."
            )
            return

    print(
        f"Setting instance {instance_id} scale in protection to {protect_from_scale_in}..."
    )
    # Enable scale in protection for the container instance
    autoscaling_client = boto3.client("autoscaling", region_name="us-east-1")
    res = autoscaling_client.set_instance_protection(
        InstanceIds=[instance_id],
        AutoScalingGroupName=auto_scaling_group_name,
        ProtectedFromScaleIn=protect_from_scale_in,
    )
    print(res)


def safely_reset_autoscaling_group_desired_and_max_capacity():
    """Set autocaling group desired and max capacity to zero if this is the only active task."""
    n_tasks = get_n_tasks_running_or_pending()
    if n_tasks > 1:
        print(
            f"There are {n_tasks} tasks running or pending, not resetting autoscaling group desired and max capacity."
        )
        return
    print("Resetting both autoscaling group desired and max capacity to zero...")
    set_desired_and_max_capacity_of_autoscaling_group(0, 0)


def get_max_capacity_of_autoscaling_group(
    auto_scaling_group_name: str = AUTOSCALING_GROUP_NAME,
):
    """Get max capacity of autoscaling group."""
    autoscaling_client = boto3.client("autoscaling", region_name="us-east-1")
    response = autoscaling_client.describe_auto_scaling_groups(
        AutoScalingGroupNames=[auto_scaling_group_name],
    )
    return response["AutoScalingGroups"][0]["MaxSize"]


def set_desired_and_max_capacity_of_autoscaling_group(
    desired_capacity: int,
    max_capacity: int,
    auto_scaling_group_name: str = AUTOSCALING_GROUP_NAME,
):
    """Set desired and max capacity of autoscaling group."""
    autoscaling_client = boto3.client("autoscaling", region_name="us-east-1")
    response = autoscaling_client.update_auto_scaling_group(
        AutoScalingGroupName=auto_scaling_group_name,
        DesiredCapacity=desired_capacity,
        MaxSize=max_capacity,
    )
    return response


def check_and_update_autoscaling_capacity():
    """Check and update autoscaling capacity. When current max capacity is zero, set it back to the max."""
    max_capacity = get_max_capacity_of_autoscaling_group()
    if max_capacity == 0:
        print("Max capacity is zero, setting it back to the max...")
        set_desired_and_max_capacity_of_autoscaling_group(
            AUTOSCALING_STARTING_DESIRED_CAPACITY, AUTOSCALING_MAX_CAPACITY
        )
    else:
        print(f"Max capacity is {max_capacity}, not changing it...")
