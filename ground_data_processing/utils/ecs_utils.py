"""Utilities for ECS tasks."""
import json
import os

import boto3

AUTOSCALING_GROUP_NAME = "prod-rogues-ecs-cluster-one-20230530192028803000000008"
CLUSTER_NAME = "prod-rogues-ecs-cluster"


def safely_set_scale_in_protection(protect_from_scale_in: bool):
    """Set the scale in protection on the host instance, taking care not to disable it if other tasks are running or pending."""
    if is_executing_on_ecs():
        try:
            # https://docs.aws.amazon.com/AmazonECS/latest/developerguide/container-metadata.html
            with open(os.environ["ECS_CONTAINER_METADATA_FILE"]) as f:
                ecs_metadata = json.load(f)
            # Get the container instance ARN
            container_instance_arn = ecs_metadata["ContainerInstanceARN"]

            # Get the Instance ID using the ARN
            ecs_client = boto3.client("ecs", region_name="us-east-1")
            res = ecs_client.describe_container_instances(
                cluster=CLUSTER_NAME,
                containerInstances=[container_instance_arn],
            )
            print(res)
            instance_id = res["containerInstances"][0]["ec2InstanceId"]
        except Exception as e:
            print(
                f"Error getting instance ID from ECS metadata: {e}, scale-in protection not set."
            )
            return
    else:
        print("Not setting scale in protection because this is not executing on ECS.")
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
        AutoScalingGroupName=AUTOSCALING_GROUP_NAME,
        ProtectedFromScaleIn=protect_from_scale_in,
    )
    print(res)


def safely_reset_autoscaling_group_desired_and_max_capacity():
    """Set autocaling group desired and max capacity to zero if this is the only active task."""
    if is_executing_on_ecs():
        n_tasks = get_n_tasks_running_or_pending()
        if n_tasks > 1:
            print(
                f"There are {n_tasks} tasks running or pending, not resetting autoscaling group desired and max capacity."
            )
            return
        print("Resetting both autoscaling group desired and max capacity to zero...")
        set_desired_and_max_capacity_of_autoscaling_group(0, 0)
    else:
        print(
            "Not resetting autoscaling group desired and max capacity because this is not executing on ECS."
        )


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


def is_executing_on_ecs() -> bool:
    """Return True if executing on ECS, False otherwise."""
    return "ECS_CONTAINER_METADATA_FILE" in os.environ
