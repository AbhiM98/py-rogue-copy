"""Utilities for working with SNS."""
import json

import boto3

ROGUES_INFERENCE_COMPLETE = "rogues-inference-complete"
ROGUES_LAUNCH_FAILURE = "rogues-launch-failure"
ROGUES_TASK_FAILURE = "rogues-task-failure"
ROGUES_GRD_ADDED = "rogues-grd-added"


def get_sns_client_if_not_exists(sns_client=None):
    """Get an SNS client if one is not provided."""
    if sns_client is None:
        sns_client = boto3.client("sns", region_name="us-east-1")
    return sns_client


def get_arn_from_name(name: str, sns_client=None) -> str:
    """Get the ARN of an SNS topic from its name."""
    sns_client = get_sns_client_if_not_exists(sns_client)
    response = sns_client.list_topics()
    for topic in response["Topics"]:
        if topic["TopicArn"].split(":")[-1] == name:
            return topic["TopicArn"]
    raise ValueError(f"Topic {name} not found.")


def publish_message_to_arn(arn: str, message: dict, sns_client=None) -> dict:
    """Publish a message to an SNS topic."""
    sns_client = get_sns_client_if_not_exists(sns_client)
    response = sns_client.publish(
        TargetArn=arn,
        Message=json.dumps({"default": json.dumps(message)}),
        MessageStructure="json",
    )
    return response


def publish_message_to_name(name: str, message: dict, sns_client=None) -> dict:
    """Publish a message to an SNS topic."""
    sns_client = get_sns_client_if_not_exists(sns_client)
    arn = get_arn_from_name(name)
    return publish_message_to_arn(arn, message)
