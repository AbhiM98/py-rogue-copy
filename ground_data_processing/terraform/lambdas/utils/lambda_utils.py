"""Lambda utils."""
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Union

import boto3
from botocore.vendored import requests


class LambdaJSONEncoder(json.JSONEncoder):
    """DynamoDB encoder to handle decimal cases."""

    def default(self, o):
        """Convert to float."""
        if isinstance(o, Decimal):
            return float(o)
        return super(LambdaJSONEncoder, self).default(o)


def get_return_block_with_cors(body, needs_encoding=True):
    """Get return block with cors."""
    # TODO just auto-detect needs_encoding
    if needs_encoding:
        body = json.dumps(body, cls=LambdaJSONEncoder)
    return {
        "statusCode": 200,
        "body": body,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "OPTIONS,PUT,POST,GET",
            "Content-Type": "application/json",
        },
    }


def get_full_lambda_name(base_name: str) -> str:
    """Attach the env prefix to a lambda base name."""
    try:
        # Second part of lambda name is the env
        env = os.environ["AWS_LAMBDA_FUNCTION_NAME"].split("-")[1]
    except KeyError:
        try:
            metadata_uri = os.environ["ECS_CONTAINER_METADATA_URI"]
            container_metadata = requests.get(metadata_uri).json()
            env = container_metadata["Name"].split("-")[0]
        except KeyError or TypeError as e:
            print(e)
            print("Not running in a lambda/ecs env, using 'prod' env by default...")
            env = "prod"
    return f"rogues-{env}-{base_name.replace('_', '-')}"


def remove_trailing_slash_and_suffix(path: Union[str, Path], suffix: str):
    """Remove trailing slash or suffix."""
    if isinstance(path, str):
        path.rstrip("/")
        path = Path(path)
    if suffix == path.name:
        # If suffix exists, remove it from the original path
        path = path.parents[0]

    return str(path)


def invoke_lambda(
    base_name: str = None, params: dict = None, run_async: bool = False
) -> dict:
    """Invoke a lambda and return the response, unless running async."""
    if params is None:
        params = {}
    if base_name is None:
        raise ValueError("Lambda invokation: base_name cannot be 'None'")
    function_name = get_full_lambda_name(base_name)
    payload_str = json.dumps(
        {"body": json.dumps(params, cls=LambdaJSONEncoder)}, cls=LambdaJSONEncoder
    )
    client = boto3.client("lambda")
    print(f"Invoking {function_name}...")
    if run_async:
        client.invoke(
            FunctionName=function_name,
            InvocationType="Event",  # No response
            Payload=payload_str,
        )
    else:
        return json.loads(
            client.invoke(
                FunctionName=function_name,
                InvocationType="RequestResponse",
                Payload=payload_str,
            )["Payload"].read()
        )["body"]
