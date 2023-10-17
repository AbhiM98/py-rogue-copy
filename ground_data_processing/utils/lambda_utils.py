"""Lambda utils."""
import json
from decimal import Decimal

import boto3


class LambdaJSONEncoder(json.JSONEncoder):
    """DynamoDB encoder to handle decimal cases."""

    def default(self, o):
        """Convert to float."""
        if isinstance(o, Decimal):
            return float(o)
        return super(LambdaJSONEncoder, self).default(o)


def invoke_lambda(
    function_name: str = None, params: dict = None, run_async: bool = False
) -> dict:
    """Invoke a lambda and return the response, unless running async."""
    if params is None:
        params = {}
    if function_name is None:
        raise ValueError("Lambda invokation: function_name cannot be 'None'")
    payload_str = json.dumps(
        {"body": json.dumps(params, cls=LambdaJSONEncoder)}, cls=LambdaJSONEncoder
    )
    client = boto3.client("lambda", region_name="us-east-1")
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
