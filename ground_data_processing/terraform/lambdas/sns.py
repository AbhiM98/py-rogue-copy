"""Lambda for sending an SNS message to a topic."""
import json

from utils.lambda_utils import get_return_block_with_cors
from utils.sns_utils import publish_message_to_name


def send_sns_message_to_name(event, context):
    """Lambda for sending an SNS message to a topic."""
    try:
        body = json.loads(event["body"])
    except Exception:
        body = event

    name = body["name"]
    message = body["message"]
    ret = publish_message_to_name(name, message)

    return get_return_block_with_cors(ret)
