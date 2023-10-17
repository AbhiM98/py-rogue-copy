"""Ground rogues data DynamoDB API."""
import json
from typing import List

import boto3

from ddb_tracking.grd_constants import GRD_DDB_TABLE_NAME, RoguesS3PathStructure
from ddb_tracking.grd_structure import GRDJSONEncoder, GRDRow


def key_query_grd_rows(
    query: dict = None,
    ddb_resource=None,
) -> List[GRDRow]:
    """Query ground rogues data rows.

    Args:
    query: The query to use. Defaults to None, which will return all rows.

    Eg:
    "field_name": "Farmer City 2022 Plot Trial Planting 1",
    "date": "2022-06-16",
    will return all rows for Farmer City 2022 Plot Trial Planting 1 on 2022-06-16.
    """
    if ddb_resource is None:
        ddb_resource = boto3.resource("dynamodb", region_name="us-east-1")
    grd_table = ddb_resource.Table(GRD_DDB_TABLE_NAME)

    # TODO: This is a hacky way to do this. We should use a query instead.
    # Load all the rows
    response = grd_table.scan()
    items = response["Items"]
    while "LastEvaluatedKey" in response:
        response = grd_table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        items.extend(response["Items"])

    all_grdrows = [
        GRDRow.from_json(json.loads(json.dumps(item, cls=GRDJSONEncoder)))
        for item in items
    ]

    if query:
        # Filter by query
        filtered_grdrows = []
        for grdrow in all_grdrows:
            if all([getattr(grdrow, key) == value for key, value in query.items()]):
                filtered_grdrows.append(grdrow)
        if len(filtered_grdrows) < 1:
            print(f"No rows found that match query = {query}.")
        return filtered_grdrows

    else:
        print("No query provided. Returning all rows.")
        return all_grdrows


def get_grd_row(
    field_name: str, row_number: int, date: str, ddb_resource=None
) -> GRDRow:
    """Get a ground rogues data row.

    Args:
        field_name: The field name to query.
        row_number: The row number to query.
        date: The date to query. YYYY-MM-DD format.
    """
    if ddb_resource is None:
        ddb_resource = boto3.resource("dynamodb", region_name="us-east-1")
    if not isinstance(row_number, int):
        # Handle string row numbers
        row_number = int(row_number)
    grd_table = ddb_resource.Table(GRD_DDB_TABLE_NAME)
    response = grd_table.get_item(
        Key={
            "field_name": field_name,
            "date#row_number": GRDRow.get_date_row_number(date, row_number),
        }
    )
    try:
        # Handle Decimal type
        return GRDRow.from_json(
            json.loads(json.dumps(response["Item"], cls=GRDJSONEncoder))
        )
    except KeyError:
        print(
            f"No item found for field_name = {field_name}, row_number = {row_number}, date = {date}."
        )
        return None


def get_grd_row_from_full_video_s3_key(s3_key: str, ddb_resource=None) -> GRDRow:
    """Get a ground rogues data row from an S3 path to the full video file."""
    (
        field_name,
        date,
        row_number,
        _,
        _,
    ) = RoguesS3PathStructure.infer_structure_and_parse_s3_key(s3_key)
    return get_grd_row(field_name, row_number, date, ddb_resource)


def put_grd_row(grd_row: GRDRow, ddb_resource=None) -> None:
    """Put a ground rogues data row."""
    if ddb_resource is None:
        ddb_resource = boto3.resource("dynamodb", region_name="us-east-1")
    grd_table = ddb_resource.Table(GRD_DDB_TABLE_NAME)
    grd_table.put_item(Item=grd_row.as_json())
