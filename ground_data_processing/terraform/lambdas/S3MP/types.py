"""Types, mostly from mypy_boto3 (boto3-stubs)."""
from pathlib import Path
from typing import List, TypeVar, Union

T = TypeVar("T")
SList = Union[List[T], T]
PathSList = SList[Path]
StrSSlist = SList[str]
