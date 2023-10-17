"""S3 Mirror pathing management."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

from S3MP.global_config import S3MPConfig
from S3MP.keys import KeySegment, get_matching_s3_keys
from S3MP.utils.local_file_utils import (
    DEFAULT_LOAD_LEDGER,
    DEFAULT_SAVE_LEDGER,
    delete_local_path,
)
from S3MP.utils.s3_utils import (
    delete_key_on_s3,
    download_key,
    key_exists_on_s3,
    key_is_file_on_s3,
    s3_list_child_keys,
    upload_to_key,
)


class MirrorPath:
    """A path representing an S3 file and its local mirror."""

    def __init__(
        self,
        key_segments: List[KeySegment],
        mirror_root: Path = None,
    ):
        """Init."""
        # Solving issues before they happen
        self.key_segments: List[KeySegment] = [seg.__copy__() for seg in key_segments]
        self._local_path_override: Path = None

        self.mirror_root = mirror_root or S3MPConfig.mirror_root

    @property
    def s3_key(self) -> str:
        """Get s3 key."""
        ret_key = "/".join([str(s.name) for s in self.key_segments])
        # We'll infer folder/file based on extension
        return ret_key if "." in self.key_segments[-1].name else f"{ret_key}/"

    @property
    def local_path(self) -> Path:
        """Get local path."""
        return self._local_path_override or Path(S3MPConfig.mirror_root) / self.s3_key

    def override_local_path(self, local_path: Path):
        """Override local path."""
        self._local_path_override = local_path

    @staticmethod
    def from_s3_key(s3_key: str, **kwargs: Dict) -> MirrorPath:
        """Create a MirrorPath from an s3 key."""
        # removesuffix is python 3.9
        # s3_key = s3_key.removesuffix('/')
        s3_key = s3_key[:-1] if s3_key.endswith("/") else s3_key
        key_segments = [KeySegment(idx, s) for idx, s in enumerate(s3_key.split("/"))]
        return MirrorPath(key_segments, **kwargs)

    @staticmethod
    def from_local_path(
        local_path: Path, mirror_root: Path = None, **kwargs: Dict
    ) -> MirrorPath:
        """Create a MirrorPath from a local path."""
        mirror_root = mirror_root or S3MPConfig.mirror_root
        s3_key = local_path.relative_to(mirror_root).as_posix()
        return MirrorPath.from_s3_key(s3_key, **kwargs)

    def __copy__(self):
        """Copy."""
        return MirrorPath(self.key_segments, **self.__dict__)

    def __repr__(self):
        """Class representation."""
        return f"{self.__class__.__name__}({self.s3_key})"

    def exists_in_mirror(self) -> bool:
        """Check if file exists in mirror."""
        return self.local_path.exists()

    def exists_on_s3(self) -> bool:
        """Check if file exists on S3."""
        return key_exists_on_s3(self.s3_key)

    def is_file_on_s3(self) -> bool:
        """Check if is a file on s3."""
        return key_is_file_on_s3(self.s3_key)

    def update_callback_on_skipped_transfer(self):
        """Update the current global callback if the transfer gets skipped."""
        if S3MPConfig.callback and self in S3MPConfig.callback._transfer_objs:
            S3MPConfig.callback(self.local_path.stat().st_size)

    def download_to_mirror(self, overwrite: bool = False):
        """Download S3 file to mirror."""
        if not overwrite and self.exists_in_mirror():
            self.update_callback_on_skipped_transfer()
            return
        download_key(self.s3_key, self.local_path)

    def download_to_mirror_if_not_present(self):
        """Download to mirror if not present in mirror."""
        self.download_to_mirror(overwrite=False)

    def upload_from_mirror(self, overwrite: bool = False):
        """Upload local file to S3."""
        if not overwrite and self.exists_on_s3():
            self.update_callback_on_skipped_transfer()
            return
        upload_to_key(self.s3_key, self.local_path)

    def upload_from_mirror_if_not_present(self):
        """Upload from mirror if not present on S3."""
        self.upload_from_mirror(overwrite=False)

    def trim(self, max_depth) -> MirrorPath:
        """Trim key from s3 key."""
        return MirrorPath(self.key_segments[:max_depth])

    def get_key_segment(self, index: int) -> KeySegment:
        """Get key segment."""
        return self.key_segments[index]

    def replace_key_segments(self, replace_segments: List[KeySegment]) -> MirrorPath:
        """Replace key segments."""
        new_segments = self.key_segments[:]
        for seg in replace_segments:
            while seg.depth >= len(new_segments):
                new_segments.append(KeySegment(len(new_segments) - 1, ""))
            new_segments[seg.depth] = seg
        return MirrorPath(new_segments)

    def replace_key_segments_at_relative_depth(
        self, replace_segments: List[KeySegment]
    ) -> MirrorPath:
        """Replace key segments at relative depth."""
        replace_segments = [seg.__copy__() for seg in replace_segments]
        for seg in replace_segments:
            seg.depth += len(self.key_segments) - 1
        return self.replace_key_segments(replace_segments)

    def get_sibling(self, sibling_name: str) -> MirrorPath:
        """Get a file with the same parent as this file."""
        return self.replace_key_segments_at_relative_depth(
            [KeySegment(0, sibling_name)]
        )

    def get_child(self, child_name: str) -> MirrorPath:
        """Get a file with the same parent as this file."""
        return self.replace_key_segments_at_relative_depth([KeySegment(1, child_name)])

    def get_children_on_s3(self) -> List[MirrorPath]:
        """Get all children on s3."""
        resp = s3_list_child_keys(self.s3_key)
        child_s3_keys = []
        # TODO decide if s3_utils is a better spot for this
        if "Contents" in resp:
            child_s3_keys = [
                obj["Key"] for obj in resp["Contents"] if obj["Key"] != self.s3_key
            ]
        if "CommonPrefixes" in resp:
            child_s3_keys += [obj["Prefix"] for obj in resp["CommonPrefixes"]]
        return [MirrorPath.from_s3_key(s3_key) for s3_key in child_s3_keys]

    def get_parent(self) -> MirrorPath:
        """Get the parent of this file."""
        return MirrorPath(self.key_segments[:-1])

    def delete_local(self):
        """Delete local file."""
        delete_local_path(self.local_path)

    def delete_s3(self):
        """Delete s3 file."""
        delete_key_on_s3(self.s3_key)

    def delete_all(self):
        """Delete all files."""
        self.delete_local()
        self.delete_s3()

    def load_local(
        self, download: bool = True, load_fn: Callable = None, overwrite: bool = False
    ):
        """
        Load local file, infer file type and load.

        Setting download to false will still download if the file is not present.
        """
        if download or overwrite or not self.exists_in_mirror():
            self.download_to_mirror(overwrite)
        if load_fn is None:
            suffix = self.local_path.suffix[1:].lower()
            load_fn = DEFAULT_LOAD_LEDGER[suffix]

        return load_fn(str(self.local_path))

    def save_local(
        self,
        data,
        upload: bool = True,
        save_fn: Callable = None,
        overwrite: bool = False,
    ):
        """Save local file, infer file type and upload."""
        try:
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
        except FileExistsError:  # handle for race conditions
            pass
        if save_fn is None:
            suffix = self.local_path.suffix[1:].lower()
            save_fn = DEFAULT_SAVE_LEDGER[suffix]

        save_fn(str(self.local_path), data)
        if upload:
            self.upload_from_mirror(overwrite)


def get_matching_s3_mirror_paths(segments: List[KeySegment]):
    """Get matching S3 mirror paths."""
    return [MirrorPath.from_s3_key(key) for key in get_matching_s3_keys(segments)]
