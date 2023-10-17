"""Merge GoPro videos."""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import ffmpeg
from tqdm import tqdm

from ground_data_processing.scripts.box_to_s3 import find_subfolder_that_exists
from ground_data_processing.utils.s3_constants import CameraViews

possible_folder_names_per_camera = {
    CameraViews.BOTTOM: ["bottom", "ground"],
    CameraViews.NADIR: ["nadir", "top"],
    CameraViews.OBLIQUE: ["oblique", "middle"],
}


@dataclass
class GoProVideoPath:
    """Dataclass to manage GoPro Video Name attributes."""

    path: Path

    _video_number: int = None
    _split_video_index: int = None

    @property
    def video_number(self) -> int:
        """Video number."""
        if self._video_number is None:
            self._video_number = int(self.stem[4:])
        return self._video_number

    @property
    def split_video_index(self) -> int:
        """Split video index."""
        if self._split_video_index is None:
            self._split_video_index = int(self.stem[3])
        return self._split_video_index

    @property
    def size_bytes(self) -> int:
        """Size in bytes."""
        return self.path.stat().st_size

    @property
    def name(self) -> str:
        """Name."""
        return self.path.name

    @property
    def stem(self) -> str:
        """Stem."""
        return self.path.stem


def merge_single_video(vid_dir: Path) -> None:
    """Merge a single video."""
    mp4_vids: List[GoProVideoPath] = [
        GoProVideoPath(mp4_file) for mp4_file in vid_dir.glob("GX*.*4")
    ]
    all_video_numbers = {vid.video_number for vid in mp4_vids}
    mp4_vids_by_video_number = {
        video_number: [vid for vid in mp4_vids if vid.video_number == video_number]
        for video_number in all_video_numbers
    }

    # Default behavior is take the most recent video
    relevant_vids = mp4_vids_by_video_number[max(all_video_numbers)]

    if len(relevant_vids) == 1:
        print("\nOnly one video, skipping.")
        return

    print(f"\nFound {len(relevant_vids)} videos, merging.")
    print("Video sizes:")
    for vid in relevant_vids:
        print(f"{vid.name}: {vid.size_bytes}")

    ffmpeg_list_str = "".join(f"file '{vid.name}'\n" for vid in relevant_vids)
    ffmpeg_list_path = vid_dir / "ffmpeg_list.txt"
    with open(ffmpeg_list_path, "w") as f:
        f.write(ffmpeg_list_str)

    ffmpeg.input(str(ffmpeg_list_path), format="concat", safe=0).output(
        str(vid_dir / "merged.mp4"), c="copy", loglevel="quiet"
    ).overwrite_output().run()


if __name__ == "__main__":
    box_dir = Path("/mnt/c/Users/nschroeder/Downloads/box_files/")

    base_dir = box_dir / "row 1"
    print(list(base_dir.glob("*")))
    for row_dir in tqdm(list(base_dir.glob("*"))):
        print(f"\nProcessing {row_dir.name}")
        for (
            camera_view,
            possible_folder_names,
        ) in tqdm(possible_folder_names_per_camera.items()):
            # TODO put this all in GoPro utils or smthn
            cam_folder = row_dir
            if not any([x in row_dir.name.lower() for x in possible_folder_names]):
                cam_folder = find_subfolder_that_exists(row_dir, possible_folder_names)
            if (cam_folder / "DCIM" / "100GOPRO").exists():
                cam_folder = cam_folder / "DCIM" / "100GOPRO"
            elif (cam_folder / "100GOPRO").exists():
                cam_folder = cam_folder / "100GOPRO"
            merge_single_video(cam_folder)
