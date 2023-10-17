"""Video utilities."""
import itertools
import os
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import ffmpeg
import imagehash
import numpy as np
import skvideo.io
from numpy.typing import ArrayLike
from PIL import Image
from tqdm import tqdm


def get_frame_count(video_path: Path) -> int:
    """Get the number of frames in a video."""
    video_data = skvideo.io.ffprobe(str(video_path))
    if "video" not in video_data:
        return 0
    return int(video_data["video"]["@nb_frames"])


def get_resolution(video_path: Path) -> Tuple[int, int]:
    """Get the resolution of a video."""
    return (
        int(skvideo.io.ffprobe(str(video_path))["video"]["@height"]),
        int(skvideo.io.ffprobe(str(video_path))["video"]["@width"]),
    )


def vid_resize(vid_path, output_path, width, overwrite=False):
    """Use ffmpeg to resize the input video to the width given, keeping aspect ratio."""
    if not (os.path.isdir(os.path.dirname(output_path))):
        raise ValueError(
            f"output_path directory does not exists: {os.path.dirname(output_path)}"
        )

    if os.path.isfile(output_path) and not overwrite:
        warnings.warn(
            f"{output_path} already exists but overwrite switch is False, nothing done."
        )
        return None

    input_vid = ffmpeg.input(str(vid_path))
    input_vid.filter("scale", width, -1).output(
        str(output_path)
    ).overwrite_output().run()
    return output_path


def get_video_iterator(
    video_path: Path, use_tqdm: bool = False
) -> Tuple[Iterable, int]:
    """Get an iterator over the frames of a video, and the frame count."""
    video = skvideo.io.vreader(str(video_path))
    n_frames = get_frame_count(video_path)
    if use_tqdm:
        video = tqdm(video, total=n_frames)
    return video, n_frames


def get_video_ffmpeg_reader(
    video_path: Path, use_tqdm: bool = False
) -> Tuple[skvideo.io.FFmpegReader, int]:
    """Get an ffmpeg reader for a video."""
    reader = skvideo.io.FFmpegReader(
        filename=str(video_path),
        # inputdict = {
        #     '-hwaccel': 'cuda'
        # }
    )
    n_frames = get_frame_count(video_path)
    if use_tqdm:
        video = tqdm(reader.nextFrame(), total=n_frames)
    else:
        video = reader.nextFrame()
    return video, n_frames


def get_ffmpeg_reader_trimmed(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float = 59.94,
    use_tqdm=False,
) -> Iterable:
    """Get an ffmpeg reader for a video."""
    start_timestamp = start_frame / fps
    end_timestamp = end_frame / fps
    reader = skvideo.io.FFmpegReader(
        filename=str(video_path),
        inputdict={"-ss": f"{start_timestamp}", "-to": f"{end_timestamp}"},
    )

    return (
        tqdm(reader.nextFrame(), total=end_frame - start_frame)
        if use_tqdm
        else reader.nextFrame()
    )


def get_frame_shape(video_path: Path) -> Tuple[int, int, int]:
    """Get the shape of a frame."""
    return (
        int(skvideo.io.ffprobe(str(video_path))["video"]["@height"]),
        int(skvideo.io.ffprobe(str(video_path))["video"]["@width"]),
        3,
    )


def get_frame_at_index(video_path: Path, index: int, fps: float = 59.94) -> np.ndarray:
    """Get a frame at a specific index."""
    frame_shape = get_frame_shape(video_path)
    out, _ = (
        ffmpeg.input(
            str(video_path),
            ss=index / fps,
        )
        .output(
            "pipe:", vframes=1, format="rawvideo", pix_fmt="rgb24", loglevel="quiet"
        )
        .run(capture_stdout=True)
    )
    return np.frombuffer(out, np.uint8).reshape(frame_shape)


def get_frames_between_indices(
    video_path: Path, start_index: int, end_index: int, fps: float = 59.94
) -> Iterable:
    """Get frames between indices."""
    return get_ffmpeg_reader_trimmed(video_path, start_index, end_index, fps)


def save_frames_between_indices(
    video_path: Path,
    save_dir: Path,
    start_index: int,
    end_index: int,
    fps: float = 59.94,
) -> List[str]:
    """Save frames between indices. Returns filenames as strings."""
    frames = get_frames_between_indices(video_path, start_index, end_index, fps)
    for idx, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(save_dir / f"{idx:03d}.png"), frame)
    return [str(fn) for fn in sorted(save_dir.glob("*"))]


def find_frame_idx(video_path: Path, frame: ArrayLike) -> int:
    """Find the index of a specified frame."""
    frame = Image.fromarray(frame)
    check_hash = imagehash.average_hash(frame)
    hash_tol = 2
    video, n_frames = get_video_ffmpeg_reader(video_path)
    for idx, frame_ in enumerate(video):
        frame_ = Image.fromarray(frame_)
        frame_hash = imagehash.average_hash(frame_)
        if frame_hash - check_hash < hash_tol:
            return idx
        # if np.array_equal(frame, frame_):
        #     return idx
    raise ValueError("Frame not found.")


def get_n_divisions(n_frames, n_divisions):
    """Get n divisions."""
    return np.linspace(0, n_frames, n_divisions + 1).astype(int)


def get_n_ffmpeg_readers(
    video_path: Path, n_readers: int, fps: float = 59.94
) -> List[skvideo.io.FFmpegReader]:
    """Get n iterators."""
    n_frames = get_frame_count(video_path)
    for start, end in itertools.pairwise(get_n_divisions(n_frames, n_readers)):
        yield get_ffmpeg_reader_trimmed(video_path, start * fps, (end - start) * fps)


# def dump_n_frames(video, n_frames, frames_to_dump, out_dir):
#     """dump frames from a video."""
#     max_val = np.max(frames_to_dump)
#     min_val = np.min(frames_to_dump)
#     video = tail(min_val, video)

#     for idx, frame in tqdm(enumerate(video), total=n_frames):
#         idx += min_val
#         if idx in frames_to_dump:
#             save_image(frame, str(out_dir / f"{idx}.png"), out_dir)
#         if idx > max_val:
#             break
