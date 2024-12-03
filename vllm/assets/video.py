from dataclasses import dataclass
from functools import lru_cache
from typing import List, Literal

import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download
from PIL import Image
import ffmpeg

from vllm.multimodal.utils import sample_frames_from_video

from .base import get_cache_dir


@lru_cache
def download_video_asset(filename: str) -> str:
    """
    Download and open an image from huggingface
    repo: raushan-testing-hf/videos-test
    """
    video_directory = get_cache_dir() / "video-eample-data"
    video_directory.mkdir(parents=True, exist_ok=True)

    video_path = video_directory / filename
    video_path_str = str(video_path)
    if not video_path.exists():
        video_path_str = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test",
            filename=filename,
            repo_type="dataset",
            cache_dir=video_directory,
        )
    return video_path_str


def extract_frames_ffmpeg(video_path: str) -> List[np.ndarray]:
    """
    Extract frames from a video using FFmpeg and return as a list of numpy arrays.
    """
    try:
        out, _ = (
            ffmpeg.input(video_path)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, capture_stderr=True)
        )
        width, height = get_video_resolution(video_path)
        frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        return frames
    except Exception as e:
        raise ValueError(f"Could not process video file {video_path}: {e}")


def get_video_resolution(video_path: str) -> (int, int):
    """
    Get the resolution of a video using FFmpeg.
    """
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")
    return int(video_stream["width"]), int(video_stream["height"])


def video_to_ndarrays(path: str, num_frames: int = -1) -> npt.NDArray:
    """
    Extract frames from a video file and convert them to a numpy array.
    """
    frames = extract_frames_ffmpeg(path)
    frames = sample_frames_from_video(frames, num_frames)
    if len(frames) < num_frames:
        raise ValueError(f"Could not read enough frames from video file {path} "
                         f"(expected {num_frames} frames, got {len(frames)})")
    return np.stack(frames)


def video_to_pil_images_list(path: str, num_frames: int = -1) -> List[Image.Image]:
    """
    Convert video frames to a list of PIL images.
    """
    frames = video_to_ndarrays(path, num_frames)
    return [Image.fromarray(frame) for frame in frames]


@dataclass(frozen=True)
class VideoAsset:
    name: Literal["sample_demo_1.mp4"]
    num_frames: int = -1

    @property
    def pil_images(self) -> List[Image.Image]:
        video_path = download_video_asset(self.name)
        return video_to_pil_images_list(video_path, self.num_frames)

    @property
    def np_ndarrays(self) -> npt.NDArray:
        video_path = download_video_asset(self.name)
        return video_to_ndarrays(video_path, self.num_frames)

