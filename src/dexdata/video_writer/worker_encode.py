# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

import os
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_fixed


class Registry:
    """Minimal name-to-class registry. Each subclass gets its own isolated dict."""

    _registry: dict[str, type]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, name: str | None = None):
        def decorator(registered_class: type) -> type:
            registration_name = name if name is not None else registered_class.__name__
            if registration_name in cls._registry:
                raise ValueError(
                    f"Class '{registration_name}' is already registered in {cls.__name__}."
                )
            cls._registry[registration_name] = registered_class
            registered_class._registry_name = registration_name
            return registered_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._registry:
            raise KeyError(
                f"Class '{name}' not found in {cls.__name__}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]


class VideoEncoderRegistry(Registry):
    _registry: dict[str, type] = {}


def _load_frames_from_args(args: dict):
    """Load frames tensor from args, supporting both mmap and direct tensor.

    If 'frames_meta' is present, loads via mmap (parallel writer path).
    Otherwise falls back to 'frames' key (serial writer path).
    Cleans up the raw mmap file after loading.
    """
    if "frames_meta" in args:
        import numpy as np
        import torch

        meta = args["frames_meta"]
        mm = np.memmap(
            meta["path"], dtype=meta["dtype"], mode="r", shape=tuple(meta["shape"])
        )
        tensor = torch.from_numpy(np.array(mm))
        del mm
        # Clean up raw file — data is now in the tensor
        try:
            os.unlink(meta["path"])
        except OSError:
            pass
        return tensor
    else:
        return args["frames"]


def _encode_pyav_inner(
    frames_tensor, fps, codec, pix_fmt, crf, gop, chunk_path: Path
) -> Path:
    """Inner encoding function for PyAV - can be retried without reloading file.

    Expects ``frames_tensor`` as ``(T, H, W, C)`` uint8 RGB — the natural
    layout that PyAV's ``rgb24`` frames consume directly. No layout
    permute happens here; the writer is responsible for delivering THWC.
    """
    import av

    # Get dimensions from tensor (T, H, W, C)
    num_frames, height, width, channels = frames_tensor.shape

    # Create output container
    container = av.open(str(chunk_path), mode="w")

    # Add video stream
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = pix_fmt

    # Set encoding options
    options = {"crf": str(crf)}
    if gop is not None:
        options["g"] = str(gop)
    stream.options = options

    try:
        # frames_tensor is already (T, H, W, C) — PyAV's rgb24 layout.
        frames_np = frames_tensor.numpy()

        for i in range(num_frames):
            frame_data = frames_np[i]  # (H, W, C), contiguous slice of THWC tensor
            av_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")

            for packet in stream.encode(av_frame):
                container.mux(packet)

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)

    finally:
        container.close()

    return chunk_path


@VideoEncoderRegistry.register("pyav")
def encode_chunk_worker_pyav(args: dict) -> Path:
    """
    Worker function to encode a single chunk of frames using PyAV.

    Must be at module level for pickling by multiprocessing.
    All imports are done inside to ensure fresh state in spawned process.

    Args:
        args: Dictionary containing all encoding parameters:
            - chunk_idx: Index of this chunk
            - frames_meta: Mmap metadata dict (path, dtype, shape) for loading frames
            - frames: (fallback) Tensor of shape (T, C, H, W) on CPU
            - fps, codec, pix_fmt, crf, gop: Encoding parameters
            - chunk_path: Output path for this chunk (as string)

    Returns:
        Path to the encoded chunk file.
    """
    chunk_path = Path(args["chunk_path"])
    frames_tensor = _load_frames_from_args(args)
    fps = args["fps"]
    codec = args["codec"]
    pix_fmt = args["pix_fmt"]
    crf = args["crf"]
    gop = args["gop"]

    # Retry the encoding part only
    encode_with_retry = retry(stop=stop_after_attempt(3), wait=wait_fixed(1))(
        _encode_pyav_inner
    )

    result = encode_with_retry(frames_tensor, fps, codec, pix_fmt, crf, gop, chunk_path)
    return result


def _encode_torchcodec_inner(
    frames_tensor, fps, codec, pix_fmt, crf, gop, chunk_path: Path
) -> Path:
    """Inner encoding function for TorchCodec - can be retried without reloading file."""
    from torchcodec.encoders import VideoEncoder

    # Encode to chunk file
    encoder = VideoEncoder(frames=frames_tensor, frame_rate=fps)
    extra_options = {"g": gop} if gop is not None else None

    encoder.to_file(
        dest=str(chunk_path),
        codec=codec,
        pixel_format=pix_fmt,
        crf=crf,
        extra_options=extra_options,
    )

    return chunk_path


@VideoEncoderRegistry.register("torchcodec")
def encode_chunk_worker_torchcodec(args: dict) -> Path:
    """
    Worker function to encode a single chunk of frames.

    Must be at module level for pickling by multiprocessing.
    All imports are done inside to ensure fresh state in spawned process.

    Args:
        args: Dictionary containing all encoding parameters:
            - chunk_idx: Index of this chunk
            - frames_meta: Mmap metadata dict (path, dtype, shape) for loading frames
            - frames: (fallback) Tensor of shape (T, C, H, W) on CPU
            - fps, codec, pix_fmt, crf, gop: Encoding parameters
            - chunk_path: Output path for this chunk (as string)

    Returns:
        Path to the encoded chunk file.
    """
    chunk_path = Path(args["chunk_path"])
    frames_tensor = _load_frames_from_args(args)
    fps = args["fps"]
    codec = args["codec"]
    pix_fmt = args["pix_fmt"]
    crf = args["crf"]
    gop = args["gop"]

    # Retry the encoding part only
    encode_with_retry = retry(stop=stop_after_attempt(3), wait=wait_fixed(1))(
        _encode_torchcodec_inner
    )

    result = encode_with_retry(frames_tensor, fps, codec, pix_fmt, crf, gop, chunk_path)
    return result
