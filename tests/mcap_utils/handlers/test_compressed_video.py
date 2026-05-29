# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""CompressedVideo handler + Writer/Reader round-trip tests.

Three layers:
  1. Codec sanity: encode/decode N synthetic frames through the handler
     directly; verify count preserved and PSNR above a per-codec floor.
  2. Writer flush: stateful encoder buffers frames and emits them only on
     close; verify a Writer that adds N frames produces N MCAP messages.
  3. Real episode round-trip: pull the first 30 frames from
     ``head_left.rgb/`` PNG sidecars (BGR on disk; converted to RGB at
     load to match the handler's RGB contract), write through
     composable, read back, check frame count and PSNR.

Codec choice: tests use h264 because libx264 is fast (avoids 30s+ av1
encode times in CI) and is the one with the more delicate ``bf=0`` setting.
A separate parametric path also exercises av1 → libdav1d on a small frame
set so the av1 path doesn't bitrot.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dexdata.handlers.containers import CompressedVideoSpec
from dexdata.mcap_utils.handlers.compressed_video import CompressedVideoHandler
from dexdata.mcap_utils.reader import Reader
from dexdata.mcap_utils.writer import Writer
from dexdata.spec import SignalSpec, Spec

REAL_EPISODE_RAW = Path(
    "/path_to_your_recordings/recordings/episode_20260508_112520_915215_raw"
)
REAL_RGB_DIR = REAL_EPISODE_RAW / "head_left.rgb"

H, W = 64, 64
N_SYNTH = 16


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Per-frame PSNR in dB; ``inf`` if identical."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 * 255.0 / mse)


def _synth_frames(n: int = N_SYNTH, h: int = H, w: int = W) -> list[np.ndarray]:
    """Smooth gradient frames with frame index baked into the brightness channel.

    Smooth content compresses much better than random noise, so PSNR floors
    on the codec-encoded result are meaningful.
    """
    out: list[np.ndarray] = []
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for i in range(n):
        b = (xx / w * 200 + i * 3) % 256
        g = (yy / h * 200 + i * 5) % 256
        r = ((xx + yy) / (w + h) * 200 + i * 7) % 256
        frame = np.stack([b, g, r], axis=-1).astype(np.uint8)
        out.append(frame)
    return out


# --- 1. Codec sanity --------------------------------------------------------


@pytest.mark.parametrize(
    "codec,preset,psnr_floor",
    [
        ("h264", "ultrafast", 24.0),
        ("av1", "8", 24.0),
    ],
)
def test_codec_round_trip_psnr(codec: str, preset: str, psnr_floor: float) -> None:
    spec = CompressedVideoSpec(codec=codec, crf=28, gop=10, preset=preset)
    enc_handler = CompressedVideoHandler(spec=spec, width=W, height=H)
    dec_handler = CompressedVideoHandler(spec=spec, width=W, height=H)

    frames = _synth_frames()
    payloads: list[bytes] = []
    for i, fr in enumerate(frames):
        for payload, pt, rt in enc_handler.serialize(fr, i, i):
            payloads.append(payload)
            assert pt == rt  # we passed identical timestamps
    for payload, _, _ in enc_handler.flush():
        payloads.append(payload)

    assert len(payloads) == len(frames), (
        f"{codec}: {len(payloads)} packets vs {len(frames)} frames"
    )

    decoded: list[np.ndarray] = []
    for p in payloads:
        for fr in dec_handler.deserialize(p):
            decoded.append(fr)
    for fr in dec_handler.flush_decode():
        decoded.append(fr)

    assert len(decoded) == len(frames)
    for i, (orig, back) in enumerate(zip(frames, decoded, strict=False)):
        psnr = _psnr(orig, back)
        assert psnr >= psnr_floor, f"{codec} frame {i}: PSNR {psnr:.2f} < {psnr_floor}"


def test_serialize_rejects_wrong_shape() -> None:
    spec = CompressedVideoSpec(codec="h264", crf=28, gop=10, preset="ultrafast")
    h = CompressedVideoHandler(spec=spec, width=W, height=H)
    bad = np.zeros((H, W + 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="frame size"):
        list(h.serialize(bad, 0, 0))


def test_serialize_rejects_wrong_dtype() -> None:
    spec = CompressedVideoSpec(codec="h264", crf=28, gop=10, preset="ultrafast")
    h = CompressedVideoHandler(spec=spec, width=W, height=H)
    bad = np.zeros((H, W, 3), dtype=np.uint16)
    with pytest.raises(ValueError, match="uint8 RGB"):
        list(h.serialize(bad, 0, 0))


def test_unsupported_codec_rejected_at_construction() -> None:
    spec = CompressedVideoSpec(codec="vp9", crf=28, gop=10, preset="0")
    with pytest.raises(ValueError, match="unsupported codec"):
        CompressedVideoHandler(spec=spec, width=W, height=H)


# --- 2. Writer flush --------------------------------------------------------


def _video_only_spec(topic: str = "/camera/head_left/rgb/video") -> Spec:
    return Spec(
        spec_id="video_only_test",
        spec_version="0.1.0",
        signals=(
            SignalSpec(
                topic=topic,
                unit="bgr",
                container=CompressedVideoSpec(
                    codec="h264", crf=28, gop=10, preset="ultrafast"
                ),
            ),
        ),
    )


def test_writer_pending_topic_blocks_write() -> None:
    spec = _video_only_spec()
    fr = np.zeros((H, W, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        with Writer(Path(tmp) / "ep", spec) as writer:
            assert writer.pending_topics == ("/camera/head_left/rgb/video",)
            with pytest.raises(RuntimeError, match="runtime metadata not set"):
                writer.write("/camera/head_left/rgb/video", fr, 0, 0)


def test_writer_set_runtime_metadata_validates() -> None:
    spec = _video_only_spec()
    topic = "/camera/head_left/rgb/video"
    with tempfile.TemporaryDirectory() as tmp:
        with Writer(Path(tmp) / "ep", spec) as writer:
            with pytest.raises(ValueError, match="missing"):
                writer.set_runtime_metadata(topic, width=W)
            with pytest.raises(ValueError, match="unexpected"):
                writer.set_runtime_metadata(topic, width=W, height=H, fps=30)
            writer.set_runtime_metadata(topic, width=W, height=H)
            assert writer.pending_topics == ()


def test_writer_flushes_encoder_on_close() -> None:
    """Writer.close() must drain the encoder so packet count == frame count."""
    spec = _video_only_spec()
    topic = "/camera/head_left/rgb/video"
    frames = _synth_frames()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ep"
        with Writer(out, spec) as writer:
            writer.set_runtime_metadata(topic, width=W, height=H)
            for i, fr in enumerate(frames):
                writer.write(topic, fr, publish_ts_ns=i, recv_ts_ns=i)

        reader = Reader(out)
        episode = reader.read_episode()
        assert episode.signals[topic].shape == (len(frames), H, W, 3)


# --- 3. Real-episode round-trip --------------------------------------------


@pytest.mark.skipif(
    not REAL_RGB_DIR.exists(),
    reason=f"real episode not available at {REAL_EPISODE_RAW}",
)
def test_real_rgb_roundtrip() -> None:
    manifest = json.loads((REAL_EPISODE_RAW / "manifest.json").read_text())
    cam = manifest["cameras"]["head_left.rgb"]
    full_h, full_w = cam["shape"][1], cam["shape"][2]

    timestamps = np.load(REAL_RGB_DIR / "timestamps.npy")
    n_test = 30
    frames: list[np.ndarray] = []
    for i in range(n_test):
        # PIL labels these PNGs RGB but the underlying bytes are the camera's
        # native BGR (manifest channel_order=bgr). Convert once at load
        # so the array fed to the handler matches the RGB contract.
        img = Image.open(REAL_RGB_DIR / f"{i:06d}.png")
        bgr = np.asarray(img)
        frames.append(bgr[..., ::-1].copy())

    spec = Spec(
        spec_id="real_rgb_test",
        spec_version="0.1.0",
        signals=(
            SignalSpec(
                topic="/camera/head_left/rgb/video",
                unit="bgr",
                container=CompressedVideoSpec(
                    codec="h264", crf=28, gop=10, preset="ultrafast"
                ),
            ),
        ),
    )
    topic = "/camera/head_left/rgb/video"

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "ep"
        with Writer(out_dir, spec) as writer:
            writer.set_runtime_metadata(topic, width=full_w, height=full_h)
            for i, fr in enumerate(frames):
                ts = int(timestamps[i])
                writer.write(topic, fr, publish_ts_ns=ts, recv_ts_ns=ts)

        reader = Reader(out_dir)
        episode = reader.read_episode()
        decoded = episode.signals[topic]
        assert decoded.shape == (n_test, full_h, full_w, 3)
        assert decoded.dtype == np.uint8

        # Per-frame PSNR floor — h264 ultrafast crf=28 on real camera data
        # comfortably clears 30 dB; we set the floor at 28 dB to absorb
        # codec/preset quirks while still catching catastrophic regressions.
        for i, (orig, back) in enumerate(zip(frames, decoded, strict=False)):
            psnr = _psnr(orig, back)
            assert psnr >= 28.0, f"frame {i}: PSNR {psnr:.2f} below floor"

        # Envelope timestamps round-trip exactly.
        assert np.array_equal(episode.publish_timestamps[topic], timestamps[:n_test])
