# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Depth codec + Writer/Reader round-trip tests.

Three layers:
  1. Codec sanity: ``encode_png16`` / ``decode_png16`` on hand-picked values
     (invalid sentinel passthrough, boundary codes, max quantization error).
  2. Writer deferred registration: ``write`` before ``set_runtime_metadata``
     raises; double-set raises; missing/extra kwargs raise.
  3. Real episode round-trip: load 10 frames from
     ``recordings/episode_..._raw/head.depth/``, write through the composable
     pipeline, read back, assert invalid mask is preserved exactly and valid
     pixels are within half-quantum of the original.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dexdata.handlers.containers import (
    DepthImageSpec,
)
from dexdata.handlers.depth_codec import decode_png16, encode_png16
from dexdata.mcap_utils.reader import Reader
from dexdata.mcap_utils.writer import Writer
from dexdata.spec import SignalSpec, Spec

REAL_EPISODE_RAW = Path(
    "/path_to_your_recordings/recordings/episode_20260508_112520_915215_raw"
)
REAL_DEPTH_DIR = REAL_EPISODE_RAW / "head.depth"


# --- 1. Codec sanity --------------------------------------------------------


def test_codec_invalid_sentinel_passthrough() -> None:
    min_r, max_r = 0.10, 8.0
    arr = np.array(
        [
            [0.0, np.nan, -1.0, 0.05, 8.5, np.inf],  # all → invalid
            [0.10, 4.05, 8.0, 4.0, 4.0, 4.0],  # valid
        ],
        dtype=np.float32,
    )
    back = decode_png16(encode_png16(arr, min_r, max_r), min_r, max_r)
    # Row 0: every element invalid → 0.0
    assert np.array_equal(back[0], np.zeros(6, dtype=np.float32))
    # Row 1: every element valid → finite, > 0
    assert np.all(back[1] > 0)


def test_codec_quantization_within_half_quantum() -> None:
    min_r, max_r = 0.10, 8.0
    quantum = (max_r - min_r) / 65534
    rng = np.random.default_rng(0)
    arr = rng.uniform(min_r, max_r, size=(64, 64)).astype(np.float32)
    back = decode_png16(encode_png16(arr, min_r, max_r), min_r, max_r)
    err = np.abs(arr - back)
    # Round-to-nearest gives at most quantum/2; allow tiny float epsilon on top.
    assert err.max() <= quantum / 2 + 1e-6


def test_codec_boundary_codes_round_trip() -> None:
    min_r, max_r = 0.10, 8.0
    arr = np.array([[min_r, max_r]], dtype=np.float32)
    back = decode_png16(encode_png16(arr, min_r, max_r), min_r, max_r)
    quantum = (max_r - min_r) / 65534
    assert abs(back[0, 0] - min_r) < quantum / 2 + 1e-6
    assert abs(back[0, 1] - max_r) < quantum / 2 + 1e-6


# --- 2. Writer deferred registration ---------------------------------------


def _depth_only_spec() -> Spec:
    return Spec(
        spec_id="depth_only_test",
        spec_version="0.1.0",
        signals=(
            SignalSpec(
                topic="/camera/head_left/depth",
                unit="m",
                container=DepthImageSpec(),
            ),
        ),
    )


def test_writer_pending_topic_blocks_write() -> None:
    spec = _depth_only_spec()
    arr = np.zeros((4, 4), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        with Writer(Path(tmp) / "ep", spec) as writer:
            assert writer.pending_topics == ("/camera/head_left/depth",)
            with pytest.raises(RuntimeError, match="runtime metadata not set"):
                writer.write("/camera/head_left/depth", arr, 0, 0)


def test_writer_set_runtime_metadata_validates() -> None:
    spec = _depth_only_spec()
    with tempfile.TemporaryDirectory() as tmp:
        with Writer(Path(tmp) / "ep", spec) as writer:
            with pytest.raises(ValueError, match="missing"):
                writer.set_runtime_metadata("/camera/head_left/depth", min_range=0.1)
            with pytest.raises(ValueError, match="unexpected"):
                writer.set_runtime_metadata(
                    "/camera/head_left/depth", min_range=0.1, max_range=8.0, foo=1
                )
            writer.set_runtime_metadata(
                "/camera/head_left/depth", min_range=0.1, max_range=8.0
            )
            assert writer.pending_topics == ()
            with pytest.raises(ValueError, match="already set"):
                writer.set_runtime_metadata(
                    "/camera/head_left/depth", min_range=0.2, max_range=9.0
                )


def test_writer_set_runtime_metadata_unknown_topic() -> None:
    spec = _depth_only_spec()
    with tempfile.TemporaryDirectory() as tmp:
        with Writer(Path(tmp) / "ep", spec) as writer:
            with pytest.raises(KeyError, match="not in spec"):
                writer.set_runtime_metadata(
                    "/no/such/topic", min_range=0.1, max_range=8.0
                )


# --- 3. Real-episode round-trip --------------------------------------------


@pytest.mark.skipif(
    not REAL_DEPTH_DIR.exists(),
    reason=f"real episode not available at {REAL_EPISODE_RAW}",
)
def test_real_depth_roundtrip() -> None:
    meta = json.loads((REAL_DEPTH_DIR / "meta.json").read_text())
    min_r = float(meta["min_range"])
    max_r = float(meta["max_range"])
    quantum = (max_r - min_r) / 65534

    timestamps = np.load(REAL_DEPTH_DIR / "timestamps.npy")
    n_test = 10
    frames = [np.load(REAL_DEPTH_DIR / f"{i:06d}.npy") for i in range(n_test)]

    spec = _depth_only_spec()
    topic = "/camera/head_left/depth"

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "ep"
        with Writer(out_dir, spec) as writer:
            writer.set_runtime_metadata(topic, min_range=min_r, max_range=max_r)
            for i, arr in enumerate(frames):
                ts = int(timestamps[i])
                writer.write(topic, arr, publish_ts_ns=ts, recv_ts_ns=ts)

        reader = Reader(out_dir)
        # Spec recovery includes the runtime min/max via the reader's metadata pull.
        assert reader.spec.signals[0].container == DepthImageSpec()
        assert reader.spec.signals[0].topic == topic

        episode = reader.read_episode()
        decoded = episode.signals[topic]
        assert decoded.shape == (n_test, *frames[0].shape)
        assert decoded.dtype == np.float32

        for i, original in enumerate(frames):
            d = decoded[i]
            orig_invalid = original == 0
            dec_invalid = d == 0
            # Invalid mask preserved exactly across the round-trip.
            assert np.array_equal(orig_invalid, dec_invalid), f"frame {i} invalid mask"
            valid = ~orig_invalid
            err = np.abs(original[valid] - d[valid])
            assert err.max() <= quantum / 2 + 1e-6, (
                f"frame {i} max err {err.max():.6e} > half quantum {quantum / 2:.6e}"
            )

        # Timestamps round-trip.
        assert np.array_equal(episode.publish_timestamps[topic], timestamps[:n_test])
        assert np.array_equal(episode.recv_timestamps[topic], timestamps[:n_test])
