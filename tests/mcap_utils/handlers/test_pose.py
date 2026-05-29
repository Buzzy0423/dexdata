# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Pose handler + Writer/Reader round-trip tests.

Two layers:
  1. Codec sanity: handler.serialize → handler.deserialize, verify byte-equal
     round-trip on random poses (positions in metres, quaternions normalized).
  2. End-to-end MCAP round-trip: write 100 poses through Writer/Reader,
     verify exact equality and timestamp preservation.

Plus three validation tests: wrong shape, wrong dtype, non-1D rejected.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dexdata.handlers.containers import PoseSpec
from dexdata.mcap_utils.handlers.pose import POSE_DTYPE, POSE_SHAPE, PoseHandler
from dexdata.mcap_utils.reader import Reader
from dexdata.mcap_utils.writer import Writer
from dexdata.spec import SignalSpec, Spec

N_FRAMES = 100
TOPIC = "/imu/right_wrist/pose"
BASE_TS_NS = 1_700_000_000_000_000_000
PERIOD_NS = 33_000_000  # ~30 Hz
RECV_DELAY_NS = 500_000


def _random_poses(n: int = N_FRAMES) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    out: list[np.ndarray] = []
    for _ in range(n):
        pos = rng.uniform(-1.0, 1.0, size=3)
        quat = rng.standard_normal(4)
        quat = quat / np.linalg.norm(quat)
        out.append(np.concatenate([pos, quat]).astype(POSE_DTYPE))
    return out


def _spec() -> Spec:
    return Spec(
        spec_id="pose_only_test",
        spec_version="0.1.0",
        signals=(SignalSpec(topic=TOPIC, unit="m+quat", container=PoseSpec()),),
    )


# --- 1. Codec sanity --------------------------------------------------------


def test_inmemory_roundtrip() -> None:
    h = PoseHandler(PoseSpec())
    for arr in _random_poses(50):
        ((payload, _, _),) = list(h.serialize(arr, 0, 0))
        (arr_back,) = list(h.deserialize(payload))
        assert arr_back.dtype == POSE_DTYPE
        assert arr_back.shape == POSE_SHAPE
        assert np.array_equal(arr_back, arr)


# --- 2. End-to-end MCAP round-trip -----------------------------------------


def test_writer_reader_roundtrip() -> None:
    spec = _spec()
    poses = _random_poses()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ep"
        with Writer(out, spec) as writer:
            for i, p in enumerate(poses):
                publish_ts = BASE_TS_NS + i * PERIOD_NS
                recv_ts = publish_ts + RECV_DELAY_NS
                writer.write(TOPIC, p, publish_ts_ns=publish_ts, recv_ts_ns=recv_ts)

        reader = Reader(out)
        assert reader.spec.signals[0].container == PoseSpec()
        episode = reader.read_episode()
        decoded = episode.signals[TOPIC]
        assert decoded.shape == (N_FRAMES, 7)
        assert decoded.dtype == POSE_DTYPE
        for i, original in enumerate(poses):
            assert np.array_equal(decoded[i], original), f"pose {i} mismatch"

        expected_pub = BASE_TS_NS + np.arange(N_FRAMES) * PERIOD_NS
        assert np.array_equal(episode.publish_timestamps[TOPIC], expected_pub)
        assert np.array_equal(
            episode.recv_timestamps[TOPIC], expected_pub + RECV_DELAY_NS
        )


# --- 3. Validation ---------------------------------------------------------


def test_serialize_rejects_wrong_shape() -> None:
    h = PoseHandler(PoseSpec())
    bad = np.zeros((6,), dtype=POSE_DTYPE)
    with pytest.raises(ValueError, match="shape mismatch"):
        list(h.serialize(bad, 0, 0))


def test_serialize_rejects_wrong_dtype() -> None:
    h = PoseHandler(PoseSpec())
    bad = np.zeros(POSE_SHAPE, dtype=np.float32)
    with pytest.raises(ValueError, match="dtype mismatch"):
        list(h.serialize(bad, 0, 0))


def test_serialize_rejects_non_1d() -> None:
    h = PoseHandler(PoseSpec())
    bad = np.zeros((1, 7), dtype=POSE_DTYPE)
    with pytest.raises(ValueError, match="shape mismatch"):
        list(h.serialize(bad, 0, 0))
