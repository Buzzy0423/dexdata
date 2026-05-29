# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""End-to-end Writer ↔ Reader for a NumericArray-only spec (Step 3).

Builds a small in-memory spec covering the kind of scalars/vectors a Vega
gripper episode would carry (qpos, qvel, wrench), writes 100 frames of
synthetic data across all channels, reads back, and asserts:

  * Reader-recovered Spec matches the constructed Spec (round-trip via
    channel metadata, no spec JSON needed at read time).
  * Per-topic numpy dicts are byte-equal to the original synthetic data.
  * Per-topic publish_time / log_time arrays round-trip exactly, with the
    simulated 0.5 ms publisher → subscriber delay preserved.
  * Streaming ``iter_messages`` yields the same values in the same order.

The fixture is built inline rather than loaded from a JSON file: this
test's purpose is the writer/reader contract, not spec parsing (covered
elsewhere). The full real-data round-trip test
(``test_real_episode_roundtrip.py``) drives the canonical
``specs/embodiment/vega_1u_gripper.json``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from dexdata.handlers.containers import NumericArraySpec
from dexdata.mcap_utils.reader import Reader
from dexdata.mcap_utils.writer import Writer
from dexdata.spec import SignalSpec, Spec

N_FRAMES = 100
BASE_TS_NS = 1_700_000_000_000_000_000
PERIOD_NS = 33_000_000  # ~30 Hz
RECV_DELAY_NS = 500_000  # 0.5 ms


def _fixture_spec() -> Spec:
    """A handful of NumericArray signals across state + action namespaces."""
    f32 = np.dtype("float32")
    return Spec(
        spec_id="numeric_smoke",
        spec_version="0.1.0",
        signals=(
            SignalSpec(
                topic="/robot/state/left_arm/qpos",
                unit="rad",
                container=NumericArraySpec(shape=(7,), dtype=f32),
            ),
            SignalSpec(
                topic="/robot/state/left_arm/qvel",
                unit="rad/s",
                container=NumericArraySpec(shape=(7,), dtype=f32),
            ),
            SignalSpec(
                topic="/robot/state/left_hand/qpos",
                unit="rad",
                container=NumericArraySpec(shape=(1,), dtype=f32),
            ),
            SignalSpec(
                topic="/robot/state/left_hand/wrench/force",
                unit="N",
                container=NumericArraySpec(shape=(3,), dtype=f32),
            ),
            SignalSpec(
                topic="/robot/state/left_hand/wrench/torque",
                unit="Nm",
                container=NumericArraySpec(shape=(3,), dtype=f32),
            ),
            SignalSpec(
                topic="/robot/action/left_arm/qpos",
                unit="rad",
                container=NumericArraySpec(shape=(7,), dtype=f32),
            ),
            SignalSpec(
                topic="/robot/action/left_hand/qpos",
                unit="rad",
                container=NumericArraySpec(shape=(1,), dtype=f32),
            ),
        ),
    )


def _synthesize(spec: Spec) -> dict[str, np.ndarray]:
    """Generate ``(N_FRAMES, *shape)`` random arrays for every signal."""
    rng = np.random.default_rng(0)
    out: dict[str, np.ndarray] = {}
    for s in spec.signals:
        assert isinstance(s.container, NumericArraySpec)
        shape = (N_FRAMES, *s.container.shape)
        out[s.topic] = rng.standard_normal(shape).astype(s.container.dtype)
    return out


def test_writer_reader_roundtrip() -> None:
    spec = _fixture_spec()
    data = _synthesize(spec)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "ep0001"

        with Writer(out_dir, spec) as writer:
            for i in range(N_FRAMES):
                publish_ts = BASE_TS_NS + i * PERIOD_NS
                recv_ts = publish_ts + RECV_DELAY_NS
                for topic, arr in data.items():
                    writer.write(topic, arr[i], publish_ts, recv_ts)

        reader = Reader(out_dir)

        # Spec recovered from channel metadata equals the constructed one.
        assert reader.spec.spec_id == spec.spec_id
        assert reader.spec.spec_version == spec.spec_version
        assert reader.spec.by_topic() == spec.by_topic()

        # Materialized read.
        episode = reader.read_episode()
        nd = episode.to_numpy_dict()
        assert set(nd) == set(data)
        for topic, expected in data.items():
            assert nd[topic].shape == expected.shape, topic
            assert nd[topic].dtype == expected.dtype, topic
            assert np.array_equal(nd[topic], expected), topic

        # Envelope timestamps round-trip exactly with the simulated delay.
        expected_publish = np.array(
            [BASE_TS_NS + i * PERIOD_NS for i in range(N_FRAMES)], dtype=np.int64
        )
        expected_recv = expected_publish + RECV_DELAY_NS
        for topic in data:
            assert np.array_equal(episode.publish_timestamps[topic], expected_publish)
            assert np.array_equal(episode.recv_timestamps[topic], expected_recv)

        # Streaming yields same values in same order.
        topic = "/robot/state/left_arm/qpos"
        streamed = list(reader.iter_messages(topic))
        assert len(streamed) == N_FRAMES
        for i, (arr, pub_ts, recv_ts) in enumerate(streamed):
            assert np.array_equal(arr, data[topic][i])
            assert pub_ts == expected_publish[i]
            assert recv_ts == expected_recv[i]
