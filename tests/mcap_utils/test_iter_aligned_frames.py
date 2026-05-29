# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Tests for Reader.iter_aligned_frames — synchronized multi-topic streaming.

We exercise the contract by writing a synthetic two-topic episode with
deliberately offset timestamps and asserting each yielded frame picks
the right nearest neighbor.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dexdata.handlers.containers import NumericArraySpec
from dexdata.mcap_utils.reader import Reader
from dexdata.mcap_utils.writer import Writer
from dexdata.spec import SignalSpec, Spec

BASE = 1_700_000_000_000_000_000


def _two_topic_spec() -> Spec:
    f32 = np.dtype("float32")
    return Spec(
        spec_id="iter_aligned",
        spec_version="0.1.0",
        signals=(
            SignalSpec(
                topic="/ref",
                unit="rad",
                container=NumericArraySpec(shape=(1,), dtype=f32),
            ),
            SignalSpec(
                topic="/other",
                unit="rad",
                container=NumericArraySpec(shape=(1,), dtype=f32),
            ),
        ),
    )


def _write_episode(out_dir: Path) -> None:
    """Reference at 20 Hz, other at 10 Hz (half as many frames).

    /ref values are i; /other values are i. The aligner should pair
    each /ref frame at index i with /other index ~i/2.
    """
    spec = _two_topic_spec()
    f32 = np.dtype("float32")
    with Writer(out_dir, spec) as w:
        for i in range(10):
            ts = BASE + i * 50_000_000  # 20 Hz
            w.write("/ref", np.array([i], dtype=f32), ts, ts)
        for j in range(5):
            ts = BASE + j * 100_000_000  # 10 Hz, aligned with even /ref
            w.write("/other", np.array([j * 10], dtype=f32), ts, ts)


def test_iter_aligned_frames_matches_nearest_neighbor() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "ep"
        _write_episode(out_dir)
        reader = Reader(out_dir)
        frames = list(reader.iter_aligned_frames("/ref"))

        assert len(frames) == 10
        for i, (ref_ts, signals) in enumerate(frames):
            assert ref_ts == BASE + i * 50_000_000
            assert set(signals) == {"/ref", "/other"}
            assert int(signals["/ref"][0]) == i
            # /other is sparser — nearest match is index i // 2.
            assert int(signals["/other"][0]) == (i // 2) * 10


def test_iter_aligned_frames_subset_of_topics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "ep"
        _write_episode(out_dir)
        reader = Reader(out_dir)
        frames = list(reader.iter_aligned_frames("/ref", topics=["/ref"]))
        assert all(set(f) == {"/ref"} for _, f in frames)


def test_iter_aligned_frames_max_gap_ns_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "ep"
        _write_episode(out_dir)
        reader = Reader(out_dir)
        # /other publishes every 100 ms, /ref every 50 ms → expected max
        # gap ~50 ms. Set 10 ms cap → must raise.
        with pytest.raises(ValueError, match="max_gap_ns"):
            list(reader.iter_aligned_frames("/ref", max_gap_ns=10_000_000))


def test_iter_aligned_frames_unknown_reference_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "ep"
        _write_episode(out_dir)
        reader = Reader(out_dir)
        with pytest.raises(KeyError, match="reference topic not in spec"):
            list(reader.iter_aligned_frames("/not_a_topic"))


def test_iter_aligned_frames_unknown_topic_in_subset_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "ep"
        _write_episode(out_dir)
        reader = Reader(out_dir)
        with pytest.raises(KeyError, match="unknown topics"):
            list(reader.iter_aligned_frames("/ref", topics=["/ref", "/missing"]))


def test_iter_aligned_frames_uses_log_time_not_publish_time() -> None:
    """Alignment axis is log_time (recorder clock), not publish_time.

    Sensor publishers in real recordings do not share a clock domain,
    so publish_time is not comparable across topics. We simulate this
    by writing ``/other`` with a 1-second publisher-clock skew: its
    publish timestamps are 1 s ahead of its log timestamps, while
    ``/ref`` keeps publish == log. The two clocks now disagree on
    what "nearest neighbor" means:

    * log-time (correct): /ref[i] pairs with /other[i // 2], identical
      to the no-skew baseline — log timestamps line up at BASE + 0,
      100ms, 200ms, ...
    * publish-time (broken): every /ref frame sits ~1 s before every
      /other publish-time, so the bisect picks /other index 0 for
      every i, and signals["/other"] == 0 across the whole episode.

    This test would fail under the old publish-time alignment.
    """
    spec = _two_topic_spec()
    f32 = np.dtype("float32")
    publisher_skew_ns = 1_000_000_000  # /other publisher clock 1 s ahead

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "ep"
        with Writer(out_dir, spec) as w:
            for i in range(10):
                log_ts = BASE + i * 50_000_000  # /ref @ 20 Hz, no skew
                w.write("/ref", np.array([i], dtype=f32), log_ts, log_ts)
            for j in range(5):
                log_ts = BASE + j * 100_000_000  # /other @ 10 Hz
                publish_ts = log_ts + publisher_skew_ns
                w.write("/other", np.array([j * 10], dtype=f32), publish_ts, log_ts)

        reader = Reader(out_dir)
        frames = list(reader.iter_aligned_frames("/ref"))

        assert len(frames) == 10
        for i, (ref_ts, signals) in enumerate(frames):
            # Yielded reference timestamp is log_time (not publish_time).
            assert ref_ts == BASE + i * 50_000_000
            # /other resolves to log-time nearest neighbor (i // 2),
            # NOT publish-time nearest neighbor (which would be 0 for all i).
            assert int(signals["/other"][0]) == (i // 2) * 10, (
                f"frame {i}: got /other={int(signals['/other'][0])}, "
                f"expected {(i // 2) * 10}. If this returned 0 across all "
                f"frames, alignment is using publish_time (wrong) instead "
                f"of log_time."
            )
