# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Tests for composable.align.align_episode.

Coverage:
  * identity: already-aligned episode is unchanged.
  * nearest: skewed source onto denser reference picks the closest frame.
  * forward_fill: sparse source onto denser reference holds last value.
  * error paths: missing reference, empty source, ref-before-source for
    forward_fill, max_gap_ns exceeded.
"""

from __future__ import annotations

import numpy as np
import pytest

from dexdata.align import align_episode
from dexdata.handlers.containers import NumericArraySpec
from dexdata.mcap_utils.episode import Episode
from dexdata.spec import SignalSpec, Spec


def _make_episode(
    *,
    state: tuple[np.ndarray, np.ndarray, np.ndarray],  # signal, pub_ts, recv_ts
    action: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Episode:
    f32 = np.dtype("float32")
    spec = Spec(
        spec_id="test",
        spec_version="0.1.0",
        signals=(
            SignalSpec(
                topic="/state",
                unit="rad",
                container=NumericArraySpec(shape=(2,), dtype=f32),
            ),
            SignalSpec(
                topic="/action",
                unit="rad",
                container=NumericArraySpec(shape=(2,), dtype=f32),
            ),
        ),
    )
    return Episode(
        spec=spec,
        signals={"/state": state[0], "/action": action[0]},
        publish_timestamps={"/state": state[1], "/action": action[1]},
        recv_timestamps={"/state": state[2], "/action": action[2]},
    )


def test_identity_already_aligned() -> None:
    """When source and reference share timestamps, alignment is a no-op."""
    ts = np.array([100, 200, 300], dtype=np.int64)
    state = np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]], dtype=np.float32)
    action = np.array([[10.0, 10.1], [20.0, 20.1], [30.0, 30.1]], dtype=np.float32)
    ep = _make_episode(state=(state, ts, ts), action=(action, ts, ts))

    aligned = align_episode(ep, "/state", mode="nearest")
    assert np.array_equal(aligned.signals["/state"], state)
    assert np.array_equal(aligned.signals["/action"], action)
    # publish_timestamps is dropped on alignment — publisher clocks are not
    # comparable across topics, so no per-row publish-time is meaningful.
    assert aligned.publish_timestamps == {}


def test_nearest_picks_closest_frame() -> None:
    """Ref [100, 200, 300]; source [110, 195, 305] → indices [0, 1, 2]."""
    ref_ts = np.array([100, 200, 300], dtype=np.int64)
    src_ts = np.array([110, 195, 305], dtype=np.int64)
    state = np.zeros((3, 2), dtype=np.float32)
    action = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    ep = _make_episode(
        state=(state, ref_ts, ref_ts),
        action=(action, src_ts, src_ts),
    )
    aligned = align_episode(ep, "/state", mode="nearest")
    assert np.array_equal(aligned.signals["/action"], action)
    assert aligned.publish_timestamps == {}


def test_nearest_chooses_lower_on_tie() -> None:
    """Distance ties resolve to the earlier source index (matches legacy)."""
    ref_ts = np.array([150], dtype=np.int64)
    src_ts = np.array([100, 200], dtype=np.int64)
    state = np.zeros((1, 2), dtype=np.float32)
    action = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    ep = _make_episode(
        state=(state, ref_ts, ref_ts),
        action=(action, src_ts, src_ts),
    )
    aligned = align_episode(ep, "/state", mode="nearest")
    # Both src timestamps are 50ns from ref; tie → earlier (index 0).
    assert np.array_equal(aligned.signals["/action"], action[:1])


def test_forward_fill_holds_last_value() -> None:
    """Sparse action onto dense state holds the last command between updates."""
    ref_ts = np.array([100, 150, 200, 250, 300], dtype=np.int64)
    # Action published only at t=100 and t=250.
    src_ts = np.array([100, 250], dtype=np.int64)
    state = np.zeros((5, 2), dtype=np.float32)
    action = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    ep = _make_episode(
        state=(state, ref_ts, ref_ts),
        action=(action, src_ts, src_ts),
    )
    aligned = align_episode(ep, "/state", mode="forward_fill")
    expected = np.array(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]],
        dtype=np.float32,
    )
    assert np.array_equal(aligned.signals["/action"], expected)
    # publish_timestamps is dropped on alignment (publisher clocks not
    # cross-topic-comparable); recv_timestamps is the only valid timeline.
    assert aligned.publish_timestamps == {}


def test_forward_fill_raises_when_ref_precedes_first_source() -> None:
    """No earlier source value → no fill candidate → explicit error."""
    ref_ts = np.array([50, 100], dtype=np.int64)
    src_ts = np.array([75], dtype=np.int64)
    state = np.zeros((2, 2), dtype=np.float32)
    action = np.array([[1.0, 1.0]], dtype=np.float32)
    ep = _make_episode(
        state=(state, ref_ts, ref_ts),
        action=(action, src_ts, src_ts),
    )
    with pytest.raises(ValueError, match="precedes first source frame"):
        align_episode(ep, "/state", mode="forward_fill")


def test_max_gap_ns_enforced_nearest() -> None:
    ref_ts = np.array([100, 1000], dtype=np.int64)
    src_ts = np.array([110], dtype=np.int64)
    state = np.zeros((2, 2), dtype=np.float32)
    action = np.array([[1.0, 1.0]], dtype=np.float32)
    ep = _make_episode(
        state=(state, ref_ts, ref_ts),
        action=(action, src_ts, src_ts),
    )
    with pytest.raises(ValueError, match="exceeds max_gap_ns"):
        align_episode(ep, "/state", mode="nearest", max_gap_ns=100)


def test_unknown_reference_raises() -> None:
    ts = np.array([100], dtype=np.int64)
    state = np.zeros((1, 2), dtype=np.float32)
    action = np.zeros((1, 2), dtype=np.float32)
    ep = _make_episode(state=(state, ts, ts), action=(action, ts, ts))
    with pytest.raises(KeyError, match="reference topic not in episode"):
        align_episode(ep, "/no/such/topic", mode="nearest")


def test_empty_reference_returns_empty_aligned() -> None:
    """Reference with zero frames produces a fully-empty aligned episode."""
    empty = np.empty((0, 2), dtype=np.float32)
    empty_ts = np.empty((0,), dtype=np.int64)
    src_ts = np.array([100, 200], dtype=np.int64)
    action = np.zeros((2, 2), dtype=np.float32)
    ep = _make_episode(
        state=(empty, empty_ts, empty_ts),
        action=(action, src_ts, src_ts),
    )
    aligned = align_episode(ep, "/state", mode="nearest")
    assert aligned.signals["/state"].shape == (0, 2)
    assert aligned.signals["/action"].shape == (0, 2)
