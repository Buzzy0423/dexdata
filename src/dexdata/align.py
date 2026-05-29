# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Multi-rate timestamp alignment for composable :class:`Episode` objects.

Real-world episodes routinely have topics published at different cadences
— state at 100 Hz, action whenever the operator moves, cameras at 30 Hz.
The composable :class:`Reader` returns each topic at its own rate (one
``np.ndarray`` per topic, lengths potentially unequal). Most downstream
consumers (training pipelines, visualizers) want one row per timestep, so
they need a *single* timeline plus per-topic values aligned onto it.

This module supplies :func:`align_episode`, a pure-numpy resampler that
takes an :class:`Episode` and a reference topic and returns a new
:class:`Episode` whose every signal is indexed by the reference's
``recv_timestamps`` (log_time).

Recv is the recorder's clock — stamped at one place for every message
regardless of where the publisher lives (PC, Jetson, SoC, etc.). It's
coherent across multi-publisher recordings even when the publishers'
clocks aren't synchronized.

Aligned episodes intentionally drop ``publish_timestamps`` (returned as
an empty dict): publisher clocks are not comparable across topics, so
exposing a per-row publish-time on the common timeline would invite
exactly the misuse this module exists to prevent. Inspect raw
``publish_timestamps`` on the pre-align :class:`Episode` if you need
per-topic sensor-capture diagnostics.

Two alignment modes:

* ``"nearest"`` — for each reference recv timestamp, take the source
  frame whose recv timestamp is closest in absolute distance
  (bisect-based, O(N+M)). Optional ``max_gap_ns`` raises if any pair is
  too far apart.
* ``"forward_fill"`` — for each reference recv timestamp, take the
  most recent source frame at or before it. Raises if the very first
  reference timestamp precedes the first source frame (no value to
  forward-fill from). Suited for *commands* that persist between
  updates (action/qpos).

Both modes leave the reference topic untouched; only its peers are
resampled.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Literal

import numpy as np

from .mcap_utils.episode import Episode

AlignMode = Literal["nearest", "forward_fill"]


def align_episode(
    episode: Episode,
    reference_topic: str,
    *,
    mode: AlignMode = "nearest",
    max_gap_ns: int | None = None,
) -> Episode:
    """Resample every topic onto the reference topic's recv timestamps.

    Args:
        episode: Source episode (per-topic arrays at each topic's own rate).
        reference_topic: The topic whose ``recv_timestamps`` (log_time)
            becomes the common timeline. Must be present in
            ``episode.signals``.
        mode: ``"nearest"`` or ``"forward_fill"``.
        max_gap_ns: Optional hard cap on the timestamp gap between a
            reference recv timestamp and its chosen source frame's recv
            timestamp. ``None`` means no cap. Raises :class:`ValueError`
            if exceeded.

    Returns:
        A new :class:`Episode` with the same ``spec`` but every topic
        re-indexed by the reference's recv timestamps. Signal arrays
        keep their per-frame shape; only the leading axis is resampled.
        ``publish_timestamps`` is returned as an empty dict — publisher
        clocks are not comparable across topics, so no per-row
        publish-time is meaningful on the aligned timeline.
    """
    if reference_topic not in episode.signals:
        raise KeyError(f"reference topic not in episode: {reference_topic}")

    ref_recv = episode.recv_timestamps[reference_topic]
    if ref_recv.size == 0:
        # Empty reference → empty aligned episode (every topic also empty).
        signals = {t: arr[:0] for t, arr in episode.signals.items()}
        recv_ts = {t: ts[:0] for t, ts in episode.recv_timestamps.items()}
        return Episode(
            spec=episode.spec,
            signals=signals,
            publish_timestamps={},
            recv_timestamps=recv_ts,
        )

    aligned_signals: dict[str, np.ndarray] = {}
    aligned_recv: dict[str, np.ndarray] = {}

    for topic, src_signal in episode.signals.items():
        src_recv = episode.recv_timestamps[topic]

        if topic == reference_topic:
            aligned_signals[topic] = src_signal
            aligned_recv[topic] = src_recv
            continue

        if src_recv.size == 0:
            # Source has no frames — can't resample. Mode determines verdict.
            if mode == "forward_fill":
                raise ValueError(
                    f"forward_fill: source topic {topic!r} is empty; "
                    f"no value to forward-fill from"
                )
            # nearest: produce zero-frame stack matching the reference cadence
            # but with zero rows is impossible — instead, fall through to
            # raise so callers don't get silently-junk-filled output.
            raise ValueError(f"nearest: source topic {topic!r} is empty; cannot align")

        idx = (
            _nearest_indices(src_recv, ref_recv, max_gap_ns)
            if mode == "nearest"
            else _forward_fill_indices(src_recv, ref_recv, max_gap_ns, topic)
        )
        aligned_signals[topic] = src_signal[idx]
        aligned_recv[topic] = src_recv[idx]

    return Episode(
        spec=episode.spec,
        signals=aligned_signals,
        publish_timestamps={},
        recv_timestamps=aligned_recv,
    )


def _nearest_indices(
    src_ts: np.ndarray, ref_ts: np.ndarray, max_gap_ns: int | None
) -> np.ndarray:
    """For each ``ref_ts``, return the index in ``src_ts`` closest in absolute distance."""
    src_list = src_ts.tolist()  # bisect on Python list is faster than np for short refs
    out = np.empty(ref_ts.shape[0], dtype=np.int64)
    for i, t in enumerate(ref_ts.tolist()):
        j = bisect_left(src_list, t)
        if j == 0:
            chosen = 0
        elif j == len(src_list):
            chosen = len(src_list) - 1
        else:
            chosen = j - 1 if (t - src_list[j - 1]) <= (src_list[j] - t) else j
        if max_gap_ns is not None and abs(src_list[chosen] - t) > max_gap_ns:
            raise ValueError(
                f"nearest: gap {abs(src_list[chosen] - t)} ns at ref index {i} "
                f"exceeds max_gap_ns={max_gap_ns}"
            )
        out[i] = chosen
    return out


def _forward_fill_indices(
    src_ts: np.ndarray,
    ref_ts: np.ndarray,
    max_gap_ns: int | None,
    topic: str,
) -> np.ndarray:
    """For each ``ref_ts``, return the index of the latest ``src_ts`` ≤ it."""
    src_list = src_ts.tolist()
    out = np.empty(ref_ts.shape[0], dtype=np.int64)
    for i, t in enumerate(ref_ts.tolist()):
        # bisect_right finds the rightmost position where t could be inserted;
        # the element at that-position-minus-one is the latest src_ts ≤ t.
        j = bisect_right(src_list, t)
        if j == 0:
            raise ValueError(
                f"forward_fill: ref timestamp at index {i} ({t} ns) precedes "
                f"first source frame on topic {topic!r} ({src_list[0]} ns); "
                f"no value to forward-fill from"
            )
        chosen = j - 1
        if max_gap_ns is not None and (t - src_list[chosen]) > max_gap_ns:
            raise ValueError(
                f"forward_fill: gap {t - src_list[chosen]} ns at ref index {i} "
                f"on topic {topic!r} exceeds max_gap_ns={max_gap_ns}"
            )
        out[i] = chosen
    return out
