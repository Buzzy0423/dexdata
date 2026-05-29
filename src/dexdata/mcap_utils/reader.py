# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Spec-driven MCAP reader.

The reader recovers the ``Spec`` from each channel's metadata (decision 2.6
+ §6.1): no spec JSON on disk is required. Runtime channel metadata
(``min_range`` / ``max_range`` for DepthImage etc.) is also pulled from the
channel record and used to construct the right handler.

``read_episode()`` materializes all messages into stacked numpy arrays +
envelope-timestamp arrays; ``iter_messages(topic)`` is the streaming
alternative for large episodes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from mcap.reader import make_reader

from ..handlers.containers import ContainerSpec, build_container_from_metadata
from ..metadata import METADATA_FILE, EpisodeMetadata, load_metadata
from ..spec import SignalSpec, Spec
from .episode import Episode
from .handlers import HANDLER_TYPES, Handler, make_handler, runtime_field_types
from .writer import EPISODE_FILE


class Reader:
    """Spec-driven MCAP reader."""

    def __init__(self, out_dir: Path | str) -> None:
        self._out_dir = Path(out_dir)
        self._path = self._out_dir / EPISODE_FILE
        if not self._path.exists():
            raise FileNotFoundError(self._path)

        self._spec, self._handlers = self._recover_spec_and_handlers()
        self._metadata: EpisodeMetadata | None = None
        self._metadata_loaded = False

    @property
    def spec(self) -> Spec:
        return self._spec

    @property
    def path(self) -> Path:
        return self._path

    @property
    def metadata(self) -> EpisodeMetadata | None:
        """Lazy-load the ``metadata.json`` sidecar, or ``None`` if absent."""
        if not self._metadata_loaded:
            sidecar = self._out_dir / METADATA_FILE
            self._metadata = load_metadata(sidecar) if sidecar.exists() else None
            self._metadata_loaded = True
        return self._metadata

    def iter_messages(self, topic: str) -> Iterator[tuple[np.ndarray, int, int]]:
        """Yield ``(value, publish_time_ns, log_time_ns)`` per *decoded value* on ``topic``.

        For stateless containers there is one decoded value per MCAP message.
        For stateful ones (video) decoders may buffer; values are paired with
        envelope timestamps positionally via a FIFO queue (decision: see
        ``handlers/compressed_video.py`` on order preservation).
        """
        if topic not in self._handlers:
            raise KeyError(f"topic not in spec: {topic}")
        handler = self._handlers[topic]
        ts_queue: list[tuple[int, int]] = []
        with open(self._path, "rb") as f:
            reader = make_reader(f)
            for _schema, _channel, message in reader.iter_messages(topics=[topic]):
                ts_queue.append((message.publish_time, message.log_time))
                for value in handler.deserialize(message.data):
                    pt_ns, rt_ns = ts_queue.pop(0)
                    yield (value, pt_ns, rt_ns)
            for value in handler.flush_decode():
                pt_ns, rt_ns = ts_queue.pop(0)
                yield (value, pt_ns, rt_ns)

    def iter_aligned_frames(
        self,
        reference_topic: str,
        *,
        topics: list[str] | None = None,
        max_gap_ns: int | None = None,
    ) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """Yield per-frame ``(reference_log_ts_ns, {topic: value})`` tuples.

        Synchronized streaming counterpart to :meth:`read_episode` + the
        batch :func:`align_episode`. For each frame on ``reference_topic``
        we attach the nearest-in-log-time value from every other topic in
        ``topics`` (or every spec topic if ``topics`` is None).

        Alignment uses ``log_time`` (subscriber receive) rather than
        ``publish_time``: publishers across an episode (PC, Jetson, SoC
        cameras, etc.) do not share a clock domain, so publish_time is
        not comparable across topics. ``log_time`` is stamped by the
        single recorder process and is the only timestamp surface that
        can serve as a common timeline.

        Implementation note: we materialize each non-reference topic's
        timestamps + decoded values up front (single decoder pass each,
        same machinery as :meth:`iter_messages`), then stream the
        reference topic and bisect for nearest neighbors. That trades
        an O(N) up-front decode-and-buffer for true single-pass
        streaming over the reference, which is what callers typically
        want for visualization / inference replay.

        Args:
            reference_topic: Topic whose log times define the output
                cadence. Must be present in the spec.
            topics: Subset of spec topics to align (defaults to every
                spec topic, including the reference).
            max_gap_ns: Optional hard cap on the log-time gap between a
                reference frame and its chosen neighbor on each other
                topic. ``None`` disables the check. Raises
                :class:`ValueError` if exceeded.

        Yields:
            One tuple per reference frame: the reference's
            ``log_time_ns`` and a dict mapping each requested topic to
            its value at that frame.
        """

        if reference_topic not in self._handlers:
            raise KeyError(f"reference topic not in spec: {reference_topic}")

        spec_topics = [s.topic for s in self._spec.signals]
        wanted = topics if topics is not None else spec_topics
        unknown = set(wanted) - set(spec_topics)
        if unknown:
            raise KeyError(f"unknown topics: {sorted(unknown)}")

        # Materialize every non-reference topic's (value, log_ts) up
        # front. The reference topic we'll stream below.
        buffered: dict[str, tuple[list[np.ndarray], list[int]]] = {}
        for topic in wanted:
            if topic == reference_topic:
                continue
            values: list[np.ndarray] = []
            log_ns: list[int] = []
            for value, _pt, rt in self.iter_messages(topic):
                values.append(value)
                log_ns.append(int(rt))
            buffered[topic] = (values, log_ns)

        for ref_value, _ref_pt, ref_rt in self.iter_messages(reference_topic):
            frame: dict[str, np.ndarray] = {}
            if reference_topic in wanted:
                frame[reference_topic] = ref_value
            for topic, (values, log_ns) in buffered.items():
                if not log_ns:
                    raise ValueError(
                        f"iter_aligned_frames: source topic {topic!r} has "
                        f"no messages; cannot align"
                    )
                idx = _nearest_index(log_ns, int(ref_rt))
                if (
                    max_gap_ns is not None
                    and abs(log_ns[idx] - int(ref_rt)) > max_gap_ns
                ):
                    raise ValueError(
                        f"iter_aligned_frames: gap "
                        f"{abs(log_ns[idx] - int(ref_rt))} ns on {topic!r} "
                        f"exceeds max_gap_ns={max_gap_ns}"
                    )
                frame[topic] = values[idx]
            yield int(ref_rt), frame

    def read_episode(self) -> Episode:
        """Materialize every topic into stacked arrays (one ``open`` pass)."""
        per_topic: dict[str, list[tuple[np.ndarray, int, int]]] = {
            s.topic: [] for s in self._spec.signals
        }
        ts_queues: dict[str, list[tuple[int, int]]] = {
            s.topic: [] for s in self._spec.signals
        }
        with open(self._path, "rb") as f:
            reader = make_reader(f)
            for _schema, channel, message in reader.iter_messages():
                handler = self._handlers.get(channel.topic)
                if handler is None:
                    continue  # foreign topic, ignore
                ts_queues[channel.topic].append(
                    (message.publish_time, message.log_time)
                )
                for value in handler.deserialize(message.data):
                    pt_ns, rt_ns = ts_queues[channel.topic].pop(0)
                    per_topic[channel.topic].append((value, pt_ns, rt_ns))
        # End-of-stream flush per topic.
        for topic, handler in self._handlers.items():
            for value in handler.flush_decode():
                pt_ns, rt_ns = ts_queues[topic].pop(0)
                per_topic[topic].append((value, pt_ns, rt_ns))

        signals: dict[str, np.ndarray] = {}
        publish_ts: dict[str, np.ndarray] = {}
        recv_ts: dict[str, np.ndarray] = {}
        for topic, items in per_topic.items():
            signal = next(s for s in self._spec.signals if s.topic == topic)
            if items:
                arrs, p_ts, r_ts = zip(*items, strict=True)
                signals[topic] = np.stack(arrs)
                publish_ts[topic] = np.asarray(p_ts, dtype=np.int64)
                recv_ts[topic] = np.asarray(r_ts, dtype=np.int64)
            else:
                signals[topic] = _empty_for(signal.container)
                publish_ts[topic] = np.empty((0,), dtype=np.int64)
                recv_ts[topic] = np.empty((0,), dtype=np.int64)

        return Episode(
            spec=self._spec,
            signals=signals,
            publish_timestamps=publish_ts,
            recv_timestamps=recv_ts,
        )

    def _recover_spec_and_handlers(self) -> tuple[Spec, dict[str, Handler]]:
        signals: list[SignalSpec] = []
        handlers: dict[str, Handler] = {}
        spec_id: str | None = None
        spec_version: str | None = None
        with open(self._path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            if summary is None:
                raise ValueError(f"{self._path}: missing summary section")
            for channel in summary.channels.values():
                md: dict[str, Any] = dict(channel.metadata)
                this_spec_id = md.get("spec_id")
                this_spec_version = md.get("spec_version")
                if spec_id is None:
                    spec_id, spec_version = this_spec_id, this_spec_version
                elif (this_spec_id, this_spec_version) != (spec_id, spec_version):
                    raise ValueError(
                        f"{self._path}: channel {channel.topic} has spec "
                        f"{this_spec_id}@{this_spec_version}, expected "
                        f"{spec_id}@{spec_version}"
                    )
                container = build_container_from_metadata(md)
                runtime_kwargs = _runtime_kwargs_from_metadata(container, md)
                signals.append(
                    SignalSpec(
                        topic=channel.topic,
                        unit=md.get("unit", ""),
                        container=container,
                    )
                )
                handlers[channel.topic] = make_handler(container, **runtime_kwargs)
        if spec_id is None or spec_version is None:
            raise ValueError(f"{self._path}: no channels found")
        return (
            Spec(
                spec_id=spec_id,
                spec_version=spec_version,
                signals=tuple(signals),
            ),
            handlers,
        )


def _runtime_kwargs_from_metadata(
    container: ContainerSpec, md: dict[str, str]
) -> dict[str, Any]:
    handler_cls = HANDLER_TYPES[type(container)]
    field_types = runtime_field_types(handler_cls)
    out: dict[str, Any] = {}
    for name, typ in field_types.items():
        if name not in md:
            raise ValueError(
                f"{handler_cls.__name__}: channel metadata missing runtime field {name!r}"
            )
        out[name] = typ(md[name])
    return out


def _empty_for(container: ContainerSpec) -> np.ndarray:
    """Zero-row numpy array shaped after ``container``'s natural per-frame dims."""
    shape: tuple[int, ...] = getattr(container, "shape", ())
    dtype = getattr(container, "dtype", np.float32)
    return np.empty((0, *shape), dtype=dtype)


def _nearest_index(sorted_ts: list[int], target: int) -> int:
    """Bisect-based nearest-neighbor lookup. Ties → earlier."""
    from bisect import bisect_left

    j = bisect_left(sorted_ts, target)
    if j == 0:
        return 0
    if j == len(sorted_ts):
        return len(sorted_ts) - 1
    return j - 1 if (target - sorted_ts[j - 1]) <= (sorted_ts[j] - target) else j
