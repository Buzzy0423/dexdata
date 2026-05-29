# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Memory-bounded reader for image-heavy composable episodes.

The standard :class:`~.reader.Reader` materializes every topic into one
stacked numpy array — fine for state/action (~MB scale per episode) but
the source-rate camera frames are the dominant RSS consumer on large
episodes. ``align_episode`` then re-indexes them onto a reference
timeline, doubling peak RAM momentarily, and ``composable_to_canonical``
adds a BGR→RGB copy.

This module keeps state/action fully materialized (cheap) and replaces
the camera read path with a frame-at-a-time iterator that walks the
source MCAP once per topic. For nearest-neighbor alignment the source
timeline is always sorted, so the consumer can pre-compute how many
reference frames each source frame should emit and walk both lists in
lockstep — no buffering of the full source-rate array.

Stats and the encoder dispatch can then consume per-chunk batches of
ref-ordered frames; peak RAM becomes
``O(batch_size × frame_size × n_cameras_in_flight)`` instead of
``O(episode_length × frame_size × n_cameras)``.

The standard ``Reader`` / ``align_episode`` / ``composable_to_canonical``
/ ``lerobot.write`` path is unchanged. Use this module when an episode
is large enough that the standard path's peak RSS hurts.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mcap.reader import make_reader

from ..handlers.containers import CompressedVideoSpec, DepthImageSpec
from ..spec import Spec
from .episode import Episode
from .reader import Reader

# Topic dispatch — anything whose container is one of these gets the
# streaming code path. Everything else stays fully materialized.
_IMAGE_CONTAINER_TYPES = (CompressedVideoSpec, DepthImageSpec)


@dataclass(frozen=True)
class TopicTimestamps:
    """Timestamps for one topic, no payload decoded."""

    publish_ns: np.ndarray  # int64 (N,)
    recv_ns: np.ndarray  # int64 (N,) — used for nearest-neighbor align


class StreamingReader:
    """Reader that materializes only non-image topics.

    Wraps a standard :class:`~.reader.Reader` for spec/handler/metadata
    recovery, then exposes:

    * :meth:`read_non_image_episode` — :class:`Episode` containing
      every non-image topic, fully stacked and ready for
      :func:`align_episode`.
    * :meth:`topic_timestamps` — int64 timestamp arrays for a topic
      with zero payload decode.
    * :meth:`iter_frames_in_ref_order` — yields frames one at a time,
      reordered onto a reference timeline via a caller-supplied
      ``ref_to_src`` index map. Single decode pass per topic.

    The original :class:`Reader`'s public API stays available via the
    forwarded properties below — callers that just want metadata or
    spec don't need to switch classes.
    """

    def __init__(self, episode_dir: Path | str) -> None:
        self._reader = Reader(episode_dir)
        # Cache the spec's container type per topic for fast image vs
        # non-image dispatch in read_non_image_episode and elsewhere.
        self._is_image: dict[str, bool] = {
            s.topic: isinstance(s.container, _IMAGE_CONTAINER_TYPES)
            for s in self._reader.spec.signals
        }

    # ---- forwarded surface ---------------------------------------------
    @property
    def spec(self) -> Spec:
        return self._reader.spec

    @property
    def path(self) -> Path:
        return self._reader.path

    @property
    def metadata(self):  # type: ignore[no-untyped-def]
        return self._reader.metadata

    def is_image_topic(self, topic: str) -> bool:
        if topic not in self._is_image:
            raise KeyError(f"topic not in spec: {topic}")
        return self._is_image[topic]

    # ---- bulk: non-image episode ---------------------------------------
    def read_non_image_episode(self) -> Episode:
        """Materialize every non-image topic into one :class:`Episode`.

        The result is shaped exactly like a normal
        :meth:`Reader.read_episode` output (same dicts, same numpy
        types) but skips the camera/depth topics entirely. Suitable
        input to :func:`align_episode` — the camera streams aren't
        needed for the alignment math; their per-frame timestamps come
        out of :meth:`topic_timestamps`.
        """
        non_image_topics = {t for t, is_img in self._is_image.items() if not is_img}
        per_topic: dict[str, list[tuple[np.ndarray, int, int]]] = {
            t: [] for t in non_image_topics
        }
        ts_queues: dict[str, list[tuple[int, int]]] = {t: [] for t in non_image_topics}
        handlers = {
            t: h
            for t, h in self._reader._handlers.items()  # noqa: SLF001
            if t in non_image_topics
        }

        with open(self._reader.path, "rb") as f:
            reader = make_reader(f)
            for _schema, channel, message in reader.iter_messages(
                topics=list(non_image_topics)
            ):
                handler = handlers.get(channel.topic)
                if handler is None:
                    continue
                ts_queues[channel.topic].append(
                    (message.publish_time, message.log_time)
                )
                for value in handler.deserialize(message.data):
                    pt_ns, rt_ns = ts_queues[channel.topic].pop(0)
                    per_topic[channel.topic].append((value, pt_ns, rt_ns))
        for topic, handler in handlers.items():
            for value in handler.flush_decode():
                pt_ns, rt_ns = ts_queues[topic].pop(0)
                per_topic[topic].append((value, pt_ns, rt_ns))

        signals: dict[str, np.ndarray] = {}
        publish_ts: dict[str, np.ndarray] = {}
        recv_ts: dict[str, np.ndarray] = {}
        for topic, items in per_topic.items():
            if items:
                arrs, p_ts, r_ts = zip(*items, strict=True)
                signals[topic] = np.stack(arrs)
                publish_ts[topic] = np.asarray(p_ts, dtype=np.int64)
                recv_ts[topic] = np.asarray(r_ts, dtype=np.int64)
            else:
                # Mirror Reader.read_episode's zero-row fallback.
                signal = next(s for s in self._reader.spec.signals if s.topic == topic)
                shape: tuple[int, ...] = getattr(signal.container, "shape", ())
                dtype = getattr(signal.container, "dtype", np.float32)
                signals[topic] = np.empty((0, *shape), dtype=dtype)
                publish_ts[topic] = np.empty((0,), dtype=np.int64)
                recv_ts[topic] = np.empty((0,), dtype=np.int64)

        return Episode(
            spec=self._reader.spec,
            signals=signals,
            publish_timestamps=publish_ts,
            recv_timestamps=recv_ts,
        )

    # ---- timestamps-only -----------------------------------------------
    def topic_timestamps(self, topic: str) -> TopicTimestamps:
        """Read every message's envelope timestamps for ``topic`` — no payload decode.

        Used to compute the ref→src index map before any frames are
        streamed. Cheap because MCAP envelopes are part of the message
        record header; the data payload is skipped entirely (we never
        invoke the handler).
        """
        if topic not in self._is_image:
            raise KeyError(f"topic not in spec: {topic}")
        publish: list[int] = []
        recv: list[int] = []
        with open(self._reader.path, "rb") as f:
            reader = make_reader(f)
            for _schema, _channel, message in reader.iter_messages(topics=[topic]):
                publish.append(message.publish_time)
                recv.append(message.log_time)
        return TopicTimestamps(
            publish_ns=np.asarray(publish, dtype=np.int64),
            recv_ns=np.asarray(recv, dtype=np.int64),
        )

    # ---- per-camera streaming ------------------------------------------
    def iter_frames_in_ref_order(
        self,
        topic: str,
        ref_to_src: np.ndarray,
    ) -> Iterator[np.ndarray]:
        """Yield decoded source frames in *reference* order, one at a time.

        Args:
            topic: An image topic in the spec.
            ref_to_src: ``int64`` array of length ``n_ref``. Index ``i``
                holds the source-message index nearest to ``ref[i]`` (as
                returned by an external nearest-neighbor lookup over
                :attr:`TopicTimestamps.recv_ns`). Must be non-decreasing.

        Yields:
            One ``(H, W, C)`` numpy frame per ref index, in ref order.
            A single source frame may yield multiple times if several
            ref indices map to it.

        Memory: at most one decoded frame at a time. The source-rate
        full array is *never* materialized.
        """
        if topic not in self._is_image or not self._is_image[topic]:
            raise KeyError(f"topic is not an image topic: {topic}")

        ref_to_src = np.asarray(ref_to_src, dtype=np.int64)
        if ref_to_src.size == 0:
            return
        if np.any(np.diff(ref_to_src) < 0):
            raise ValueError(
                "ref_to_src must be non-decreasing; "
                "nearest-neighbor align over sorted timelines satisfies this"
            )
        last_needed_src = int(ref_to_src[-1])

        # Pre-compute, for each source index in [0, last_needed_src], how
        # many ref indices map to it. Stored as a parallel array so the
        # inner loop is one O(1) lookup per src frame.
        emits_per_src = np.zeros(last_needed_src + 1, dtype=np.int64)
        for src_idx in ref_to_src:
            emits_per_src[src_idx] += 1

        # iter_messages drives the handler's stateful decoder. We pull
        # frames in order, emit each one as many times as its src index
        # appears in ref_to_src, then move on. After src_idx exceeds the
        # last needed index we exit early — the rest of the source video
        # is unused.
        src_idx = 0
        for value, _pt, _rt in self._reader.iter_messages(topic):
            if src_idx > last_needed_src:
                break
            count = int(emits_per_src[src_idx])
            for _ in range(count):
                yield value
            src_idx += 1


def nearest_ref_to_src(ref_ts: np.ndarray, src_ts: np.ndarray) -> np.ndarray:
    """Build the ``ref_to_src`` index map used by streaming align.

    For each reference timestamp, picks the source index whose timestamp
    is closest in absolute distance — same rule as
    :func:`align_episode` in ``"nearest"`` mode. Both timelines must be
    sorted (composable's writer guarantees this within a single topic).

    Returns:
        ``int64`` array shape ``(len(ref_ts),)``. Empty if either side
        is empty (and the caller should skip the topic).
    """
    if ref_ts.size == 0 or src_ts.size == 0:
        return np.empty((0,), dtype=np.int64)
    # bisect_left on each ref against src
    idx_right = np.searchsorted(src_ts, ref_ts, side="left")
    idx_right_clamped = np.minimum(idx_right, src_ts.size - 1)
    idx_left = np.maximum(idx_right_clamped - 1, 0)
    # Prefer left when its distance is ≤ right's (matches align_episode's
    # tie-break: closer-or-equal earlier neighbor wins).
    dist_left = np.abs(ref_ts - src_ts[idx_left])
    dist_right = np.abs(src_ts[idx_right_clamped] - ref_ts)
    take_left = dist_left <= dist_right
    return np.where(take_left, idx_left, idx_right_clamped).astype(np.int64)
