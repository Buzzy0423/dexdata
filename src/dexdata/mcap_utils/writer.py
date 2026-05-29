# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Spec-driven MCAP writer.

Construction registers every signal in the spec as one MCAP channel
(decision 2.4). Per-write the handler dispatched off ``topic`` validates
and serializes ``value`` to bytes; the writer hands those to MCAP along with
``publish_time`` (sensor-claimed capture, decision 2.13) and ``log_time``
(our subscriber receive time).

**Eager vs deferred registration** (DESIGN §5.1). Containers with no
runtime channel metadata (NumericArray) register at construction time.
Containers that need sensor-derived metadata (DepthImage's ``min_range`` /
``max_range``) defer until :meth:`set_runtime_metadata` supplies it; until
then, ``write`` on those topics raises.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
from mcap.writer import Writer as McapWriter

from ..metadata import (
    METADATA_FILE,
    EpisodeMetadata,
    compute_file_checksum_and_size,
    save_metadata,
)
from ..spec import SignalSpec, Spec
from .handlers import HANDLER_TYPES, Handler, make_handler, runtime_field_types
from .proto import build_file_descriptor_set

EPISODE_FILE = "episode.mcap"


class Writer:
    """Spec-driven MCAP writer.

    Usage::

        writer = Writer(out_dir, spec)
        writer.set_runtime_metadata("/camera/head_left/depth", min_range=0.1, max_range=8.0)
        writer.write("/robot/state/left_arm/qpos", arr, publish_ts_ns, recv_ts_ns)
        ...
        writer.close()

    Or as a context manager::

        with Writer(out_dir, spec) as writer:
            writer.set_runtime_metadata(...)
            writer.write(...)
    """

    def __init__(
        self,
        out_dir: Path | str,
        spec: Spec,
        *,
        metadata: EpisodeMetadata | None = None,
        length_topic: str | None = None,
    ) -> None:
        self._spec = spec
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._out_dir / EPISODE_FILE
        if self._path.exists():
            raise FileExistsError(f"{self._path} already exists")

        self._file = open(self._path, "wb")
        self._mcap = McapWriter(self._file)
        self._mcap.start()

        # Schemas live one-per-container-type, shared across channels.
        self._schema_ids: dict[str, int] = {}
        self._channels: dict[str, int] = {}
        self._handlers: dict[str, Handler] = {}
        # Topics whose channel hasn't been registered yet (deferred).
        self._pending: dict[str, SignalSpec] = {}
        # Topics that have been explicitly finalized via finalize_topic —
        # their handler is dropped and further writes are rejected, but
        # the MCAP channel and its already-written messages stay intact.
        self._finalized: set[str] = set()

        # Optional sidecar metadata, finalized at close() with derivable
        # fields (duration_s, length, size_bytes, checksum, uri, topics).
        self._metadata = metadata
        # If set, this topic's message count is taken as
        # ``collection.length``; else the longest stream wins.
        self._length_topic = length_topic
        # Recorder-clock (log_time) counters. ``duration_s`` is derived from
        # these; recv is the only timestamp domain that's coherent across
        # multi-publisher recordings (each publisher has its own clock; the
        # recorder stamps log_time uniformly).
        self._first_recv_ns: int | None = None
        self._last_recv_ns: int | None = None
        self._topic_counts: dict[str, int] = {}

        for signal in spec.signals:
            handler_cls = HANDLER_TYPES.get(type(signal.container))
            if handler_cls is None:
                raise NotImplementedError(
                    f"no handler for {type(signal.container).__name__} "
                    f"(topic {signal.topic})"
                )
            if runtime_field_types(handler_cls):
                self._pending[signal.topic] = signal
            else:
                self._register(signal, make_handler(signal.container))

        # Save metadata up front so the sidecar exists even if the writer
        # crashes mid-recording. Derivable fields (size_bytes, duration_s,
        # length, topics, checksum) keep their dataclass defaults until
        # close() rewrites the file with the real values.
        if self._metadata is not None:
            save_metadata(self._metadata, self._out_dir / METADATA_FILE)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def pending_topics(self) -> tuple[str, ...]:
        """Topics whose runtime metadata hasn't been supplied yet."""
        return tuple(self._pending)

    def set_runtime_metadata(self, topic: str, **kwargs: Any) -> None:
        """Supply sensor-derived runtime metadata for a deferred channel.

        After this call the channel is registered and ``write(topic, ...)``
        is valid. Calling twice on the same topic raises.
        """
        signal = self._pending.pop(topic, None)
        if signal is None:
            if topic in self._channels:
                raise ValueError(f"{topic}: runtime metadata already set")
            raise KeyError(f"{topic}: not in spec")
        handler_cls = HANDLER_TYPES[type(signal.container)]
        expected = set(runtime_field_types(handler_cls))
        provided = set(kwargs)
        missing = expected - provided
        extra = provided - expected
        if missing or extra:
            self._pending[topic] = signal  # restore so caller can retry
            parts: list[str] = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if extra:
                parts.append(f"unexpected {sorted(extra)}")
            raise ValueError(f"{topic}: {', '.join(parts)}")
        self._register(signal, make_handler(signal.container, **kwargs))

    def write(
        self,
        topic: str,
        value: np.ndarray,
        publish_ts_ns: int,
        recv_ts_ns: int,
    ) -> None:
        if topic in self._pending:
            raise RuntimeError(
                f"{topic}: runtime metadata not set; "
                f"call set_runtime_metadata({topic!r}, ...) first"
            )
        if topic in self._finalized:
            raise RuntimeError(f"{topic}: already finalized; no more writes accepted")
        try:
            handler = self._handlers[topic]
            channel_id = self._channels[topic]
        except KeyError:
            raise KeyError(f"topic not in spec: {topic}") from None
        for payload, pt_ns, rt_ns in handler.serialize(
            value, publish_ts_ns, recv_ts_ns
        ):
            self._emit(channel_id, topic, payload, pt_ns, rt_ns)

    def finalize_topic(self, topic: str) -> None:
        """Drain and release one topic's handler early.

        Calls ``handler.flush()`` to emit any encoder-buffered packets,
        then drops the handler from the writer's state so the underlying
        codec context can be released. Use this for memory-heavy
        handlers — CompressedVideo holds a multi-frame SVT-AV1 buffer
        pool (~1 GB per encoder) that otherwise stays live until
        :meth:`close`.

        After this call, ``write(topic, ...)`` raises. The MCAP channel
        and every message already written to it remain intact, and the
        topic continues to count toward ``data.topics`` and
        ``collection.length`` at close.

        Args:
            topic: A topic previously registered via the spec.

        Raises:
            ValueError: ``topic`` is still pending runtime metadata, or
                has already been finalized.
            KeyError: ``topic`` is not in the spec.
        """
        if topic in self._pending:
            raise ValueError(f"{topic}: cannot finalize before set_runtime_metadata")
        if topic in self._finalized:
            raise ValueError(f"{topic}: already finalized")
        try:
            handler = self._handlers[topic]
            channel_id = self._channels[topic]
        except KeyError:
            raise KeyError(f"topic not in spec: {topic}") from None
        for payload, pt_ns, rt_ns in handler.flush():
            self._emit(channel_id, topic, payload, pt_ns, rt_ns)
        # Drop the handler reference so the codec context (PyAV
        # CodecContext, holding SVT-AV1's frame pools) can be released
        # by refcount-driven dealloc.
        del self._handlers[topic]
        self._finalized.add(topic)

    def close(self) -> None:
        if self._file.closed:
            return
        finalized_ok = False
        try:
            # Drain any handler-side buffers (video encoders) before finishing.
            # Already-finalized topics aren't in self._handlers and are skipped.
            for topic, handler in list(self._handlers.items()):
                channel_id = self._channels[topic]
                for payload, pt_ns, rt_ns in handler.flush():
                    self._emit(channel_id, topic, payload, pt_ns, rt_ns)
            self._mcap.finish()
            finalized_ok = True
        finally:
            self._file.close()

        # Only write the sidecar if the MCAP itself closed cleanly — a
        # partial MCAP shouldn't get a checksum that pretends it's whole.
        if finalized_ok and self._metadata is not None:
            self._finalize_metadata()

    def __enter__(self) -> Writer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _emit(
        self,
        channel_id: int,
        topic: str,
        payload: bytes,
        pt_ns: int,
        rt_ns: int,
    ) -> None:
        """Single chokepoint for MCAP writes — also drives metadata counters."""
        self._mcap.add_message(
            channel_id=channel_id,
            log_time=rt_ns,
            publish_time=pt_ns,
            data=payload,
        )
        if self._metadata is None:
            return
        if self._first_recv_ns is None or rt_ns < self._first_recv_ns:
            self._first_recv_ns = rt_ns
        if self._last_recv_ns is None or rt_ns > self._last_recv_ns:
            self._last_recv_ns = rt_ns
        self._topic_counts[topic] = self._topic_counts.get(topic, 0) + 1

    def _finalize_metadata(self) -> None:
        """Fill derivable fields and write the sidecar.

        The writer is authoritative for these fields — caller-supplied
        values are overwritten because they describe what's actually on
        disk, not what the caller intended.

        - ``data.uri``           → ``"episode.mcap"`` (relative to sidecar).
        - ``data.size_bytes``    → real file size after ``finish()``.
        - ``data.checksum_sha256`` → SHA-256 of the closed MCAP.
        - ``data.topics``        → flat sorted list of topics that
          received at least one message.
        - ``collection.duration_s`` → (last − first) ``log_time`` (recv)
          across every message written. Recv is the recorder's clock —
          coherent across multi-publisher episodes; publish_time is
          per-publisher and would yield a misleading global span when
          publisher clocks aren't synchronized.
        - ``collection.length``  → message count of the writer's
          ``length_topic`` if set and written-to; else the longest
          stream; else 0.
        """
        assert self._metadata is not None
        meta = self._metadata

        checksum, size = compute_file_checksum_and_size(self._path)
        meta.data.uri = EPISODE_FILE
        meta.data.size_bytes = size
        meta.data.checksum_sha256 = checksum
        meta.data.topics = sorted(self._topic_counts)

        # If the caller didn't pre-fill data_spec_id, derive it from the
        # Spec we're writing against. The Spec's spec_id is the natural
        # join key between metadata and the on-disk schema. We don't
        # overwrite a caller-supplied value — they might have a richer
        # registry identifier than the spec_id we'd guess.
        if meta.robot.data_spec_id is None:
            meta.robot.data_spec_id = f"{self._spec.spec_id}_v{self._spec.spec_version}"

        if self._first_recv_ns is not None and self._last_recv_ns is not None:
            meta.collection.duration_s = (
                self._last_recv_ns - self._first_recv_ns
            ) / 1e9
        else:
            meta.collection.duration_s = 0.0

        if self._length_topic is not None and self._length_topic in self._topic_counts:
            meta.collection.length = self._topic_counts[self._length_topic]
        elif self._topic_counts:
            meta.collection.length = max(self._topic_counts.values())
        else:
            meta.collection.length = 0

        save_metadata(meta, self._out_dir / METADATA_FILE)

    def _register(self, signal: SignalSpec, handler: Handler) -> None:
        if handler.schema_name not in self._schema_ids:
            self._schema_ids[handler.schema_name] = self._mcap.register_schema(
                name=handler.schema_name,
                encoding="protobuf",
                data=build_file_descriptor_set(handler.proto_class),
            )
        channel_id = self._mcap.register_channel(
            topic=signal.topic,
            schema_id=self._schema_ids[handler.schema_name],
            message_encoding="protobuf",
            metadata=_build_channel_metadata(self._spec, signal, handler),
        )
        self._handlers[signal.topic] = handler
        self._channels[signal.topic] = channel_id


def _build_channel_metadata(
    spec: Spec, signal: SignalSpec, handler: Handler
) -> dict[str, str]:
    """Channel metadata = container fields + runtime fields + signal unit + spec identity."""
    md: dict[str, Any] = {
        "spec_id": spec.spec_id,
        "spec_version": spec.spec_version,
        "unit": signal.unit,
    }
    md.update(signal.container.channel_metadata())
    # Runtime fields stored as strings, mirroring container.channel_metadata().
    for name in runtime_field_types(type(handler)):
        md[name] = getattr(handler, name)
    return {k: str(v) for k, v in md.items()}
