# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Handler protocol — the bytes-level contract for one container type.

A handler is the encode/decode adapter between a numpy value and its
container's proto wire format. It is constructed from the corresponding
:class:`ContainerSpec` and uses the spec's fields (and runtime channel
metadata, where applicable) to enforce shape/dtype invariants.

The protocol is **streaming**. ``serialize`` yields zero or more
``(payload_bytes, publish_ts_ns, recv_ts_ns)`` records per input value, and
``flush`` drains any trailing records when the episode ends. Stateless
containers (``NumericArray``, ``DepthImage``) yield exactly one record per
input and have an empty flush; stateful ones (``CompressedVideo``) buffer
frames internally and may emit 0/1/2+ packets per call.

Decode mirrors the encode side: ``deserialize`` yields zero or more numpy
values per packet, ``flush_decode`` drains the decoder. The reader pairs
each yielded value with the envelope timestamps of the corresponding source
record positionally — the codecs we support (h264 with ``bf=0``, av1 via
libaom + libdav1d) preserve frame order end-to-end, verified empirically.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np

from ...handlers.containers import ContainerSpec


class Handler(Protocol):
    """numpy ↔ proto-bytes for a single container type."""

    spec: ContainerSpec

    def serialize(
        self,
        value: np.ndarray,
        publish_ts_ns: int,
        recv_ts_ns: int,
    ) -> Iterable[tuple[bytes, int, int]]:
        """Encode ``value`` into zero or more proto payloads.

        Yields ``(payload_bytes, publish_ts_ns, recv_ts_ns)`` per output
        record. Stateless handlers yield exactly one element with the input
        timestamps unchanged. Stateful handlers may buffer and emit packets
        from earlier inputs; the timestamps yielded reference the *original*
        source frame's timestamps, not the current call's.
        """
        ...

    def flush(self) -> Iterable[tuple[bytes, int, int]]:
        """Drain any buffered records at end of stream.

        Empty for stateless handlers. Stateful handlers (e.g. video) call
        their underlying encoder's flush here.
        """
        ...

    def deserialize(self, payload: bytes) -> Iterable[np.ndarray]:
        """Decode one proto payload into zero or more numpy values."""
        ...

    def flush_decode(self) -> Iterable[np.ndarray]:
        """Drain any buffered values from the decoder at end of stream."""
        ...
