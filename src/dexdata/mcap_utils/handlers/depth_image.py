# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Handler for the ``DepthImage`` container.

The handler is constructed with the spec **and** runtime channel metadata
(``min_range`` / ``max_range`` from the depth sensor). At spec-load time we
don't know these — they're sensor-derived (decision 2.13 / DESIGN §5.1
deferred registration). The :class:`Writer` learns them via
``set_runtime_metadata`` and the :class:`Reader` reads them back from MCAP
channel metadata.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ...handlers.containers import DepthImageSpec
from ...handlers.depth_codec import decode_png16, encode_png16
from ..proto.depth_image_pb2 import DepthImage


@dataclass(frozen=True)
class DepthImageHandler:
    """Encode/decode a single-camera depth stream as PNG-16."""

    spec: DepthImageSpec
    min_range: float
    max_range: float

    proto_class: ClassVar[type] = DepthImage
    schema_name: ClassVar[str] = "dexdata.composable.DepthImage"

    def serialize(
        self, value: np.ndarray, publish_ts_ns: int, recv_ts_ns: int
    ) -> Iterable[tuple[bytes, int, int]]:
        msg = DepthImage()
        msg.data = encode_png16(value, self.min_range, self.max_range)
        yield (msg.SerializeToString(), publish_ts_ns, recv_ts_ns)

    def flush(self) -> Iterable[tuple[bytes, int, int]]:
        return ()

    def deserialize(self, payload: bytes) -> Iterable[np.ndarray]:
        msg = DepthImage()
        msg.ParseFromString(payload)
        yield decode_png16(msg.data, self.min_range, self.max_range)

    def flush_decode(self) -> Iterable[np.ndarray]:
        return ()
