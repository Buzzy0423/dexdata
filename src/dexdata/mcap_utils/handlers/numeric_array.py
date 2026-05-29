# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Handler for the ``NumericArray`` container."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ...handlers.containers import NumericArraySpec
from ..proto.numeric_array_pb2 import NumericArray


@dataclass(frozen=True)
class NumericArrayHandler:
    """Encode/decode N-dim numeric tensors against a fixed shape + dtype."""

    spec: NumericArraySpec

    proto_class: ClassVar[type] = NumericArray
    schema_name: ClassVar[str] = "dexdata.composable.NumericArray"

    def serialize(
        self, value: np.ndarray, publish_ts_ns: int, recv_ts_ns: int
    ) -> Iterable[tuple[bytes, int, int]]:
        if value.shape != self.spec.shape:
            raise ValueError(
                f"NumericArray shape mismatch: expected {self.spec.shape}, "
                f"got {value.shape}"
            )
        if value.dtype != self.spec.dtype:
            raise ValueError(
                f"NumericArray dtype mismatch: expected {self.spec.dtype.name}, "
                f"got {value.dtype.name}"
            )
        msg = NumericArray()
        msg.data = np.ascontiguousarray(value).tobytes()
        yield (msg.SerializeToString(), publish_ts_ns, recv_ts_ns)

    def flush(self) -> Iterable[tuple[bytes, int, int]]:
        return ()

    def deserialize(self, payload: bytes) -> Iterable[np.ndarray]:
        msg = NumericArray()
        msg.ParseFromString(payload)
        yield np.frombuffer(msg.data, dtype=self.spec.dtype).reshape(self.spec.shape)

    def flush_decode(self) -> Iterable[np.ndarray]:
        return ()
