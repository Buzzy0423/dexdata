# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Handler for the bare ``foxglove.Pose`` container.

We use the bare ``foxglove.Pose`` (position + orientation only), not
``foxglove.PoseInFrame`` — the latter's ``frame_id`` is redundant with the
topic-path identity convention from §2.2 (decision 2.13). The proto's
defunct ``timestamp`` etc. is not present on bare ``Pose``.

**Wire layout.** Foxglove Pose has two sub-messages, each holding doubles:

  Vector3 position    : x, y, z
  Quaternion orientation : x, y, z, w

We accept a 1-D numpy array of length 7, dtype float64, in the order
``[px, py, pz, qx, qy, qz, qw]``. The xyzw quaternion ordering matches the
proto field order; we don't auto-normalize — the source is authoritative.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from foxglove_schemas_protobuf import Pose_pb2

from ...handlers.containers import PoseSpec

POSE_SHAPE: tuple[int, ...] = (7,)
POSE_DTYPE: np.dtype = np.dtype("float64")


@dataclass(frozen=True)
class PoseHandler:
    """Encode/decode a single ``foxglove.Pose`` per call."""

    spec: PoseSpec

    proto_class: ClassVar[type] = Pose_pb2.Pose
    schema_name: ClassVar[str] = Pose_pb2.Pose.DESCRIPTOR.full_name

    def serialize(
        self, value: np.ndarray, publish_ts_ns: int, recv_ts_ns: int
    ) -> Iterable[tuple[bytes, int, int]]:
        if value.shape != POSE_SHAPE:
            raise ValueError(
                f"Pose shape mismatch: expected {POSE_SHAPE}, got {value.shape}"
            )
        if value.dtype != POSE_DTYPE:
            raise ValueError(
                f"Pose dtype mismatch: expected {POSE_DTYPE.name}, "
                f"got {value.dtype.name}"
            )
        msg = Pose_pb2.Pose()
        msg.position.x = float(value[0])
        msg.position.y = float(value[1])
        msg.position.z = float(value[2])
        msg.orientation.x = float(value[3])
        msg.orientation.y = float(value[4])
        msg.orientation.z = float(value[5])
        msg.orientation.w = float(value[6])
        yield (msg.SerializeToString(), publish_ts_ns, recv_ts_ns)

    def flush(self) -> Iterable[tuple[bytes, int, int]]:
        return ()

    def deserialize(self, payload: bytes) -> Iterable[np.ndarray]:
        msg = Pose_pb2.Pose()
        msg.ParseFromString(payload)
        yield np.array(
            [
                msg.position.x,
                msg.position.y,
                msg.position.z,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ],
            dtype=POSE_DTYPE,
        )

    def flush_decode(self) -> Iterable[np.ndarray]:
        return ()
