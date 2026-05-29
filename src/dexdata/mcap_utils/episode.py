# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""``Episode`` — materialized view of a recorded MCAP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..spec import Spec


@dataclass(frozen=True)
class Episode:
    """All messages of a single episode, materialized into stacked arrays.

    For each topic, ``signals[topic]`` is a ``(T, ...)`` ndarray holding
    every message body in receive order, and the two timestamp dicts hold
    parallel ``(T,)`` int64 ns arrays — the MCAP envelope's
    ``publish_time`` (sensor capture) and ``log_time`` (subscriber receive).

    Sensor timestamps are independent across signals (decision 2.8); no
    cross-topic alignment is performed here.
    """

    spec: Spec
    signals: dict[str, np.ndarray]
    publish_timestamps: dict[str, np.ndarray]
    recv_timestamps: dict[str, np.ndarray]

    def to_numpy_dict(self) -> dict[str, np.ndarray]:
        """Topic-path → stacked ``(T, ...)`` array. Spec naming preserved."""
        return dict(self.signals)
