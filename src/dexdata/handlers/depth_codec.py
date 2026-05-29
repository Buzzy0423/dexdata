# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Lossless PNG-16 codec for the ``DepthImage`` container.

Float32 meters ↔ PNG-16 bytes, with `0.0`/`NaN` (and out-of-range values) as
the invalid sentinel on the float32 side and uint16 `0` as the invalid
sentinel on the wire. The 65535 valid uint16 codes (1..65535) span
``[min_range, max_range]`` linearly; quantum is ``(max - min) / 65534``.

DESIGN §3.3 has the rationale for the asymmetry — depth cameras emit `0.0 =
invalid` natively in float32, while every other container in the framework
follows the project-wide ``NaN = invalid`` convention.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

_QUANTA_MAX_CODE = 65534  # gap count between adjacent valid codes (1..65535)


def encode_png16(arr_m: np.ndarray, min_range: float, max_range: float) -> bytes:
    """Encode a float32 depth array (meters) to PNG-16 bytes.

    Args:
        arr_m: ``(H, W)`` float32 depth in meters. ``0.0``, ``NaN``, and any
            value outside ``[min_range, max_range]`` are mapped to the
            invalid sentinel (uint16 0).
        min_range: Lower edge of the representable range, in meters.
        max_range: Upper edge of the representable range, in meters.
    """
    if arr_m.ndim != 2:
        raise ValueError(f"depth array must be 2-D (H, W); got {arr_m.shape}")
    if arr_m.dtype != np.float32:
        raise ValueError(f"depth array must be float32; got {arr_m.dtype}")
    if not (max_range > min_range):
        raise ValueError(
            f"max_range must be > min_range; got {max_range} <= {min_range}"
        )

    quantum = (max_range - min_range) / _QUANTA_MAX_CODE
    valid = np.isfinite(arr_m) & (arr_m >= min_range) & (arr_m <= max_range)
    quantized = np.zeros(arr_m.shape, dtype=np.uint16)
    if valid.any():
        # round to nearest code in [0, 65534], then shift to [1, 65535]
        codes = np.round((arr_m[valid] - min_range) / quantum)
        codes = np.clip(codes, 0, _QUANTA_MAX_CODE).astype(np.uint16) + 1
        quantized[valid] = codes

    buf = io.BytesIO()
    Image.fromarray(quantized).save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def decode_png16(payload: bytes, min_range: float, max_range: float) -> np.ndarray:
    """Decode PNG-16 bytes to a float32 depth array (meters).

    Invalid pixels (uint16 0 on the wire) come back as float32 ``0.0``.
    """
    if not (max_range > min_range):
        raise ValueError(
            f"max_range must be > min_range; got {max_range} <= {min_range}"
        )

    quantum = (max_range - min_range) / _QUANTA_MAX_CODE
    img = Image.open(io.BytesIO(payload))
    quantized = np.asarray(img, dtype=np.uint16)
    arr = np.zeros(quantized.shape, dtype=np.float32)
    valid = quantized > 0
    arr[valid] = (quantized[valid].astype(np.float32) - 1.0) * quantum + min_range
    return arr
