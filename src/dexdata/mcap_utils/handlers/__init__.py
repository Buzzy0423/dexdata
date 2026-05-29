# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Container handlers: numpy ↔ proto bytes (one per container type).

Public surface:
  * :class:`Handler` — bytes-level protocol every handler implements.
  * :class:`NumericArrayHandler`, :class:`DepthImageHandler`,
    :class:`CompressedVideoHandler` — concrete handlers.
  * :func:`make_handler` — factory dispatching off a :class:`ContainerSpec` instance.
  * :func:`runtime_field_types` — names + types of the constructor kwargs that
    a handler needs beyond ``spec`` (its sensor-derived runtime metadata).

Adding a new container = subclass `Handler`, register in `HANDLER_TYPES`.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import get_type_hints

from ...handlers.containers import (
    CompressedVideoSpec,
    ContainerSpec,
    DepthImageSpec,
    NumericArraySpec,
    PoseSpec,
)
from .base import Handler
from .compressed_video import CompressedVideoHandler
from .depth_image import DepthImageHandler
from .numeric_array import NumericArrayHandler
from .pose import PoseHandler

HANDLER_TYPES: dict[type[ContainerSpec], type] = {
    NumericArraySpec: NumericArrayHandler,
    DepthImageSpec: DepthImageHandler,
    CompressedVideoSpec: CompressedVideoHandler,
    PoseSpec: PoseHandler,
}


def make_handler(spec: ContainerSpec, **runtime_kwargs) -> Handler:
    """Build the handler for a given ContainerSpec instance.

    ``runtime_kwargs`` carries any sensor-derived fields the handler needs
    (e.g. ``min_range`` / ``max_range`` for ``DepthImage``,
    ``width`` / ``height`` for ``CompressedVideo``). For handlers with eager
    registration (no runtime metadata), pass nothing.
    """
    cls = HANDLER_TYPES.get(type(spec))
    if cls is None:
        raise NotImplementedError(
            f"no handler registered for {type(spec).__name__}; "
            f"known: {sorted(c.__name__ for c in HANDLER_TYPES)}"
        )
    return cls(spec, **runtime_kwargs)


def runtime_field_types(handler_cls: type) -> dict[str, type]:
    """Return ``{field_name: type}`` for the handler's runtime kwargs.

    Computed from the dataclass fields, excluding ``spec``. Empty for
    handlers that register eagerly (NumericArray); non-empty for handlers
    that need sensor-derived metadata before they can serialize
    (DepthImage's ``min_range``/``max_range``,
    CompressedVideo's ``width``/``height``).
    """
    hints = get_type_hints(handler_cls)
    return {
        f.name: hints[f.name] for f in dataclass_fields(handler_cls) if f.name != "spec"
    }


__all__ = [
    "HANDLER_TYPES",
    "CompressedVideoHandler",
    "DepthImageHandler",
    "Handler",
    "NumericArrayHandler",
    "PoseHandler",
    "make_handler",
    "runtime_field_types",
]
