# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Per-container spec dataclasses for the composable schema.

Each `ContainerSpec` subclass is a *thin declaration* — fields and their types,
nothing else. The base class supplies generic ``from_dict`` / ``to_dict`` /
``channel_metadata`` driven by the dataclass field annotations. Adding a new
container type is one block: ``@dataclass class FooSpec(ContainerSpec)`` with
the fields it carries.

There are no field defaults: a JSON spec must list every field of every
container by name. Hidden defaults make specs ambiguous; explicit is better.

The five container types are the alphabet for any spec:

| name                        | role                                            |
|-----------------------------|-------------------------------------------------|
| ``NumericArray``            | 1D vectors: qpos, qvel, tactile, wrench, …      |
| ``foxglove.CompressedVideo``| RGB video                                       |
| ``foxglove.CompressedImage``| depth                                           |
| ``foxglove.Pose``           | IMU / SE3 wrist pose                            |
| ``foxglove.PointCloud``     | lidar                                           |

Adding a new container = subclass `ContainerSpec`, register in
`CONTAINER_TYPES`. No edits to `spec.py` needed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, get_origin, get_type_hints

import numpy as np

# ---- type codecs (annotation → JSON value, JSON value → annotation) -------

_DTYPE_NAMES: dict[str, np.dtype] = {
    name: np.dtype(name)
    for name in (
        "float32",
        "float64",
        "uint8",
        "uint16",
        "uint32",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
    )
}


def _decode_dtype(v: Any) -> np.dtype:
    if not isinstance(v, str) or v not in _DTYPE_NAMES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPE_NAMES)}, got {v!r}")
    return _DTYPE_NAMES[v]


def _decode_shape(v: Any) -> tuple[int, ...]:
    if not isinstance(v, list) or not all(
        isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in v
    ):
        raise ValueError(f"shape must be a list of positive ints, got {v!r}")
    return tuple(v)


def _decode(annotation: Any, value: Any) -> Any:
    if annotation is np.dtype:
        return _decode_dtype(value)
    if get_origin(annotation) is tuple:
        return _decode_shape(value)
    return value


def _encode(annotation: Any, value: Any) -> Any:
    if annotation is np.dtype:
        return value.name
    if get_origin(annotation) is tuple:
        return list(value)
    return value


def _decode_from_str(annotation: Any, s: str) -> Any:
    """Inverse of :func:`_stringify` for ContainerSpec field types.

    Used by :meth:`ContainerSpec.from_channel_metadata` to rebuild a typed
    spec from the str-keyed MCAP channel metadata dict.
    """
    if annotation is np.dtype:
        return _decode_dtype(s)
    if get_origin(annotation) is tuple:
        if not s:
            return ()
        return tuple(int(x) for x in s.split(","))
    if annotation is int:
        return int(s)
    if annotation is bool:
        return s.lower() == "true"
    if annotation is str:
        return s
    raise TypeError(
        f"unsupported field annotation for channel metadata: {annotation!r}"
    )


def _stringify(v: Any) -> str:
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


# ---- ContainerSpec base ---------------------------------------------------


@dataclass(frozen=True)
class ContainerSpec:
    """Base class for any container type.

    Subclasses declare fields and their types only. The base handles JSON
    round-trip and channel-metadata stringification by introspecting field
    annotations — no per-class accessor methods needed.
    """

    type_name: ClassVar[str] = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContainerSpec:
        expected = {f.name for f in fields(cls)}
        provided = set(d) - {"type"}
        missing = expected - provided
        extra = provided - expected
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if extra:
                parts.append(f"unexpected {sorted(extra)}")
            raise ValueError(f"{cls.type_name}: {', '.join(parts)}")
        hints = get_type_hints(cls)
        return cls(**{f.name: _decode(hints[f.name], d[f.name]) for f in fields(cls)})

    def to_dict(self) -> dict[str, Any]:
        hints = get_type_hints(type(self))
        return {
            "type": self.type_name,
            **{
                f.name: _encode(hints[f.name], getattr(self, f.name))
                for f in fields(self)
            },
        }

    def channel_metadata(self) -> dict[str, str]:
        """JSON-equivalent dict, stringified for MCAP channel metadata.

        ``container_type`` carries the type name; remaining keys are the
        container's fields encoded the same way as :meth:`to_dict`.
        """
        d = self.to_dict()
        ct = d.pop("type")
        return {"container_type": ct, **{k: _stringify(v) for k, v in d.items()}}

    @classmethod
    def from_channel_metadata(cls, md: dict[str, str]) -> ContainerSpec:
        """Inverse of :meth:`channel_metadata`: recover a typed spec from MCAP.

        Extra keys in ``md`` (``unit``, ``spec_id``, etc. populated alongside
        the container's fields by the writer) are tolerated — only the keys
        named on this dataclass are read.
        """
        hints = get_type_hints(cls)
        missing = [f.name for f in fields(cls) if f.name not in md]
        if missing:
            raise ValueError(
                f"{cls.type_name}: channel metadata missing {sorted(missing)}"
            )
        return cls(
            **{f.name: _decode_from_str(hints[f.name], md[f.name]) for f in fields(cls)}
        )


# ---- concrete container types --------------------------------------------


@dataclass(frozen=True)
class NumericArraySpec(ContainerSpec):
    type_name: ClassVar[str] = "NumericArray"
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class CompressedVideoSpec(ContainerSpec):
    """RGB video, h264/av1 packets via the foxglove proto.

    Spec carries only the user-decided compression knobs (codec, CRF, GOP,
    preset). Camera-derived dimensions (``width``, ``height``) are populated
    into channel metadata at writer register time, not by the spec author.
    """

    type_name: ClassVar[str] = "foxglove.CompressedVideo"
    codec: str
    crf: int
    gop: int
    preset: str


@dataclass(frozen=True)
class DepthImageSpec(ContainerSpec):
    """Lossless PNG-16 depth (custom proto we own).

    Spec carries no user-authored fields — depth has no compression knobs and
    no codec choice. The sensor's ``min_range`` and ``max_range`` are
    populated into channel metadata at writer register time (the writer
    learns them from the sensor handle), not declared by the spec author.

    Wire format: float32 array (meters, NaN = invalid) → uint16 PNG, with
    ``0`` reserved as the invalid sentinel and the 65534 remaining quanta
    spread linearly across ``[min_range, max_range]``. Quantum size
    therefore equals ``(max_range - min_range) / 65534`` — derived at decode
    time from channel metadata.
    """

    type_name: ClassVar[str] = "DepthImage"


@dataclass(frozen=True)
class PoseSpec(ContainerSpec):
    type_name: ClassVar[str] = "foxglove.Pose"


@dataclass(frozen=True)
class PointCloudSpec(ContainerSpec):
    type_name: ClassVar[str] = "foxglove.PointCloud"


# ---- registry / dispatch -------------------------------------------------

CONTAINER_TYPES: dict[str, type[ContainerSpec]] = {
    NumericArraySpec.type_name: NumericArraySpec,
    DepthImageSpec.type_name: DepthImageSpec,
    CompressedVideoSpec.type_name: CompressedVideoSpec,
    PoseSpec.type_name: PoseSpec,
    PointCloudSpec.type_name: PointCloudSpec,
}


def build_container(d: dict[str, Any]) -> ContainerSpec:
    """Resolve a JSON ``container`` block to its ContainerSpec subclass."""
    if "type" not in d:
        raise ValueError(f"container block missing 'type': {d!r}")
    type_name = d["type"]
    if type_name not in CONTAINER_TYPES:
        raise ValueError(
            f"unknown container type {type_name!r}; expected one of "
            f"{sorted(CONTAINER_TYPES)}"
        )
    return CONTAINER_TYPES[type_name].from_dict(d)


def build_container_from_metadata(md: dict[str, str]) -> ContainerSpec:
    """Resolve an MCAP channel metadata dict to its ContainerSpec subclass."""
    if "container_type" not in md:
        raise ValueError(f"channel metadata missing 'container_type': {md!r}")
    type_name = md["container_type"]
    if type_name not in CONTAINER_TYPES:
        raise ValueError(
            f"unknown container_type {type_name!r}; expected one of "
            f"{sorted(CONTAINER_TYPES)}"
        )
    return CONTAINER_TYPES[type_name].from_channel_metadata(md)
