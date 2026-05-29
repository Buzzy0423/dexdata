# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Composable spec: JSON → flat list of `(topic, signal)` pairs.

Tree position of a leaf becomes its topic path:
``channels.robot.state.left_arm.qpos`` → ``/robot/state/left_arm/qpos``.

Invariant (decision §2.3): every node is **either** a namespace (object whose
values are all dicts) **or** a leaf (object containing a ``container`` key).
No node is both. The loader uses this to flatten ``channels`` in a single pass
to ``{topic: leaf}``; everything else is a flat iteration.

Composition: a spec file may declare an ``includes`` array of paths (relative
to the file). The loader resolves them depth-first and deep-merges their
``channels`` trees in array order; a local ``channels`` block (optional)
merges on top. Namespaces merge; same-topic leaf collisions raise.

Fail-fast: unknown container types, missing fields, non-positive shape entries,
duplicate topics, unexpected sibling keys, leaf/namespace collisions during
include-merge, or circular includes all raise at load.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .handlers.containers import ContainerSpec, build_container


@dataclass(frozen=True)
class SignalSpec:
    topic: str
    unit: str
    container: ContainerSpec


@dataclass(frozen=True)
class Spec:
    spec_id: str
    spec_version: str
    signals: tuple[SignalSpec, ...]

    def by_topic(self) -> dict[str, SignalSpec]:
        return {s.topic: s for s in self.signals}

    def topics(self) -> list[str]:
        return [s.topic for s in self.signals]


def restrict(spec: Spec, keep_topics: Iterable[str]) -> Spec:
    """Return a copy of ``spec`` containing only signals whose topic is in
    ``keep_topics``. Signal order is preserved.

    Raises ``ValueError`` if ``keep_topics`` contains any topic the spec does
    not declare — callers see typos at restrict-time, not at write-time.
    Component-level filtering (e.g., omniteleop's ``record_components``)
    lives in the caller; this helper only knows about topics.
    """
    keep = set(keep_topics)
    declared = {s.topic for s in spec.signals}
    unknown = keep - declared
    if unknown:
        raise ValueError(
            f"restrict: topics not in spec {spec.spec_id!r}: {sorted(unknown)}"
        )
    kept = tuple(s for s in spec.signals if s.topic in keep)
    return replace(spec, signals=kept)


_LEAF_KEYS = frozenset({"unit", "container"})


def flatten_channels(
    node: dict[str, Any], path: tuple[str, ...] = ()
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Walk a `channels` tree once; yield `(topic, leaf_dict)` per leaf.

    A leaf is any node containing a ``container`` key. The walker assumes the
    spec invariant — every other node is a namespace whose values are all
    dicts — and raises on violations (non-dict child of a namespace).
    """
    if "container" in node:
        yield "/" + "/".join(path), node
        return
    for key, child in node.items():
        if not isinstance(child, dict):
            raise ValueError(
                f"non-object at /{'/'.join((*path, key))}: {type(child).__name__}"
            )
        yield from flatten_channels(child, (*path, key))


def _is_leaf(node: Any) -> bool:
    return isinstance(node, dict) and "container" in node


def _deep_merge_channels(
    into: dict[str, Any], from_: dict[str, Any], path: tuple[str, ...] = ()
) -> None:
    """Deep-merge ``from_`` into ``into``. Namespaces merge; leaf collisions raise.

    A "leaf collision" is either:
      * the same topic path appearing in both trees (both sides are leaves), or
      * one side being a leaf and the other a namespace at the same path
        (the spec invariant says a node is one or the other, never both).
    """
    for key, value in from_.items():
        if key not in into:
            into[key] = copy.deepcopy(value)
            continue
        existing = into[key]
        if not isinstance(existing, dict) or not isinstance(value, dict):
            raise ValueError(f"non-object at /{'/'.join((*path, key))}: cannot merge")
        if _is_leaf(existing) or _is_leaf(value):
            raise ValueError(
                f"channel collision at /{'/'.join((*path, key))}: "
                f"leaf/namespace conflict or duplicate topic"
            )
        _deep_merge_channels(existing, value, (*path, key))


def _resolve_includes(
    path: Path, raw: dict[str, Any], stack: tuple[Path, ...]
) -> dict[str, Any]:
    """Return the composed ``channels`` dict for ``raw``, resolving includes.

    Includes are paths relative to ``path``'s parent. Resolved depth-first;
    later includes merge over earlier ones; a local ``channels`` (if present)
    merges over all of them. Cycles raise.
    """
    resolved = path.resolve()
    if resolved in {p.resolve() for p in stack}:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"circular include: {chain}")
    composed: dict[str, Any] = {}
    for inc in raw.get("includes", []):
        inc_path = (path.parent / inc).resolve()
        with open(inc_path) as f:
            inc_raw = json.load(f)
        inc_channels = _resolve_includes(inc_path, inc_raw, (*stack, path))
        _deep_merge_channels(composed, inc_channels)
    if "channels" in raw:
        _deep_merge_channels(composed, raw["channels"])
    return composed


def load_spec(path: str | Path) -> Spec:
    """Load and validate a spec JSON file."""
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    for key in ("spec_id", "spec_version"):
        if key not in raw:
            raise ValueError(f"{path}: missing top-level key {key!r}")
    if "channels" not in raw and "includes" not in raw:
        raise ValueError(f"{path}: must declare 'channels' or 'includes'")

    channels = _resolve_includes(path, raw, ())

    signals: list[SignalSpec] = []
    seen: set[str] = set()
    for topic, leaf in flatten_channels(channels):
        if topic in seen:
            raise ValueError(f"duplicate topic: {topic}")
        seen.add(topic)
        extra = set(leaf) - _LEAF_KEYS
        if extra:
            raise ValueError(f"leaf {topic} has unexpected keys: {sorted(extra)}")
        if "unit" not in leaf:
            raise ValueError(f"leaf {topic} missing 'unit'")
        signals.append(
            SignalSpec(
                topic=topic,
                unit=str(leaf["unit"]),
                container=build_container(leaf["container"]),
            )
        )

    return Spec(
        spec_id=str(raw["spec_id"]),
        spec_version=str(raw["spec_version"]),
        signals=tuple(signals),
    )


# ---- CLI ------------------------------------------------------------------


def _print(spec: Spec) -> None:
    print(f"spec: {spec.spec_id} @ {spec.spec_version}  ({len(spec.signals)} signals)")
    if not spec.signals:
        return
    width = max(len(s.topic) for s in spec.signals)
    for s in spec.signals:
        md = s.container.channel_metadata()
        extras = " ".join(f"{k}={v}" for k, v in md.items() if k != "container_type")
        print(
            f"  {s.topic.ljust(width)}  unit={s.unit!r:<6}  "
            f"{s.container.type_name}  {extras}".rstrip()
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Load a composable spec JSON and print its topic-path tree."
    )
    p.add_argument("paths", nargs="+", type=Path)
    args = p.parse_args()
    for i, path in enumerate(args.paths):
        if i:
            print()
        _print(load_spec(path))


if __name__ == "__main__":
    main()
