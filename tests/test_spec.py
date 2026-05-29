# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Spec loader: `includes` composition, leaf collisions, cycle detection.

The flat-channels path is exercised indirectly throughout the rest of the
suite; this module covers the composition behavior added with the
``embodiment/`` + ``module/`` reorganization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dexdata.spec import load_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
EMBODIMENT_DIR = REPO_ROOT / "src/dexdata/specs/embodiment"
EMBODIMENT_SPEC = EMBODIMENT_DIR / "vega_1u_gripper.json"

# (filename, expected spec_id, expected signal count) per shipped embodiment.
# Counts are derived from the humanoid+hand module pair each one composes;
# update when modules change.
EMBODIMENTS = [
    ("example.json", "all_container_types", 5),
    ("vega_1_f5d6.json", "vega_1_f5d6", 22),
    ("vega_1_gripper.json", "vega_1_gripper", 24),
    ("vega_1u_gripper.json", "vega_1u_gripper", 21),
]

# vega_1u is upper-body only — no torso, no chassis.
EXPECTED_TOPICS = {
    "/robot/state/left_arm/qpos",
    "/robot/state/left_arm/qvel",
    "/robot/state/right_arm/qpos",
    "/robot/state/right_arm/qvel",
    "/robot/state/head/qpos",
    "/robot/state/left_hand/qpos",
    "/robot/state/left_hand/wrench/force",
    "/robot/state/left_hand/wrench/torque",
    "/robot/state/right_hand/qpos",
    "/robot/state/right_hand/wrench/force",
    "/robot/state/right_hand/wrench/torque",
    "/robot/action/left_arm/qpos",
    "/robot/action/right_arm/qpos",
    "/robot/action/head/qpos",
    "/robot/action/left_hand/qpos",
    "/robot/action/right_hand/qpos",
    "/camera/head_left/rgb/video",
    "/camera/head_right/rgb/video",
    "/camera/head_left/depth",
    "/camera/left_wrist/rgb/video",
    "/camera/right_wrist/rgb/video",
}


def test_vega_1u_gripper_composes_from_modules():
    """The shipped vega_1u_gripper spec is two `includes` and nothing else.

    Verifies the composer reproduces the full topic set from the
    humanoid + dex_gripper modules, with no collisions.
    """
    spec = load_spec(EMBODIMENT_SPEC)
    assert spec.spec_id == "vega_1u_gripper"
    assert {s.topic for s in spec.signals} == EXPECTED_TOPICS


@pytest.mark.parametrize(("filename", "spec_id", "n_signals"), EMBODIMENTS)
def test_embodiment_loads(filename: str, spec_id: str, n_signals: int):
    """Every shipped embodiment spec composes cleanly with the expected size.

    Catches accidental collisions or missing modules when new humanoid/hand
    combinations are added. Update the EMBODIMENTS table above when modules
    change shape.
    """
    spec = load_spec(EMBODIMENT_DIR / filename)
    assert spec.spec_id == spec_id
    assert len(spec.signals) == n_signals


def _leaf(shape: list[int]) -> dict:
    return {
        "unit": "rad",
        "container": {"type": "NumericArray", "shape": shape, "dtype": "float32"},
    }


def test_local_channels_merge_on_top_of_includes(tmp_path: Path):
    """A spec can both include modules and add its own `channels` block."""
    module = tmp_path / "mod.json"
    module.write_text(
        json.dumps(
            {
                "spec_id": "mod",
                "spec_version": "0.1.0",
                "channels": {"a": {"x": _leaf([1])}},
            }
        )
    )
    top = tmp_path / "top.json"
    top.write_text(
        json.dumps(
            {
                "spec_id": "top",
                "spec_version": "0.1.0",
                "includes": ["mod.json"],
                "channels": {"b": {"y": _leaf([2])}},
            }
        )
    )
    spec = load_spec(top)
    assert {s.topic for s in spec.signals} == {"/a/x", "/b/y"}


def test_leaf_collision_raises(tmp_path: Path):
    """Two includes claiming the same topic must fail loud."""
    m1 = tmp_path / "m1.json"
    m1.write_text(
        json.dumps(
            {
                "spec_id": "m1",
                "spec_version": "0.1.0",
                "channels": {"a": {"x": _leaf([1])}},
            }
        )
    )
    m2 = tmp_path / "m2.json"
    m2.write_text(
        json.dumps(
            {
                "spec_id": "m2",
                "spec_version": "0.1.0",
                "channels": {"a": {"x": _leaf([2])}},
            }
        )
    )
    top = tmp_path / "top.json"
    top.write_text(
        json.dumps(
            {
                "spec_id": "top",
                "spec_version": "0.1.0",
                "includes": ["m1.json", "m2.json"],
            }
        )
    )
    with pytest.raises(ValueError, match="channel collision"):
        load_spec(top)


def test_circular_include_raises(tmp_path: Path):
    """A -> B -> A must raise rather than recurse forever."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps({"spec_id": "a", "spec_version": "0.1.0", "includes": ["b.json"]})
    )
    b.write_text(
        json.dumps({"spec_id": "b", "spec_version": "0.1.0", "includes": ["a.json"]})
    )
    with pytest.raises(ValueError, match="circular include"):
        load_spec(a)


def test_missing_channels_and_includes_raises(tmp_path: Path):
    """A spec must declare at least one of `channels` or `includes`."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"spec_id": "bad", "spec_version": "0.1.0"}))
    with pytest.raises(ValueError, match="channels.*includes"):
        load_spec(bad)
