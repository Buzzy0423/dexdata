# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Tests for composable.discovery.discover_episodes."""

from __future__ import annotations

from pathlib import Path

import pytest

from dexdata.mcap_utils.discovery import discover_episodes


def test_discovers_nested_episodes(tmp_path: Path) -> None:
    """Sub-directories anywhere under root are reported, sorted."""
    (tmp_path / "a/ep1").mkdir(parents=True)
    (tmp_path / "a/ep1/episode.mcap").touch()
    (tmp_path / "b/run/ep2").mkdir(parents=True)
    (tmp_path / "b/run/ep2/episode.mcap").touch()
    (tmp_path / "no_episode_here").mkdir()
    (tmp_path / "no_episode_here/notes.txt").touch()

    found = discover_episodes(tmp_path)
    assert found == [tmp_path / "a/ep1", tmp_path / "b/run/ep2"]


def test_empty_root_returns_empty(tmp_path: Path) -> None:
    assert discover_episodes(tmp_path) == []


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_episodes(tmp_path / "does_not_exist")
