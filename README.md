# dexdata

Data recording and replay utilities for Dexmate robots. Reads and writes MCAP episode data (robot state/action, RGB cameras, depth) with Protobuf serialization, and encodes cameras to video.

## Install

```bash
pip install -e .            # core: MCAP read/write
pip install -e ".[convert]"   # adds torch + torchcodec for video_writer
```

Requires Python 3.11.

## Layout

- `dexdata.mcap_utils` — `TeleopMcapWriter`, `TeleopMcapReader`, `Episode`, `EpisodeMetadata`. Self-contained.
- `dexdata.video_writer` — torch + torchcodec-backed parallel/serial video encoder (opt-in via `[video]` extra).

## Quick read

```python
from pathlib import Path
from dexdata.mcap_utils import TeleopMcapReader

reader = TeleopMcapReader(Path("/path/to/episode_dir"))
episode = reader.load_episode()
data = episode.to_numpy_dict()   # flat {key: np.ndarray} for training
fps = reader.metadata.record_rate_hz
```

An episode directory contains `state_action.mcap`, zero or more `camera_*.mcap` / `depth_*.mcap`, and `info.json`.

## Scripts

- `scripts/example_mcap_read.py <episode_dir>` — dump an episode to its numpy-dict form and print key shapes/dtypes.
- `scripts/mcap_to_video.py --episode_path <dir>` — render a 2×2 camera grid to `concatenated.mp4`. FPS defaults to `robot.record_rate_hz` from `info.json`.
- `scripts/mcap_viz.py` — Rerun-based episode visualization helper (`RerunViz`).

## Protobuf schemas

Schemas live in `src/dexdata/mcap_utils/proto/`. Compiled `*_pb2.py` modules are checked in; regenerate with:

```bash
cd src/dexdata/mcap_utils/proto && ./compile_proto.sh
```
