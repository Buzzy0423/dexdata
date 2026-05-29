# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Round-trip tests for the NumericArray handler + proto.

Per decision 2.13, the handler is purely value ↔ bytes — timestamps live on
the MCAP envelope (``publish_time`` / ``log_time``) and are set by the writer
at ``add_message`` time, not by the handler.

Exercises three cases:
  1. In-memory: handler.serialize → handler.deserialize over 1000 frames.
  2. MCAP: write 1000 messages with explicit publish_time + log_time on the
     envelope, read back, assert byte-equal arrays and round-tripped envelope
     timestamps.
  3. Validation: shape and dtype mismatches each raise ValueError.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dexdata.handlers.containers import NumericArraySpec
from dexdata.mcap_utils.handlers.numeric_array import NumericArrayHandler
from dexdata.mcap_utils.proto.numeric_array_pb2 import NumericArray

SHAPE = (7,)
DTYPE = np.dtype("float32")
N_FRAMES = 1000
BASE_TS_NS = 1_700_000_000_000_000_000
PERIOD_NS = 33_000_000  # ~30 Hz
RECV_DELAY_NS = 500_000  # 0.5ms simulated subscriber latency


def _spec() -> NumericArraySpec:
    return NumericArraySpec(shape=SHAPE, dtype=DTYPE)


def _random_arrays(n: int = N_FRAMES) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.standard_normal(SHAPE).astype(DTYPE) for _ in range(n)]


def test_inmemory_roundtrip() -> None:
    h = NumericArrayHandler(_spec())
    arrays = _random_arrays()
    for arr in arrays:
        # serialize/deserialize are both iterables; for NumericArray each
        # yields exactly one record.
        ((payload, _, _),) = list(h.serialize(arr, 0, 0))
        (arr_back,) = list(h.deserialize(payload))
        assert arr_back.dtype == DTYPE
        assert arr_back.shape == SHAPE
        assert np.array_equal(arr_back, arr)


def test_mcap_roundtrip() -> None:
    # mcap-protobuf-support is a test-only convenience; production code uses
    # bare mcap + generated pb2 modules. Skip rather than declare a runtime
    # dep we don't actually need.
    pytest.importorskip("mcap_protobuf")
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    from mcap_protobuf.writer import Writer as ProtoWriter

    spec = _spec()
    h = NumericArrayHandler(spec)
    arrays = _random_arrays()
    topic = "/robot/state/left_arm/qpos"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.mcap"
        with open(path, "wb") as f:
            with ProtoWriter(f) as writer:
                for i, arr in enumerate(arrays):
                    publish_ts = BASE_TS_NS + i * PERIOD_NS
                    recv_ts = publish_ts + RECV_DELAY_NS
                    msg = NumericArray()
                    ((payload, _, _),) = list(h.serialize(arr, 0, 0))
                    msg.ParseFromString(payload)
                    writer.write_message(
                        topic=topic,
                        message=msg,
                        log_time=recv_ts,
                        publish_time=publish_ts,
                    )

        received: list[tuple[np.ndarray, int, int]] = []
        with open(path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _schema, _channel, message, proto_msg in reader.iter_decoded_messages():
                (arr_back,) = list(h.deserialize(proto_msg.SerializeToString()))
                received.append((arr_back, message.publish_time, message.log_time))

    assert len(received) == N_FRAMES
    for i, ((arr_back, pub_back, recv_back), arr) in enumerate(
        zip(received, arrays, strict=False)
    ):
        expected_pub = BASE_TS_NS + i * PERIOD_NS
        assert pub_back == expected_pub
        assert recv_back == expected_pub + RECV_DELAY_NS
        assert np.array_equal(arr_back, arr)


def test_serialize_rejects_wrong_shape() -> None:
    h = NumericArrayHandler(_spec())
    bad = np.zeros((6,), dtype=DTYPE)
    with pytest.raises(ValueError, match="shape mismatch"):
        # Generator: error fires when we attempt to consume the first record.
        list(h.serialize(bad, 0, 0))


def test_serialize_rejects_wrong_dtype() -> None:
    h = NumericArrayHandler(_spec())
    bad = np.zeros(SHAPE, dtype=np.float64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        list(h.serialize(bad, 0, 0))
