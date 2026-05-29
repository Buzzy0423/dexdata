# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Handler for the ``foxglove.CompressedVideo`` container.

Stateful encoder/decoder built on PyAV. The handler is the only one in the
alphabet whose :meth:`serialize` does not output one-record-per-call:
codecs buffer frames internally for B-frame analysis / lookahead, so a
single input frame may produce 0, 1, or 2+ output packets, and the final
batch only emerges from :meth:`flush`. Decode mirrors this asymmetry.

**Color convention: RGB uint8 on both ends.** The handler's numpy
contract is ``(H, W, 3)`` uint8 in **RGB** byte order at both
:meth:`serialize` (input) and :meth:`deserialize` (output). PyAV/swscale
goes RGB↔YUV with the same correctness regardless of which byte order
the numpy buffer uses; the format flag is purely a memory-layout
descriptor. Previously the handler used ``bgr24`` end-to-end (cv2's
native order at the publisher); switching both sides to ``rgb24``
eliminates the downstream LeRobot BGR→RGB copy without re-encoding any
on-disk mp4s — the YUV bitstream is identical between the two
conventions, only the numpy byte order at the boundary changes. So
existing episodes encoded with the old ``bgr24`` convention now decode
to RGB-ordered numpy under this handler, and that's the correct
semantic content (because the swscale math used at original encode time
was R-channel-aware regardless of the byte layout it read R from).
Callers handing BGR numpy to :meth:`serialize` must ``cv2.cvtColor``
(or equivalent) before the call.

**pts → envelope-timestamp tracking.** Each input frame is assigned a
monotonic ``pts`` (0, 1, 2, ...). The handler stores
``pts → (publish_ts_ns, recv_ts_ns)`` so that when the encoder later emits
a packet whose ``pkt.pts`` references frame N, we can look up frame N's
*original* timestamps and emit them with the packet — not the timestamps
of the call that happened to flush the encoder.

**Order preservation.** With ``bf=0`` for h264 and the default settings of
``libaom-av1`` + ``libdav1d``, packet pts and decoded frame pts are
monotonic — verified empirically before adopting these defaults. The
reader can therefore pair decoded frames with envelope timestamps
positionally; we don't need to round-trip pts through the decoder.

**Codec / decoder names.** PyAV decoders aren't always named the same as
their encoders; in particular the av1 decoder is ``libdav1d``, not
``av1``. ``_DECODER_FOR_FORMAT`` maps the spec's codec name → PyAV decoder.

**Spec field caveats.** ``preset`` is a codec-specific knob, forwarded under
each encoder's native option name: ``preset`` for libx264 (named strings
``ultrafast``..``veryslow``) and for libsvtav1 (digits ``0``..``13``,
slowest..fastest). The two scales are not interchangeable but the field
shape (a string) is uniform.

**SVT-AV1 vs libaom-av1.** We use ``libsvtav1`` because it matches the
existing recorder's encoder and offers materially better RD performance at
the same preset. ``libsvtav1`` outputs packets in display order (verified
empirically with default ``random access`` pred-struct), so the reader's
positional FIFO pairing still holds; no ``pred-struct=1`` low-delay
override needed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, ClassVar

import av
import numpy as np
from foxglove_schemas_protobuf import CompressedVideo_pb2

from ...handlers.containers import CompressedVideoSpec

_ENCODER_FOR_CODEC = {
    "h264": "libx264",
    "av1": "libsvtav1",
}

_DECODER_FOR_CODEC = {
    "h264": "h264",
    "av1": "libdav1d",
}

# Frame rate is informational for the encoder's rate-control; we don't have
# a meaningful FPS at the spec layer (frames may arrive at any cadence) so
# we pin a placeholder. Timestamps in MCAP carry the actual cadence.
_PLACEHOLDER_FPS = 30


def _encoder_options(codec: str, crf: int, gop: int, preset: str) -> dict[str, str]:
    """Map spec fields onto the encoder's native option names.

    Both supported encoders take ``preset`` under that same name (libx264's
    named presets and libsvtav1's digit presets), so the only codec-specific
    quirk is ``bf=0`` for h264 to keep packet pts == input pts.
    """
    if codec not in _ENCODER_FOR_CODEC:
        raise ValueError(f"unsupported codec: {codec!r}")
    opts: dict[str, str] = {"crf": str(crf), "g": str(gop), "preset": preset}
    if codec == "h264":
        opts["bf"] = "0"  # disable B-frames → packet pts == input pts
    return opts


@dataclass(eq=False)
class CompressedVideoHandler:
    """Encode/decode an RGB camera stream via PyAV.

    See the module docstring for the RGB-uint8 contract that callers
    on both sides must honor.

    Not ``frozen=True``: encoder/decoder contexts are stateful and must
    mutate. ``eq=False`` because identity-comparison is the only sensible
    semantics for a handler with live codec state. ``spec``, ``width``,
    ``height`` are dataclass fields so handler-introspection in
    ``handlers/__init__.py`` (``runtime_field_types``) discovers them
    uniformly with the other handlers.
    """

    spec: CompressedVideoSpec
    width: int
    height: int

    proto_class: ClassVar[type] = CompressedVideo_pb2.CompressedVideo
    schema_name: ClassVar[str] = (
        CompressedVideo_pb2.CompressedVideo.DESCRIPTOR.full_name
    )

    def __post_init__(self) -> None:
        if self.spec.codec not in _ENCODER_FOR_CODEC:
            raise ValueError(
                f"unsupported codec {self.spec.codec!r}; "
                f"expected one of {sorted(_ENCODER_FOR_CODEC)}"
            )
        self.width = int(self.width)
        self.height = int(self.height)
        self._encoder: Any | None = None
        self._decoder: Any | None = None
        self._next_pts: int = 0
        self._pts_to_ts: dict[int, tuple[int, int]] = {}

    def _ensure_encoder(self) -> Any:
        if self._encoder is None:
            enc = av.CodecContext.create(_ENCODER_FOR_CODEC[self.spec.codec], "w")
            enc.width = self.width
            enc.height = self.height
            enc.pix_fmt = "yuv420p"
            enc.framerate = Fraction(_PLACEHOLDER_FPS, 1)
            enc.time_base = Fraction(1, _PLACEHOLDER_FPS)
            enc.options = _encoder_options(
                self.spec.codec, self.spec.crf, self.spec.gop, self.spec.preset
            )
            self._encoder = enc
        return self._encoder

    def _ensure_decoder(self) -> Any:
        if self._decoder is None:
            self._decoder = av.CodecContext.create(
                _DECODER_FOR_CODEC[self.spec.codec], "r"
            )
        return self._decoder

    def serialize(
        self, value: np.ndarray, publish_ts_ns: int, recv_ts_ns: int
    ) -> Iterable[tuple[bytes, int, int]]:
        if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
            raise ValueError(
                f"CompressedVideo expects (H, W, 3) uint8 RGB; got "
                f"shape={value.shape} dtype={value.dtype}"
            )
        h, w, _ = value.shape
        if (h, w) != (self.height, self.width):
            raise ValueError(
                f"frame size {(h, w)} does not match handler ({self.height}, {self.width})"
            )

        encoder = self._ensure_encoder()
        pts = self._next_pts
        self._next_pts += 1
        self._pts_to_ts[pts] = (publish_ts_ns, recv_ts_ns)

        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(value), format="rgb24")
        vf = vf.reformat(format="yuv420p")
        vf.pts = pts
        for pkt in encoder.encode(vf):
            yield from self._wrap_packet(pkt)

    def flush(self) -> Iterable[tuple[bytes, int, int]]:
        if self._encoder is None:
            return
        for pkt in self._encoder.encode(None):
            yield from self._wrap_packet(pkt)

    def _wrap_packet(self, pkt: Any) -> Iterable[tuple[bytes, int, int]]:
        pkt_pts = pkt.pts
        if pkt_pts is None or pkt_pts not in self._pts_to_ts:
            # Fall back to oldest pending pts; with bf=0 / libaom default this
            # branch is unreachable, but it keeps us safe if the codec ever
            # drops the pts.
            if not self._pts_to_ts:
                return
            pkt_pts = min(self._pts_to_ts)
        publish_ts_ns, recv_ts_ns = self._pts_to_ts.pop(pkt_pts)
        msg = self.proto_class()
        msg.format = self.spec.codec
        msg.data = bytes(pkt)
        # `timestamp` and `frame_id` left at proto3 defaults (decision 2.13).
        yield (msg.SerializeToString(), publish_ts_ns, recv_ts_ns)

    def deserialize(self, payload: bytes) -> Iterable[np.ndarray]:
        msg = self.proto_class()
        msg.ParseFromString(payload)
        decoder = self._ensure_decoder()
        for frame in decoder.decode(av.Packet(msg.data)):
            yield frame.to_ndarray(format="rgb24")

    def flush_decode(self) -> Iterable[np.ndarray]:
        if self._decoder is None:
            return
        for frame in self._decoder.decode(None):
            yield frame.to_ndarray(format="rgb24")
