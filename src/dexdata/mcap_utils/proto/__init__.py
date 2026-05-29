# Copyright (C) 2026 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Owned proto schemas for the composable framework.

Re-compile via ``compile_proto.sh`` whenever ``*.proto`` changes. The
generated ``*_pb2.py`` files are checked in.

Also exposes :func:`build_file_descriptor_set`, a generic helper that walks
a protobuf message's descriptor (and its imports) to produce the
FileDescriptorSet bytes that MCAP wants for its protobuf schema records.
"""

from __future__ import annotations

from google.protobuf.descriptor_pb2 import FileDescriptorProto, FileDescriptorSet

from . import depth_image_pb2, numeric_array_pb2


def build_file_descriptor_set(message_class: type) -> bytes:
    """Serialize a FileDescriptorSet covering ``message_class`` + its imports.

    MCAP's protobuf schema record stores a serialized ``FileDescriptorSet``;
    this helper handles both owned protos (no imports) and foxglove-style
    protos (which import google/protobuf/Timestamp etc.).
    """
    fds = FileDescriptorSet()
    seen: set[str] = set()

    def _add(file_descriptor) -> None:
        if file_descriptor.name in seen:
            return
        seen.add(file_descriptor.name)
        for dep in file_descriptor.dependencies:
            _add(dep)
        fdp = FileDescriptorProto()
        file_descriptor.CopyToProto(fdp)
        fds.file.append(fdp)

    _add(message_class.DESCRIPTOR.file)
    return fds.SerializeToString()


__all__ = ["build_file_descriptor_set", "depth_image_pb2", "numeric_array_pb2"]
