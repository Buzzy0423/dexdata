#!/bin/bash
# Compile composable owned-proto schemas to Python.
# Requires protoc on PATH (apt: protobuf-compiler, conda: protobuf).
#
# Re-run whenever a *.proto file in this directory changes; the generated
# *_pb2.py files are checked in.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

protoc --python_out=. numeric_array.proto depth_image.proto

echo "Compiled:"
echo "  - numeric_array_pb2.py"
echo "  - depth_image_pb2.py"
