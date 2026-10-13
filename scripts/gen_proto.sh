#!/usr/bin/env bash
# Regenerate the Python protobuf / gRPC bindings from pllm/proto/pipeline.proto.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
"$PY" -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. pllm/proto/pipeline.proto
echo "generated pllm/proto/pipeline_pb2.py, pipeline_pb2.pyi, pipeline_pb2_grpc.py"
