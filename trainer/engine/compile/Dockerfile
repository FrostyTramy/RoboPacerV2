# Hailo DFC 3.34.0 compilation environment (Ubuntu 22.04 + Python 3.10)
# Build context: <trainer>/engine/
#
# The Hailo Dataflow Compiler wheel is NOT downloadable automatically - it
# requires a (free) Hailo Developer Zone account. Download
# hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl from there and
# place it at engine/compile/resources/ before building this image.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        wget \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Pre-install DFC runtime dependencies
RUN pip install \
    "numpy>=1.23.5,<2.0" \
    scipy \
    matplotlib \
    pillow \
    networkx \
    flatbuffers \
    PyYAML \
    tqdm \
    onnx \
    onnxruntime \
    "tensorflow-cpu==2.12.0"

# Install Hailo Dataflow Compiler (Linux x86_64 wheel - see note above)
COPY compile/resources/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl /tmp/
RUN pip install /tmp/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl

# Ubuntu's pip installs to dist-packages, but the DFC's own path detection
# only checks for "site-packages" - patch it so it recognizes its own
# install location.
RUN python3.10 -c "\
import os, glob; \
p = '/usr/local/lib/python3.10/dist-packages/hailo_sdk_common/paths_manager/paths.py'; \
src = open(p).read(); \
old = '\"site-packages\" in os.path.dirname(os.path.dirname(hailo_sdk_common.origin))'; \
new = '(\"site-packages\" in os.path.dirname(os.path.dirname(hailo_sdk_common.origin)) or \"dist-packages\" in os.path.dirname(os.path.dirname(hailo_sdk_common.origin)))'; \
open(p, 'w').write(src.replace(old, new)) if old in src else None; \
[os.remove(f) for f in glob.glob(os.path.join(os.path.dirname(p), '__pycache__', 'paths.cpython-*.pyc'))]"

RUN python3.10 -c "import hailo_sdk_client; print('DFC OK')"

WORKDIR /workspace
