#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-date-string

rapids-logger "Configure static cpp build"

python -m pip install cmake ninja
pyenv rehash
cmake -S cpp -B build_static -GNinja -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC=OFF
