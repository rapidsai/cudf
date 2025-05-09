#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

if [[ ! -d "/tmp/gha-tools" ]]; then
  git clone \
    --branch "gha-artifacts/consolidate-scripts" \
    https://github.com/rapidsai/gha-tools.git \
    /tmp/gha-tools

  export PATH="/tmp/gha-tools/tools":$PATH
fi
