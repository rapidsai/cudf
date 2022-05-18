#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.
# Only add the license notice to libcudf and not our examples / tests
if [[ "$PKG_NAME" == "libcudf" ]]; then
  cat ./nvlink.txt >> $PREFIX/.messages.txt
fi
