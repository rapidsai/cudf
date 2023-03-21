#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
# Only add the license notice to libcudf and not our examples / tests
if [[ "$PKG_NAME" == "libcudf" ]]; then
  cat ./nvcomp.txt >> $PREFIX/.messages.txt
fi
