#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

export GTEST_CUDF_RMM_MODE=cuda
export GTEST_CUDF_MEMORY_PEAK=1
export GTEST_BRIEF=1
for gt in gtests/*_TEST ; do
  test_name=$(basename "${gt}")
  echo -n "$test_name"
  # dependent on the string output from testing_main.hpp
  bytes=$(${gt} 2>/dev/null | grep Peak | cut -d' ' -f4)
  echo ",${bytes}"
done
unset GTEST_BRIEF
unset GTEST_CUDF_RMM_MODE
unset GTEST_CUDF_MEMORY_PEAK
