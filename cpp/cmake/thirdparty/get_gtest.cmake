# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds gtest and sets any additional necessary environment variables.
function(find_and_configure_gtest)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)

  # Find or install GoogleTest
  rapids_cpm_gtest(BUILD_STATIC)

endfunction()

find_and_configure_gtest()
