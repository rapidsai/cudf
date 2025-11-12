# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds nvbench and applies any needed patches.
function(find_and_configure_nvbench)

  include(${rapids-cmake-dir}/cpm/nvbench.cmake)
  include(${rapids-cmake-dir}/cpm/package_override.cmake)

  rapids_cpm_nvbench(BUILD_STATIC)

endfunction()

find_and_configure_nvbench()
