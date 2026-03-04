# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds cuCollections and performs any additional configuration.
function(find_and_configure_cucollections)
  include(${rapids-cmake-dir}/cpm/cuco.cmake)

  rapids_cpm_cuco(BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports)
endfunction()

find_and_configure_cucollections()
