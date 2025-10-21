# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Need to call rapids_cpm_nvtx3 to get support for an installed version of nvtx3 and to support
# installing it ourselves
function(find_and_configure_nvtx)
  include(${rapids-cmake-dir}/cpm/nvtx3.cmake)

  # Find or install nvtx3
  rapids_cpm_nvtx3(BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports)

endfunction()

find_and_configure_nvtx()
