# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Use CPM to find or clone CCCL
function(find_and_configure_cccl)
  include(${rapids-cmake-dir}/cpm/cccl.cmake)
  rapids_cpm_cccl(BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports)
endfunction()

find_and_configure_cccl()
