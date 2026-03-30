# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Use CPM to find or clone CCCL
function(find_and_configure_cccl)
  include(${rapids-cmake-dir}/cpm/cccl.cmake)

  include(${rapids-cmake-dir}/cpm/package_override.cmake)

  set(rmm_patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches")
  rapids_cpm_package_override("${rmm_patch_dir}/cccl_override.json")

  rapids_cpm_cccl(BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports)
endfunction()

find_and_configure_cccl()
