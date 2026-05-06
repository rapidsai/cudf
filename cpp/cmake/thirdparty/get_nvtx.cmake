# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Need to call rapids_cpm_nvtx3 to get support for an installed version of nvtx3 and to support
# installing it ourselves
function(find_and_configure_nvtx)
  include(${rapids-cmake-dir}/cpm/nvtx3.cmake)

  # nvtx3 is private for cudf, but it is a public dependency of rmm. When rmm is absorbed into
  # libcudf via whole-archive, rmm's public transitive deps (including nvtx3-cpp) are promoted into
  # cudf's public interface. CMake's export validation requires that nvtx3-cpp be in an export set.
  rapids_cpm_nvtx3(
    BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports
                                                     ${CUDF_EXCLUDE_DEPS_FROM_ALL_FLAG}
  )

endfunction()

find_and_configure_nvtx()
