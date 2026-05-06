# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Need to call rapids_cpm_nvtx3 to get the nvtx3 target available at configure time.
function(find_and_configure_nvtx)
  include(${rapids-cmake-dir}/cpm/nvtx3.cmake)

  # nvtx3 headers are bundled directly into cudf's install tree, so we only need the build-side
  # export set for configure-time target resolution. No INSTALL_EXPORT_SET — consumers get headers
  # from cudf's include directory without needing find_dependency(nvtx3).
  rapids_cpm_nvtx3(BUILD_EXPORT_SET cudf-exports ${CUDF_EXCLUDE_DEPS_FROM_ALL_FLAG})

  # Propagate source dir to parent scope for header installation.
  set(nvtx3_SOURCE_DIR
      "${nvtx3_SOURCE_DIR}"
      PARENT_SCOPE
  )

endfunction()

find_and_configure_nvtx()
