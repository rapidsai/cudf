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

  # TODO: nvtx is linked privately, so we shouldn't have to export it, but it is part of rmm's
  # public export set and we run into ordering/first find wins issues if we don't export it in cudf
  # as well. We should be able to remove this once we have a better solution for handling rmm's
  # export sets consistently from cudf.
  rapids_cpm_nvtx3(
    BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports
                                                     ${CUDF_EXCLUDE_DEPS_FROM_ALL_FLAG}
  )

endfunction()

find_and_configure_nvtx()
