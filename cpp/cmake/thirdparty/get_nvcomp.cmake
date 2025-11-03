# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds nvcomp and sets any additional necessary environment variables.
function(find_and_configure_nvcomp)

  include(${rapids-cmake-dir}/cpm/nvcomp.cmake)
  set(export_args)
  if(CUDF_EXPORT_NVCOMP)
    set(export_args BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports)
  endif()
  rapids_cpm_nvcomp(${export_args} USE_PROPRIETARY_BINARY ON)

  # Per-thread default stream
  if(TARGET nvcomp AND CUDF_USE_PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(nvcomp PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)
  endif()
endfunction()

find_and_configure_nvcomp()
