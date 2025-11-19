# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds nanoarrow and sets any additional necessary environment variables.
function(find_and_configure_nanoarrow)
  if(NOT BUILD_SHARED_LIBS)
    set(_exclude_from_all EXCLUDE_FROM_ALL FALSE)
  else()
    set(_exclude_from_all EXCLUDE_FROM_ALL TRUE)
  endif()

  # Currently we need to always build nanoarrow so we don't pickup a previous installed version
  set(CPM_DOWNLOAD_nanoarrow ON)
  rapids_cpm_find(
    nanoarrow 0.7.0.dev
    GLOBAL_TARGETS nanoarrow
    CPM_ARGS
    GIT_REPOSITORY https://github.com/apache/arrow-nanoarrow.git
    GIT_TAG 4bf5a9322626e95e3717e43de7616c0a256179eb
    GIT_SHALLOW FALSE
    OPTIONS "BUILD_SHARED_LIBS OFF" "NANOARROW_NAMESPACE cudf" ${_exclude_from_all}
  )
  set_target_properties(nanoarrow PROPERTIES POSITION_INDEPENDENT_CODE ON)
  rapids_export_find_package_root(BUILD nanoarrow "${nanoarrow_BINARY_DIR}" EXPORT_SET cudf-exports)
endfunction()

find_and_configure_nanoarrow()
