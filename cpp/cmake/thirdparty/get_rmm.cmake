# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds rmm and sets any additional necessary environment variables.
function(find_and_configure_rmm BUILD_SHARED EXCLUDE_FROM_ALL)
  include(${rapids-cmake-dir}/cpm/rmm.cmake)

  if(EXCLUDE_FROM_ALL)
    set(_exclude_flag EXCLUDE_FROM_ALL)
  else()
    set(_exclude_flag)
  endif()

  # Find or install RMM.
  set(_rmm_args BUILD_EXPORT_SET cudf-exports)
  if(NOT EXCLUDE_FROM_ALL)
    list(APPEND _rmm_args INSTALL_EXPORT_SET cudf-exports)
  endif()
  rapids_cpm_rmm(${_rmm_args} ${_exclude_flag} CPM_ARGS OPTIONS "BUILD_SHARED_LIBS ${BUILD_SHARED}")

endfunction()

find_and_configure_rmm(${CUDF_DEPS_BUILD_SHARED} ${CUDF_EXCLUDE_DEPS_FROM_ALL})
