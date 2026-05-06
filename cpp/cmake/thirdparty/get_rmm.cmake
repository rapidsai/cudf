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
  # Remove nvtx3 from rmm's public interface. Since rmm is absorbed into libcudf via whole-archive
  # and we bundle nvtx3 headers directly, consumers don't need the nvtx3 target. This prevents the
  # absorption loop from promoting nvtx3 into cudf's installed interface.
  get_target_property(_rmm_real_target rmm::rmm ALIASED_TARGET)
  if(_rmm_real_target)
    get_target_property(_rmm_iface_libs ${_rmm_real_target} INTERFACE_LINK_LIBRARIES)
    if(_rmm_iface_libs)
      list(REMOVE_ITEM _rmm_iface_libs nvtx3::nvtx3-cpp)
      set_property(TARGET ${_rmm_real_target} PROPERTY INTERFACE_LINK_LIBRARIES ${_rmm_iface_libs})
    endif()
  endif()

  # If rmm was found as a pre-existing shared library (e.g. conda), we need find_dependency(rmm) in
  # the installed config even when EXCLUDE_FROM_ALL is set. EXCLUDE_FROM_ALL controls whether CPM-
  # built targets are installed, but a pre-existing shared library is not absorbed into libcudf and
  # must be findable by downstream consumers.
  if(EXCLUDE_FROM_ALL)
    get_target_property(_rmm_type rmm::rmm TYPE)
    if(NOT _rmm_type STREQUAL "STATIC_LIBRARY")
      include("${rapids-cmake-dir}/export/package.cmake")
      rapids_export_package(INSTALL rmm cudf-exports VERSION ${rmm_VERSION})
    endif()
  endif()

  # Propagate rmm source/binary dirs to parent scope for header installation
  set(rmm_SOURCE_DIR
      "${rmm_SOURCE_DIR}"
      PARENT_SCOPE
  )
  set(rmm_BINARY_DIR
      "${rmm_BINARY_DIR}"
      PARENT_SCOPE
  )
endfunction()

find_and_configure_rmm(${CUDF_DEPS_BUILD_SHARED} ${CUDF_EXCLUDE_DEPS_FROM_ALL})
