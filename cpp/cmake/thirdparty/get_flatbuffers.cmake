# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Use CPM to find or clone flatbuffers
function(find_and_configure_flatbuffers VERSION)

  if(NOT BUILD_SHARED_LIBS)
    set(_exclude_from_all EXCLUDE_FROM_ALL FALSE)
  else()
    set(_exclude_from_all EXCLUDE_FROM_ALL TRUE)
  endif()

  rapids_cpm_find(
    flatbuffers ${VERSION}
    GLOBAL_TARGETS flatbuffers
    CPM_ARGS
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v${VERSION}
    GIT_SHALLOW TRUE ${_exclude_from_all}
  )

  rapids_export_find_package_root(
    BUILD flatbuffers "${flatbuffers_BINARY_DIR}" EXPORT_SET cudf-exports
  )

endfunction()

find_and_configure_flatbuffers(24.3.25)
