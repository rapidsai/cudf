# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Use CPM to clone CRoaring and set up the necessary targets and include directories.
function(find_and_configure_roaring VERSION)
  rapids_cpm_find(
    roaring ${VERSION}
    GLOBAL_TARGETS roaring
    CPM_ARGS
    GIT_REPOSITORY https://github.com/RoaringBitmap/CRoaring.git
    GIT_TAG v${VERSION}
    GIT_SHALLOW TRUE
    OPTIONS "ROARING_BUILD_STATIC ON"
            "BUILD_SHARED_LIBS OFF"
            "ENABLE_ROARING_TESTS OFF"
            "ENABLE_ROARING_MICROBENCHMARKS OFF"
            "ROARING_DISABLE_NEON ON"
            "ROARING_DISABLE_X64 ON"
            "ROARING_DISABLE_AVX2 ON"
            "ROARING_DISABLE_AVX512 ON"
  )
  if(roaring_ADDED)
    set_target_properties(roaring PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()

  if(DEFINED roaring_SOURCE_DIR)
    set(roaring_INCLUDE_DIR
        "${roaring_SOURCE_DIR}"
        PARENT_SCOPE
    )
  endif()

endfunction()

set(roaring_VERSION_cudf "4.3.11")
find_and_configure_roaring(${roaring_VERSION_cudf})
