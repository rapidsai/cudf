# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Use CPM to clone CRoaring and set up the necessary targets and include directories.
function(find_and_configure_croaring VERSION)
  rapids_cpm_find(
    croaring ${VERSION}
    GLOBAL_TARGETS croaring
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
  if(croaring_ADDED)
    # Position independent code to link the static library
    target_compile_options(roaring PRIVATE -fPIC)
  endif()

  if(DEFINED croaring_SOURCE_DIR)
  set(croaring_INCLUDE_DIR
        "${croaring_SOURCE_DIR}"
        PARENT_SCOPE
    )
  endif()

endfunction()

set(croaring_VERSION_cudf "4.3.10")
find_and_configure_croaring(${croaring_VERSION_cudf})
