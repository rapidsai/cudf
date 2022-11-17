# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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

set(cython_sources
    aggregation.pyx
    avro.pyx
    binaryop.pyx
    column.pyx
    concat.pyx
    copying.pyx
    csv.pyx
    datetime.pyx
    expressions.pyx
    filling.pyx
    groupby.pyx
    hash.pyx
    interop.pyx
    join.pyx
    json.pyx
    labeling.pyx
    lists.pyx
    merge.pyx
    null_mask.pyx
    orc.pyx
    parquet.pyx
    partitioning.pyx
    quantiles.pyx
    reduce.pyx
    replace.pyx
    reshape.pyx
    rolling.pyx
    round.pyx
    scalar.pyx
    search.pyx
    sort.pyx
    stream_compaction.pyx
    string_casting.pyx
    text.pyx
    transform.pyx
    transpose.pyx
    types.pyx
    unary.pyx
    utils.pyx
)
set(linked_libraries cudf::cudf)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" ASSOCIATED_TARGETS cudf
)

# TODO: Finding NumPy currently requires finding Development due to a bug in CMake. This bug was
# fixed in https://gitlab.kitware.com/cmake/cmake/-/merge_requests/7410 and will be available in
# CMake 3.24, so we can remove the Development component once we upgrade to CMake 3.24.
find_package(Python REQUIRED COMPONENTS Development NumPy)
set(targets_using_numpy interop avro csv orc json parquet)
foreach(target IN LISTS targets_using_numpy)
  target_include_directories(${target} PRIVATE "${Python_NumPy_INCLUDE_DIRS}")
endforeach()

set(targets_using_dlpack interop)
foreach(target IN LISTS targets_using_dlpack)
  target_include_directories(${target} PRIVATE "${DLPACK_INCLUDE_DIR}")
endforeach()

add_subdirectory(io)
add_subdirectory(nvtext)
add_subdirectory(strings)
