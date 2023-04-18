# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
    strings_udf.pyx
    text.pyx
    timezone.pyx
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

# All modules need to include the header containing the exception handler.
foreach(target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
endforeach()

target_link_libraries(strings_udf cudf_strings_udf)

# TODO: Finding NumPy currently requires finding Development due to a bug in CMake. This bug was
# fixed in https://gitlab.kitware.com/cmake/cmake/-/merge_requests/7410 and will be available in
# CMake 3.24, so we can remove the Development component once we upgrade to CMake 3.24.
# find_package(Python REQUIRED COMPONENTS Development NumPy)

# Note: The bug noted above prevents us from finding NumPy successfully using FindPython.cmake
# inside the manylinux images used to build wheels because manylinux images do not contain
# libpython.so and therefore Development cannot be found. Until we upgrade to CMake 3.24, we should
# use FindNumpy.cmake instead (provided by scikit-build). When we switch to 3.24 we can try
# switching back, but it may not work if that implicitly still requires Python libraries. In that
# case we'll need to follow up with the CMake team to remove that dependency.  The stopgap solution
# is to unpack the static lib tarballs in the wheel building jobs so that there are at least static
# libs to be found, but that should be a last resort since it implies a dependency that isn't really
# necessary. The relevant command is tar -xf /opt/_internal/static-libs-for-embedding-only.tar.xz -C
# /opt/_internal"
find_package(NumPy REQUIRED)
set(targets_using_numpy interop avro csv orc json parquet)
foreach(target IN LISTS targets_using_numpy)
  target_include_directories(${target} PRIVATE "${NumPy_INCLUDE_DIRS}")
  # Switch to the line below when we switch back to FindPython.cmake in CMake 3.24.
  # target_include_directories(${target} PRIVATE "${Python_NumPy_INCLUDE_DIRS}")
endforeach()

set(targets_using_dlpack interop)
foreach(target IN LISTS targets_using_dlpack)
  target_include_directories(${target} PRIVATE "${DLPACK_INCLUDE_DIR}")
endforeach()

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pyarrow; print(pyarrow.get_include())"
  OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(targets_using_arrow_headers interop avro csv orc json parquet)
foreach(target IN LISTS targets_using_arrow_headers)
  target_include_directories(${target} PRIVATE "${PYARROW_INCLUDE_DIR}")
endforeach()

add_subdirectory(io)
add_subdirectory(nvtext)
add_subdirectory(strings)
