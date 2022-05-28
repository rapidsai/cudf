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
    gpuarrow.pyx
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
  LINKED_LIBRARIES "${linked_libraries}"
)

# TODO: Finding NumPy requires finding Development as well. Once this is fixed in CMake (no date
# yet) we can remove the extra component spec.
find_package(Python REQUIRED COMPONENTS Development NumPy)
set(targets_using_numpy gpuarrow interop avro csv orc json parquet)
foreach(target IN LISTS targets_using_numpy)
  target_include_directories(${target} PRIVATE "${Python_NumPy_INCLUDE_DIRS}")
endforeach()

# PyArrow relies on the C++ Arrow library already being installed, so we can just find the C++
# library directly and link to the same one. We rely on libcudf's exports to provide the
# arrow_shared_lib and arrow_cuda_shared_lib libraries. That just leaves us to find the ArrowPython
# library on our own.
find_library(arrow_python_shared_library arrow_python REQUIRED)
target_link_libraries(gpuarrow ${arrow_python_shared_library})

foreach(cython_module IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  set_target_properties(${cython_module} PROPERTIES INSTALL_RPATH "\$ORIGIN;\$ORIGIN/cpp")
endforeach()

add_subdirectory(io)
add_subdirectory(nvtext)
add_subdirectory(strings)
