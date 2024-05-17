# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
    binaryop.pyx
    column.pyx
    concatenate.pyx
    copying.pyx
    filling.pyx
    gpumemoryview.pyx
    groupby.pyx
    interop.pyx
    join.pyx
    lists.pyx
    merge.pyx
    reduce.pyx
    replace.pyx
    rolling.pyx
    scalar.pyx
    search.pyx
    stream_compaction.pyx
    sorting.pyx
    table.pyx
    types.pyx
    unary.pyx
    utils.pyx
)
set(linked_libraries cudf::cudf)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX pylibcudf_ ASSOCIATED_TARGETS cudf
)
link_to_pyarrow_headers(pylibcudf_interop)

add_subdirectory(libcudf)
add_subdirectory(strings)
