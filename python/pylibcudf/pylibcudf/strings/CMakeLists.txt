# =============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
    attributes.pyx
    capitalize.pyx
    case.pyx
    char_types.pyx
    contains.pyx
    combine.pyx
    extract.pyx
    find.pyx
    find_multiple.pyx
    findall.pyx
    padding.pyx
    regex_flags.pyx
    regex_program.pyx
    repeat.pyx
    replace.pyx
    replace_re.pyx
    reverse.pyx
    side_type.pyx
    slice.pyx
    strip.pyx
    translate.pyx
    wrap.pyx
)

set(linked_libraries cudf::cudf)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX pylibcudf_strings_ ASSOCIATED_TARGETS cudf
)

add_subdirectory(convert)
add_subdirectory(split)
