# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
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
