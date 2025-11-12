# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources
    convert_booleans.pyx convert_datetime.pyx convert_durations.pyx convert_fixed_point.pyx
    convert_floats.pyx convert_integers.pyx convert_ipv4.pyx convert_lists.pyx convert_urls.pyx
)

set(linked_libraries cudf::cudf)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX pylibcudf_strings_ ASSOCIATED_TARGETS cudf
)
