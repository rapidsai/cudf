# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(cython_sources partition.pyx)
set(linked_libraries cudf_streaming::cudf_streaming)

rapids_cython_create_modules(
  CXX ASSOCIATED_TARGETS cudf_streaming
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX cudf_streaming_integrations_
)

foreach(target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
endforeach()
