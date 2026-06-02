# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(cython_sources bloom_filter.pyx channel_metadata.pyx parquet.pyx partition.pyx table_chunk.pyx)
set(linked_libraries cudf_streaming::cudf_streaming rapidsmpf::rapidsmpf)

rapids_cython_create_modules(
  CXX ASSOCIATED_TARGETS cudf_streaming
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}"
  MODULE_PREFIX cudf_streaming_streaming_
)
