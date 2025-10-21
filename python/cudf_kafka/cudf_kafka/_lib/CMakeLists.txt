# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

set(cython_sources kafka.pyx)
set(linked_libraries cudf_kafka::cudf_kafka)

rapids_cython_create_modules(
  CXX ASSOCIATED_TARGETS cudf_kafka
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}"
)
