# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/thirdparty/get_cudf.cmake")

set(CUDF_STREAMING_MIN_VERSION
    "${CUDF_STREAMING_VERSION_MAJOR}.${CUDF_STREAMING_VERSION_MINOR}.${CUDF_STREAMING_VERSION_PATCH}"
)
find_and_configure_cudf(${CUDF_STREAMING_MIN_VERSION} cudf_streaming-exports)

if(cudf_REQUIRES_CUDA)
  rapids_cuda_init_architectures(CUDF_STREAMING)

  # Since we are building cudf as part of ourselves we need to enable the CUDA language in the
  # top-most scope
  enable_language(CUDA)

  # Since CUDF_STREAMING only enables CUDA optionally we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` calling
  if(DEFINED CMAKE_PROJECT_CUDF_STREAMING_INCLUDE)
    include("${CMAKE_PROJECT_CUDF_STREAMING_INCLUDE}")
  endif()
endif()
