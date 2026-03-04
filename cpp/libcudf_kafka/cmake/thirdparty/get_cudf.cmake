# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds cudf and sets any additional necessary environment variables.
function(find_and_configure_cudf VERSION)
  rapids_cmake_parse_version(MAJOR_MINOR ${VERSION} major_minor)
  rapids_cpm_find(
    cudf ${VERSION}
    BUILD_EXPORT_SET cudf_kafka-exports
    INSTALL_EXPORT_SET cudf_kafka-exports
    CPM_ARGS
    GIT_REPOSITORY https://github.com/rapidsai/cudf.git
    GIT_TAG "${RAPIDS_BRANCH}"
    GIT_SHALLOW TRUE SOURCE_SUBDIR cpp
    OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF"
  )
  # If after loading cudf we now have the CMAKE_CUDA_COMPILER variable we know that we need to
  # re-enable the cuda language
  if(CMAKE_CUDA_COMPILER)
    set(cudf_REQUIRES_CUDA
        TRUE
        PARENT_SCOPE
    )
  endif()
endfunction()

set(CUDF_KAFKA_MIN_VERSION
    "${CUDF_KAFKA_VERSION_MAJOR}.${CUDF_KAFKA_VERSION_MINOR}.${CUDF_KAFKA_VERSION_PATCH}"
)
find_and_configure_cudf(${CUDF_KAFKA_MIN_VERSION})

if(cudf_REQUIRES_CUDA)
  rapids_cuda_init_architectures(CUDF_KAFKA)

  # Since we are building cudf as part of ourselves we need to enable the CUDA language in the
  # top-most scope
  enable_language(CUDA)

  # Since CUDF_KAFKA only enables CUDA optionally we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` calling
  if(DEFINED CMAKE_PROJECT_CUDF_KAFKA_INCLUDE)
    include("${CMAKE_PROJECT_CUDF_KAFKA_INCLUDE}")
  endif()
endif()
