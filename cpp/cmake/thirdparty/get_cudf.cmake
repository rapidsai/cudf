# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds cudf and sets any additional necessary environment variables. Arguments:
# VERSION      - The cudf version to find EXPORT_SET   - The CMake export set name PROJECT_NAME -
# The project name in uppercase
function(find_and_configure_cudf VERSION EXPORT_SET PROJECT_NAME)
  rapids_cmake_parse_version(MAJOR_MINOR ${VERSION} major_minor)
  rapids_cpm_find(
    cudf ${VERSION}
    BUILD_EXPORT_SET ${EXPORT_SET}
    INSTALL_EXPORT_SET ${EXPORT_SET}
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
