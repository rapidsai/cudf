# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include(${CMAKE_CURRENT_LIST_DIR}/versions.cmake)

set(CPM_DOWNLOAD_VERSION v0.38.5)
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/${CPM_DOWNLOAD_VERSION}/get_cpm.cmake
  ${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake
)
include(${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake)

# find or build it via CPM
CPMFindPackage(
  NAME cudf
  FIND_PACKAGE_ARGUMENTS "PATHS ${cudf_ROOT} ${cudf_ROOT}/latest" GIT_REPOSITORY
                         https://github.com/rapidsai/cudf
  GIT_TAG ${CUDF_TAG}
  GIT_SHALLOW
    TRUE
    SOURCE_SUBDIR
    cpp
)
