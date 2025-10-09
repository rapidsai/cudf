# =============================================================================
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(pylibcudf)

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

project(
  pylibcudf
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(cudf "${RAPIDS_VERSION}" REQUIRED)

# an installed version of libcudf doesn't provide the dlpack headers so we need to download dlpack
# for the interop.pyx
include(rapids-cpm)
rapids_cpm_init()
include(../../cpp/cmake/thirdparty/get_dlpack.cmake)

include(rapids-cython-core)

rapids_cython_init()

add_subdirectory(pylibcudf)
