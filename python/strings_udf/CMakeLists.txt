# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

set(strings_udf_version 22.10.00)

include(../../fetch_rapids.cmake)

include(rapids-cuda)

# TODO: I believe this needs to come before the `project` call, but this is unfortunate because CUDA
# is really only required in the cpp subdirectory so this is a bit out of place. I'm not sure if
# there's a better solution here, though.
rapids_cuda_init_architectures(STRINGS_UDF)

project(
  strings-udf-python
  VERSION ${strings_udf_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA
)

find_package(cudf ${strings_udf_version} REQUIRED)

add_subdirectory(cpp)

include(rapids-cython)
rapids_cython_init()

add_subdirectory(strings_udf/_lib)
