#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_cudf VERSION)
    rapids_cmake_parse_version(MAJOR_MINOR ${VERSION} major_minor)
    rapids_cpm_find(cudf ${VERSION}
        BUILD_EXPORT_SET cudf_kafka-exports
        INSTALL_EXPORT_SET cudf_kafka-exports
        CPM_ARGS
            GIT_REPOSITORY  https://github.com/rapidsai/cudf.git
            GIT_TAG         branch-${major_minor}
            GIT_SHALLOW     TRUE
            SOURCE_SUBDIR   cpp
            OPTIONS         "BUILD_TESTS OFF"
                            "BUILD_BENCHMARKS OFF")
    # If after loading cudf we now have the CMAKE_CUDA_COMPILER
    # variable we know that we need to re-enable the cuda language
    if(CMAKE_CUDA_COMPILER)
        set(cudf_REQUIRES_CUDA TRUE PARENT_SCOPE)
    endif()
endfunction()

set(CUDA_KAFKA_MIN_VERSION_cudf "${CUDA_KAFKA_VERSION_MAJOR}.${CUDA_KAFKA_VERSION_MINOR}.${CUDA_KAFKA_VERSION_PATCH}")
find_and_configure_cudf(${CUDA_KAFKA_MIN_VERSION_cudf})

if(cudf_REQUIRES_CUDA)
    rapids_cuda_init_architectures(CUDA_KAFKA)

    # Since we are building cudf as part of ourselves we need
    # to enable the CUDA language in the top-most scope
    enable_language(CUDA)

    # Since CUDA_KAFKA only enables CUDA optionally we need to manually include the file that
    # rapids_cuda_init_architectures relies on `project` calling
    if(DEFINED CMAKE_PROJECT_CUDA_KAFKA_INCLUDE)
        include("${CMAKE_PROJECT_CUDA_KAFKA_INCLUDE}")
    endif()
endif()
