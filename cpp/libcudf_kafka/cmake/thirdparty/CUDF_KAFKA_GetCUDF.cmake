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
    CPMFindPackage(NAME cudf
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/rapidsai/cudf.git
        GIT_TAG         branch-${VERSION}
        GIT_SHALLOW     TRUE
        SOURCE_SUBDIR   cpp
        OPTIONS         "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF"
                        "USE_NVTX ${USE_NVTX}"
                        "ARROW_STATIC_LIB ${ARROW_STATIC_LIB}"
                        "JITIFY_USE_CACHE ${JITIFY_USE_CACHE}"
                        "CUDA_STATIC_RUNTIME ${CUDA_STATIC_RUNTIME}"
                        "PER_THREAD_DEFAULT_STREAM ${PER_THREAD_DEFAULT_STREAM}"
                        "DISABLE_DEPRECATION_WARNING ${DISABLE_DEPRECATION_WARNING}")

    if(NOT cudf_BINARY_DIR IN_LIST CMAKE_PREFIX_PATH)
        list(APPEND CMAKE_PREFIX_PATH "${cudf_BINARY_DIR}")
        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
    endif()
endfunction()

set(CUDF_KAFKA_MIN_VERSION_cudf 0.19)

find_and_configure_cudf(${CUDF_KAFKA_MIN_VERSION_cudf})
