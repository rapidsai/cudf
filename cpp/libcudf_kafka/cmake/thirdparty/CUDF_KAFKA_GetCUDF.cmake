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
                        "BUILD_BENCHMARKS OFF")
    if(cudf_ADDED)
        set(cudf_ADDED TRUE PARENT_SCOPE)
    endif()
endfunction()

set(CUDA_KAFKA_MIN_VERSION_cudf "${CUDA_KAFKA_VERSION_MAJOR}.${CUDA_KAFKA_VERSION_MINOR}")
find_and_configure_cudf(${CUDA_KAFKA_MIN_VERSION_cudf})

if(cudf_ADDED)
    # Since we are building cudf as part of ourselves we need
    # to enable the CUDA language in the top-most scope
    enable_language(CUDA)
endif()
