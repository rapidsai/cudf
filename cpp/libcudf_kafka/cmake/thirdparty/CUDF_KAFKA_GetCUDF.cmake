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

function(cudfkafka_save_if_enabled var)
    if(CUDF_KAFKA_${var})
        unset(${var} PARENT_SCOPE)
        unset(${var} CACHE)
    endif()
endfunction()

function(cudfkafka_restore_if_enabled var)
    if(CUDF_KAFKA_${var})
        set(${var} ON CACHE INTERNAL "" FORCE)
    endif()
endfunction()

function(find_and_configure_cudf VERSION)
    cudfkafka_save_if_enabled(BUILD_TESTS)
    cudfkafka_save_if_enabled(BUILD_BENCHMARKS)
    CPMFindPackage(NAME cudf
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/rapidsai/cudf.git
        GIT_TAG         branch-${VERSION}
        GIT_SHALLOW     TRUE
        SOURCE_SUBDIR   cpp
        OPTIONS         "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF")
    cudfkafka_restore_if_enabled(BUILD_TESTS)
    cudfkafka_restore_if_enabled(BUILD_BENCHMARKS)
endfunction()

set(CUDF_KAFKA_MIN_VERSION_cudf 0.19)
find_and_configure_cudf(${CUDF_KAFKA_MIN_VERSION_cudf})
