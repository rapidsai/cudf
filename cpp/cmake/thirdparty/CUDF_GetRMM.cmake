#=============================================================================
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

function(cudf_save_if_enabled var)
    if(CUDF_${var})
        unset(${var} PARENT_SCOPE)
        unset(${var} CACHE)
    endif()
endfunction()

function(cudf_restore_if_enabled var)
    if(CUDF_${var})
        set(${var} ON CACHE INTERNAL "" FORCE)
    endif()
endfunction()

function(find_and_configure_rmm VERSION)

    if(TARGET rmm::rmm)
        return()
    endif()

    # Consumers have two options for local source builds:
    # 1. Pass `-D CPM_rmm_SOURCE=/path/to/rmm` to build a local RMM source tree
    # 2. Pass `-D CMAKE_PREFIX_PATH=/path/to/rmm/build` to use an existing local
    #    RMM build directory as the install location for find_package(rmm)
    cudf_save_if_enabled(BUILD_TESTS)
    cudf_save_if_enabled(BUILD_BENCHMARKS)

    CPMFindPackage(NAME rmm
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/rapidsai/rmm.git
        GIT_TAG         branch-${VERSION}
        GIT_SHALLOW     TRUE
        OPTIONS         "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF"
                        "CUDA_STATIC_RUNTIME ${CUDA_STATIC_RUNTIME}"
                        "DISABLE_DEPRECATION_WARNING ${DISABLE_DEPRECATION_WARNING}"
    )
    cudf_restore_if_enabled(BUILD_TESTS)
    cudf_restore_if_enabled(BUILD_BENCHMARKS)

    # Make sure consumers of cudf can also see rmm::rmm
    fix_cmake_global_defaults(rmm::rmm)
endfunction()

set(CUDF_MIN_VERSION_rmm "${CUDF_VERSION_MAJOR}.${CUDF_VERSION_MINOR}")

find_and_configure_rmm(${CUDF_MIN_VERSION_rmm})
