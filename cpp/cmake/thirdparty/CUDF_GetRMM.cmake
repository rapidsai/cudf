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

function(find_and_configure_rmm VERSION)

    # If building against a local RMM source repo without installing,
    # set the RMM_HOME and RMM_ROOT environment variables to the source
    # and binary dirs, i.e. the git repo root and build dir, respectively.
    if(ARGC GREATER 2 AND (ARGV1 AND ARGV2))
        set(BUILD_TESTS OFF)
        set(BUILD_BENCHMARKS OFF)
        add_subdirectory("${ARGV1}" "${ARGV2}" EXCLUDE_FROM_ALL)
        return()
    endif()

    # Alteratively, set `-DCPM_rmm_SOURCE=/path/to/rmm` in the CMake configure flags.

    CPMFindPackage(NAME rmm
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/rapidsai/rmm.git
        GIT_TAG         branch-${VERSION}
        GIT_SHALLOW     TRUE
        OPTIONS         "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF"
                        "CUDA_STATIC_RUNTIME ${CUDA_STATIC_RUNTIME}"
                        "CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}"
                        "DISABLE_DEPRECATION_WARNING ${DISABLE_DEPRECATION_WARNING}"
    )
endfunction()

set(CUDF_MIN_VERSION_rmm "${CUDF_VERSION_MAJOR}.${CUDF_VERSION_MINOR}")

find_and_configure_rmm(${CUDF_MIN_VERSION_rmm} "$ENV{RMM_HOME}" "$ENV{RMM_ROOT}")
