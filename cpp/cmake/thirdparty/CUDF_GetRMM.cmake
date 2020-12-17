#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

    if(ARGC GREATER 2 AND (ARGV1 AND ARGV2))
        add_subdirectory("${ARGV1}" "${ARGV2}" EXCLUDE_FROM_ALL)
        return()
    endif()

    CPMFindPackage(NAME rmm
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/rapidsai/rmm.git
        GIT_TAG         branch-${VERSION}
        GIT_SHALLOW     TRUE
    )
endfunction()

set(CUDF_MIN_VERSION_rmm "${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}")

find_and_configure_rmm(${CUDF_MIN_VERSION_rmm} "$ENV{RMM_HOME}" "$ENV{RMM_ROOT}")
