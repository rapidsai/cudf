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

function(find_and_configure_thrust VERSION)
    # We only want to set `UPDATE_DISCONNECTED` while
    # the GIT tag hasn't moved from the last time we cloned
    set(cpm_thrust_disconnect_update "UPDATE_DISCONNECTED TRUE")
    set(CPM_THRUST_CURRENT_VERSION ${VERSION} CACHE STRING "version of thrust we checked out")
    if(NOT VERSION VERSION_EQUAL CPM_THRUST_CURRENT_VERSION)
        set(CPM_THRUST_CURRENT_VERSION ${VERSION} CACHE STRING "version of thrust we checked out" FORCE)
        set(cpm_thrust_disconnect_update "")
    endif()

    CPMAddPackage(NAME Thrust
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/NVIDIA/thrust.git
        GIT_TAG         ${VERSION}
        GIT_SHALLOW     TRUE
        ${cpm_thrust_disconnect_update}
        PATCH_COMMAND   patch --reject-file=- -p1 -N < ${CUDF_SOURCE_DIR}/cmake/thrust.patch || true
        )

    thrust_create_target(cudf::Thrust FROM_OPTIONS)
    set(THRUST_LIBRARY "cudf::Thrust" PARENT_SCOPE)

    include(GNUInstallDirs)
    install(DIRECTORY "${Thrust_SOURCE_DIR}/thrust"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/libcudf/Thrust/"
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.inl")
    install(DIRECTORY "${Thrust_SOURCE_DIR}/dependencies/cub/cub"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/libcudf/Thrust/dependencies/"
        FILES_MATCHING
            PATTERN "*.cuh")

    install(DIRECTORY "${Thrust_SOURCE_DIR}/thrust/cmake"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/libcudf/Thrust/thrust/")
    install(DIRECTORY "${Thrust_SOURCE_DIR}/dependencies/cub/cub/cmake"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/libcudf/Thrust/dependencies/cub/")

endfunction()

set(CUDF_MIN_VERSION_Thrust 1.12.0)

find_and_configure_thrust(${CUDF_MIN_VERSION_Thrust})
