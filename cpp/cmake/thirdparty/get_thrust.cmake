# =============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

# This function finds thrust and sets any additional necessary environment variables.
function(find_and_configure_thrust VERSION)
  # We only want to set `UPDATE_DISCONNECTED` while the GIT tag hasn't moved from the last time we
  # cloned
  set(cpm_thrust_disconnect_update "UPDATE_DISCONNECTED TRUE")
  set(CPM_THRUST_CURRENT_VERSION
      ${VERSION}
      CACHE STRING "version of thrust we checked out"
  )
  if(NOT VERSION VERSION_EQUAL CPM_THRUST_CURRENT_VERSION)
    set(CPM_THRUST_CURRENT_VERSION
        ${VERSION}
        CACHE STRING "version of thrust we checked out" FORCE
    )
    set(cpm_thrust_disconnect_update "")
  endif()

  # We currently require cuDF to always build with a custom version of thrust. This is needed so
  # that build times of of cudf are kept reasonable, without this CI builds of cudf will be killed
  # as some source file can take over 45 minutes to build
  #
  set(CPM_DOWNLOAD_ALL TRUE)
  rapids_cpm_find(
    Thrust ${VERSION}
    BUILD_EXPORT_SET cudf-exports
    INSTALL_EXPORT_SET cudf-exports
    CPM_ARGS
    GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
    GIT_TAG ${VERSION}
    GIT_SHALLOW TRUE ${cpm_thrust_disconnect_update}
    PATCH_COMMAND patch --reject-file=- -p1 -N < ${CUDF_SOURCE_DIR}/cmake/thrust.patch || true
    OPTIONS "THRUST_INSTALL TRUE"
  )

  if(NOT TARGET cudf::Thrust)
    thrust_create_target(cudf::Thrust FROM_OPTIONS)
  endif()

  if(Thrust_SOURCE_DIR) # only install thrust when we have an in-source version
    include(GNUInstallDirs)
    install(
      DIRECTORY "${Thrust_SOURCE_DIR}/thrust"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/libcudf/Thrust/"
      FILES_MATCHING
      REGEX "\\.(h|inl)$"
    )
    install(
      DIRECTORY "${Thrust_SOURCE_DIR}/dependencies/cub/cub"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/libcudf/Thrust/dependencies/"
      FILES_MATCHING
      PATTERN "*.cuh"
    )

    install(DIRECTORY "${Thrust_SOURCE_DIR}/thrust/cmake"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/libcudf/Thrust/thrust/"
    )
    install(DIRECTORY "${Thrust_SOURCE_DIR}/dependencies/cub/cub/cmake"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/libcudf/Thrust/dependencies/cub/"
    )

    # Store where CMake can find our custom Thrust install
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(
      INSTALL Thrust [=[${CMAKE_CURRENT_LIST_DIR}/../../../include/libcudf/Thrust/]=] cudf-exports
    )
  endif()
endfunction()

set(CUDF_MIN_VERSION_Thrust 1.17.0)

find_and_configure_thrust(${CUDF_MIN_VERSION_Thrust})
