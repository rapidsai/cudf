# =============================================================================
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

# Finding arrow is far more complex than it should be, and as a result we violate multiple linting
# rules aiming to limit complexity. Since all our other CMake scripts conform to expectations
# without undue difficulty, disabling those rules for just this function is our best approach for
# now. The spacing between this comment, the cmake-lint directives, and the function docstring is
# necessary to prevent cmake-format from trying to combine the lines.

# cmake-lint: disable=R0912,R0913,R0915

include_guard(GLOBAL)

# This function finds arrow and sets any additional necessary environment variables.
function(find_and_configure_arrow VERSION BUILD_STATIC)

  if(BUILD_STATIC)
    if(TARGET arrow_static)
      set(ARROW_FOUND
          TRUE
          PARENT_SCOPE
      )
      set(ARROW_LIBRARIES
          arrow_static
          PARENT_SCOPE
      )
      return()
    endif()
  else()
    if(TARGET arrow_shared)
      set(ARROW_FOUND
          TRUE
          PARENT_SCOPE
      )
      set(ARROW_LIBRARIES
          arrow_shared
          PARENT_SCOPE
      )
      return()
    endif()
  endif()

  if(NOT ARROW_SIMD_LEVEL)
    set(ARROW_SIMD_LEVEL "NONE")
  endif()

  if(BUILD_STATIC)
    set(ARROW_BUILD_STATIC ON)
    set(ARROW_BUILD_SHARED OFF)
    set(ARROW_DEPENDENCY_USE_SHARED ON)
    # Turn off CPM using `find_package` so we always download and make sure we get proper static
    # library.
    set(CPM_DOWNLOAD_Arrow TRUE)
  else()
    set(ARROW_BUILD_SHARED ON)
    set(ARROW_BUILD_STATIC OFF)
    set(ARROW_DEPENDENCY_USE_SHARED ON)
  endif()

  rapids_cpm_find(
    Arrow ${VERSION}
    GLOBAL_TARGETS arrow_shared arrow_static
    CPM_ARGS
    GIT_REPOSITORY https://github.com/apache/arrow.git
    GIT_TAG apache-arrow-${VERSION}
    GIT_SHALLOW TRUE SOURCE_SUBDIR cpp
    OPTIONS "CMAKE_VERBOSE_MAKEFILE ON"
            "ARROW_ACERO ON"
            "ARROW_IPC ON"
            "ARROW_DATASET ON"
            "ARROW_WITH_BACKTRACE ON"
            "ARROW_CXXFLAGS -w"
            "ARROW_JEMALLOC OFF"
            "ARROW_S3 OFF"
            "ARROW_ORC OFF"
            "ARROW_PARQUET OFF"
            "ARROW_FILESYSTEM OFF"
            "ARROW_PYTHON OFF"
            # Arrow modifies CMake's GLOBAL RULE_LAUNCH_COMPILE unless this is off
            "ARROW_USE_CCACHE OFF"
            "ARROW_ARMV8_ARCH ${ARROW_ARMV8_ARCH}"
            "ARROW_SIMD_LEVEL ${ARROW_SIMD_LEVEL}"
            "ARROW_BUILD_STATIC ${ARROW_BUILD_STATIC}"
            "ARROW_BUILD_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_POSITION_INDEPENDENT_CODE ON"
            "ARROW_DEPENDENCY_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_BOOST_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_BROTLI_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_GFLAGS_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_GRPC_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_PROTOBUF_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_ZSTD_USE_SHARED ${ARROW_BUILD_SHARED}"
            "xsimd_SOURCE AUTO"
  )

  set(ARROW_FOUND
      TRUE
      PARENT_SCOPE
  )

  if(BUILD_STATIC)
    set(ARROW_LIBRARIES arrow_static)
  else()
    set(ARROW_LIBRARIES arrow_shared)
  endif()

  # Arrow_ADDED: set if CPM downloaded Arrow from Github
  if(Arrow_ADDED)
    # Copy these files so we can avoid adding paths in Arrow_BINARY_DIR to
    # target_include_directories. That defeats ccache.
    file(INSTALL "${Arrow_BINARY_DIR}/src/arrow/util/config.h"
         DESTINATION "${Arrow_SOURCE_DIR}/cpp/src/arrow/util"
    )
    # Arrow populates INTERFACE_INCLUDE_DIRECTORIES for the `arrow_static` and `arrow_shared`
    # targets in FindArrow, so for static source-builds, we have to do it after-the-fact.
    #
    # This only works because we know exactly which components we're using. Don't forget to update
    # this list if we add more!
    #
    foreach(ARROW_LIBRARY ${ARROW_LIBRARIES})
      target_include_directories(
        ${ARROW_LIBRARY}
        INTERFACE "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/src>"
                  "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/src/generated>"
                  "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/thirdparty/hadoop/include>"
                  "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/thirdparty/flatbuffers/include>"
      )
    endforeach()
  elseif(NOT Arrow_DIR)
    set(ARROW_FOUND
        FALSE
        PARENT_SCOPE
    )
    message(FATAL_ERROR "CUDF: Arrow library not found or downloaded.")
  endif()

  if(Arrow_ADDED)

    set(arrow_code_string
        [=[
          if (TARGET cudf::arrow_shared AND (NOT TARGET arrow_shared))
              add_library(arrow_shared ALIAS cudf::arrow_shared)
          endif()
          if (TARGET cudf::arrow_static AND (NOT TARGET arrow_static))
              add_library(arrow_static ALIAS cudf::arrow_static)
          endif()
          if (NOT TARGET arrow::flatbuffers)
            add_library(arrow::flatbuffers INTERFACE IMPORTED)
          endif()
          if (NOT TARGET arrow::hadoop)
            add_library(arrow::hadoop INTERFACE IMPORTED)
          endif()
        ]=]
    )
    if(NOT TARGET xsimd)
      string(
        APPEND
        arrow_code_string
        "
          if(NOT TARGET arrow::xsimd)
            add_library(arrow::xsimd INTERFACE IMPORTED)
            target_include_directories(arrow::xsimd INTERFACE \"${Arrow_BINARY_DIR}/xsimd_ep/src/xsimd_ep-install/include\")
          endif()
        "
      )
    endif()
    rapids_cmake_install_lib_dir(lib_dir)
    if(TARGET arrow_static)
      get_target_property(interface_libs arrow_static INTERFACE_LINK_LIBRARIES)
      # The `arrow_static` library is leaking a dependency on the object libraries it was built with
      # we need to remove this from the interface, since keeping them around would cause duplicate
      # symbols and CMake export errors
      if(interface_libs MATCHES "arrow_array" AND interface_libs MATCHES "arrow_compute")
        string(REPLACE "BUILD_INTERFACE:" "BUILD_LOCAL_INTERFACE:" interface_libs
                       "${interface_libs}"
        )
        set_target_properties(arrow_static PROPERTIES INTERFACE_LINK_LIBRARIES "${interface_libs}")
        get_target_property(interface_libs arrow_static INTERFACE_LINK_LIBRARIES)
      endif()
    endif()
    rapids_export(
      BUILD Arrow
      VERSION ${VERSION}
      EXPORT_SET arrow_targets
      GLOBAL_TARGETS arrow_shared arrow_static
      NAMESPACE cudf::
      FINAL_CODE_BLOCK arrow_code_string
    )

  endif()
  # We generate the arrow-configfiles when we built arrow locally, so always do `find_dependency`
  rapids_export_package(BUILD Arrow cudf-exports)
  rapids_export_package(INSTALL Arrow cudf-exports)

  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(
    BUILD Arrow [=[${CMAKE_CURRENT_LIST_DIR}]=] EXPORT_SET cudf-exports
  )
  set(ARROW_LIBRARIES
      "${ARROW_LIBRARIES}"
      PARENT_SCOPE
  )
endfunction()

if(NOT DEFINED CUDF_VERSION_Arrow)
  set(CUDF_VERSION_Arrow
      # This version must be kept in sync with the libarrow version pinned for builds in
      # dependencies.yaml.
      16.1.0
      CACHE STRING "The version of Arrow to find (or build)"
  )
endif()

find_and_configure_arrow(${CUDF_VERSION_Arrow} ${CUDF_USE_ARROW_STATIC})
