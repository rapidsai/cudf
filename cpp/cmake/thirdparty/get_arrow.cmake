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

function(find_and_configure_arrow VERSION BUILD_STATIC ENABLE_S3 ENABLE_ORC ENABLE_PYTHON ENABLE_PARQUET)

    if(BUILD_STATIC)
        if(TARGET arrow_static AND TARGET arrow_cuda_static)
            list(APPEND ARROW_LIBRARIES arrow_static)
            list(APPEND ARROW_LIBRARIES arrow_cuda_static)
            set(ARROW_FOUND TRUE PARENT_SCOPE)
            set(ARROW_LIBRARIES ${ARROW_LIBRARIES} PARENT_SCOPE)
            return()
        endif()
    else()
        if(TARGET arrow_shared AND TARGET arrow_cuda_shared)
            list(APPEND ARROW_LIBRARIES arrow_shared)
            list(APPEND ARROW_LIBRARIES arrow_cuda_shared)
            set(ARROW_FOUND TRUE PARENT_SCOPE)
            set(ARROW_LIBRARIES ${ARROW_LIBRARIES} PARENT_SCOPE)
            return()
        endif()
    endif()

    set(ARROW_BUILD_SHARED ON)
    set(ARROW_BUILD_STATIC OFF)
    set(CPMAddOrFindPackage CPMFindPackage)

    if(NOT ARROW_ARMV8_ARCH)
        set(ARROW_ARMV8_ARCH "armv8-a")
    endif()

    if(NOT ARROW_SIMD_LEVEL)
        set(ARROW_SIMD_LEVEL "NONE")
    endif()

    if(BUILD_STATIC)
        set(ARROW_BUILD_STATIC ON)
        set(ARROW_BUILD_SHARED OFF)
        # Turn off CPM using `find_package` so we always download
        # and make sure we get proper static library
        set(CPM_DOWNLOAD_ALL TRUE)
    endif()

    set(ARROW_PYTHON_OPTIONS "")
    if(ENABLE_PYTHON)
        list(APPEND ARROW_PYTHON_OPTIONS "ARROW_PYTHON ON")
        # Arrow's logic to build Boost from source is busted, so we have to get it from the system.
        list(APPEND ARROW_PYTHON_OPTIONS "BOOST_SOURCE SYSTEM")
        list(APPEND ARROW_PYTHON_OPTIONS "Thrift_SOURCE BUNDLED")
        list(APPEND ARROW_PYTHON_OPTIONS "ARROW_DEPENDENCY_SOURCE AUTO")
    endif()

    # Set this so Arrow correctly finds the CUDA toolkit when the build machine
    # does not have the CUDA driver installed. This must be an env var.
    set(ENV{CUDA_LIB_PATH} "${CUDAToolkit_LIBRARY_DIR}/stubs")

    rapids_cpm_find(Arrow ${VERSION}
        GLOBAL_TARGETS arrow_shared arrow_cuda_shared
        CPM_ARGS
        GIT_REPOSITORY  https://github.com/apache/arrow.git
        GIT_TAG         apache-arrow-${VERSION}
        GIT_SHALLOW     TRUE
        SOURCE_SUBDIR   cpp
        OPTIONS         "CMAKE_VERBOSE_MAKEFILE ON"
                        "CUDA_USE_STATIC_CUDA_RUNTIME ${CUDA_STATIC_RUNTIME}"
                        "ARROW_IPC ON"
                        "ARROW_CUDA ON"
                        "ARROW_DATASET ON"
                        "ARROW_WITH_BACKTRACE ON"
                        "ARROW_CXXFLAGS -w"
                        "ARROW_JEMALLOC OFF"
                        "ARROW_S3 ${ENABLE_S3}"
                        "ARROW_ORC ${ENABLE_ORC}"
                        # e.g. needed by blazingsql-io
                        "ARROW_PARQUET ${ENABLE_PARQUET}"
                        ${ARROW_PYTHON_OPTIONS}
                        # Arrow modifies CMake's GLOBAL RULE_LAUNCH_COMPILE unless this is off
                        "ARROW_USE_CCACHE OFF"
                        "ARROW_ARMV8_ARCH ${ARROW_ARMV8_ARCH}"
                        "ARROW_SIMD_LEVEL ${ARROW_SIMD_LEVEL}"
                        "ARROW_BUILD_STATIC ${ARROW_BUILD_STATIC}"
                        "ARROW_BUILD_SHARED ${ARROW_BUILD_SHARED}"
                        "ARROW_DEPENDENCY_USE_SHARED ${ARROW_BUILD_SHARED}"
                        "ARROW_BOOST_USE_SHARED ${ARROW_BUILD_SHARED}"
                        "ARROW_BROTLI_USE_SHARED ${ARROW_BUILD_SHARED}"
                        "ARROW_GFLAGS_USE_SHARED ${ARROW_BUILD_SHARED}"
                        "ARROW_GRPC_USE_SHARED ${ARROW_BUILD_SHARED}"
                        "ARROW_PROTOBUF_USE_SHARED ${ARROW_BUILD_SHARED}"
                        "ARROW_ZSTD_USE_SHARED ${ARROW_BUILD_SHARED}")

    set(ARROW_FOUND TRUE)
    set(ARROW_LIBRARIES "")

    # Arrow_ADDED: set if CPM downloaded Arrow from Github
    # Arrow_DIR:   set if CPM found Arrow on the system/conda/etc.
    if(Arrow_ADDED OR Arrow_DIR)
        if(BUILD_STATIC)
            list(APPEND ARROW_LIBRARIES arrow_static)
            list(APPEND ARROW_LIBRARIES arrow_cuda_static)
        else()
            list(APPEND ARROW_LIBRARIES arrow_shared)
            list(APPEND ARROW_LIBRARIES arrow_cuda_shared)
        endif()

        if(Arrow_DIR)
            # Set this to enable `find_package(ArrowCUDA)`
            set(ArrowCUDA_DIR "${Arrow_DIR}")
            find_package(Arrow REQUIRED QUIET)
            find_package(ArrowCUDA REQUIRED QUIET)
        elseif(Arrow_ADDED)
            # Copy these files so we can avoid adding paths in
            # Arrow_BINARY_DIR to target_include_directories.
            # That defeats ccache.
            file(INSTALL "${Arrow_BINARY_DIR}/src/arrow/util/config.h"
                 DESTINATION "${Arrow_SOURCE_DIR}/cpp/src/arrow/util")
            file(INSTALL "${Arrow_BINARY_DIR}/src/arrow/gpu/cuda_version.h"
                 DESTINATION "${Arrow_SOURCE_DIR}/cpp/src/arrow/gpu")
            if(ENABLE_PARQUET)
                file(INSTALL "${Arrow_BINARY_DIR}/src/parquet/parquet_version.h"
                     DESTINATION "${Arrow_SOURCE_DIR}/cpp/src/parquet")
            endif()
            ###
            # This shouldn't be necessary!
            #
            # Arrow populates INTERFACE_INCLUDE_DIRECTORIES for the `arrow_static`
            # and `arrow_shared` targets in FindArrow and FindArrowCUDA respectively,
            # so for static source-builds, we have to do it after-the-fact.
            #
            # This only works because we know exactly which components we're using.
            # Don't forget to update this list if we add more!
            ###
            foreach(ARROW_LIBRARY ${ARROW_LIBRARIES})
                target_include_directories(${ARROW_LIBRARY}
                    INTERFACE "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/src>"
                              "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/src/generated>"
                              "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/thirdparty/hadoop/include>"
                              "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/thirdparty/flatbuffers/include>"
                )
            endforeach()
        endif()
    else()
        set(ARROW_FOUND FALSE)
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
        ]=]
        )
        set(arrow_cuda_code_string
        [=[
        if (TARGET cudf::arrow_cuda_shared AND (NOT TARGET arrow_cuda_shared))
            add_library(arrow_cuda_shared ALIAS cudf::arrow_cuda_shared)
        endif()
        if (TARGET cudf::arrow_cuda_static AND (NOT TARGET arrow_cuda_static))
            add_library(arrow_cuda_static ALIAS cudf::arrow_cuda_static)
        endif()
        ]=]
        )

        rapids_export(BUILD Arrow
          VERSION ${VERSION}
          EXPORT_SET arrow_targets
          GLOBAL_TARGETS arrow_shared cud
          NAMESPACE cudf::
          FINAL_CODE_BLOCK arrow_code_string)

        rapids_export(BUILD ArrowCUDA
          VERSION ${VERSION}
          EXPORT_SET arrow_cuda_targets
          GLOBAL_TARGETS arrow_cuda_shared arrow_cuda_static
          NAMESPACE cudf::
          FINAL_CODE_BLOCK arrow_cuda_code_string)
    endif()
    # We generate the arrow-config and arrowcuda-config files
    # when we built arrow locally, so always do `find_dependency`
    rapids_export_package(BUILD Arrow cudf-exports)
    rapids_export_package(INSTALL Arrow cudf-exports)

    # We have to generate the find_dependency(ArrowCUDA) ourselves
    # since we need to specify ArrowCUDA_DIR to be where Arrow
    # was found, since Arrow packages ArrowCUDA.config in a non-standard
    # location
    rapids_export_package(BUILD ArrowCUDA cudf-exports)

    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD Arrow [=[${CMAKE_CURRENT_LIST_DIR}]=] cudf-exports)
    rapids_export_find_package_root(BUILD ArrowCUDA [=[${CMAKE_CURRENT_LIST_DIR}]=] cudf-exports)

    set(ARROW_FOUND "${ARROW_FOUND}" PARENT_SCOPE)
    set(ARROW_LIBRARIES "${ARROW_LIBRARIES}" PARENT_SCOPE)

endfunction()

set(CUDF_VERSION_Arrow 5.0.0)

find_and_configure_arrow(
    ${CUDF_VERSION_Arrow}
    ${CUDF_USE_ARROW_STATIC}
    ${CUDF_ENABLE_ARROW_S3}
    ${CUDF_ENABLE_ARROW_ORC}
    ${CUDF_ENABLE_ARROW_PYTHON}
    ${CUDF_ENABLE_ARROW_PARQUET}
)
