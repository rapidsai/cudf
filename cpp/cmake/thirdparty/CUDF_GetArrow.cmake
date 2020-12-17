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

set(CUDF_VERSION_Arrow 1.0.1)

function(find_and_configure_arrow VERSION BUILD_STATIC)

    set(ARROW_BUILD_SHARED ON)
    set(ARROW_BUILD_STATIC OFF)
    set(CPMAddOrFindPackage CPMFindPackage)

    if(BUILD_STATIC)
        set(ARROW_BUILD_STATIC ON)
        set(ARROW_BUILD_SHARED OFF)
        # Use CPMAddPackage if static linking
        set(CPMAddOrFindPackage CPMAddPackage)
    endif()

    cmake_language(CALL ${CPMAddOrFindPackage}
        NAME            Arrow
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/apache/arrow.git
        GIT_TAG         apache-arrow-${VERSION}
        GIT_SHALLOW     TRUE
        SOURCE_SUBDIR   cpp
        OPTIONS         "CMAKE_VERBOSE_MAKEFILE         ON"
                        "ARROW_IPC                      ON"
                        "ARROW_CUDA                     ON"
                        "ARROW_DATASET                  ON"
                        "ARROW_WITH_BACKTRACE           ON"
                        "ARROW_JEMALLOC                 OFF"
                        "ARROW_SIMD_LEVEL               NONE"
                        "ARROW_BUILD_STATIC             ${ARROW_BUILD_STATIC}"
                        "ARROW_BUILD_SHARED             ${ARROW_BUILD_SHARED}"
                        "ARROW_DEPENDENCY_USE_SHARED    ${ARROW_BUILD_SHARED}"
                        "ARROW_BOOST_USE_SHARED         ${ARROW_BUILD_SHARED}"
                        "ARROW_BROTLI_USE_SHARED        ${ARROW_BUILD_SHARED}"
                        "ARROW_GFLAGS_USE_SHARED        ${ARROW_BUILD_SHARED}"
                        "ARROW_GRPC_USE_SHARED          ${ARROW_BUILD_SHARED}"
                        "ARROW_PROTOBUF_USE_SHARED      ${ARROW_BUILD_SHARED}"
                        "ARROW_ZSTD_USE_SHARED          ${ARROW_BUILD_SHARED}")

    set(Arrow_DIR "${Arrow_DIR}" PARENT_SCOPE)
    set(Arrow_ADDED "${Arrow_ADDED}" PARENT_SCOPE)
    set(Arrow_BINARY_DIR "${Arrow_BINARY_DIR}" PARENT_SCOPE)

    set(ARROW_FOUND TRUE PARENT_SCOPE)
    set(ARROW_LIBRARIES "" PARENT_SCOPE)

    # This will be set if CPM had to download Arrow from Github
    if(Arrow_ADDED)
        include(${Arrow_BINARY_DIR}/src/arrow/ArrowConfig.cmake)
        if(BUILD_STATIC)
            list(APPEND ARROW_LIBRARIES arrow_static PARENT_SCOPE)
        else()
            list(APPEND ARROW_LIBRARIES arrow_shared PARENT_SCOPE)
        endif()
    # This will be set if CPM found Arrow on the system (or in conda, etc.)
    elseif(Arrow_DIR)
        # Set this for find_package(ArrowCUDA) to work
        set(ArrowCUDA_DIR ${Arrow_DIR})
        find_package(Arrow REQUIRED QUIET)
        find_package(ArrowCUDA REQUIRED QUIET)
        if(BUILD_STATIC)
            list(APPEND ARROW_LIBRARIES ${ARROW_STATIC_LIB} PARENT_SCOPE)
            list(APPEND ARROW_LIBRARIES ${ARROW_CUDA_STATIC_LIB} PARENT_SCOPE)
        else()
            list(APPEND ARROW_LIBRARIES ${ARROW_SHARED_LIB} PARENT_SCOPE)
            list(APPEND ARROW_LIBRARIES ${ARROW_CUDA_SHARED_LIB} PARENT_SCOPE)
        endif()
    else()
        set(ARROW_FOUND FALSE PARENT_SCOPE)
        message(FATAL_ERROR "Arrow library not found or downloaded.")
    endif()

endfunction()

find_and_configure_arrow(${CUDF_VERSION_Arrow} ${ARROW_STATIC_LIB})
