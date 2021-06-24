#=============================================================================
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Use GPU_ARCHS if it is defined
if(DEFINED GPU_ARCHS)
  set(CMAKE_CUDA_ARCHITECTURES "${GPU_ARCHS}")
endif()

# If `CMAKE_CUDA_ARCHITECTURES` is not defined, build for all supported architectures. If
# `CMAKE_CUDA_ARCHITECTURES` is set to an empty string (""), build for only the current
# architecture. If `CMAKE_CUDA_ARCHITECTURES` is specified by the user, use user setting.

# This needs to be run before enabling the CUDA language due to the default initialization behavior
# of `CMAKE_CUDA_ARCHITECTURES`, https://gitlab.kitware.com/cmake/cmake/-/issues/21302
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "ALL")
  set(CUDF_JNI_BUILD_FOR_ALL_ARCHS TRUE)
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CUDF_JNI_BUILD_FOR_DETECTED_ARCHS TRUE)
endif()

project(CUDF_JNI VERSION 21.08.00 LANGUAGES C CXX)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
option(USE_GDS "Build with GPUDirect Storage (GDS)/cuFile support" OFF)
option(CUDF_JNI_ARROW_STATIC "Statically link Arrow" ON)

message(VERBOSE "CUDF_JNI: Build with NVTX support: ${USE_NVTX}")
message(VERBOSE "CUDF_JNI: Configure CMake to build tests: ${BUILD_TESTS}")
message(VERBOSE "CUDF_JNI: Build with per-thread default stream: ${PER_THREAD_DEFAULT_STREAM}")
message(VERBOSE "CUDF_JNI: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")
message(VERBOSE "CUDF_JNI: Build with GPUDirect Storage support: ${USE_GDS}")
message(VERBOSE "CUDF_JNI: Build with static Arrow library: ${CUDF_JNI_ARROW_STATIC}")

set(CUDF_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../../../cpp")
if (DEFINED ENV{CUDF_CPP_BUILD_DIR}) 
  set(CUDF_CPP_BUILD_DIR "$ENV{CUDF_CPP_BUILD_DIR}")
else()
  set(CUDF_CPP_BUILD_DIR "${CUDF_SOURCE_DIR}/build")
endif()

set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/"
    "${CUDF_SOURCE_DIR}/cmake/Modules/"
    ${CMAKE_MODULE_PATH})

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CUDF_CXX_FLAGS "")
set(CUDF_CUDA_FLAGS "")
set(CUDF_CXX_DEFINITIONS "")
set(CUDF_CUDA_DEFINITIONS "")

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
include(ConfigureCUDA)

# Disable NVTX if necessary
if(NOT USE_NVTX)
    target_compile_definitions(cudfjni PUBLIC NVTX_DISABLE)
endif()

if(PER_THREAD_DEFAULT_STREAM)
    message(STATUS "Using per-thread default stream")
    add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif(PER_THREAD_DEFAULT_STREAM)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

###################################################################################################
# - cudart options --------------------------------------------------------------------------------
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking

if(CUDA_STATIC_RUNTIME)
    message(STATUS "Enabling static linking of cudart")
    set(CUDART_LIBRARY "cudart_static")
else()
    set(CUDART_LIBRARY "cudart")
endif(CUDA_STATIC_RUNTIME)

###################################################################################################
# - Thrust/CUB/libcudacxx ------------------------------------------------------------------------------------
find_path(THRUST_INCLUDE "thrust"
    HINTS "$ENV{CUDF_ROOT}/_deps/thrust-src"
          "${CUDF_CPP_BUILD_DIR}/_deps/thrust-src"
          "$ENV{CONDA_PREFIX}/include")

message(STATUS "THRUST: THRUST_INCLUDE set to ${THRUST_INCLUDE}")

find_path(CUB_INCLUDE "cub"
    HINTS "$ENV{CUDF_ROOT}/_deps/thrust-src"
          "${CUDF_CPP_BUILD_DIR}/_deps/thrust-src"
          "$ENV{CONDA_PREFIX}/include")

message(STATUS "CUB: CUB_INCLUDE set to ${CUB_INCLUDE}")

find_path(LIBCUDACXX_INCLUDE "cuda"
    HINTS "$ENV{CUDF_ROOT}/_deps/libcudacxx-src/include"
          "${CUDF_CPP_BUILD_DIR}/_deps/libcudacxx-src/include")

message(STATUS "LIBCUDACXX: LIBCUDACXX_INCLUDE set to ${LIBCUDACXX_INCLUDE}")

find_path(SPDLOG_INCLUDE "spdlog"
    HINTS "${CUDF_CPP_BUILD_DIR}/_deps/spdlog-src/include"
          "$ENV{RMM_ROOT}/_deps/spdlog-src/include"
          "$ENV{RMM_ROOT}/include"
          "$ENV{CONDA_PREFIX}/include")

message(STATUS "SPDLOG: SPDLOG_INCLUDE set to ${SPDLOG_INCLUDE}")
###################################################################################################
# - CUDF ------------------------------------------------------------------------------------------

set(CUDF_INCLUDE "${PROJECT_SOURCE_DIR}/../../../../cpp/include"
                 "${PROJECT_SOURCE_DIR}/../../../../cpp/src/")

set(CUDF_LIB_HINTS
    HINTS "$ENV{CUDF_ROOT}"
          "$ENV{CUDF_ROOT}/lib"
          "$ENV{CONDA_PREFIX}/lib"
          "${CUDF_CPP_BUILD_DIR}")

find_library(CUDF_LIB "cudf" REQUIRED HINTS ${CUDF_LIB_HINTS})

###################################################################################################
# - RMM -------------------------------------------------------------------------------------------

find_path(RMM_INCLUDE "rmm"
          HINTS "${CUDF_CPP_BUILD_DIR}/_deps/rmm-src/include"
                "$ENV{RMM_ROOT}/include"
                "$ENV{RMM_HOME}/include"
                "$ENV{CONDA_PREFIX}/include/rmm"
                "$ENV{CONDA_PREFIX}/include")

message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

###################################################################################################
# - ARROW -----------------------------------------------------------------------------------------

find_path(ARROW_INCLUDE "arrow"
          HINTS "$ENV{ARROW_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include"
                "${CUDF_CPP_BUILD_DIR}/_deps/arrow-src/cpp/src")

message(STATUS "ARROW: ARROW_INCLUDE set to ${ARROW_INCLUDE}")

if(CUDF_JNI_ARROW_STATIC)
  # Find static version of Arrow lib
  set(CUDF_JNI_ARROW_LIBNAME "libarrow.a")
else()
  set(CUDF_JNI_ARROW_LIBNAME "arrow")
endif(CUDF_JNI_ARROW_STATIC)

find_library(ARROW_LIBRARY ${CUDF_JNI_ARROW_LIBNAME} REQUIRED
  HINTS "$ENV{ARROW_ROOT}/lib"
        "$ENV{CONDA_PREFIX}/lib"
        "${CUDF_CPP_BUILD_DIR}/_deps/arrow-build/release")

if(NOT ARROW_LIBRARY)
  if(CUDF_JNI_ARROW_STATIC)
    message(FATAL_ERROR "Arrow static library not found. Was libcudf built with CUDF_USE_ARROW_STATIC=ON?")
  else()
    message(FATAL_ERROR "Arrow dynamic library not found.")
  endif(CUDF_JNI_ARROW_STATIC)
else()
  message(STATUS "ARROW: ARROW_LIBRARY set to ${ARROW_LIBRARY}")
endif(NOT ARROW_LIBRARY)

###################################################################################################
# - find JNI -------------------------------------------------------------------------------------
find_package(JNI REQUIRED)
if(JNI_FOUND)
    message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "JDK with JNI not found, please check your settings.")
endif(JNI_FOUND)

###################################################################################################
# - nvcomp ----------------------------------------------------------------------------------------

include(ConfigureNvcomp)
if(NVCOMP_FOUND)
    message(STATUS "nvcomp compression library found in ${NVCOMP_ROOT}")
else()
    message(FATAL_ERROR "nvcomp compression library not found.")
endif(NVCOMP_FOUND)

add_library(nvcomp STATIC IMPORTED ${NVCOMP_LIB})

###################################################################################################
# - GDS/cufile ------------------------------------------------------------------------------------

if(USE_GDS)
    message(STATUS "Building with GPUDirect Storage (GDS)/cuFile support")
    find_package(cuFile REQUIRED)
endif(USE_GDS)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${THRUST_INCLUDE}"
                    "${CUB_INCLUDE}"
                    "${LIBCUDACXX_INCLUDE}"
                    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${NVCOMP_INCLUDE_DIR}"
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${SPDLOG_INCLUDE}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${JNI_INCLUDE_DIRS}"
                    "${CUDF_INCLUDE}"
                    "${RMM_INCLUDE}"
                    "${ARROW_INCLUDE}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib")


###################################################################################################
# - library targets -------------------------------------------------------------------------------

set(SOURCE_FILES
    "src/row_conversion.cu"
    "src/AggregationJni.cpp"
    "src/CudfJni.cpp"
    "src/CudaJni.cpp"
    "src/ColumnVectorJni.cpp"
    "src/ColumnViewJni.cpp"
    "src/ContiguousTableJni.cpp"
    "src/HostMemoryBufferNativeUtilsJni.cpp"
    "src/NvcompJni.cpp"
    "src/NvtxRangeJni.cpp"
    "src/RmmJni.cpp"
    "src/ScalarJni.cpp"
    "src/TableJni.cpp"
    "src/map_lookup.cu")
add_library(cudfjni SHARED ${SOURCE_FILES})

#Override RPATH for cudfjni
SET_TARGET_PROPERTIES(cudfjni
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
)

target_compile_options(cudfjni
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

target_compile_definitions(cudfjni
            PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_DEFINITIONS}>"
                   "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_DEFINITIONS}>"
)

if(USE_GDS)
    add_library(cufilejni SHARED "src/CuFileJni.cpp")
    target_include_directories(cufilejni PRIVATE "${cuFile_INCLUDE_DIRS}")
    target_link_libraries(cufilejni PRIVATE cudfjni "${cuFile_LIBRARIES}")
endif(USE_GDS)

###################################################################################################
# - rmm logging level -----------------------------------------------------------------------------

set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
# Set the possible values of build type for cmake-gui
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS
        "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(STATUS "RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

target_compile_definitions(cudfjni PUBLIC SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL})

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(cudfjni ${NVCOMP_LIB} ${CUDF_LIB} ${ARROW_LIBRARY} ${CUDART_LIBRARY} cuda)
