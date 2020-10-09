#=============================================================================
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.17...3.18 FATAL_ERROR)

project(CUDA_DATAFRAME VERSION 0.16.0 LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif(NOT CMAKE_CUDA_COMPILER)

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
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------
include(cmake/CUDF_CompilerOptions.cmake)

###################################################################################################
# - CPM options --------------------------------------------------------------------------------
include(cmake/CPM.cmake)

###################################################################################################
# - cudart options --------------------------------------------------------------------------------
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking

option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

if(CUDA_STATIC_RUNTIME)
    message(STATUS "Enabling static linking of cudart")
    set(CUDART_LIBRARY "cudart_static")
else()
    set(CUDART_LIBRARY "cudart")
endif(CUDA_STATIC_RUNTIME)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - third party dependencies-----------------------------------------------------------------------
# add third party dependencies using CMake FetchContent

add_subdirectory(thirdparty)

###################################################################################################
# - conda environment -----------------------------------------------------------------------------

if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{BUILD_PREFIX}/include" "$ENV{PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{BUILD_PREFIX}/lib" "$ENV{PREFIX}/lib")
    message(STATUS "Conda build detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Conda environment detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
endif("$ENV{CONDA_BUILD}" STREQUAL "1")

###################################################################################################
# - find arrow ------------------------------------------------------------------------------------

option(ARROW_STATIC_LIB "Build and statically link with Arrow libraries" OFF)

if(ARROW_STATIC_LIB)
  message(STATUS "BUILDING ARROW")
  include(ConfigureArrow)

  if(ARROW_FOUND)
    message(STATUS "Apache Arrow found in ${ARROW_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "Apache Arrow not found, please check your settings.")
  endif(ARROW_FOUND)

  add_library(arrow STATIC IMPORTED ${ARROW_LIB})
  add_library(arrow_cuda STATIC IMPORTED ${ARROW_CUDA_LIB})
else()
  find_path(ARROW_INCLUDE_DIR "arrow"
      HINTS "$ENV{ARROW_ROOT}/include")

  find_library(ARROW_LIB "arrow"
      NAMES libarrow
      HINTS "$ENV{ARROW_ROOT}/lib" "$ENV{ARROW_ROOT}/build")

  find_library(ARROW_CUDA_LIB "arrow_cuda"
      NAMES libarrow_cuda
      HINTS "$ENV{ARROW_ROOT}/lib" "$ENV{ARROW_ROOT}/build")

  message(STATUS "ARROW: ARROW_INCLUDE_DIR set to ${ARROW_INCLUDE_DIR}")
  message(STATUS "ARROW: ARROW_LIB set to ${ARROW_LIB}")
  message(STATUS "ARROW: ARROW_CUDA_LIB set to ${ARROW_CUDA_LIB}")

  add_library(arrow SHARED IMPORTED ${ARROW_LIB})
  add_library(arrow_cuda SHARED IMPORTED ${ARROW_CUDA_LIB})
endif(ARROW_STATIC_LIB)

if(ARROW_INCLUDE_DIR AND ARROW_LIB AND ARROW_CUDA_LIB)
  set_target_properties(arrow PROPERTIES IMPORTED_LOCATION ${ARROW_LIB})
  set_target_properties(arrow_cuda PROPERTIES IMPORTED_LOCATION ${ARROW_CUDA_LIB})
endif(ARROW_INCLUDE_DIR AND ARROW_LIB AND ARROW_CUDA_LIB)

###################################################################################################
# - copy libcu++ ------------------------------------------------------------------------------------

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/libcudacxx")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/libcxx/include")

file(INSTALL ${LIBCUDACXX_INCLUDE_DIR} DESTINATION ${CMAKE_BINARY_DIR}/include/libcudacxx/)
file(INSTALL ${LIBCXX_INCLUDE_DIR}     DESTINATION ${CMAKE_BINARY_DIR}/include/libcxx/include/)

###################################################################################################
# - find zlib -------------------------------------------------------------------------------------

CPMFindPackage(
  NAME ZLIB
  GITHUB_REPOSITORY madler/zlib
  VERSION 1.2.11
)

message(STATUS "ZLIB: ZLIB_LIBRARY set to ${ZLIB_LIBRARY}")
message(STATUS "ZLIB: ZLIB_INCLUDE_DIR set to ${ZLIB_INCLUDE_DIR}")

if(ZLIB_INCLUDE_DIR)
    message(STATUS "ZLib found in ${ZLIB_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "ZLib not found, please check your settings.")
endif(ZLIB_INCLUDE_DIR)


###################################################################################################
# - find boost ------------------------------------------------------------------------------------

# Don't look for a CMake configuration file
set(Boost_NO_BOOST_CMAKE ON)

CPMFindPackage(
  NAME Boost
  GITHUB_REPOSITORY Orphis/boost-cmake@60fb5fa
  VERSION 1.70.0
  FIND_PACKAGE_ARGUMENTS "COMPONENTS filesystem"
)

message(STATUS "BOOST: Boost_LIBRARY set to ${Boost_LIBRARY}")
message(STATUS "BOOST: Boost_INCLUDE_DIR set to ${Boost_INCLUDE_DIR}")

if(Boost_INCLUDE_DIR)
    message(STATUS "Boost found in ${Boost_INCLUDE_DIR}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
else()
    message(FATAL_ERROR "Boost not found, please check your settings.")
endif(Boost_INCLUDE_DIR)

###################################################################################################
# - RMM -------------------------------------------------------------------------------------------

# FindCUDA doesn't quite set the variables it should. RMM needs this to be set to build from CPM
set(CUDAToolkit_INCLUDE_DIR "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

find_path(RMM_INCLUDE "rmm"
          HINTS "$ENV{RMM_ROOT}/include")

message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

#CPMFindPackage(
#  NAME RMM
#  GITHUB_REPOSITORY rapidsai/rmm
#  VERSION ${CMAKE_PROJECT_VERSION}
#  GIT_TAG "branch-${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}"
#)

#message(STATUS "RMM: RMM_SOURCE_DIR set to ${RMM_SOURCE_DIR}")

###################################################################################################
# - DLPACK -------------------------------------------------------------------------------------------

CPMFindPackage(
  NAME DLPACK
  GITHUB_REPOSITORY dmlc/dlpack
  VERSION 0.3
)

message(STATUS "DLPACK: dlpack_SOURCE_DIR set to ${dlpack_SOURCE_DIR}")


###################################################################################################
# - jitify ----------------------------------------------------------------------------------------

option(JITIFY_USE_CACHE "Use a file cache for JIT compiled kernels" ON)
if(JITIFY_USE_CACHE)
    message(STATUS "Using file cache for JIT compiled kernels")
    add_definitions("-DJITIFY_USE_CACHE -DCUDF_VERSION=${CMAKE_PROJECT_VERSION}")
endif(JITIFY_USE_CACHE)

###################################################################################################
# - per-thread default stream option --------------------------------------------------------------
# This needs to be defined first so tests and benchmarks can inherit it.

option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
if(PER_THREAD_DEFAULT_STREAM)
    message(STATUS "Using per-thread default stream")
    add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif(PER_THREAD_DEFAULT_STREAM)

###################################################################################################
# - add gtest -------------------------------------------------------------------------------------

if(BUILD_TESTS)

  CPMFindPackage(
    NAME GTEST
    GITHUB_REPOSITORY google/googletest
    GIT_TAG release-1.10.0
    VERSION 1.10.0
    OPTIONS
      "INSTALL_GTEST OFF"
      "gtest_force_shared_crt"
  )
  enable_testing()
  
  include_directories("${gtest_SOURCE_DIR}/include"
                      "${gmock_SOURCE_DIR}/include")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif(BUILD_TESTS)


message(STATUS "CUDF_TEST_LIST set to: ${CUDF_TEST_LIST}")

###################################################################################################
# - add google benchmark --------------------------------------------------------------------------

if(BUILD_BENCHMARKS)

  CPMFindPackage(
    NAME benchmark
    GITHUB_REPOSITORY google/benchmark
    VERSION 1.5.1
    OPTIONS
    "BENCHMARK_ENABLE_TESTING OFF"
    # The REGEX feature test fails when gbench's cmake is run under CPM because it doesn't assume C++11
    # Additionally, attempting to set the CMAKE_CXX_VERSION here doesn't propogate to the feature test build
    # Therefore, we just disable the feature test and assume platforms we care about have a regex impl available
    "RUN_HAVE_GNU_POSIX_REGEX 0" #
  )

  include_directories("${benchmark_INCLUDE_DIR}")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)

  endif(BUILD_BENCHMARKS)

message(STATUS "benchmark_SOURCE_DIR: ${benchmark_SOURCE_DIR}")
message(STATUS "BENCHMARK_LIST set to: ${BENCHMARK_LIST}")

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${THRUST_INCLUDE_DIR}"
                    "${JITIFY_INCLUDE_DIR}"
                    "${LIBCUDACXX_INCLUDE_DIR}")

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_BINARY_DIR}/include/jit"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${ARROW_INCLUDE_DIR}")

if(CONDA_INCLUDE_DIRS)
    include_directories("${CONDA_INCLUDE_DIRS}")
endif(CONDA_INCLUDE_DIRS)

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}")

if(CONDA_LINK_DIRS)
    link_directories("${CONDA_LINK_DIRS}")
endif(CONDA_LINK_DIRS)

###################################################################################################
# - library targets -------------------------------------------------------------------------------
include(cmake/CUDF_LibraryTargets.cmake)

###################################################################################################
# - jitify ----------------------------------------------------------------------------------------
include(cmake/CUDF_Jitify.cmake)


###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
else()
    # Makes NVTX APIs no-ops
    add_compile_definitions(NVTX_DISABLE)
endif(USE_NVTX)

option(HT_DEFAULT_ALLOCATOR "Use the default allocator for hash tables" ON)
if(HT_DEFAULT_ALLOCATOR)
    message(STATUS "Using default allocator for hash tables")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DHT_DEFAULT_ALLOCATOR")
endif(HT_DEFAULT_ALLOCATOR)

###################################################################################################
# - rmm logging level -----------------------------------------------------------------------------

set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
# Set the possible values of build type for cmake-gui
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS
        "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(STATUS "RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

target_compile_definitions(cudf PUBLIC SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL})

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

# link targets for cuDF
target_link_libraries(cudf 
                      arrow 
                      arrow_cuda 
                      nvrtc 
                      ${CUDART_LIBRARY} 
                      cuda 
                      ZLIB::ZLIB
                      Boost::filesystem)

###################################################################################################
# - install targets -------------------------------------------------------------------------------

# install targets for cuDF
install(TARGETS cudf
        DESTINATION lib
        COMPONENT cudf)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf
        DESTINATION include
        COMPONENT cudf)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf_test
        DESTINATION include
        COMPONENT cudf)

install(DIRECTORY ${CMAKE_BINARY_DIR}/include/libcxx
        DESTINATION include/libcudf
        COMPONENT cudf)

install(DIRECTORY ${CMAKE_BINARY_DIR}/include/libcudacxx
        DESTINATION include/libcudf
        COMPONENT cudf)

add_custom_target(install_cudf
                  COMMAND "${CMAKE_COMMAND}" -DCOMPONENT=cudf -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
                  DEPENDS cudf)

if(BUILD_TESTS)
    add_dependencies(install_cudf cudftestutil)
endif(BUILD_TESTS)

add_custom_target(build_tests_cudf
                  DEPENDS ${CUDF_TEST_LIST})

add_custom_target(test_cudf
                  COMMAND ctest
                  DEPENDS build_tests_cudf)

add_custom_target(build_benchmarks_cudf
                  DEPENDS ${BENCHMARK_LIST})

###################################################################################################
# - make documentation ----------------------------------------------------------------------------

# doc targets for cuDF
add_custom_command(OUTPUT CUDF_DOXYGEN
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                   COMMAND doxygen Doxyfile
                   VERBATIM)

add_custom_target(docs_cudf DEPENDS CUDF_DOXYGEN)
