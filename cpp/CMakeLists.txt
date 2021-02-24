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

cmake_minimum_required(VERSION 3.14...3.17 FATAL_ERROR)

project(CUDA_DATAFRAME VERSION 0.18.0 LANGUAGES C CXX CUDA)

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

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")
    # Suppress parentheses warning which causes gmock to fail
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-parentheses")
endif(CMAKE_COMPILER_IS_GNUCXX)

if(CMAKE_CUDA_COMPILER_VERSION)
  # Compute the version. from  CMAKE_CUDA_COMPILER_VERSION
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR ${CMAKE_CUDA_COMPILER_VERSION})
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR ${CMAKE_CUDA_COMPILER_VERSION})
  set(CUDA_VERSION "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" CACHE STRING "Version of CUDA as computed from nvcc.")
  mark_as_advanced(CUDA_VERSION)
endif()

message(STATUS "CUDA_VERSION_MAJOR: ${CUDA_VERSION_MAJOR}")
message(STATUS "CUDA_VERSION_MINOR: ${CUDA_VERSION_MINOR}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")

# Always set this convenience variable
set(CUDA_VERSION_STRING "${CUDA_VERSION}")

# Auto-detect available GPU compute architectures
set(GPU_ARCHS "ALL" CACHE STRING
  "List of GPU architectures (semicolon-separated) to be compiled for. Pass 'ALL' if you want to compile for all supported GPU architectures. Empty string means to auto-detect the GPUs on the current system")

if("${GPU_ARCHS}" STREQUAL "")
  include(cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()

if("${GPU_ARCHS}" STREQUAL "ALL")
  
  # Check for embedded vs workstation architectures
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # This is being built for Linux4Tegra or SBSA ARM64
    set(GPU_ARCHS "62")
    if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
      set(GPU_ARCHS "${GPU_ARCHS};72")
    endif()
    if((CUDA_VERSION_MAJOR EQUAL 11) OR (CUDA_VERSION_MAJOR GREATER 11))
      set(GPU_ARCHS "${GPU_ARCHS};75;80")
    endif()

  else()
    # This is being built for an x86 or x86_64 architecture
    set(GPU_ARCHS "60")
    if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
      set(GPU_ARCHS "${GPU_ARCHS};70")
    endif()
    if((CUDA_VERSION_MAJOR EQUAL 10) OR (CUDA_VERSION_MAJOR GREATER 10))
      set(GPU_ARCHS "${GPU_ARCHS};75")
    endif()
    if((CUDA_VERSION_MAJOR EQUAL 11) OR (CUDA_VERSION_MAJOR GREATER 11))
      set(GPU_ARCHS "${GPU_ARCHS};80")
    endif()

  endif()
  
endif()
message("GPU_ARCHS = ${GPU_ARCHS}")

foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${arch},code=sm_${arch}")
endforeach()

list(GET GPU_ARCHS -1 ptx)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${ptx},code=compute_${ptx}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# set warnings as errors
# TODO: remove `no-maybe-unitialized` used to suppress warnings in rmm::exec_policy
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror=cross-execution-space-call -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations")

option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
if(DISABLE_DEPRECATION_WARNING)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif(DISABLE_DEPRECATION_WARNING)

# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
if(CMAKE_CUDA_LINEINFO)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif(CMAKE_CUDA_LINEINFO)

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -Xcompiler=-rdynamic")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

# To apply RUNPATH to transitive dependencies (this is a temporary solution)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--disable-new-dtags")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)

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
# - copy libcu++ ----------------------------------------------------------------------------------

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")

# `${LIBCUDACXX_INCLUDE_DIR}/` specifies that the contents of this directory will be installed (not the directory itself)
file(INSTALL "${LIBCUDACXX_INCLUDE_DIR}/" DESTINATION "${CMAKE_BINARY_DIR}/include/libcudacxx")  
file(INSTALL ${LIBCXX_INCLUDE_DIR}     DESTINATION "${CMAKE_BINARY_DIR}/include/libcxx")

###################################################################################################
# - find zlib -------------------------------------------------------------------------------------

find_package(ZLIB REQUIRED)

message(STATUS "ZLIB: ZLIB_LIBRARIES set to ${ZLIB_LIBRARIES}")
message(STATUS "ZLIB: ZLIB_INCLUDE_DIRS set to ${ZLIB_INCLUDE_DIRS}")

if(ZLIB_FOUND)
    message(STATUS "ZLib found in ${ZLIB_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "ZLib not found, please check your settings.")
endif(ZLIB_FOUND)

###################################################################################################
# - find boost ------------------------------------------------------------------------------------

# Don't look for a CMake configuration file
set(Boost_NO_BOOST_CMAKE ON)

find_package(
    Boost REQUIRED MODULE
    COMPONENTS filesystem
)

message(STATUS "BOOST: Boost_LIBRARIES set to ${Boost_LIBRARIES}")
message(STATUS "BOOST: Boost_INCLUDE_DIRS set to ${Boost_INCLUDE_DIRS}")

if(Boost_FOUND)
    message(STATUS "Boost found in ${Boost_INCLUDE_DIRS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
else()
    message(FATAL_ERROR "Boost not found, please check your settings.")
endif(Boost_FOUND)

###################################################################################################
# - RMM -------------------------------------------------------------------------------------------

find_path(RMM_INCLUDE "rmm"
          HINTS "$ENV{RMM_ROOT}/include")

message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

###################################################################################################
# - DLPACK -------------------------------------------------------------------------------------------

find_path(
    DLPACK_INCLUDE "dlpack"
    HINTS "$ENV{DLPACK_ROOT}/include"
)

message(STATUS "DLPACK: DLPACK_INCLUDE set to ${DLPACK_INCLUDE}")

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
endif(PER_THREAD_DEFAULT_STREAM)

###################################################################################################
# - add gtest -------------------------------------------------------------------------------------

if(BUILD_TESTS)
    include(CTest)
    include(ConfigureGoogleTest)

    if(GTEST_FOUND)
        message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
        include_directories(${GTEST_INCLUDE_DIR})
        add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
    else()
        message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
    endif(GTEST_FOUND)
endif(BUILD_TESTS)

message(STATUS "CUDF_TEST_LIST set to: ${CUDF_TEST_LIST}")

###################################################################################################
# - add google benchmark --------------------------------------------------------------------------

if(BUILD_BENCHMARKS)

  include(ConfigureGoogleBenchmark)

  if(GBENCH_FOUND)
    message(STATUS "Google C++ Benchmarking Framework (Google Benchmark) found in ${GBENCH_ROOT}")
    include_directories(${GBENCH_INCLUDE_DIR})
    add_subdirectory(${CMAKE_SOURCE_DIR}/benchmarks)
  else()
    message(AUTHOR_WARNING "Google C++ Benchmarking Framework (Google Benchmark) not found: automated tests are disabled.")
  endif(GBENCH_FOUND)

endif(BUILD_BENCHMARKS)

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
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${ARROW_INCLUDE_DIR}"
                    "${ZLIB_INCLUDE_DIRS}"
                    "${Boost_INCLUDE_DIRS}"
                    "${RMM_INCLUDE}"
                    "${DLPACK_INCLUDE}")

if(CONDA_INCLUDE_DIRS)
    include_directories("${CONDA_INCLUDE_DIRS}")
endif(CONDA_INCLUDE_DIRS)

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}"
                 "${GTEST_LIBRARY_DIR}")

if(CONDA_LINK_DIRS)
    link_directories("${CONDA_LINK_DIRS}")
endif(CONDA_LINK_DIRS)

###################################################################################################
# - jitify ----------------------------------------------------------------------------------------

# Creates executable stringify and uses it to convert types.h to c-str for use in JIT code
add_executable(stringify "${JITIFY_INCLUDE_DIR}/stringify.cpp")
execute_process(WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include
    )

add_custom_command(WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
                   COMMENT "Stringify headers for use in JIT compiled code"
                   DEPENDS stringify
                   OUTPUT ${CMAKE_BINARY_DIR}/include/jit/types.h.jit
                          ${CMAKE_BINARY_DIR}/include/jit/types.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/bit.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/timestamps.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/fixed_point.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/durations.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/chrono.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/climits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/cstddef.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/cstdint.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/ctime.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/limits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/ratio.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/type_traits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/version.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__config.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__pragma_pop.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__pragma_push.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__config.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_pop.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_push.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__undef_macros.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/chrono.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/climits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/cstddef.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/cstdint.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/ctime.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/limits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/ratio.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/type_traits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/version.jit
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/types.h
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/types.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/utilities/bit.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/wrappers/timestamps.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/fixed_point/fixed_point.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/wrappers/durations.hpp
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/chrono
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/climits
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/cstddef
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/cstdint
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/ctime
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/limits
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/ratio
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/type_traits
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/version
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__config
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__pragma_pop
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__pragma_push
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__config
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__pragma_pop
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__pragma_push
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__undef_macros
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/chrono
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/climits
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/cstddef
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/cstdint
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/ctime
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/limits
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/ratio
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/type_traits
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/version

                   # stringified headers are placed underneath the bin include jit directory and end in ".jit"
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/types.h > ${CMAKE_BINARY_DIR}/include/jit/types.h.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/types.hpp > ${CMAKE_BINARY_DIR}/include/jit/types.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/utilities/bit.hpp > ${CMAKE_BINARY_DIR}/include/jit/bit.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ../src/rolling/rolling_jit_detail.hpp > ${CMAKE_BINARY_DIR}/include/jit/rolling_jit_detail.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/wrappers/timestamps.hpp > ${CMAKE_BINARY_DIR}/include/jit/timestamps.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/fixed_point/fixed_point.hpp > ${CMAKE_BINARY_DIR}/include/jit/fixed_point.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/wrappers/durations.hpp > ${CMAKE_BINARY_DIR}/include/jit/durations.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/chrono cuda_std_chrono > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/chrono.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/climits cuda_std_climits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/climits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/cstddef cuda_std_cstddef > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/cstddef.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/cstdint cuda_std_cstdint > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/cstdint.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/ctime cuda_std_ctime > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/ctime.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/limits cuda_std_limits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/limits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/ratio cuda_std_ratio > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/ratio.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/type_traits cuda_std_type_traits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/type_traits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/version cuda_std_version > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/version.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__config cuda_std_detail___config > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__config.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__pragma_pop cuda_std_detail___pragma_pop > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__pragma_pop.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__pragma_push cuda_std_detail___pragma_push > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__pragma_push.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__config cuda_std_detail_libcxx_include___config > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__config.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__pragma_pop cuda_std_detail_libcxx_include___pragma_pop > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_pop.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__pragma_push cuda_std_detail_libcxx_include___pragma_push > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_push.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__undef_macros cuda_std_detail_libcxx_include___undef_macros > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__undef_macros.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/chrono cuda_std_detail_libcxx_include_chrono > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/chrono.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/climits cuda_std_detail_libcxx_include_climits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/climits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/cstddef cuda_std_detail_libcxx_include_cstddef > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/cstddef.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/cstdint cuda_std_detail_libcxx_include_cstdint > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/cstdint.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/ctime cuda_std_detail_libcxx_include_ctime > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/ctime.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/limits cuda_std_detail_libcxx_include_limits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/limits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/ratio cuda_std_detail_libcxx_include_ratio > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/ratio.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/type_traits cuda_std_detail_libcxx_include_type_traits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/type_traits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/version cuda_std_detail_libcxx_include_version > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/version.jit
                   )

add_custom_target(stringify_run DEPENDS
                  ${CMAKE_BINARY_DIR}/include/jit/types.h.jit
                  ${CMAKE_BINARY_DIR}/include/jit/types.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/bit.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/timestamps.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/fixed_point.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/durations.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/chrono.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/climits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/cstddef.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/cstdint.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/ctime.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/limits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/ratio.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/type_traits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/version.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__config.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__pragma_pop.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__pragma_push.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__config.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_pop.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_push.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__undef_macros.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/chrono.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/climits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/cstddef.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/cstdint.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/ctime.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/limits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/ratio.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/type_traits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/version.jit
                  )

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
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

###################################################################################################
# - library targets -------------------------------------------------------------------------------

set(CUDF_NAMESPACE "cudf")

# define a function that populates the namespace with modules from the cudf source tree (cpp/src)
function(add_library_module NAMESPACE MODULE MODULE_SOURCE_DIRS)
    foreach(MODULE_SOURCE_DIR IN LISTS MODULE_SOURCE_DIRS)
        file(GLOB_RECURSE MODULE_SOURCE "${MODULE_SOURCE_DIR}/*")
        list(APPEND MODULE_SRC ${MODULE_SOURCE})
    endforeach(MODULE_SOURCE_DIR IN LISTS MODULE_SOURCE_DIRS)
    message(STATUS "Adding library module with name ${MODULE} under namespace ${NAMESPACE} ...")
    message(STATUS "Using sources: ${MODULE_SRC}")
    add_library("${NAMESPACE}_${MODULE}" "${MODULE_SRC}")
    set_target_properties("${NAMESPACE}_${MODULE}" PROPERTIES BUILD_RPATH "\$ORIGIN")
    # Disable Jitify log printing. See https://github.com/NVIDIA/jitify/issues/79
    target_compile_definitions("${NAMESPACE}_${MODULE}" PRIVATE "JITIFY_PRINT_LOG=0")
    # per-thread default stream
    if(PER_THREAD_DEFAULT_STREAM)
        target_compile_definitions("${NAMESPACE}_${MODULE}" PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
    endif(PER_THREAD_DEFAULT_STREAM)
    # nvtx
    if(NOT USE_NVTX)
        target_compile_definitions("${NAMESPACE}_${MODULE}" PUBLIC NVTX_DISABLE)
    endif(NOT USE_NVTX)
    # spdlog level
    target_compile_definitions("${NAMESPACE}_${MODULE}" PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${LOGGING_LEVEL}")
    add_dependencies("${NAMESPACE}_${MODULE}" stringify_run)
    target_link_libraries("${NAMESPACE}_${MODULE}" arrow arrow_cuda nvrtc ${CUDART_LIBRARY} cuda ${ZLIB_LIBRARIES} ${Boost_LIBRARIES})
    add_library("${NAMESPACE}::${MODULE}" ALIAS "${NAMESPACE}_${MODULE}")
endfunction()

# define a list of modules for the base library
set(
    CUDF_BASE_MODULE_LIST
        "${CMAKE_SOURCE_DIR}/src/aggregation"
        "${CMAKE_SOURCE_DIR}/src/binaryop"
        "${CMAKE_SOURCE_DIR}/src/bitmask"
        "${CMAKE_SOURCE_DIR}/src/column"
        "${CMAKE_SOURCE_DIR}/src/copying"
        "${CMAKE_SOURCE_DIR}/src/datetime"
        "${CMAKE_SOURCE_DIR}/src/dictionary"
        "${CMAKE_SOURCE_DIR}/src/filling"
        "${CMAKE_SOURCE_DIR}/src/groupby"
        "${CMAKE_SOURCE_DIR}/src/jit"
        "${CMAKE_SOURCE_DIR}/src/lists"
        "${CMAKE_SOURCE_DIR}/src/quantiles"
        "${CMAKE_SOURCE_DIR}/src/reshape"
        "${CMAKE_SOURCE_DIR}/src/round"
        "${CMAKE_SOURCE_DIR}/src/scalar"
        "${CMAKE_SOURCE_DIR}/src/search"
        "${CMAKE_SOURCE_DIR}/src/sort"
        "${CMAKE_SOURCE_DIR}/src/stream_compaction"
        "${CMAKE_SOURCE_DIR}/src/strings"
        "${CMAKE_SOURCE_DIR}/src/structs"
        "${CMAKE_SOURCE_DIR}/src/table"
        "${CMAKE_SOURCE_DIR}/src/text"
        "${CMAKE_SOURCE_DIR}/src/transform"
        "${CMAKE_SOURCE_DIR}/src/unary"
        "${CMAKE_SOURCE_DIR}/src/utilities"
)

set(CUDF_BASE_NAME "base")
add_library_module("${CUDF_NAMESPACE}" "${CUDF_BASE_NAME}" "${CUDF_BASE_MODULE_LIST}")

set(
    CUDF_MODULE_LIST
        "${CMAKE_SOURCE_DIR}/src/ast"
        "${CMAKE_SOURCE_DIR}/src/comms"
        "${CMAKE_SOURCE_DIR}/src/hash"
        "${CMAKE_SOURCE_DIR}/src/interop"
        "${CMAKE_SOURCE_DIR}/src/io"
        "${CMAKE_SOURCE_DIR}/src/join"
        "${CMAKE_SOURCE_DIR}/src/merge"
        "${CMAKE_SOURCE_DIR}/src/partitioning"
        "${CMAKE_SOURCE_DIR}/src/reductions"
        "${CMAKE_SOURCE_DIR}/src/replace"
        "${CMAKE_SOURCE_DIR}/src/reshape"
        "${CMAKE_SOURCE_DIR}/src/rolling"
        "${CMAKE_SOURCE_DIR}/src/transpose"
)

# generate names of cudf library modules based on source tree under cpp/src
foreach(CUDF_MODULE_DIR IN LISTS CUDF_MODULE_LIST)
    get_filename_component(CUDF_MODULE_NAME ${CUDF_MODULE_DIR} NAME)
    add_library_module(${CUDF_NAMESPACE} ${CUDF_MODULE_NAME} ${CUDF_MODULE_DIR})
    target_link_libraries("${CUDF_NAMESPACE}_${CUDF_MODULE_NAME}" "${CUDF_NAMESPACE}::${CUDF_BASE_NAME}")
    list(APPEND CUDF_MODULES "${CUDF_NAMESPACE}_${CUDF_MODULE_NAME}")
endforeach()

set(CUDF_PROXY_NAME "${CUDF_NAMESPACE}")
set(CUDF_PROXY_SRC "${CMAKE_BINARY_DIR}/cudf.cpp")  # this file will be generated
message(STATUS "Generating proxy library: cudf")
file(WRITE "${CUDF_PROXY_SRC}" "// This file is generated by ${CMAKE_CURRENT_LIST_FILE} \n")
add_library("${CUDF_PROXY_NAME}" "${CUDF_PROXY_SRC}")
# `-Wl,--no-as-needed`: link libraries to cudf even if no symbols in that library are called
target_link_options("${CUDF_PROXY_NAME}" PRIVATE "LINKER:--no-as-needed")
set_target_properties("${CUDF_PROXY_NAME}" PROPERTIES BUILD_RPATH "\$ORIGIN")
add_library("${CUDF_PROXY_NAME}::${CUDF_PROXY_NAME}" ALIAS "${CUDF_PROXY_NAME}")

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries("${CUDF_PROXY_NAME}" "${CUDF_NAMESPACE}::${CUDF_BASE_NAME}" "${CUDF_MODULES}")

###################################################################################################
# - install targets -------------------------------------------------------------------------------

# install targets for cuDF
install(TARGETS cudf_base
        DESTINATION lib
        COMPONENT cudf)

install(TARGETS ${CUDF_MODULES}
        DESTINATION lib
        COMPONENT cudf)

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
