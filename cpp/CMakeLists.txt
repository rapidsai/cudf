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

project(CUDA_DATAFRAME VERSION 0.17.0 LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif(NOT CMAKE_CUDA_COMPILER)

###################################################################################################
# - CPM options --------------------------------------------------------------------------------

include(cmake/CPM.cmake)

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
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

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
# - copy libcu++ ------------------------------------------------------------------------------------

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")

# `${LIBCUDACXX_INCLUDE_DIR}/` specifies that the contents of this directory will be installed (not the directory itself)
#file(INSTALL "${LIBCUDACXX_INCLUDE_DIR}/" DESTINATION "${CMAKE_BINARY_DIR}/include/libcudacxx")  
file(INSTALL ${LIBCXX_INCLUDE_DIR}     DESTINATION "${CMAKE_BINARY_DIR}/include/libcxx")


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

# jitify and libcudacxx are not CPM compatible yet
add_subdirectory(thirdparty)
include(cmake/CUDF_Thirdparty.cmake)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${JITIFY_INCLUDE_DIR}"
                    "${LIBCUDACXX_INCLUDE_DIR}")

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)

include_directories("${CMAKE_BINARY_DIR}/include"
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
#add_library("${CUDF_PROXY_NAME}" "${CUDF_PROXY_SRC}")
# `-Wl,--no-as-needed`: link libraries to cudf even if no symbols in that library are called
target_link_options("${CUDF_PROXY_NAME}" PRIVATE "LINKER:--no-as-needed")
set_target_properties("${CUDF_PROXY_NAME}" PROPERTIES BUILD_RPATH "\$ORIGIN")
add_library("${CUDF_PROXY_NAME}::${CUDF_PROXY_NAME}" ALIAS "${CUDF_PROXY_NAME}")

###################################################################################################
# - library targets -------------------------------------------------------------------------------

if(CUDA_STATIC_RUNTIME)
    message(STATUS "Enabling static linking of cudart")
    set(CUDART_LIBRARY "cudart_static")
    target_link_libraries(cudf INTERFACE cudart_static)
else()
    set(CUDART_LIBRARY "cudart")
    target_link_libraries(cudf INTERFACE cudart)
endif(CUDA_STATIC_RUNTIME)

target_link_libraries(cudf INTERFACE ZLIB::ZLIB Boost::filesystem)
target_link_libraries(cudf INTERFACE arrow arrow_cuda)
target_link_libraries(cudf INTERFACE nvrtc cuda)
target_link_libraries(cudf INTERFACE rmm::rmm)

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
