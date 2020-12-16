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

cmake_minimum_required(VERSION 3.14...3.18 FATAL_ERROR)

project(CUDA_DATAFRAME VERSION 0.18.0 LANGUAGES C CXX CUDA)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)
option(ARROW_STATIC_LIB "Build and statically link with Arrow libraries" OFF)
option(JITIFY_USE_CACHE "Use a file cache for JIT compiled kernels" ON)
option(HT_DEFAULT_ALLOCATOR "Use the default allocator for hash tables" ON)
option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING  "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)

# Set RMM logging level
set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(STATUS "RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
endif()

if(JITIFY_USE_CACHE)
    message(STATUS "Using file cache for JIT compiled kernels")
endif()

if(HT_DEFAULT_ALLOCATOR)
    message(STATUS "Using default allocator for hash tables")
endif()

if(PER_THREAD_DEFAULT_STREAM)
    message(STATUS "Using per-thread default stream")
endif(PER_THREAD_DEFAULT_STREAM)

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
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# find CUDAToolkit package, set CUDA compilation flags
include(cmake/Modules/ConfigureCUDA.cmake)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CUDA_DATAFRAME_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(cmake/Modules/CPM.cmake)

###################################################################################################
# - third party dependencies-----------------------------------------------------------------------

# add third party dependencies using CPM
include(cmake/Modules/CUDF_thirdparty.cmake)

if(Boost_FOUND)
    message(STATUS "Boost found in ${Boost_INCLUDE_DIRS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
else()
    message(FATAL_ERROR "Boost not found, please check your settings.")
endif(Boost_FOUND)

# Stringify libcudf and libcudacxx headers used in JIT operations
include(cmake/Modules/StringifyJITHeaders.cmake)

###################################################################################################
# - add tests -------------------------------------------------------------------------------------

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif(BUILD_TESTS)

message(STATUS "CUDF_TEST_LIST set to: ${CUDF_TEST_LIST}")

###################################################################################################
# - add benchmarks --------------------------------------------------------------------------------

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif(BUILD_BENCHMARKS)

message(STATUS "BENCHMARK_LIST set to: ${BENCHMARK_LIST}")

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
    # Specify include paths for the current target and dependents
    target_include_directories("${NAMESPACE}_${MODULE}"
        PUBLIC      "$<BUILD_INTERFACE:${RMM_INCLUDE}>"
                    "$<BUILD_INTERFACE:${DLPACK_INCLUDE}>"
                    "$<BUILD_INTERFACE:${ARROW_INCLUDE_DIR}>"
                    "$<BUILD_INTERFACE:${JITIFY_INCLUDE_DIR}>"
                    "$<BUILD_INTERFACE:${LIBCUDACXX_INCLUDE_DIR}>"
                    "$<BUILD_INTERFACE:${CUDA_DATAFRAME_SOURCE_DIR}/include>"
                    "$<BUILD_INTERFACE:${CUDA_DATAFRAME_BINARY_DIR}/include>"
        PRIVATE     "$<BUILD_INTERFACE:${CUDA_DATAFRAME_SOURCE_DIR}/src>"
        INTERFACE   "$<INSTALL_INTERFACE:include>")
    # Add Conda library paths if specified
    if(CONDA_LINK_DIRS)
        target_link_directories("${NAMESPACE}_${MODULE}" PUBLIC "$<BUILD_INTERFACE:${CONDA_LINK_DIRS}>")
    endif()
    # Add Conda include paths if specified
    if(CONDA_INCLUDE_DIRS)
        target_include_directories("${NAMESPACE}_${MODULE}" PUBLIC "$<BUILD_INTERFACE:${CONDA_INCLUDE_DIRS}>")
    endif(CONDA_INCLUDE_DIRS)
    # Instruct jitify to use the kernel JIT cache
    if(JITIFY_USE_CACHE)
        target_compile_definitions("${NAMESPACE}_${MODULE}" PUBLIC
            JITIFY_USE_CACHE CUDF_VERSION=${CMAKE_PROJECT_VERSION})
    endif(JITIFY_USE_CACHE)
    # Set hash table allocator
    if(HT_DEFAULT_ALLOCATOR)
        target_compile_definitions("${NAMESPACE}_${MODULE}" PUBLIC HT_DEFAULT_ALLOCATOR)
    endif(HT_DEFAULT_ALLOCATOR)
    # Per-thread default stream
    if(PER_THREAD_DEFAULT_STREAM)
        target_compile_definitions("${NAMESPACE}_${MODULE}" PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
    endif(PER_THREAD_DEFAULT_STREAM)
    # Disable NVTX if necessary
    if(NOT USE_NVTX)
        target_compile_definitions("${NAMESPACE}_${MODULE}" PUBLIC NVTX_DISABLE)
    endif(NOT USE_NVTX)
    # Define spdlog level
    target_compile_definitions("${NAMESPACE}_${MODULE}" PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${LOGGING_LEVEL}")
    # Compile stringified JIT sources first
    add_dependencies("${NAMESPACE}_${MODULE}" stringify_run)
    # Specify the target module library dependencies
    target_link_libraries("${NAMESPACE}_${MODULE}"
                   PUBLIC ZLIB::ZLIB
                          Boost::filesystem
                          arrow arrow_cuda
                          cudf::Thrust
                          CUDA::nvrtc
                          ${CUDART_LIBRARY}
                          CUDA::cuda_driver)
    # Add alias target cudf::<module_name> for exporting
    add_library("${NAMESPACE}::${MODULE}" ALIAS "${NAMESPACE}_${MODULE}")
endfunction()

# define a list of modules for the base library
set(
    CUDF_BASE_MODULE_LIST
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/aggregation"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/binaryop"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/bitmask"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/column"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/copying"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/datetime"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/dictionary"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/filling"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/groupby"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/jit"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/lists"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/quantiles"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/reshape"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/round"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/scalar"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/search"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/sort"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/stream_compaction"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/strings"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/structs"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/table"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/text"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/transform"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/unary"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/utilities"
)

set(CUDF_BASE_NAME "base")
add_library_module("${CUDF_NAMESPACE}" "${CUDF_BASE_NAME}" "${CUDF_BASE_MODULE_LIST}")

set(
    CUDF_MODULE_LIST
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/ast"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/comms"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/hash"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/interop"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/io"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/join"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/merge"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/partitioning"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/reductions"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/replace"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/reshape"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/rolling"
        "${CUDA_DATAFRAME_SOURCE_DIR}/src/transpose"
)

# generate names of cudf library modules based on source tree under cpp/src
foreach(CUDF_MODULE_DIR IN LISTS CUDF_MODULE_LIST)
    get_filename_component(CUDF_MODULE_NAME ${CUDF_MODULE_DIR} NAME)
    add_library_module(${CUDF_NAMESPACE} ${CUDF_MODULE_NAME} ${CUDF_MODULE_DIR})
    target_link_libraries("${CUDF_NAMESPACE}_${CUDF_MODULE_NAME}"
                   PUBLIC "${CUDF_NAMESPACE}::${CUDF_BASE_NAME}")
    list(APPEND CUDF_MODULES "${CUDF_NAMESPACE}_${CUDF_MODULE_NAME}")
endforeach()

set(CUDF_PROXY_NAME "${CUDF_NAMESPACE}")
set(CUDF_PROXY_SRC "${CUDA_DATAFRAME_BINARY_DIR}/cudf.cpp")  # this file will be generated
message(STATUS "Generating proxy library: cudf")
file(WRITE "${CUDF_PROXY_SRC}" "// This file is generated by ${CMAKE_CURRENT_LIST_FILE} \n")
add_library("${CUDF_PROXY_NAME}" "${CUDF_PROXY_SRC}")
# `-Wl,--no-as-needed`: link libraries to cudf even if no symbols in that library are called
target_link_options("${CUDF_PROXY_NAME}" PRIVATE "LINKER:--no-as-needed")
set_target_properties("${CUDF_PROXY_NAME}" PROPERTIES BUILD_RPATH "\$ORIGIN")
add_library("${CUDF_PROXY_NAME}::${CUDF_PROXY_NAME}" ALIAS "${CUDF_PROXY_NAME}")
# Link the proxy libcudf.so to libcudf_base.so and each `libcudf_<module>.so`
target_link_libraries("${CUDF_PROXY_NAME}" PUBLIC "${CUDF_NAMESPACE}::${CUDF_BASE_NAME}" "${CUDF_MODULES}")

###################################################################################################
# - install targets -------------------------------------------------------------------------------

include(GNUInstallDirs)

set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/cudf)

# install target for cudf_base
install(TARGETS "${CUDF_NAMESPACE}_${CUDF_BASE_NAME}"
        DESTINATION lib
        COMPONENT cudf
        EXPORT cudf-targets)

# install target for each cudf_<module>
install(TARGETS ${CUDF_MODULES}
        DESTINATION lib
        COMPONENT cudf
        EXPORT cudf-targets)

# install target for the proxy libcudf.so
install(TARGETS ${CUDF_PROXY_NAME}
        DESTINATION lib
        COMPONENT cudf
        EXPORT cudf-targets)

install(DIRECTORY ${CUDA_DATAFRAME_SOURCE_DIR}/include/cudf
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT cudf)

install(DIRECTORY ${CUDA_DATAFRAME_SOURCE_DIR}/include/cudf_test
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT cudf)

install(DIRECTORY ${CUDA_DATAFRAME_BINARY_DIR}/include/libcxx
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libcudf
        COMPONENT cudf)

install(DIRECTORY ${CUDA_DATAFRAME_BINARY_DIR}/include/libcudacxx
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libcudf
        COMPONENT cudf)

add_custom_target(install_cudf
                  COMMAND "${CMAKE_COMMAND}" -DCOMPONENT=cudf -P "${CUDA_DATAFRAME_BINARY_DIR}/cmake_install.cmake"
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

include(CMakePackageConfigHelpers)

configure_package_config_file(cmake/cudf-config.cmake.in "${CUDA_DATAFRAME_BINARY_DIR}/cudf-config.cmake"
                              INSTALL_DESTINATION "${INSTALL_CONFIGDIR}")

write_basic_package_version_file("${CUDA_DATAFRAME_BINARY_DIR}/cudf-config-version.cmake"
                                 COMPATIBILITY SameMinorVersion)

install(FILES   "${CUDA_DATAFRAME_BINARY_DIR}/cudf-config.cmake"
                "${CUDA_DATAFRAME_BINARY_DIR}/cudf-config-version.cmake"
                "${CUDA_DATAFRAME_SOURCE_DIR}/cmake/install/FindThrust.cmake"
    DESTINATION "${INSTALL_CONFIGDIR}")

install(EXPORT  cudf-targets
    FILE        cudf-targets.cmake
    NAMESPACE   cudf::
    DESTINATION "${INSTALL_CONFIGDIR}")

###################################################################################################
# - make documentation ----------------------------------------------------------------------------

# doc targets for cuDF
add_custom_command(OUTPUT CUDF_DOXYGEN
                   WORKING_DIRECTORY ${CUDA_DATAFRAME_SOURCE_DIR}/doxygen
                   COMMAND doxygen Doxyfile
                   VERBATIM)

add_custom_target(docs_cudf DEPENDS CUDF_DOXYGEN)
