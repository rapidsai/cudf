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
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)
option(BUILD_SHARED_LIBS "Build cuDF shared libraries" ON)
option(JITIFY_USE_CACHE "Use a file cache for JIT compiled kernels" ON)
option(ARROW_STATIC_LIB "Build and statically link Arrow libraries" OFF)
option(HT_DEFAULT_ALLOCATOR "Use the default allocator for hash tables" ON)
option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(AUTO_DETECT_CUDA_ARCHITECTURES "Auto-detect CUDA architectures to build from the GPUs on the current system" OFF)
option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

message(STATUS "Build with NVTX support: ${USE_NVTX}")
message(STATUS "Configure CMake to build tests: ${BUILD_TESTS}")
message(STATUS "Configure CMake to build (google) benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "Build cuDF shared libraries: ${BUILD_SHARED_LIBS}")
message(STATUS "Use a file cache for JIT compiled kernels: ${JITIFY_USE_CACHE}")
message(STATUS "Build and statically link Arrow libraries: ${ARROW_STATIC_LIB}")
message(STATUS "Use the default allocator for hash tables: ${HT_DEFAULT_ALLOCATOR}")
message(STATUS "Build with per-thread default stream: ${PER_THREAD_DEFAULT_STREAM}")
message(STATUS "Auto-detect CUDA architectures to build from the GPUs on the current system: ${AUTO_DETECT_CUDA_ARCHITECTURES}")
message(STATUS "Disable warnings generated from deprecated declarations: ${DISABLE_DEPRECATION_WARNING}")
message(STATUS "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler: ${CMAKE_CUDA_LINEINFO}")
message(STATUS "Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")
set(CUDF_BUILD_TESTS ${BUILD_TESTS})
set(CUDF_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})

set(CUDF_CXX_FLAGS "")
set(CUDF_CUDA_FLAGS "")
set(CUDF_CXX_DEFINITIONS "")
set(CUDF_CUDA_DEFINITIONS "")

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

if (NOT CUDF_GENERATED_INCLUDE_DIR)
    set(CUDF_GENERATED_INCLUDE_DIR ${CUDA_DATAFRAME_BINARY_DIR})
endif()

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

set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})

# find CUDAToolkit package, set CUDA compilation flags
include(cmake/Modules/ConfigureCUDA.cmake)

###################################################################################################
# - dependencies ----------------------------------------------------------------------------------

# find zlib
find_package(ZLIB REQUIRED)
# add third party dependencies using CPM
include(cmake/thirdparty/CUDF_GetCPM.cmake)
# find boost
include(cmake/thirdparty/CUDF_FindBoost.cmake)
# find jitify
include(cmake/thirdparty/CUDF_GetJitify.cmake)
# find thrust/cub
include(cmake/thirdparty/CUDF_GetThrust.cmake)
# find rmm
include(cmake/thirdparty/CUDF_GetRMM.cmake)
# find arrow
include(cmake/thirdparty/CUDF_GetArrow.cmake)
# find dlpack
include(cmake/thirdparty/CUDF_GetDLPack.cmake)
# find libcu++
include(cmake/thirdparty/CUDF_GetLibcudacxx.cmake)
# Stringify libcudf and libcudacxx headers used in JIT operations
include(cmake/Modules/StringifyJITHeaders.cmake)

###################################################################################################
# - add tests -------------------------------------------------------------------------------------

if(CUDF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "CUDF_TEST_LIST set to: ${CUDF_TEST_LIST}")

###################################################################################################
# - add benchmarks --------------------------------------------------------------------------------

if(CUDF_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

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
    message(STATUS "Adding library module ${NAMESPACE}_${MODULE} ...")
    message(TRACE "Using sources: ${MODULE_SRC}")
    add_library("${NAMESPACE}_${MODULE}" "${MODULE_SRC}")
    set_target_properties("${NAMESPACE}_${MODULE}"
        PROPERTIES BUILD_RPATH                         "\$ORIGIN"
                   # set target compile options
                   CXX_STANDARD                        14
                   CXX_STANDARD_REQUIRED               ON
                   CUDA_STANDARD                       14
                   CUDA_STANDARD_REQUIRED              ON
                   POSITION_INDEPENDENT_CODE           ON
                   INTERFACE_POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options("${NAMESPACE}_${MODULE}"
                    PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                           "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
    )
    target_compile_definitions("${NAMESPACE}_${MODULE}"
                    PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_DEFINITIONS}>"
                           "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_DEFINITIONS}>"
    )
    # Disable Jitify log printing. See https://github.com/NVIDIA/jitify/issues/79
    target_compile_definitions("${NAMESPACE}_${MODULE}" PRIVATE "JITIFY_PRINT_LOG=0")
    # Specify include paths for the current target and dependents
    target_include_directories("${NAMESPACE}_${MODULE}"
                   PUBLIC      "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>"
                               "$<BUILD_INTERFACE:${JITIFY_INCLUDE_DIR}>"
                               "$<BUILD_INTERFACE:${LIBCUDACXX_INCLUDE_DIR}>"
                               "$<BUILD_INTERFACE:${CUDA_DATAFRAME_SOURCE_DIR}/include>"
                               "$<BUILD_INTERFACE:${CUDF_GENERATED_INCLUDE_DIR}/include>"
                   PRIVATE     "$<BUILD_INTERFACE:${CUDA_DATAFRAME_SOURCE_DIR}/src>"
                   INTERFACE   "$<INSTALL_INTERFACE:include>")
    # Add Conda library paths if specified
    if(CONDA_LINK_DIRS)
        target_link_directories("${NAMESPACE}_${MODULE}" PUBLIC "$<BUILD_INTERFACE:${CONDA_LINK_DIRS}>")
    endif()
    # Add Conda include paths if specified
    if(CONDA_INCLUDE_DIRS)
        target_include_directories("${NAMESPACE}_${MODULE}" PUBLIC "$<BUILD_INTERFACE:${CONDA_INCLUDE_DIRS}>")
    endif()
    # Instruct jitify to use the kernel JIT cache
    if(JITIFY_USE_CACHE)
        target_compile_definitions("${NAMESPACE}_${MODULE}" PUBLIC JITIFY_USE_CACHE "CUDF_VERSION=${CMAKE_PROJECT_VERSION}")
    endif()
    # Set hash table allocator
    if(HT_DEFAULT_ALLOCATOR)
        target_compile_definitions("${NAMESPACE}_${MODULE}" PUBLIC HT_DEFAULT_ALLOCATOR)
    endif()
    # Per-thread default stream
    if(PER_THREAD_DEFAULT_STREAM)
        target_compile_definitions("${NAMESPACE}_${MODULE}" PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
    endif()
    # Disable NVTX if necessary
    if(NOT USE_NVTX)
        target_compile_definitions("${NAMESPACE}_${MODULE}" PUBLIC NVTX_DISABLE)
    endif()
    # Define spdlog level
    target_compile_definitions("${NAMESPACE}_${MODULE}" PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${LOGGING_LEVEL}")
    # Compile stringified JIT sources first
    add_dependencies("${NAMESPACE}_${MODULE}" stringify_run)
    # Specify the target module library dependencies
    target_link_libraries("${NAMESPACE}_${MODULE}"
                   PUBLIC ZLIB::ZLIB
                          Boost::filesystem
                          ${ARROW_LIBRARIES}
                          cudf::Thrust
                          rmm::rmm)
    if(CUDA_STATIC_RUNTIME)
        target_link_libraries("${NAMESPACE}_${MODULE}" PUBLIC CUDA::nvrtc CUDA::cudart_static CUDA::cuda_driver)
    else()
        target_link_libraries("${NAMESPACE}_${MODULE}" PUBLIC CUDA::nvrtc CUDA::cudart CUDA::cuda_driver)
    endif()
    # `-Wl,--disable-new-dtags`: apply RUNPATH to transitive dependencies (this is a temporary solution)
    target_link_options("${NAMESPACE}_${MODULE}" PRIVATE "LINKER:--disable-new-dtags")
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
# `-Wl,--disable-new-dtags`: apply RUNPATH to transitive dependencies (this is a temporary solution)
target_link_options("${CUDF_PROXY_NAME}" PRIVATE "LINKER:--no-as-needed,--disable-new-dtags")
set_target_properties("${CUDF_PROXY_NAME}"
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        14
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       14
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)
add_library("${CUDF_PROXY_NAME}::${CUDF_PROXY_NAME}" ALIAS "${CUDF_PROXY_NAME}")
# Link the proxy libcudf.so to libcudf_base.so and each `libcudf_<module>.so`
target_link_libraries("${CUDF_PROXY_NAME}" PUBLIC "${CUDF_NAMESPACE}::${CUDF_BASE_NAME}" "${CUDF_MODULES}")

###################################################################################################
# - install targets -------------------------------------------------------------------------------

include(GNUInstallDirs)

set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/cudf)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME cudf)

# install target for cudf_base
install(TARGETS "${CUDF_NAMESPACE}_${CUDF_BASE_NAME}"
        DESTINATION lib
        EXPORT cudf-targets)

# install target for each cudf_<module>
install(TARGETS ${CUDF_MODULES}
        DESTINATION lib
        EXPORT cudf-targets)

# install target for the proxy libcudf.so
install(TARGETS ${CUDF_PROXY_NAME}
        DESTINATION lib
        EXPORT cudf-targets)

install(DIRECTORY ${CUDA_DATAFRAME_SOURCE_DIR}/include/cudf
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY ${CUDA_DATAFRAME_SOURCE_DIR}/include/cudf_test
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY ${CUDF_GENERATED_INCLUDE_DIR}/include/libcxx
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libcudf)

install(DIRECTORY ${CUDF_GENERATED_INCLUDE_DIR}/include/libcudacxx
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libcudf)

add_custom_target(install_cudf
                  COMMAND "${CMAKE_COMMAND}" -DCOMPONENT=cudf -P "${CUDA_DATAFRAME_BINARY_DIR}/cmake_install.cmake"
                  DEPENDS cudf)

if(CUDF_BUILD_TESTS)
    add_dependencies(install_cudf cudftestutil)
endif()

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
