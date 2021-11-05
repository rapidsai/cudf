# =============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-21.12/RAPIDS.cmake
     ${CMAKE_BINARY_DIR}/RAPIDS.cmake
)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUDF)

project(
  CUDF
  VERSION 21.12.00
  LANGUAGES C CXX CUDA
)

# Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to
# have different values for the `Threads::Threads` target. Setting this flag ensures
# `Threads::Threads` is the same value in first run and subsequent runs.
set(THREADS_PREFER_PTHREAD_FLAG ON)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google & nvbench) benchmarks" OFF)
option(BUILD_SHARED_LIBS "Build cuDF shared libraries" ON)
option(JITIFY_USE_CACHE "Use a file cache for JIT compiled kernels" ON)
option(CUDF_USE_ARROW_STATIC "Build and statically link Arrow libraries" OFF)
option(CUDF_ENABLE_ARROW_ORC "Build the Arrow ORC adapter" OFF)
option(CUDF_ENABLE_ARROW_PYTHON "Find (or build) Arrow with Python support" OFF)
option(CUDF_ENABLE_ARROW_PARQUET "Find (or build) Arrow with Parquet support" OFF)
option(CUDF_ENABLE_ARROW_S3 "Build/Enable AWS S3 Arrow filesystem support" ON)
option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
option(CUDA_ENABLE_LINEINFO
       "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF
)
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

message(VERBOSE "CUDF: Build with NVTX support: ${USE_NVTX}")
message(VERBOSE "CUDF: Configure CMake to build tests: ${BUILD_TESTS}")
message(VERBOSE "CUDF: Configure CMake to build (google & nvbench) benchmarks: ${BUILD_BENCHMARKS}")
message(VERBOSE "CUDF: Build cuDF shared libraries: ${BUILD_SHARED_LIBS}")
message(VERBOSE "CUDF: Use a file cache for JIT compiled kernels: ${JITIFY_USE_CACHE}")
message(VERBOSE "CUDF: Build and statically link Arrow libraries: ${CUDF_USE_ARROW_STATIC}")
message(VERBOSE "CUDF: Build and enable S3 filesystem support for Arrow: ${CUDF_ENABLE_ARROW_S3}")
message(VERBOSE "CUDF: Build with per-thread default stream: ${PER_THREAD_DEFAULT_STREAM}")
message(
  VERBOSE
  "CUDF: Disable warnings generated from deprecated declarations: ${DISABLE_DEPRECATION_WARNING}"
)
message(
  VERBOSE
  "CUDF: Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler: ${CUDA_ENABLE_LINEINFO}"
)
message(VERBOSE "CUDF: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")

# Set a default build type if none was specified
rapids_cmake_build_type("Release")
set(CUDF_BUILD_TESTS ${BUILD_TESTS})
set(CUDF_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})

set(CUDF_CXX_FLAGS "")
set(CUDF_CUDA_FLAGS "")
set(CUDF_CXX_DEFINITIONS "")
set(CUDF_CUDA_DEFINITIONS "")

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
)
message(VERBOSE "CUDF: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

if(NOT CUDF_GENERATED_INCLUDE_DIR)
  set(CUDF_GENERATED_INCLUDE_DIR ${CUDF_BINARY_DIR})
endif()

# ##################################################################################################
# * conda environment -----------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET cudf-exports
  INSTALL_EXPORT_SET cudf-exports
)
include(cmake/Modules/ConfigureCUDA.cmake) # set other CUDA compilation flags

# ctest cuda memcheck
find_program(CUDA_SANITIZER compute-sanitizer)
set(MEMORYCHECK_COMMAND ${CUDA_SANITIZER})
set(MEMORYCHECK_TYPE CudaSanitizer)
set(CUDA_SANITIZER_COMMAND_OPTIONS "--tool memcheck")

# ##################################################################################################
# * dependencies ----------------------------------------------------------------------------------

# find zlib
rapids_find_package(ZLIB REQUIRED)

# find Threads (needed by cudftestutil)
rapids_find_package(
  Threads REQUIRED
  BUILD_EXPORT_SET cudf-exports
  INSTALL_EXPORT_SET cudf-exports
)

# add third party dependencies using CPM
rapids_cpm_init()
# find jitify
include(cmake/thirdparty/get_jitify.cmake)
# find nvCOMP
include(cmake/thirdparty/get_nvcomp.cmake)
# find thrust/cub
include(cmake/thirdparty/get_thrust.cmake)
# find rmm
include(cmake/thirdparty/get_rmm.cmake)
# find arrow
include(cmake/thirdparty/get_arrow.cmake)
# find dlpack
include(cmake/thirdparty/get_dlpack.cmake)
# find libcu++
include(cmake/thirdparty/get_libcudacxx.cmake)
# find cuCollections Should come after including thrust and libcudacxx
include(cmake/thirdparty/get_cucollections.cmake)
# find or install GoogleTest
include(cmake/thirdparty/get_gtest.cmake)
# preprocess jitify-able kernels
include(cmake/Modules/JitifyPreprocessKernels.cmake)
# find cuFile
include(cmake/Modules/FindcuFile.cmake)

# ##################################################################################################
# * library targets -------------------------------------------------------------------------------

add_library(
  cudf
  src/groupby/test.cu
)

set_target_properties(
  cudf
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  cudf PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
               "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

# Specify include paths for the current target and dependents
target_include_directories(
  cudf
  PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>"
         "$<BUILD_INTERFACE:${JITIFY_INCLUDE_DIR}>"
         "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${CUDF_GENERATED_INCLUDE_DIR}/include>"
  PRIVATE "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
  INTERFACE "$<INSTALL_INTERFACE:include>"
)

target_compile_definitions(
  cudf PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_DEFINITIONS}>"
              "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_DEFINITIONS}>>"
)

# Disable Jitify log printing. See https://github.com/NVIDIA/jitify/issues/79
target_compile_definitions(cudf PRIVATE "JITIFY_PRINT_LOG=0")

# Instruct jitify to use the kernel JIT cache
if(JITIFY_USE_CACHE)
  target_compile_definitions(cudf PUBLIC JITIFY_USE_CACHE "CUDF_VERSION=${PROJECT_VERSION}")
endif()

# Per-thread default stream
if(PER_THREAD_DEFAULT_STREAM)
  target_compile_definitions(cudf PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif()

# Disable NVTX if necessary
if(NOT USE_NVTX)
  target_compile_definitions(cudf PUBLIC NVTX_DISABLE)
endif()

# Define spdlog level
target_compile_definitions(cudf PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL}")

# Compile stringified JIT sources first
add_dependencies(cudf jitify_preprocess_run)

# Specify the target module library dependencies
target_link_libraries(
  cudf
  PUBLIC ${ARROW_LIBRARIES} libcudacxx::libcudacxx cudf::Thrust rmm::rmm
  PRIVATE cuco::cuco ZLIB::ZLIB nvcomp::nvcomp
)

# Add Conda library, and include paths if specified
if(TARGET conda_env)
  target_link_libraries(cudf PRIVATE conda_env)
endif()

# Add cuFile interface if available
if(TARGET cuFile::cuFile_interface)
  target_link_libraries(cudf PRIVATE cuFile::cuFile_interface)
endif()

if(CUDA_STATIC_RUNTIME)
  # Tell CMake what CUDA language runtime to use
  set_target_properties(cudf PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  # Make sure to export to consumers what runtime we used
  target_link_libraries(cudf PUBLIC CUDA::cudart_static)
else()
  # Tell CMake what CUDA language runtime to use
  set_target_properties(cudf PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  # Make sure to export to consumers what runtime we used
  target_link_libraries(cudf PUBLIC CUDA::cudart)
endif()

# The CUDA::cuda_driver is needed due to JITIFY sources which directly call the cuda driver API
if(NOT TARGET CUDA::cuda_driver)
  message(
    FATAL_ERROR
      "Building libcudf requires `libcuda.so` to be present.
        This error often occurs when trying to build libcudf from a container without the NVIDIA runtime loaded."
  )
endif()
target_link_libraries(cudf PUBLIC CUDA::cuda_driver)

file(
  WRITE "${CUDF_BINARY_DIR}/fatbin.ld"
  [=[
SECTIONS
{
  .nvFatBinSegment : { *(.nvFatBinSegment) }
  .nv_fatbin : { *(.nv_fatbin) }
}
]=]
)
target_link_options(cudf PRIVATE "$<HOST_LINK:${CUDF_BINARY_DIR}/fatbin.ld>")

add_library(cudf::cudf ALIAS cudf)

# ##################################################################################################
# * tests and benchmarks --------------------------------------------------------------------------
# ##################################################################################################

# ##################################################################################################
# * build cudftestutil ----------------------------------------------------------------------------

add_library(
  cudftestutil STATIC
  tests/utilities/base_fixture.cpp tests/utilities/column_utilities.cu
  tests/utilities/table_utilities.cu tests/io/metadata_utilities.cpp tests/strings/utilities.cpp
)

set_target_properties(
  cudftestutil
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  cudftestutil PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                      "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>>"
)

target_link_libraries(cudftestutil PUBLIC GTest::gmock GTest::gtest Threads::Threads cudf)

target_include_directories(
  cudftestutil PUBLIC "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}>"
                      "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
)

add_library(cudf::cudftestutil ALIAS cudftestutil)

# ##################################################################################################
# * add tests -------------------------------------------------------------------------------------

if(CUDF_BUILD_TESTS)
  # include CTest module -- automatically calls enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * add benchmarks --------------------------------------------------------------------------------

if(CUDF_BUILD_BENCHMARKS)
  # Find or install GoogleBench
  rapids_cpm_find(
    benchmark 1.5.2
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.5.2
    GIT_SHALLOW TRUE
    OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "BENCHMARK_ENABLE_INSTALL OFF"
  )

  # Find or install NVBench
  include(${rapids-cmake-dir}/cpm/nvbench.cmake)
  rapids_cpm_nvbench()
  add_subdirectory(benchmarks)
endif()

# ##################################################################################################
# * install targets -------------------------------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
include(CPack)
include(GNUInstallDirs)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME cudf)

# install target for cudf_base and the proxy libcudf.so
install(
  TARGETS cudf
  DESTINATION ${lib_dir}
  EXPORT cudf-exports
)

install(DIRECTORY ${CUDF_SOURCE_DIR}/include/cudf ${CUDF_SOURCE_DIR}/include/cudf_test
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  TARGETS cudftestutil
  DESTINATION ${lib_dir}
  EXPORT cudf-testing-exports
)

install(
  EXPORT cudf-testing-exports
  FILE cudf-testing-targets.cmake
  NAMESPACE cudf::
  DESTINATION "${lib_dir}/cmake/cudf"
)

include("${rapids-cmake-dir}/export/write_dependencies.cmake")
rapids_export_write_dependencies(
  INSTALL cudf-testing-exports
  "${PROJECT_BINARY_DIR}/rapids-cmake/cudf/export/cudf-testing-dependencies.cmake"
)

set(doc_string
    [=[
Provide targets for the cudf library.

Built based on the Apache Arrow columnar memory format, cuDF is a GPU DataFrame
library for loading, joining, aggregating, filtering, and otherwise
manipulating data.

cuDF provides a pandas-like API that will be familiar to data engineers &
data scientists, so they can use it to easily accelerate their workflows
without going into the details of CUDA programming.


Imported Targets
^^^^^^^^^^^^^^^^

If cudf is found, this module defines the following IMPORTED GLOBAL
targets:

 cudf::cudf             - The main cudf library.

This module offers an optional testing component which defines the
following IMPORTED GLOBAL  targets:

 cudf::cudftestutil     - The main cudf testing library
    ]=]
)

set(common_code_string
    [=[
if(NOT TARGET cudf::Thrust)
  thrust_create_target(cudf::Thrust FROM_OPTIONS)
endif()
]=]
)

set(install_code_string
    [=[
set(ArrowCUDA_DIR "${Arrow_DIR}")
find_dependency(ArrowCUDA)
if(testing IN_LIST cudf_FIND_COMPONENTS)
  enable_language(CUDA)
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cudf-testing-dependencies.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/cudf-testing-dependencies.cmake")
  endif()
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cudf-testing-targets.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/cudf-testing-targets.cmake")
  endif()
endif()
]=]
)
string(APPEND install_code_string "${common_code_string}")

rapids_export(
  INSTALL cudf
  EXPORT_SET cudf-exports
  GLOBAL_TARGETS cudf
  NAMESPACE cudf::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK install_code_string
)

# ##################################################################################################
# * build export -------------------------------------------------------------------------------
set(build_code_string
    [=[
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cudf-testing-dependencies.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/cudf-testing-dependencies.cmake")
endif()
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cudf-testing-targets.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/cudf-testing-targets.cmake")
endif()
]=]
)
string(APPEND build_code_string "${common_code_string}")

rapids_export(
  BUILD cudf
  EXPORT_SET cudf-exports
  GLOBAL_TARGETS cudf
  NAMESPACE cudf::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK build_code_string
)

export(
  EXPORT cudf-testing-exports
  FILE ${CUDF_BINARY_DIR}/cudf-testing-targets.cmake
  NAMESPACE cudf::
)
rapids_export_write_dependencies(
  BUILD cudf-testing-exports "${CUDF_BINARY_DIR}/cudf-testing-dependencies.cmake"
)

# ##################################################################################################
# * make documentation ----------------------------------------------------------------------------

# doc targets for cuDF
add_custom_command(
  OUTPUT CUDF_DOXYGEN
  WORKING_DIRECTORY ${CUDF_SOURCE_DIR}/doxygen
  COMMAND doxygen Doxyfile
  VERBATIM
  COMMENT "Custom command for building cudf doxygen docs."
)

add_custom_target(
  docs_cudf
  DEPENDS CUDF_DOXYGEN
  COMMENT "Custom command for building cudf doxygen docs."
)
