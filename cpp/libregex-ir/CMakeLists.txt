# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.23)

project(
  regex_ir
  VERSION 0.1.0
  LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(REGEX_IR_BUILD_TESTS "Build Regex IR tests" ON)
option(REGEX_IR_BUILD_EXAMPLES "Build Regex IR examples" ON)
option(REGEX_IR_BUILD_BENCHMARKS "Build the Regex IR interpreter/RE2 CPU comparison" OFF)
option(REGEX_IR_BUILD_GPU_BENCHMARKS "Build the cuDF/NVVM GPU comparison" OFF)
set(REGEX_IR_CUDF_ROOT
    ""
    CACHE PATH "cuDF installation prefix for GPU benchmarks"
)
set(REGEX_IR_BENCHMARK_CORPUS_DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/benchmark-corpora"
    CACHE PATH "Directory for verified external benchmark corpora"
)
option(REGEX_IR_BUILD_EXPLORER "Build the interactive regex-to-IR explorer" ON)
option(REGEX_IR_BUILD_CODEGEN "Build the NVVM IR generator command" ON)
option(REGEX_IR_BUILD_FUZZERS "Build Clang libFuzzer targets" OFF)
option(REGEX_IR_BUILD_DOCS "Build Doxygen and Sphinx documentation" OFF)

if(REGEX_IR_BUILD_GPU_BENCHMARKS AND CMAKE_VERSION VERSION_LESS 4.0)
  message(FATAL_ERROR "GPU benchmarks require CMake 4.0 or newer for NVBench")
endif()

if(REGEX_IR_BUILD_GPU_BENCHMARKS)
  find_path(
    REGEX_IR_FMT_INCLUDE_DIR
    NAMES fmt/format.h
    HINTS ${REGEX_IR_CUDF_ROOT}/include REQUIRED
  )
  find_library(
    REGEX_IR_FMT_LIBRARY
    NAMES fmt
    HINTS ${REGEX_IR_CUDF_ROOT}/lib ${REGEX_IR_CUDF_ROOT}/lib64 REQUIRED
  )
else()
  find_package(fmt CONFIG REQUIRED)
endif()

add_library(regex_ir regex_ir.cpp)
add_library(regex_ir::regex_ir ALIAS regex_ir)

target_compile_features(regex_ir PUBLIC cxx_std_20)
if(REGEX_IR_BUILD_GPU_BENCHMARKS)
  target_include_directories(regex_ir PRIVATE ${REGEX_IR_FMT_INCLUDE_DIR})
  target_link_libraries(regex_ir PRIVATE ${REGEX_IR_FMT_LIBRARY})
else()
  target_link_libraries(regex_ir PRIVATE fmt::fmt)
endif()
target_include_directories(
  regex_ir PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(
  regex_ir
  PROPERTIES CXX_EXTENSIONS OFF
             EXPORT_NAME regex_ir
             VERSION ${PROJECT_VERSION}
             SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(regex_ir PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

install(
  TARGETS regex_ir
  EXPORT regex_ir-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES regex_ir.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  EXPORT regex_ir-targets
  FILE regex_ir-targets.cmake
  NAMESPACE regex_ir::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/regex_ir
)

configure_package_config_file(
  cmake/regex_ir-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/regex_ir-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/regex_ir
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/regex_ir-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/regex_ir-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/regex_ir-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/regex_ir
)

if(REGEX_IR_BUILD_EXPLORER)
  add_executable(regex_ir_explorer tools/explorer.cpp)
  target_link_libraries(regex_ir_explorer PRIVATE regex_ir::regex_ir)
  if(REGEX_IR_BUILD_GPU_BENCHMARKS)
    target_include_directories(regex_ir_explorer PRIVATE ${REGEX_IR_FMT_INCLUDE_DIR})
    target_link_libraries(regex_ir_explorer PRIVATE ${REGEX_IR_FMT_LIBRARY})
  else()
    target_link_libraries(regex_ir_explorer PRIVATE fmt::fmt)
  endif()
  target_compile_features(regex_ir_explorer PRIVATE cxx_std_20)
  set_target_properties(regex_ir_explorer PROPERTIES OUTPUT_NAME regex-ir-explorer)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(regex_ir_explorer PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
  install(TARGETS regex_ir_explorer RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(REGEX_IR_BUILD_CODEGEN)
  add_executable(regex_ir_codegen tools/codegen.cpp)
  target_link_libraries(regex_ir_codegen PRIVATE regex_ir::regex_ir)
  target_compile_features(regex_ir_codegen PRIVATE cxx_std_20)
  set_target_properties(regex_ir_codegen PROPERTIES OUTPUT_NAME regex-ir-codegen)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(regex_ir_codegen PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
  install(TARGETS regex_ir_codegen RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

include(CTest)
if(BUILD_TESTING AND REGEX_IR_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(REGEX_IR_BUILD_EXAMPLES)
  add_executable(regex_ir_example examples/compile.cpp)
  target_link_libraries(regex_ir_example PRIVATE regex_ir::regex_ir)
  target_compile_features(regex_ir_example PRIVATE cxx_std_20)
endif()

if(REGEX_IR_BUILD_BENCHMARKS)
  find_path(REGEX_IR_RE2_INCLUDE_DIR NAMES re2/re2.h REQUIRED)
  find_library(REGEX_IR_RE2_LIBRARY NAMES re2 REQUIRED)
  add_executable(regex_ir_cpu_benchmark benchmarks/cpu_benchmark.cpp)
  target_include_directories(regex_ir_cpu_benchmark SYSTEM PRIVATE ${REGEX_IR_RE2_INCLUDE_DIR})
  target_link_libraries(regex_ir_cpu_benchmark PRIVATE regex_ir::regex_ir ${REGEX_IR_RE2_LIBRARY})
  target_compile_features(regex_ir_cpu_benchmark PRIVATE cxx_std_20)
  set_target_properties(regex_ir_cpu_benchmark PROPERTIES OUTPUT_NAME regex-ir-cpu-benchmark)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
      regex_ir_cpu_benchmark PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
  endif()
endif()

if(REGEX_IR_BUILD_GPU_BENCHMARKS)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  include(${PROJECT_SOURCE_DIR}/cmake/EmbedCudaLto.cmake)
  find_package(nvbench CONFIG QUIET)
  if(NOT TARGET nvbench::main)
    include(FetchContent)
    set(NVBench_ENABLE_NVML
        OFF
        CACHE BOOL "" FORCE
    )
    set(NVBench_ENABLE_CUPTI
        OFF
        CACHE BOOL "" FORCE
    )
    set(NVBench_ENABLE_TESTING
        OFF
        CACHE BOOL "" FORCE
    )
    set(NVBench_ENABLE_HEADER_TESTING
        OFF
        CACHE BOOL "" FORCE
    )
    set(NVBench_ENABLE_DEVICE_TESTING
        OFF
        CACHE BOOL "" FORCE
    )
    set(NVBench_ENABLE_EXAMPLES
        OFF
        CACHE BOOL "" FORCE
    )
    set(NVBench_ENABLE_INSTALL_RULES
        OFF
        CACHE BOOL "" FORCE
    )
    FetchContent_Declare(
      nvbench
      GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
      GIT_TAG b88a45f4170af4e907e69af22a55af67859d3b49
    )
    FetchContent_MakeAvailable(nvbench)
  endif()
  find_path(
    REGEX_IR_CUDF_INCLUDE_DIR
    NAMES cudf/strings/contains.hpp
    HINTS ${REGEX_IR_CUDF_ROOT}/include
  )
  find_library(
    REGEX_IR_CUDF_LIBRARY
    NAMES cudf
    HINTS ${REGEX_IR_CUDF_ROOT}/lib ${REGEX_IR_CUDF_ROOT}/lib64
  )
  find_library(
    REGEX_IR_RMM_LIBRARY
    NAMES rmm
    HINTS ${REGEX_IR_CUDF_ROOT}/lib ${REGEX_IR_CUDF_ROOT}/lib64
  )
  find_path(
    REGEX_IR_CCCL_INCLUDE_DIR
    NAMES cuda/std/iterator
    HINTS ${REGEX_IR_CUDF_ROOT}/include/rapids
          ${REGEX_IR_CUDF_ROOT}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/include
          ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES cccl
  )
  find_path(
    REGEX_IR_NVVM_INCLUDE_DIR
    NAMES nvvm.h
    HINTS ${CUDAToolkit_LIBRARY_ROOT}/nvvm/include
  )
  find_library(
    REGEX_IR_NVVM_LIBRARY
    NAMES nvvm
    HINTS ${CUDAToolkit_LIBRARY_ROOT}/nvvm/lib64
  )
  find_path(
    REGEX_IR_NVJITLINK_INCLUDE_DIR
    NAMES nvJitLink.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
  )
  find_library(
    REGEX_IR_NVJITLINK_LIBRARY
    NAMES nvJitLink
    HINTS ${CUDAToolkit_LIBRARY_DIR}
          ${CUDAToolkit_LIBRARY_ROOT}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib
  )
  if(NOT REGEX_IR_CUDF_INCLUDE_DIR
     OR NOT REGEX_IR_CUDF_LIBRARY
     OR NOT REGEX_IR_RMM_LIBRARY
     OR NOT REGEX_IR_CCCL_INCLUDE_DIR
  )
    message(FATAL_ERROR "REGEX_IR_BUILD_GPU_BENCHMARKS requires cuDF; set REGEX_IR_CUDF_ROOT")
  endif()
  if(NOT REGEX_IR_NVVM_INCLUDE_DIR
     OR NOT REGEX_IR_NVVM_LIBRARY
     OR NOT REGEX_IR_NVJITLINK_INCLUDE_DIR
     OR NOT REGEX_IR_NVJITLINK_LIBRARY
  )
    message(FATAL_ERROR "GPU benchmarks require libNVVM and nvJitLink")
  endif()

  set(REGEX_IR_BENCHMARK_FRAGMENT_DIRECTORY ${PROJECT_SOURCE_DIR}/benchmarks/fragments)
  set(REGEX_IR_BENCHMARK_KERNEL_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)
  regex_ir_embed_cuda_lto(
    REGEX_IR_BENCHMARK_CONTAINS_KERNEL
    regex_ir_benchmark_contains
    ${REGEX_IR_BENCHMARK_FRAGMENT_DIRECTORY}/regex_ir_contains_kernel.cu
    regex_ir_benchmark_contains_fatbin
  )
  regex_ir_embed_cuda_lto(
    REGEX_IR_BENCHMARK_EXTRACT_KERNEL
    regex_ir_benchmark_extract
    ${REGEX_IR_BENCHMARK_FRAGMENT_DIRECTORY}/regex_ir_extract_kernel.cu
    regex_ir_benchmark_extract_fatbin
  )
  regex_ir_embed_cuda_lto(
    REGEX_IR_BENCHMARK_COUNT_KERNEL
    regex_ir_benchmark_count
    ${REGEX_IR_BENCHMARK_FRAGMENT_DIRECTORY}/regex_ir_count_kernel.cu
    regex_ir_benchmark_count_fatbin
  )
  regex_ir_embed_cuda_lto(
    REGEX_IR_BENCHMARK_REPLACE_KERNEL
    regex_ir_benchmark_replace
    ${REGEX_IR_BENCHMARK_FRAGMENT_DIRECTORY}/regex_ir_replace_kernel.cu
    regex_ir_benchmark_replace_fatbin
  )
  regex_ir_embed_cuda_lto(
    REGEX_IR_BENCHMARK_SPLIT_KERNEL
    regex_ir_benchmark_split
    ${REGEX_IR_BENCHMARK_FRAGMENT_DIRECTORY}/regex_ir_split_kernel.cu
    regex_ir_benchmark_split_fatbin
  )
  add_custom_target(
    regex_ir_benchmark_kernels
    COMMENT "Compiling and embedding Regex IR benchmark kernels"
    DEPENDS ${REGEX_IR_BENCHMARK_CONTAINS_KERNEL}
            ${REGEX_IR_BENCHMARK_EXTRACT_KERNEL}
            ${REGEX_IR_BENCHMARK_COUNT_KERNEL}
            ${REGEX_IR_BENCHMARK_REPLACE_KERNEL}
            ${REGEX_IR_BENCHMARK_SPLIT_KERNEL}
  )
  add_custom_target(
    regex_ir_benchmark_corpora
    COMMAND ${CMAKE_COMMAND} -DOUTPUT_DIRECTORY=${REGEX_IR_BENCHMARK_CORPUS_DIRECTORY} -P
            ${PROJECT_SOURCE_DIR}/cmake/FetchBenchmarkCorpora.cmake
    COMMENT "Verifying complete external benchmark corpora"
    VERBATIM
  )

  # Add an NVBench executable that links Regex IR, cuDF, and the generated CUDA fragments.
  function(regex_ir_add_gpu_benchmark target output_name)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
    add_executable(${target} ${ARGN})
    add_dependencies(${target} regex_ir_benchmark_kernels)
    target_link_libraries(
      ${target}
      PRIVATE regex_ir::regex_ir
              ${REGEX_IR_FMT_LIBRARY}
              ${REGEX_IR_CUDF_LIBRARY}
              ${REGEX_IR_RMM_LIBRARY}
              ${REGEX_IR_NVVM_LIBRARY}
              ${REGEX_IR_NVJITLINK_LIBRARY}
              nvbench::main
              CUDA::cuda_driver
              CUDA::cudart
              ${CMAKE_DL_LIBS}
    )
    target_include_directories(
      ${target} PRIVATE ${REGEX_IR_BENCHMARK_KERNEL_DIRECTORY} ${REGEX_IR_FMT_INCLUDE_DIR}
    )
    target_include_directories(
      ${target} SYSTEM PRIVATE ${REGEX_IR_CUDF_INCLUDE_DIR} ${REGEX_IR_CCCL_INCLUDE_DIR}
                               ${REGEX_IR_NVVM_INCLUDE_DIR} ${REGEX_IR_NVJITLINK_INCLUDE_DIR}
    )
    target_compile_features(${target} PRIVATE cxx_std_20 cuda_std_20)
    set_target_properties(
      ${target}
      PROPERTIES OUTPUT_NAME ${output_name}
                 CUDA_EXTENSIONS OFF
                 BUILD_RPATH "${REGEX_IR_CUDF_ROOT}/lib;${CUDAToolkit_LIBRARY_ROOT}/nvvm/lib64"
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(
        ${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>
      )
    endif()
  endfunction()

  regex_ir_add_gpu_benchmark(
    regex_ir_gpu_benchmark regex-ir-gpu-benchmark benchmarks/gpu_benchmark.cpp
  )
  regex_ir_add_gpu_benchmark(
    regex_ir_corpus_benchmark
    regex-ir-corpus-benchmark
    benchmarks/openresty_benchmark.cpp
    benchmarks/leipzig_benchmark.cpp
    benchmarks/boost_benchmark.cpp
    benchmarks/mariomka_benchmark.cpp
  )
  add_dependencies(regex_ir_corpus_benchmark regex_ir_benchmark_corpora)
  target_compile_definitions(
    regex_ir_corpus_benchmark
    PRIVATE REGEX_IR_BENCHMARK_CORPUS_DIRECTORY="${REGEX_IR_BENCHMARK_CORPUS_DIRECTORY}"
  )
endif()

if(REGEX_IR_BUILD_FUZZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "REGEX_IR_BUILD_FUZZERS requires Clang")
  endif()
  add_executable(regex_ir_compile_execute_fuzzer tests/fuzz_tests.cpp)
  target_link_libraries(regex_ir_compile_execute_fuzzer PRIVATE regex_ir::regex_ir)
  target_compile_features(regex_ir_compile_execute_fuzzer PRIVATE cxx_std_20)
  target_compile_options(
    regex_ir_compile_execute_fuzzer PRIVATE -fsanitize=fuzzer,address,undefined
                                            -fno-omit-frame-pointer
  )
  target_link_options(
    regex_ir_compile_execute_fuzzer PRIVATE -fsanitize=fuzzer,address,undefined
    -fno-omit-frame-pointer
  )
endif()

if(REGEX_IR_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_custom_target(
    regex_ir_docs
    COMMAND ${CMAKE_COMMAND} -E make_directory docs/build/doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} docs/Doxyfile
    COMMAND ${Python3_EXECUTABLE} -m sphinx -n -W --keep-going -b html docs docs/build/html
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Regex IR Doxygen and Sphinx documentation"
    VERBATIM
  )
endif()
