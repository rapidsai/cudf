# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(string_transforms_examples)

project(
  string_transforms_examples
  VERSION 0.0.1
  LANGUAGES CXX CUDA ASM
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

# Build and embed the AOT CUDA fragments consumed by cudf::transform_lto.
find_package(zstd CONFIG REQUIRED)
find_package(bs_thread_pool CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(nvcomp CONFIG REQUIRED)
find_package(kvikio CONFIG REQUIRED)
if(NOT TARGET cuco::cuco)
  add_library(cuco::cuco INTERFACE IMPORTED)
endif()
if(NOT TARGET nanoarrow::nanoarrow)
  add_library(nanoarrow::nanoarrow INTERFACE IMPORTED)
endif()
if(NOT TARGET zstd)
  add_library(zstd INTERFACE)
  target_link_libraries(zstd INTERFACE zstd::libzstd)
endif()

if(NOT TARGET xxhash)
  find_library(XXHASH_LIBRARY NAMES xxhash REQUIRED)
  find_path(XXHASH_INCLUDE_DIR NAMES xxhash.h REQUIRED)
  add_library(xxhash UNKNOWN IMPORTED)
  set_target_properties(
    xxhash PROPERTIES IMPORTED_LOCATION "${XXHASH_LIBRARY}"
                      INTERFACE_INCLUDE_DIRECTORIES "${XXHASH_INCLUDE_DIR}")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../librtcx/embed.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/Modules/AddFragment.cmake)
set(CUDF_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
set(CUDF_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

# Creates a string transforms example executable and installs it.
#
# Arguments: target_name  - Name of the CMake target and resulting executable source_file  - Source
# file to compile for the example
function(add_string_transforms_example target_name source_file)
  add_executable(${target_name} ${source_file})
  target_compile_features(${target_name} PRIVATE cxx_std_20)
  target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
  target_link_libraries(${target_name} PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>)
  install(TARGETS ${target_name} DESTINATION bin/examples/libcudf/string_transformers)
endfunction()

add_string_transforms_example(compute_checksum_jit compute_checksum_jit.cpp)
add_string_transforms_example(extract_email_jit extract_email_jit.cpp)
add_string_transforms_example(extract_email_precompiled extract_email_precompiled.cpp)
add_string_transforms_example(format_phone_jit format_phone_jit.cpp)
add_string_transforms_example(format_phone_precompiled format_phone_precompiled.cpp)
add_string_transforms_example(localize_phone_jit localize_phone_jit.cpp)
add_string_transforms_example(localize_phone_precompiled localize_phone_precompiled.cpp)

rtcx_add_embed(http_log_fragments)
add_fragment(http_log_fragments FRAGMENT http_request_line_sizes SOURCE
             fragments/http_request_line_sizes.cu)
add_fragment(http_log_fragments FRAGMENT http_request_line_output SOURCE
             fragments/http_request_line_output.cu)
add_fragment(http_log_fragments FRAGMENT http_combined_log_sizes SOURCE
             fragments/http_combined_log_sizes.cu)
add_fragment(http_log_fragments FRAGMENT http_combined_log_output SOURCE
             fragments/http_combined_log_output.cu)

foreach(
  fragment_target
  http_log_fragments_http_request_line_sizes
  http_log_fragments_http_request_line_output
  http_log_fragments_http_combined_log_sizes
  http_log_fragments_http_combined_log_output)
  target_include_directories(${fragment_target}
                             PRIVATE ${CMAKE_CURRENT_LIST_DIR})
endforeach()

rtcx_embed(http_log_fragments COMPRESSION none OUTPUT_DIRECTORY
           "${CUDF_GENERATED_INCLUDE_DIR}/rtcx_embed")

add_string_transforms_example(http_log_transforms http_log_transforms.cpp)
target_sources(
  http_log_transforms PRIVATE ${http_log_fragments_SOURCE_DIR}/http_log_fragments.s
)
target_include_directories(http_log_transforms
                           PRIVATE ${http_log_fragments_SOURCE_DIR})
add_dependencies(http_log_transforms http_log_fragments)

install(FILES ${CMAKE_CURRENT_LIST_DIR}/info.csv
              ${CMAKE_CURRENT_LIST_DIR}/http_logs.csv
        DESTINATION bin/examples/libcudf/string_transformers)
