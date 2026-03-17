# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(string_transforms_examples)

project(
  string_transforms_examples
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

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

install(FILES ${CMAKE_CURRENT_LIST_DIR}/info.csv
        DESTINATION bin/examples/libcudf/string_transformers
)
