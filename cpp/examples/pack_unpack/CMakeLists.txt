# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# Initialize CUDA architecture
rapids_cuda_init_architectures(basic_example)

project(
  pack_unpack_examples
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

# Add executables
add_executable(device_pack_unpack_example device_pack_unpack.cpp common_utils.cpp)
target_link_libraries(device_pack_unpack_example PRIVATE cudf::cudf)
target_compile_features(device_pack_unpack_example PRIVATE cxx_std_20)
install(TARGETS device_pack_unpack_example DESTINATION bin/examples/libcudf/pack_unpack)

add_executable(host_pack_unpack_example host_pack_unpack.cpp common_utils.cpp)
target_link_libraries(host_pack_unpack_example PRIVATE cudf::cudf)
target_compile_features(host_pack_unpack_example PRIVATE cxx_std_20)
install(TARGETS host_pack_unpack_example DESTINATION bin/examples/libcudf/pack_unpack)

add_executable(host_pack_copy_unpack_example host_pack_copy_unpack.cpp common_utils.cpp)
target_link_libraries(host_pack_copy_unpack_example PRIVATE cudf::cudf)
target_compile_features(host_pack_copy_unpack_example PRIVATE cxx_std_20)
install(TARGETS host_pack_copy_unpack_example DESTINATION bin/examples/libcudf/pack_unpack)
