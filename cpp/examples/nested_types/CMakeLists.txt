# Copyright (c) 2023-2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(nested_types)
rapids_cuda_set_architectures(RAPIDS)

project(
  nested_types
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

# Configure your project here
add_executable(deduplication deduplication.cpp)
target_link_libraries(deduplication PRIVATE cudf::cudf)
target_compile_features(deduplication PRIVATE cxx_std_17)

install(TARGETS deduplication DESTINATION bin/examples/libcudf)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/example.json DESTINATION bin/examples/libcudf)
