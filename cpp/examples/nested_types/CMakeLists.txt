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

# Configure your project here
add_executable(deduplication deduplication.cpp)
target_link_libraries(deduplication PRIVATE cudf::cudf)
target_compile_features(deduplication PRIVATE cxx_std_17)

# Install the example only if CMAKE_INSTALL_PREFIX is set
if (NOT $ENV{CMAKE_INSTALL_PREFIX} STREQUAL "")
  # set the CMAKE_INSTALL_PREFIX to the environment variable value
  set(CMAKE_INSTALL_PREFIX "$ENV{CMAKE_INSTALL_PREFIX}" CACHE STRING "Force CMAKE_INSTALL_PREFIX to environment variable" FORCE)
  install(TARGETS deduplication DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/libcudf)
  install(FILES ${CMAKE_CURRENT_LIST_DIR}/example.json DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/libcudf)
endif()