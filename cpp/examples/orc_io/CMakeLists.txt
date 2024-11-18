# Copyright (c) 2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(orc_io)
rapids_cuda_set_architectures(RAPIDS)

project(
  orc_io
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

add_library(orc_io_utils OBJECT common_utils.cpp io_source.cpp)
target_compile_features(orc_io_utils PRIVATE cxx_std_17)
target_link_libraries(orc_io_utils PRIVATE cudf::cudf)

# Build and install orc_io
add_executable(orc_io orc_io.cpp)
target_link_libraries(orc_io PRIVATE cudf::cudf $<TARGET_OBJECTS:orc_io_utils>)
target_compile_features(orc_io PRIVATE cxx_std_17)
install(TARGETS orc_io DESTINATION bin/examples/libcudf)