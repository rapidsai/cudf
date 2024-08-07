# Copyright (c) 2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(drop_duplicates)
rapids_cuda_set_architectures(RAPIDS)

project(
  drop_duplicates
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

# Configure your project here
add_executable(drop_duplicates drop_duplicates.cpp)
target_link_libraries(drop_duplicates PRIVATE cudf::cudf)
target_compile_features(drop_duplicates PRIVATE cxx_std_17)

install(TARGETS drop_duplicates DESTINATION bin/examples/libcudf)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/transactions.csv DESTINATION bin/examples/libcudf)
