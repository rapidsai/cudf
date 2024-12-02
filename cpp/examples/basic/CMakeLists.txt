# Copyright (c) 2020-2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(basic_example)
rapids_cuda_set_architectures(RAPIDS)

project(
  basic_example
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

# Configure your project here
add_executable(basic_example src/process_csv.cpp)
target_link_libraries(basic_example PRIVATE cudf::cudf)
target_compile_features(basic_example PRIVATE cxx_std_17)

install(TARGETS basic_example DESTINATION bin/examples/libcudf)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/4stock_5day.csv DESTINATION bin/examples/libcudf)
