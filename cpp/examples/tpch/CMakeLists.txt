# Copyright (c) 2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

rapids_cuda_init_architectures(tpch_example)
rapids_cuda_set_architectures(RAPIDS)

project(
  tpch_example
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

function(ConfigureTPCHQuery CMAKE_QUERY_NAME)
  add_executable(${CMAKE_QUERY_NAME} ${ARGN})
  target_link_libraries(${CMAKE_QUERY_NAME} PUBLIC cudf::cudf)
  target_compile_features(${CMAKE_QUERY_NAME} PUBLIC cxx_std_17)
endfunction()

ConfigureTPCHQuery(tpch_q1 q1.cpp)
ConfigureTPCHQuery(tpch_q5 q5.cpp)
ConfigureTPCHQuery(tpch_q6 q6.cpp)
ConfigureTPCHQuery(tpch_q9 q9.cpp)
ConfigureTPCHQuery(tpch_q10 q10.cpp)
