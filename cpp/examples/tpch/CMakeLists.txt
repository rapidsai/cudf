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

add_executable(dbgen dbgen.cpp)
target_link_libraries(dbgen PRIVATE cudf::cudf cudf::cudf_tpch_datagen)
target_compile_features(dbgen PRIVATE cxx_std_17)

add_executable(tpch_q1 q1.cpp)
target_link_libraries(tpch_q1 PRIVATE cudf::cudf)
target_compile_features(tpch_q1 PRIVATE cxx_std_17)

add_executable(tpch_q5 q5.cpp)
target_link_libraries(tpch_q5 PRIVATE cudf::cudf)
target_compile_features(tpch_q5 PRIVATE cxx_std_17)

add_executable(tpch_q6 q6.cpp)
target_link_libraries(tpch_q6 PRIVATE cudf::cudf)
target_compile_features(tpch_q6 PRIVATE cxx_std_17)

add_executable(tpch_q9 q9.cpp)
target_link_libraries(tpch_q9 PRIVATE cudf::cudf)
target_compile_features(tpch_q9 PRIVATE cxx_std_17)

add_executable(tpch_q10 q10.cpp)
target_link_libraries(tpch_q10 PRIVATE cudf::cudf)
target_compile_features(tpch_q10 PRIVATE cxx_std_17)
