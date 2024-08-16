# Copyright (c) 2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(billion)
rapids_cuda_set_architectures(RAPIDS)

project(
  billion
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

add_library(groupby_results OBJECT groupby_results.cpp)
target_link_libraries(groupby_results PRIVATE cudf::cudf)

add_executable(load_csv load_csv.cpp)
target_link_libraries(load_csv PRIVATE cudf::cudf nvToolsExt $<TARGET_OBJECTS:groupby_results>)
install(TARGETS load_csv DESTINATION bin/examples/libcudf)

add_executable(load_csv_chunks load_csv_chunks.cpp)
target_link_libraries(
  load_csv_chunks PRIVATE cudf::cudf nvToolsExt $<TARGET_OBJECTS:groupby_results>
)
install(TARGETS load_csv_chunks DESTINATION bin/examples/libcudf)

add_executable(load_csv_pipeline load_csv_pipeline.cpp)
target_link_libraries(
  load_csv_pipeline PRIVATE cudf::cudf nvToolsExt $<TARGET_OBJECTS:groupby_results>
)
install(TARGETS load_csv_pipeline DESTINATION bin/examples/libcudf)

