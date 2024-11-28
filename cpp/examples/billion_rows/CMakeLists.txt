# Copyright (c) 2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(billion_rows)
rapids_cuda_set_architectures(RAPIDS)

project(
  billion_rows
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

add_library(groupby_results OBJECT groupby_results.cpp)
target_link_libraries(groupby_results PRIVATE cudf::cudf)

add_executable(brc brc.cpp)
target_link_libraries(brc PRIVATE cudf::cudf nvToolsExt $<TARGET_OBJECTS:groupby_results>)
install(TARGETS brc DESTINATION bin/examples/libcudf)

add_executable(brc_chunks brc_chunks.cpp)
target_link_libraries(brc_chunks PRIVATE cudf::cudf nvToolsExt $<TARGET_OBJECTS:groupby_results>)
install(TARGETS brc_chunks DESTINATION bin/examples/libcudf)

add_executable(brc_pipeline brc_pipeline.cpp)
target_link_libraries(brc_pipeline PRIVATE cudf::cudf nvToolsExt $<TARGET_OBJECTS:groupby_results>)
install(TARGETS brc_pipeline DESTINATION bin/examples/libcudf)
