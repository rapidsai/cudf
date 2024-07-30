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

add_executable(load_csv load_csv.cpp)
target_compile_features(load_csv PRIVATE cxx_std_17)
target_link_libraries(load_csv PRIVATE cudf::cudf nvToolsExt cudf::cudftestutil)
install(TARGETS load_csv DESTINATION bin/examples/libcudf)

add_executable(load_csv_chunks load_csv_chunks.cpp)
target_compile_features(load_csv_chunks PRIVATE cxx_std_17)
target_link_libraries(load_csv_chunks PRIVATE cudf::cudf nvToolsExt cudf::cudftestutil)
install(TARGETS load_csv_chunks DESTINATION bin/examples/libcudf)

add_executable(load_text load_text.cpp)
target_compile_features(load_text PRIVATE cxx_std_17)
target_link_libraries(load_text PRIVATE cudf::cudf nvToolsExt cudf::cudftestutil)
install(TARGETS load_text DESTINATION bin/examples/libcudf)

add_executable(load_text_chunks load_text_chunks.cpp)
target_compile_features(load_text_chunks PRIVATE cxx_std_17)
target_link_libraries(load_text_chunks PRIVATE cudf::cudf nvToolsExt cudf::cudftestutil)
install(TARGETS load_text_chunks DESTINATION bin/examples/libcudf)

add_executable(load_text_pipeline load_text_pipeline.cpp)
target_compile_features(load_text_pipeline PRIVATE cxx_std_17)
target_link_libraries(load_text_pipeline PRIVATE cudf::cudf nvToolsExt cudf::cudftestutil)
install(TARGETS load_text_pipeline DESTINATION bin/examples/libcudf)
