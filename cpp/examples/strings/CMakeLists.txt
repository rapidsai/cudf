# Copyright (c) 2022-2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(strings_examples)
rapids_cuda_set_architectures(RAPIDS)

project(
  strings_examples
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(libcudf_apis libcudf_apis.cpp)
target_compile_features(libcudf_apis PRIVATE cxx_std_17)
target_link_libraries(libcudf_apis PRIVATE cudf::cudf nvToolsExt)
install(TARGETS libcudf_apis DESTINATION bin/examples/libcudf)

add_executable(custom_with_malloc custom_with_malloc.cu)
target_compile_features(custom_with_malloc PRIVATE cxx_std_17)
target_compile_options(custom_with_malloc PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(custom_with_malloc PRIVATE cudf::cudf nvToolsExt)
install(TARGETS custom_with_malloc DESTINATION bin/examples/libcudf)

add_executable(custom_prealloc custom_prealloc.cu)
target_compile_features(custom_prealloc PRIVATE cxx_std_17)
target_compile_options(custom_prealloc PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(custom_prealloc PRIVATE cudf::cudf nvToolsExt)
install(TARGETS custom_prealloc DESTINATION bin/examples/libcudf)

add_executable(custom_optimized custom_optimized.cu)
target_compile_features(custom_optimized PRIVATE cxx_std_17)
target_compile_options(custom_optimized PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(custom_optimized PRIVATE cudf::cudf nvToolsExt)
install(TARGETS custom_optimized DESTINATION bin/examples/libcudf)

install(FILES ${CMAKE_CURRENT_LIST_DIR}/names.csv DESTINATION bin/examples/libcudf)
