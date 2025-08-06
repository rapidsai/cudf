# Copyright (c) 2025, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(string_transforms_examples)

project(
  string_transforms_examples
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(compute_checksum_jit compute_checksum_jit.cpp)
target_compile_features(compute_checksum_jit PRIVATE cxx_std_20)
target_compile_options(
  compute_checksum_jit PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)
target_link_libraries(
  compute_checksum_jit PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
)
install(TARGETS compute_checksum_jit DESTINATION bin/examples/libcudf)

add_executable(extract_email_jit extract_email_jit.cpp)
target_compile_features(extract_email_jit PRIVATE cxx_std_20)
target_compile_options(extract_email_jit PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(
  extract_email_jit PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
)
install(TARGETS extract_email_jit DESTINATION bin/examples/libcudf)

add_executable(extract_email_precompiled extract_email_precompiled.cpp)
target_compile_features(extract_email_precompiled PRIVATE cxx_std_20)
target_compile_options(
  extract_email_precompiled PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)
target_link_libraries(
  extract_email_precompiled PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
)
install(TARGETS extract_email_precompiled DESTINATION bin/examples/libcudf)

add_executable(format_phone_jit format_phone_jit.cpp)
target_compile_features(format_phone_jit PRIVATE cxx_std_20)
target_compile_options(format_phone_jit PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(format_phone_jit PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>)
install(TARGETS format_phone_jit DESTINATION bin/examples/libcudf)

add_executable(format_phone_precompiled format_phone_precompiled.cpp)
target_compile_features(format_phone_precompiled PRIVATE cxx_std_20)
target_compile_options(
  format_phone_precompiled PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)
target_link_libraries(
  format_phone_precompiled PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
)
install(TARGETS format_phone_precompiled DESTINATION bin/examples/libcudf)

add_executable(localize_phone_jit localize_phone_jit.cpp)
target_compile_features(localize_phone_jit PRIVATE cxx_std_20)
target_compile_options(localize_phone_jit PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(
  localize_phone_jit PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
)
install(TARGETS localize_phone_jit DESTINATION bin/examples/libcudf)

add_executable(localize_phone_precompiled localize_phone_precompiled.cpp)
target_compile_features(localize_phone_precompiled PRIVATE cxx_std_20)
target_compile_options(
  localize_phone_precompiled PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)
target_link_libraries(
  localize_phone_precompiled PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
)
install(TARGETS localize_phone_precompiled DESTINATION bin/examples/libcudf)

install(FILES ${CMAKE_CURRENT_LIST_DIR}/info.csv DESTINATION bin/examples/libcudf)
