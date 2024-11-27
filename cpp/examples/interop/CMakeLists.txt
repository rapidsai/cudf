# Copyright (c) 2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

rapids_cuda_init_architectures(interop_example)
rapids_cuda_set_architectures(RAPIDS)

project(
  interop_example
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

# The Arrow CMake is currently broken if the build type is not set
set(CMAKE_BUILD_TYPE Release)
# No need to install Arrow libs when only the final example executable is shipped.
set(CUDF_EXCLUDE_ARROW_FROM_ALL ON)
include(../../cmake/thirdparty/get_arrow.cmake)

add_executable(interop interop.cpp)
target_link_libraries(interop PRIVATE cudf::cudf)
target_compile_features(interop PRIVATE cxx_std_17)
target_link_libraries(interop PRIVATE ${ARROW_LIBRARIES})
