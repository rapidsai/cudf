# Copyright (c) 2024-2025, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

rapids_cuda_init_architectures(interop_example)

project(
  interop_example
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

# For now, disable CMake's automatic module scanning for C++ files. There is an
# sccache bug in the version RAPIDS uses in CI that causes it to handle the
# resulting -M* flags incorrectly with gcc>=14. We can remove this once we
# upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# The Arrow CMake is currently broken if the build type is not set
set(CMAKE_BUILD_TYPE Release)
# No need to install Arrow libs when only the final example executable is shipped.
set(CUDF_EXCLUDE_ARROW_FROM_ALL ON)
include(../../cmake/thirdparty/get_arrow.cmake)

add_executable(interop interop.cpp)
target_link_libraries(interop PRIVATE cudf::cudf)
target_compile_features(interop PRIVATE cxx_std_20)
target_link_libraries(interop PRIVATE ${ARROW_LIBRARIES})
