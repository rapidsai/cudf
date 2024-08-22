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

# Try hardcoding build type for arrow
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build."
  )
endif()
include(../../tests/cmake/thirdparty/get_arrow.cmake)

add_executable(interop interop.cpp)
target_link_libraries(interop PRIVATE cudf::cudf)
target_compile_features(interop PRIVATE cxx_std_17)
target_link_libraries(interop PRIVATE ${ARROW_LIBRARIES})
