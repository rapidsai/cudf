# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.20.1)

include(rapids-cmake)
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(GROUPBY_UDF)

# Create a project so that we can enable CUDA architectures in this file.
project(
  groupby-udf-cpp
  VERSION 0.0.0 # Placeholder since this isn't a real project
  LANGUAGES CUDA
)

rapids_find_package(CUDAToolkit REQUIRED)

# include(${rapids-cmake-dir}/cpm/libcudacxx.cmake) rapids_cpm_libcudacxx(BUILD_EXPORT_SET
# strings-udf-exports INSTALL_EXPORT_SET strings-udf-exports)

# add_library(groupby_udf_cpp SHARED function.cu) target_include_directories( groupby_udf_cpp PUBLIC
# "$<BUILD_INTERFACE:${strings-udf-cpp_SOURCE_DIR}/include>" )

# set_target_properties( groupby_udf_cpp PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN"
# CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
# POSITION_INDEPENDENT_CODE ON INTERFACE_POSITION_INDEPENDENT_CODE ON )
#
# set(UDF_CXX_FLAGS) set(UDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
# target_compile_options( groupby_udf_cpp PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${UDF_CXX_FLAGS}>"
# "$<$<COMPILE_LANGUAGE:CUDA>:${UDF_CUDA_FLAGS}>"
# ) target_link_libraries(groupby_udf_cpp PUBLIC cudf::cudf CUDA::nvrtc) install(TARGETS
# groupby_udf_cpp DESTINATION ./cudf/core/udf/)

# This function will copy the generated PTX file from its generator-specific location in the build
# tree into a specified location in the build tree from which we can install it.
function(copy_ptx_to_location target destination)
  set(cmake_generated_file
      "${CMAKE_CURRENT_BINARY_DIR}/cmake/cp_${target}_$<LOWER_CASE:$<CONFIG>>_ptx.cmake"
  )
  file(
    GENERATE
    OUTPUT "${cmake_generated_file}"
    CONTENT
      "
set(ptx_paths \"$<TARGET_OBJECTS:${target}>\")
file(COPY \${ptx_paths} DESTINATION \"${destination}\")"
  )

  add_custom_target(
    ${target}_cp_ptx ALL
    COMMAND ${CMAKE_COMMAND} -P "${cmake_generated_file}"
    DEPENDS $<TARGET_OBJECTS:${target}>
    COMMENT "Copying PTX files to '${destination}'"
  )
endfunction()

# Create the shim library for each architecture.
set(GROUPBY_FUNCTION_CUDA_FLAGS --expt-relaxed-constexpr -rdc=true)

foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  set(tgt function_${arch})

  add_library(${tgt} OBJECT function.cu)
  set_target_properties(
    ${tgt}
    PROPERTIES CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               CUDA_ARCHITECTURES ${arch}
               CUDA_PTX_COMPILATION ON
  )

  target_include_directories(${tgt} PUBLIC include)
  target_compile_options(
    ${tgt} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${GROUPBY_FUNCTION_CUDA_FLAGS}>"
  )
  target_link_libraries(${tgt} PUBLIC CUDA::nvrtc)

  copy_ptx_to_location(${tgt} "${CMAKE_CURRENT_BINARY_DIR}/")
  install(
    FILES $<TARGET_OBJECTS:${tgt}>
    DESTINATION ./cudf/core/udf/
    RENAME ${tgt}.ptx
  )
endforeach()
