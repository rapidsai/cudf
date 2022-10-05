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

cmake_minimum_required(VERSION 3.23.1)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(strings-udf-cpp)

# Create a project so that we can enable CUDA architectures in this file.
project(
  strings-udf-cpp
  VERSION ${strings_udf_version}
  LANGUAGES CUDA
)

rapids_cpm_init()

rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET strings-udf-exports
  INSTALL_EXPORT_SET strings-udf-exports
)

include(${rapids-cmake-dir}/cpm/libcudacxx.cmake)
rapids_cpm_libcudacxx(BUILD_EXPORT_SET strings-udf-exports INSTALL_EXPORT_SET strings-udf-exports)

add_library(cudf_strings_udf SHARED src/strings/udf/udf_apis.cu)
target_include_directories(
  cudf_strings_udf PUBLIC "$<BUILD_INTERFACE:${strings-udf-cpp_SOURCE_DIR}/include>"
)

set_target_properties(
  cudf_strings_udf
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

set(UDF_CXX_FLAGS)
set(UDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
target_compile_options(
  cudf_strings_udf PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${UDF_CXX_FLAGS}>"
                           "$<$<COMPILE_LANGUAGE:CUDA>:${UDF_CUDA_FLAGS}>"
)
target_link_libraries(cudf_strings_udf PUBLIC cudf::cudf CUDA::nvrtc)
install(TARGETS cudf_strings_udf DESTINATION ./strings_udf/_lib/)

# This function will copy the generated PTX file from its generator-specific location in the build
# tree into a specified location in the build tree from which we can install it.
function(copy_ptx_to_location target destination new_name)
  set(cmake_generated_file
      "${CMAKE_CURRENT_BINARY_DIR}/cmake/cp_${target}_$<LOWER_CASE:$<CONFIG>>_ptx.cmake"
  )
  file(
    GENERATE
    OUTPUT "${cmake_generated_file}"
    CONTENT
      "
set(ptx_path \"$<TARGET_OBJECTS:${target}>\")
file(MAKE_DIRECTORY \"${destination}\")
file(COPY_FILE \${ptx_path} \"${destination}/${new_name}\")"
  )

  add_custom_target(
    ${target}_cp_ptx ALL
    COMMAND ${CMAKE_COMMAND} -P "${cmake_generated_file}"
    DEPENDS $<TARGET_OBJECTS:${target}>
    COMMENT "Copying PTX files to '${destination}'"
  )
endfunction()

# Create the shim library for each architecture.
set(SHIM_CUDA_FLAGS --expt-relaxed-constexpr -rdc=true)

list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "-real" "")
list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "-virtual" "")
list(SORT CMAKE_CUDA_ARCHITECTURES)
list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)

foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  set(tgt shim_${arch})

  add_library(${tgt} OBJECT src/strings/udf/shim.cu)

  set_target_properties(${tgt} PROPERTIES CUDA_ARCHITECTURES ${arch} CUDA_PTX_COMPILATION ON)

  target_include_directories(${tgt} PUBLIC include)
  target_compile_options(${tgt} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SHIM_CUDA_FLAGS}>")
  target_link_libraries(${tgt} PUBLIC cudf::cudf)

  copy_ptx_to_location(${tgt} "${CMAKE_CURRENT_BINARY_DIR}/../strings_udf" ${tgt}.ptx)
  install(
    FILES $<TARGET_OBJECTS:${tgt}>
    DESTINATION ./strings_udf
    RENAME ${tgt}.ptx
  )
endforeach()
