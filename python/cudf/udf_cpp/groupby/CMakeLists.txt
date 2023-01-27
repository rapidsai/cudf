# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

include(rapids-find)

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
file(COPY_FILE \${ptx_paths} \"${destination}/${target}.ptx\")"
  )

  add_custom_target(
    ${target}_cp_ptx ALL
    COMMAND ${CMAKE_COMMAND} -P "${cmake_generated_file}"
    DEPENDS $<TARGET_OBJECTS:${target}>
    COMMENT "Copying PTX files to '${destination}'"
  )
endfunction()

# Create the shim library for each architecture.
set(GROUPBY_FUNCTION_CUDA_FLAGS --expt-relaxed-constexpr)

# always build a default PTX file in case RAPIDS_NO_INITIALIZE is set and the device cc can't be
# safely queried through a context
list(INSERT CMAKE_CUDA_ARCHITECTURES 0 "60")

list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "-real" "")
list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "-virtual" "")
list(SORT CMAKE_CUDA_ARCHITECTURES)
list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)

foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  set(tgt function_${arch})

  add_library(${tgt} OBJECT function.cu)
  set_target_properties(
    ${tgt}
    PROPERTIES CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               CUDA_ARCHITECTURES ${arch}
               CUDA_PTX_COMPILATION ON
               CUDA_SEPARABLE_COMPILATION ON
  )

  target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(
    ${tgt} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${GROUPBY_FUNCTION_CUDA_FLAGS}>"
  )
  target_link_libraries(${tgt} PUBLIC cudf::cudf)

  copy_ptx_to_location(${tgt} "${CMAKE_CURRENT_BINARY_DIR}/")
  install(
    FILES $<TARGET_OBJECTS:${tgt}>
    DESTINATION ./cudf/core/udf/
    RENAME ${tgt}.ptx
  )
endforeach()
