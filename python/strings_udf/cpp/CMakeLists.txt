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
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cpm_init()

rapids_cuda_init_architectures(STRINGS_UDF)

project(
  strings-udf-cpp
  VERSION ${strings_udf_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            CUDA
)

rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET strings-udf-exports
  INSTALL_EXPORT_SET strings-udf-exports
)

include(${rapids-cmake-dir}/cpm/libcudacxx.cmake)
rapids_cpm_libcudacxx(BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports)

set(UDF_CXX_FLAGS "")
set(UDF_CUDA_FLAGS "")

list(APPEND UDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

add_library(cudf_strings_udf SHARED src/strings/udf/udf_apis.cu)
target_include_directories(cudf_strings_udf PUBLIC include)

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
             PUBLIC_HEADER "include/cudf/strings/udf/udf_apis.hpp"
)

target_compile_options(
  cudf_strings_udf PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${UDF_CXX_FLAGS}>"
                           "$<$<COMPILE_LANGUAGE:CUDA>:${UDF_CUDA_FLAGS}>"
)
target_link_libraries(cudf_strings_udf PUBLIC cudf::cudf CUDA::nvrtc)

# Shim libraries. TODO: Will need to create separate shim libraries for different architectures.
add_library(shim OBJECT src/strings/udf/shim.cu)
set(SHIM_CUDA_FLAGS "")
list(APPEND SHIM_CUDA_FLAGS --expt-relaxed-constexpr -rdc=true)

set_property(TARGET shim PROPERTY CUDA_PTX_COMPILATION ON)

target_include_directories(shim PUBLIC include)
target_compile_options(shim PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SHIM_CUDA_FLAGS}>")
target_link_libraries(shim PUBLIC cudf::cudf)

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

copy_ptx_to_location(shim "${CMAKE_CURRENT_BINARY_DIR}/../strings_udf")
install(FILES $<TARGET_OBJECTS:shim> DESTINATION ./strings_udf)
