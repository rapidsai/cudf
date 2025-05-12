# =============================================================================
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-find)

rapids_cpm_init()

rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET udf-exports
  INSTALL_EXPORT_SET udf-exports
)

# The below in vendored/inspired by legate dataframe
execute_process(
  COMMAND "python" -c "from numba.cuda.runtime.nrt import get_include; print(get_include())"
  OUTPUT_VARIABLE NRT_INCLUDE_DIR
  ERROR_VARIABLE NRT_ERROR
  RESULT_VARIABLE NRT_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(${NRT_RESULT})
  message(FATAL_ERROR "Error while trying to obtain NRT include directory:\n${NRT_ERROR}")
endif()

message(STATUS "NRT_INCLUDE_DIR: ${NRT_INCLUDE_DIR}")

include(${rapids-cmake-dir}/cpm/cccl.cmake)
rapids_cpm_cccl(BUILD_EXPORT_SET udf-exports INSTALL_EXPORT_SET udf-exports)

add_library(cudf_strings_udf SHARED strings/src/strings/udf/udf_apis.cu)
target_include_directories(
  cudf_strings_udf PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/strings/include>"
                          "${NRT_INCLUDE_DIR}"
)

set_target_properties(
  cudf_strings_udf
  PROPERTIES BUILD_RPATH "\$ORIGIN/../"
             INSTALL_RPATH "\$ORIGIN/../"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

set(UDF_CXX_FLAGS)
set(UDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr -g -G)
target_compile_options(
  cudf_strings_udf PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${UDF_CXX_FLAGS}>"
                           "$<$<COMPILE_LANGUAGE:CUDA>:${UDF_CUDA_FLAGS}>"
)
target_link_libraries(cudf_strings_udf PUBLIC cudf::cudf)
install(TARGETS cudf_strings_udf DESTINATION ./cudf/_lib/)

# Create the shim library for each architecture.
set(SHIM_CUDA_FLAGS --expt-relaxed-constexpr -rdc=true --expt-extended-lambda -g -G)

# always build a default PTX file in case RAPIDS_NO_INITIALIZE is set and the device cc can't be
# safely queried through a context
list(INSERT CMAKE_CUDA_ARCHITECTURES 0 "70")

list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "-real" "")
list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "-virtual" "")
list(SORT CMAKE_CUDA_ARCHITECTURES)
list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)

foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  set(tgt shim_${arch})

  add_library(${tgt} OBJECT shim.cu)

  set_target_properties(${tgt} PROPERTIES CUDA_ARCHITECTURES ${arch} CUDA_PTX_COMPILATION ON)

  target_include_directories(
    ${tgt} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/strings/include>"
                  "${NRT_INCLUDE_DIR}"
  )
  target_compile_options(${tgt} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SHIM_CUDA_FLAGS}>")
  target_link_libraries(${tgt} PUBLIC cudf::cudf)

  install(
    FILES $<TARGET_OBJECTS:${tgt}>
    DESTINATION cudf/core/udf/
    RENAME ${tgt}.ptx
  )
endforeach()

foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  set(tgt udf_apis_${arch})

  add_library(${tgt} OBJECT strings/src/strings/udf/udf_apis.cu)

  set_target_properties(${tgt} PROPERTIES CUDA_ARCHITECTURES ${arch} CUDA_PTX_COMPILATION ON)

  target_include_directories(
    ${tgt} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/strings/include>"
                  "${NRT_INCLUDE_DIR}"
  )
  target_compile_options(${tgt} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SHIM_CUDA_FLAGS}>")
  target_link_libraries(${tgt} PUBLIC cudf::cudf)

  install(
    FILES $<TARGET_OBJECTS:${tgt}>
    DESTINATION cudf/core/udf/
    RENAME ${tgt}.ptx
  )
endforeach()
