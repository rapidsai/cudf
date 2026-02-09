# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Create `jitify_preprocess` executable
add_executable(jitify_preprocess "${JITIFY_INCLUDE_DIR}/jitify2_preprocess.cpp")

target_compile_definitions(jitify_preprocess PRIVATE "_FILE_OFFSET_BITS=64")
rapids_cuda_set_runtime(jitify_preprocess USE_STATIC ${CUDA_STATIC_RUNTIME})
target_link_libraries(jitify_preprocess PUBLIC ${CMAKE_DL_LIBS})

# Take a list of files to JIT-compile and run them through jitify_preprocess.
function(jit_preprocess_files)
  cmake_parse_arguments(ARG "" "SOURCE_DIRECTORY" "FILES" ${ARGN})

  get_target_property(libcudacxx_raw_includes CCCL::libcudacxx INTERFACE_INCLUDE_DIRECTORIES)
  set(includes)
  foreach(inc IN LISTS libcudacxx_raw_includes CUDAToolkit_INCLUDE_DIRS)
    list(APPEND includes "-I${inc}")
  endforeach()
  foreach(ARG_FILE ${ARG_FILES})
    set(ARG_OUTPUT ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_preprocessed_files/${ARG_FILE}.jit.hpp)
    get_filename_component(jit_output_directory "${ARG_OUTPUT}" DIRECTORY)
    list(APPEND JIT_PREPROCESSED_FILES "${ARG_OUTPUT}")

    get_filename_component(ARG_OUTPUT_DIR "${ARG_OUTPUT}" DIRECTORY)

    # Note: need to pass _FILE_OFFSET_BITS=64 in COMMAND due to a limitation in how conda builds
    # glibc
    add_custom_command(
      OUTPUT ${ARG_OUTPUT}
      DEPENDS jitify_preprocess "${ARG_SOURCE_DIRECTORY}/${ARG_FILE}"
      WORKING_DIRECTORY ${ARG_SOURCE_DIRECTORY}
      VERBATIM
      COMMAND ${CMAKE_COMMAND} -E make_directory "${jit_output_directory}"
      COMMAND
        "${CMAKE_COMMAND}" -E env LD_LIBRARY_PATH=${CUDAToolkit_LIBRARY_DIR}
        $<TARGET_FILE:jitify_preprocess> ${ARG_FILE} -o ${ARG_OUTPUT_DIR} -i -std=c++20
        -remove-unused-globals -D_FILE_OFFSET_BITS=64 -D__CUDACC_RTC__ -DCUDF_RUNTIME_JIT
        -I${CUDF_SOURCE_DIR}/include -I${CUDF_SOURCE_DIR}/src ${includes}
        --no-preinclude-workarounds --no-replace-pragma-once --diag-suppress=47 --device-int128
      COMMENT "Custom command to JIT-compile files."
    )
  endforeach()
  set(JIT_PREPROCESSED_FILES
      "${JIT_PREPROCESSED_FILES}"
      PARENT_SCOPE
  )
endfunction()

if(NOT (EXISTS "${CUDF_GENERATED_INCLUDE_DIR}/include"))
  make_directory("${CUDF_GENERATED_INCLUDE_DIR}/include")
endif()

jit_preprocess_files(
  SOURCE_DIRECTORY ${CUDF_SOURCE_DIR}/src FILES binaryop/jit/kernel.cu rolling/jit/kernel.cu
  stream_compaction/filter/jit/kernel.cu transform/jit/kernel.cu
)

add_custom_target(
  jitify_preprocess_run
  DEPENDS ${JIT_PREPROCESSED_FILES}
  COMMENT "Target representing jitified files."
)

# when a user requests CMake to clean the build directory
#
# * `cmake --build <dir> --target clean`
# * `cmake --build <dir> --clean-first`
# * ninja clean
#
# We also remove the jitify2 program cache as well. This ensures that we don't keep older versions
# of the programs in cache
set(cache_path "$ENV{HOME}/.cudf")
if(ENV{LIBCUDF_KERNEL_CACHE_PATH})
  set(cache_path "$ENV{LIBCUDF_KERNEL_CACHE_PATH}")
endif()
cmake_path(APPEND cache_path "${CUDF_VERSION}/")
set_target_properties(jitify_preprocess_run PROPERTIES ADDITIONAL_CLEAN_FILES "${cache_path}")
