#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

cmake_minimum_required(VERSION 3.18)

file(MAKE_DIRECTORY "${CUDF_GENERATED_INCLUDE_DIR}/include/jit_preprocessed_files")

# Create `jitify_preprocess` executable
project(jitify_preprocess VERSION 2.0 LANGUAGES CXX CUDA)
add_executable(jitify_preprocess "${JITIFY_INCLUDE_DIR}/jitify2_preprocess.cpp")

target_link_libraries(jitify_preprocess CUDA::cudart ${CMAKE_DL_LIBS})

function(jit_preprocess_files)
    cmake_parse_arguments(ARG
                          ""
                          "SOURCE_DIRECTORY"
                          "FILES"
                          ${ARGN}
                          )

    foreach(ARG_FILE ${ARG_FILES})
        set(ARG_OUTPUT ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_preprocessed_files/${ARG_FILE}.jit)
        list(APPEND JIT_PREPROCESSED_FILES "${ARG_OUTPUT}")
        add_custom_command(WORKING_DIRECTORY ${ARG_SOURCE_DIRECTORY}
                           DEPENDS jitify_preprocess
                           OUTPUT ${ARG_OUTPUT}
                           VERBATIM
                           COMMAND jitify_preprocess ${ARG_FILE}
                                    -o ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_preprocessed_files
                                    -v
                                    -i
                                    -m
                                    -std=c++14
                                    -remove-unused-globals
                                    -D__CUDACC_RTC__
                                    -I${CUDF_SOURCE_DIR}/include
                                    -I${CUDF_SOURCE_DIR}/src
                                    -I${LIBCUDACXX_INCLUDE_DIR}
                                    -I${CUDAToolkit_INCLUDE_DIRS}
                                    --no-preinclude-workarounds
                                    --no-replace-pragma-once
                           )
    endforeach()
    set(JIT_PREPROCESSED_FILES "${JIT_PREPROCESSED_FILES}" PARENT_SCOPE)
endfunction()

jit_preprocess_files(SOURCE_DIRECTORY      ${CUDF_SOURCE_DIR}/src
                     FILES                 binaryop/jit/kernel.cu
                                           transform/jit/kernel.cu
                                           rolling/jit/kernel.cu
                     )

add_custom_target(jitify_preprocess_run DEPENDS ${JIT_PREPROCESSED_FILES})

file(COPY "${LIBCUDACXX_INCLUDE_DIR}/" DESTINATION "${CUDF_GENERATED_INCLUDE_DIR}/include/libcudacxx")
file(COPY "${LIBCXX_INCLUDE_DIR}"      DESTINATION "${CUDF_GENERATED_INCLUDE_DIR}/include/libcxx")
