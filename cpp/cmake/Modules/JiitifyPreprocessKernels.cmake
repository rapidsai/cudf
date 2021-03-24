#=============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
target_link_libraries(jitify_preprocess CUDA::cudart_static)


set(JIT_PREPROCESSED_FILES)

function(jit_preprocess_files)
    cmake_parse_arguments(ARG
                          ""
                          "SOURCE_DIRECTORY"
                          "FILES"
                          ${ARGN}
                          )

    foreach(ARG_FILE ${ARG_FILES})
        set(ARG_OUTPUT ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_preprocessed_files/${ARG_FILE}.jit)
        set(JIT_PREPROCESSED_FILES "${ARG_OUTPUT};${JIT_PREPROCESSED_FILES}")
        add_custom_command(WORKING_DIRECTORY ${ARG_SOURCE_DIRECTORY}
                           DEPENDS stringify
                           OUTPUT ${ARG_OUTPUT}
                           COMMAND ${CUDF_BINARY_DIR}/stringify ${ARG_FILE} > ${ARG_OUTPUT}_old
                           )
    endforeach()
    set(JIT_PREPROCESSED_FILES "${JIT_PREPROCESSED_FILES}" PARENT_SCOPE)
endfunction()

jit_preprocess_files(SOURCE_DIRECTORY      ${CUDF_SOURCE_DIR}/cudf/src
                     FILES                 binaryop/jit/kernel.cu
                     )

add_custom_target(stringify_run DEPENDS ${JIT_PREPROCESSED_FILES})

###################################################################################################
# - copy libcu++ ----------------------------------------------------------------------------------

# `${LIBCUDACXX_INCLUDE_DIR}/` specifies that the contents of this directory will be installed (not the directory itself)
file(INSTALL "${LIBCUDACXX_INCLUDE_DIR}/" DESTINATION "${CUDF_GENERATED_INCLUDE_DIR}/include/libcudacxx")
file(INSTALL "${LIBCXX_INCLUDE_DIR}"      DESTINATION "${CUDF_GENERATED_INCLUDE_DIR}/include/libcxx")
