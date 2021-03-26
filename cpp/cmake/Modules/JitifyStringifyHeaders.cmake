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

file(MAKE_DIRECTORY "${CUDF_GENERATED_INCLUDE_DIR}/include/jit_stringified")

# Create `stringify` executable
add_executable(stringify "${JITIFY_INCLUDE_DIR}/stringify.cpp")

execute_process(WORKING_DIRECTORY ${CUDF_GENERATED_INCLUDE_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_stringified/cuda/std/detail/libcxx/include
    )

set(JIT_STRINGIFIED_FILES)

function(jit_stringify_files)
    cmake_parse_arguments(ARG
                          ""
                          "SOURCE_DIRECTORY"
                          "FILES"
                          ${ARGN}
                          )

    foreach(ARG_FILE ${ARG_FILES})
        set(ARG_OUTPUT ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_stringified/${ARG_FILE}.jit)
        set(JIT_STRINGIFIED_FILES "${ARG_OUTPUT};${JIT_STRINGIFIED_FILES}")
        add_custom_command(WORKING_DIRECTORY ${ARG_SOURCE_DIRECTORY}
                           DEPENDS stringify
                           OUTPUT ${ARG_OUTPUT}
                           COMMAND ${CUDF_BINARY_DIR}/stringify ${ARG_FILE} > ${ARG_OUTPUT}
                           )
    endforeach()
    set(JIT_STRINGIFIED_FILES "${JIT_STRINGIFIED_FILES}" PARENT_SCOPE)
endfunction()

jit_stringify_files(SOURCE_DIRECTORY      ${CUDF_SOURCE_DIR}/include
                    FILES                 cudf/types.hpp
                                          cudf/utilities/bit.hpp
                                          cudf/wrappers/timestamps.hpp
                                          cudf/fixed_point/fixed_point.hpp
                                          cudf/wrappers/durations.hpp
                                          cudf/detail/utilities/assert.cuh
                    )

jit_stringify_files(SOURCE_DIRECTORY      ${LIBCUDACXX_INCLUDE_DIR}
                    FILES                 cuda/std/chrono
                                          cuda/std/climits
                                          cuda/std/cstddef
                                          cuda/std/cstdint
                                          cuda/std/ctime
                                          cuda/std/limits
                                          cuda/std/ratio
                                          cuda/std/type_traits
                                          cuda/std/version
                                          cuda/std/detail/__config
                                          cuda/std/detail/__pragma_pop
                                          cuda/std/detail/__pragma_push
                                          cuda/std/detail/libcxx/include/__config
                                          cuda/std/detail/libcxx/include/__pragma_pop
                                          cuda/std/detail/libcxx/include/__pragma_push
                                          cuda/std/detail/libcxx/include/__undef_macros
                                          cuda/std/detail/libcxx/include/chrono
                                          cuda/std/detail/libcxx/include/climits
                                          cuda/std/detail/libcxx/include/cstddef
                                          cuda/std/detail/libcxx/include/cstdint
                                          cuda/std/detail/libcxx/include/ctime
                                          cuda/std/detail/libcxx/include/limits
                                          cuda/std/detail/libcxx/include/ratio
                                          cuda/std/detail/libcxx/include/type_traits
                                          cuda/std/detail/libcxx/include/version
                    )

# hacky way around using internal headers in jit files. should probably be moved to a public header.
jit_stringify_files(SOURCE_DIRECTORY      ${CUDF_SOURCE_DIR}/..
                    FILES                 cudf/src/rolling/rolling_jit_detail.hpp
                    )

add_custom_target(jitify_stringify_run DEPENDS ${JIT_STRINGIFIED_FILES})

###################################################################################################
# - copy libcu++ ----------------------------------------------------------------------------------

# `${LIBCUDACXX_INCLUDE_DIR}/` specifies that the contents of this directory will be installed (not the directory itself)
file(COPY "${LIBCUDACXX_INCLUDE_DIR}/" DESTINATION "${CUDF_GENERATED_INCLUDE_DIR}/include/libcudacxx")
file(COPY "${LIBCXX_INCLUDE_DIR}"      DESTINATION "${CUDF_GENERATED_INCLUDE_DIR}/include/libcxx")
