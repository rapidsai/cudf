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

file(MAKE_DIRECTORY "${CUDF_GENERATED_INCLUDE_DIR}/include")

# Create `stringify` executable
add_executable(stringify "${JITIFY_INCLUDE_DIR}/stringify.cpp")

execute_process(WORKING_DIRECTORY ${CUDF_GENERATED_INCLUDE_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include
    )

function(jit_pre)
    cmake_parse_arguments(ARG
                          ""
                          "SOURCE_DIRECTORY"
                          "FILES"
                          ${ARGN}
                          )

    # if(NOT ARG_SOURCE_DIRECTORY)
    #     message(FATAL_ERROR "You must provide a source directory ${ARG_UNPARSED_ARGUMENTS}")
    # endif(NOT ARG_SOURCE_DIRECTORY)

    # if(NOT ARG_DESTINATION_DIRECTORY)
    #     message(FATAL_ERROR "You must provide a destination directory")
    # endif(NOT ARG_DESTINATION_DIRECTORY)

    message("Provided sources are:")
    foreach(ARG_FILE ${ARG_FILES})
        message("- ${src}")
        add_custom_command(WORKING_DIRECTORY ${ARG_SOURCE_DIRECTORY}
                           DEPENDS stringify
                           OUTPUT ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/${ARG_FILE}.jit
                           COMMAND ${CUDF_BINARY_DIR}/stringify ${ARG_FILE} > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/${ARG_FILE}.jit
                           )
    endforeach()
endfunction()

# jit_pre(SOURCE_DIRECTORY      ${CUDF_SOURCE_DIR}
#         DESTINATION_DIRECTORY ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf_src
#         FILES                 rolling/rolling_jit_detail.hpp
#         )

jit_pre(SOURCE_DIRECTORY      ${CUDF_SOURCE_DIR}/include
        FILES                 cudf/types.hpp
                              cudf/utilities/bit.hpp
                              cudf/wrappers/timestamps.hpp
                              cudf/fixed_point/fixed_point.hpp
                              cudf/wrappers/durations.hpp
        )

# jit_pre(SOURCE_DIRECTORY      ${LIBCUDACXX_INCLUDE_DIR}
#         FILES                 cuda/std/climits
#                               cuda/std/cstddef
#                               cuda/std/cstdint
#                               cuda/std/ctime
#                               cuda/std/limits
#                               cuda/std/ratio
#                               cuda/std/type_traits
#                               cuda/std/version
#                               cuda/std/detail/__config
#                               cuda/std/detail/__pragma_pop
#                               cuda/std/detail/__pragma_push
#                               cuda/std/detail/libcxx/include/__config
#                               cuda/std/detail/libcxx/include/__pragma_pop
#                               cuda/std/detail/libcxx/include/__pragma_pus
#                               cuda/std/detail/libcxx/include/__undef_macr
#                               cuda/std/detail/libcxx/include/chrono
#                               cuda/std/detail/libcxx/include/climits
#                               cuda/std/detail/libcxx/include/cstddef
#                               cuda/std/detail/libcxx/include/cstdint
#                               cuda/std/detail/libcxx/include/ctime
#                               cuda/std/detail/libcxx/include/limits
#                               cuda/std/detail/libcxx/include/ratio
#                               cuda/std/detail/libcxx/include/type_traits
#                               cuda/std/detail/libcxx/include/version
#         )

add_custom_command(WORKING_DIRECTORY ${CUDF_SOURCE_DIR}/include
                   COMMENT "Stringify headers for use in JIT compiled code"
                   DEPENDS stringify
                   OUTPUT # ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/types.hpp.jit
                        #   ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/utilities/bit.hpp.jit
                        #   ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/wrappers/timestamps.hpp.jit
                        #   ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/fixed_point/fixed_point.hpp.jit
                        #   ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/wrappers/durations.hpp.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/chrono.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/climits.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/cstddef.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/cstdint.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/ctime.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/limits.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/ratio.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/type_traits.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/version.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/__config.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/__pragma_pop.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/__pragma_push.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/__config.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/__pragma_pop.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/__pragma_push.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/__undef_macros.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/chrono.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/climits.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/cstddef.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/cstdint.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/ctime.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/limits.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/ratio.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/type_traits.jit
                          ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/version.jit
                   MAIN_DEPENDENCY # ${CUDF_SOURCE_DIR}/include/cudf/types.hpp
                                #    ${CUDF_SOURCE_DIR}/include/cudf/utilities/bit.hpp
                                #    ${CUDF_SOURCE_DIR}/include/cudf/wrappers/timestamps.hpp
                                #    ${CUDF_SOURCE_DIR}/include/cudf/fixed_point/fixed_point.hpp
                                #    ${CUDF_SOURCE_DIR}/include/cudf/wrappers/durations.hpp
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/chrono
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/climits
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/cstddef
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/cstdint
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/ctime
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/limits
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/ratio
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/type_traits
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/version
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__config
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__pragma_pop
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__pragma_push
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__config
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__pragma_pop
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__pragma_push
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__undef_macros
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/chrono
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/climits
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/cstddef
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/cstdint
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/ctime
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/limits
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/ratio
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/type_traits
                                   ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/version

                   # stringified headers are placed underneath the bin include jit directory and end in ".jit"
                #   COMMAND ${CUDF_BINARY_DIR}/stringify cudf/types.hpp > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/types.hpp.jit
                #    COMMAND ${CUDF_BINARY_DIR}/stringify cudf/utilities/bit.hpp > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/utilities/bit.hpp.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ../src/rolling/rolling_jit_detail.hpp > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/rolling_jit_detail.hpp.jit
                #    COMMAND ${CUDF_BINARY_DIR}/stringify cudf/wrappers/timestamps.hpp > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/wrappers/timestamps.hpp.jit
                #    COMMAND ${CUDF_BINARY_DIR}/stringify cudf/fixed_point/fixed_point.hpp > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/fixed_point/fixed_point.hpp.jit
                #    COMMAND ${CUDF_BINARY_DIR}/stringify cudf/wrappers/durations.hpp > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/wrappers/durations.hpp.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/chrono cuda_std_chrono > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/chrono.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/climits cuda_std_climits > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/climits.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/cstddef cuda_std_cstddef > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/cstddef.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/cstdint cuda_std_cstdint > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/cstdint.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/ctime cuda_std_ctime > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/ctime.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/limits cuda_std_limits > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/limits.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/ratio cuda_std_ratio > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/ratio.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/type_traits cuda_std_type_traits > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/type_traits.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/version cuda_std_version > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/version.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__config cuda_std_detail___config > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/__config.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__pragma_pop cuda_std_detail___pragma_pop > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/__pragma_pop.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__pragma_push cuda_std_detail___pragma_push > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/__pragma_push.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__config cuda_std_detail_libcxx_include___config > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/__config.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__pragma_pop cuda_std_detail_libcxx_include___pragma_pop > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/__pragma_pop.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__pragma_push cuda_std_detail_libcxx_include___pragma_push > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/__pragma_push.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__undef_macros cuda_std_detail_libcxx_include___undef_macros > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/__undef_macros.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/chrono cuda_std_detail_libcxx_include_chrono > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/chrono.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/climits cuda_std_detail_libcxx_include_climits > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/climits.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/cstddef cuda_std_detail_libcxx_include_cstddef > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/cstddef.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/cstdint cuda_std_detail_libcxx_include_cstdint > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/cstdint.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/ctime cuda_std_detail_libcxx_include_ctime > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/ctime.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/limits cuda_std_detail_libcxx_include_limits > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/limits.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/ratio cuda_std_detail_libcxx_include_ratio > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/ratio.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/type_traits cuda_std_detail_libcxx_include_type_traits > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/type_traits.jit
                   COMMAND ${CUDF_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/version cuda_std_detail_libcxx_include_version > ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/version.jit
                   )

add_custom_target(stringify_run DEPENDS
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/types.hpp.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/utilities/bit.hpp.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/wrappers/timestamps.hpp.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/fixed_point/fixed_point.hpp.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cudf/wrappers/durations.hpp.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/chrono.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/climits.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/cstddef.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/cstdint.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/ctime.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/limits.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/ratio.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/type_traits.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/version.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/__config.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/__pragma_pop.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/__pragma_push.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/__config.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/__pragma_pop.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/__pragma_push.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/__undef_macros.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/chrono.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/climits.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/cstddef.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/cstdint.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/ctime.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/limits.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/ratio.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/type_traits.jit
                  ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_pre/cuda/std/detail/libcxx/include/version.jit
                  )

###################################################################################################
# - copy libcu++ ----------------------------------------------------------------------------------

# `${LIBCUDACXX_INCLUDE_DIR}/` specifies that the contents of this directory will be installed (not the directory itself)
file(INSTALL "${LIBCUDACXX_INCLUDE_DIR}/" DESTINATION "${CUDF_GENERATED_INCLUDE_DIR}/include/libcudacxx")
file(INSTALL "${LIBCXX_INCLUDE_DIR}"      DESTINATION "${CUDF_GENERATED_INCLUDE_DIR}/include/libcxx")
