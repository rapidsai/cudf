#=============================================================================
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

# Creates executable stringify and uses it to convert types.h to c-str for use in JIT code
add_executable(stringify "${JITIFY_INCLUDE_DIR}/stringify.cpp")

execute_process(WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include
    )

add_custom_command(WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
                   COMMENT "Stringify headers for use in JIT compiled code"
                   DEPENDS stringify
                   OUTPUT ${CMAKE_BINARY_DIR}/include/jit/types.h.jit
                          ${CMAKE_BINARY_DIR}/include/jit/types.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/bit.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/timestamps.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/fixed_point.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/durations.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/chrono.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/climits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/cstddef.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/cstdint.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/ctime.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/limits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/ratio.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/type_traits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/version.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__config.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__pragma_pop.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__pragma_push.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__config.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_pop.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_push.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__undef_macros.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/chrono.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/climits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/cstddef.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/cstdint.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/ctime.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/limits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/ratio.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/type_traits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/version.jit
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/types.h
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/types.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/utilities/bit.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/wrappers/timestamps.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/fixed_point/fixed_point.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/wrappers/durations.hpp
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
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/types.h > ${CMAKE_BINARY_DIR}/include/jit/types.h.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/types.hpp > ${CMAKE_BINARY_DIR}/include/jit/types.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/utilities/bit.hpp > ${CMAKE_BINARY_DIR}/include/jit/bit.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ../src/rolling/rolling_jit_detail.hpp > ${CMAKE_BINARY_DIR}/include/jit/rolling_jit_detail.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/wrappers/timestamps.hpp > ${CMAKE_BINARY_DIR}/include/jit/timestamps.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/fixed_point/fixed_point.hpp > ${CMAKE_BINARY_DIR}/include/jit/fixed_point.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/wrappers/durations.hpp > ${CMAKE_BINARY_DIR}/include/jit/durations.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/chrono cuda_std_chrono > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/chrono.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/climits cuda_std_climits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/climits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/cstddef cuda_std_cstddef > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/cstddef.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/cstdint cuda_std_cstdint > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/cstdint.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/ctime cuda_std_ctime > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/ctime.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/limits cuda_std_limits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/limits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/ratio cuda_std_ratio > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/ratio.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/type_traits cuda_std_type_traits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/type_traits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/version cuda_std_version > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/version.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__config cuda_std_detail___config > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__config.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__pragma_pop cuda_std_detail___pragma_pop > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__pragma_pop.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/__pragma_push cuda_std_detail___pragma_push > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__pragma_push.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__config cuda_std_detail_libcxx_include___config > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__config.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__pragma_pop cuda_std_detail_libcxx_include___pragma_pop > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_pop.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__pragma_push cuda_std_detail_libcxx_include___pragma_push > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_push.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/__undef_macros cuda_std_detail_libcxx_include___undef_macros > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__undef_macros.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/chrono cuda_std_detail_libcxx_include_chrono > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/chrono.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/climits cuda_std_detail_libcxx_include_climits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/climits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/cstddef cuda_std_detail_libcxx_include_cstddef > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/cstddef.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/cstdint cuda_std_detail_libcxx_include_cstdint > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/cstdint.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/ctime cuda_std_detail_libcxx_include_ctime > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/ctime.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/limits cuda_std_detail_libcxx_include_limits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/limits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/ratio cuda_std_detail_libcxx_include_ratio > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/ratio.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/type_traits cuda_std_detail_libcxx_include_type_traits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/type_traits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/cuda/std/detail/libcxx/include/version cuda_std_detail_libcxx_include_version > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/version.jit
                   )

add_custom_target(stringify_run DEPENDS
                  ${CMAKE_BINARY_DIR}/include/jit/types.h.jit
                  ${CMAKE_BINARY_DIR}/include/jit/types.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/bit.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/timestamps.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/fixed_point.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/durations.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/chrono.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/climits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/cstddef.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/cstdint.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/ctime.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/limits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/ratio.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/type_traits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/version.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__config.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__pragma_pop.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/__pragma_push.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__config.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_pop.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_push.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/__undef_macros.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/chrono.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/climits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/cstddef.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/cstdint.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/ctime.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/limits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/ratio.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/type_traits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/cuda/std/detail/libcxx/include/version.jit
                  )
