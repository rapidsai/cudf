###################################################################################################
# - jitify ----------------------------------------------------------------------------------------

# Creates executable stringify and uses it to convert types.h to c-str for use in JIT code
add_executable(stringify "${JITIFY_INCLUDE_DIR}/stringify.cpp")
execute_process(WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/include/jit
        ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/details
        ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt
        ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include)

add_custom_command(WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
                   COMMENT "Stringify headers for use in JIT compiled code"
                   DEPENDS stringify
                   OUTPUT ${CMAKE_BINARY_DIR}/include/jit/types.h.jit
                          ${CMAKE_BINARY_DIR}/include/jit/types.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/bit.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/timestamps.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/fixed_point.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/durations.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/details/__config.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/limits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cfloat.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/chrono.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/ctime.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/ratio.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/type_traits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/version.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cmath.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cassert.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/__config.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/__undef_macros.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cfloat.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/chrono.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/ctime.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/limits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/ratio.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/type_traits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cmath.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cassert.jit
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/types.h
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/types.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/utilities/bit.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/wrappers/timestamps.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/fixed_point/fixed_point.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/wrappers/durations.hpp
                                   ${LIBCUDACXX_INCLUDE_DIR}/details/__config
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/limits
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/cfloat
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/chrono
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/ctime
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/ratio
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/type_traits
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/version
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/cmath
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/cassert
                                   ${LIBCXX_INCLUDE_DIR}/__config
                                   ${LIBCXX_INCLUDE_DIR}/__undef_macros
                                   ${LIBCXX_INCLUDE_DIR}/cfloat
                                   ${LIBCXX_INCLUDE_DIR}/chrono
                                   ${LIBCXX_INCLUDE_DIR}/ctime
                                   ${LIBCXX_INCLUDE_DIR}/limits
                                   ${LIBCXX_INCLUDE_DIR}/ratio
                                   ${LIBCXX_INCLUDE_DIR}/type_traits
                                   ${LIBCXX_INCLUDE_DIR}/cmath
                                   ${LIBCXX_INCLUDE_DIR}/cassert

                   # stringified headers are placed underneath the bin include jit directory and end in ".jit"
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/types.h > ${CMAKE_BINARY_DIR}/include/jit/types.h.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/types.hpp > ${CMAKE_BINARY_DIR}/include/jit/types.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/utilities/bit.hpp > ${CMAKE_BINARY_DIR}/include/bit.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ../src/rolling/rolling_jit_detail.hpp > ${CMAKE_BINARY_DIR}/include/rolling_jit_detail.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/wrappers/timestamps.hpp > ${CMAKE_BINARY_DIR}/include/jit/timestamps.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/fixed_point/fixed_point.hpp > ${CMAKE_BINARY_DIR}/include/jit/fixed_point.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/wrappers/durations.hpp > ${CMAKE_BINARY_DIR}/include/jit/durations.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/details/__config libcudacxx_details_config > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/details/__config.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/limits libcudacxx_simt_limits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/limits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/cfloat libcudacxx_simt_cfloat > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cfloat.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/chrono libcudacxx_simt_chrono > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/chrono.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/ctime libcudacxx_simt_ctime > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/ctime.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/ratio libcudacxx_simt_ratio > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/ratio.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/type_traits libcudacxx_simt_type_traits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/type_traits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/cmath libcudacxx_simt_cmath > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cmath.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/cassert libcudacxx_simt_cassert > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cassert.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/version libcudacxx_simt_version > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/version.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/__config libcxx_config > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/__config.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/__undef_macros libcxx_undef_macros > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/__undef_macros.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/cfloat libcxx_cfloat > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cfloat.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/chrono libcxx_chrono > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/chrono.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/ctime libcxx_ctime > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/ctime.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/limits libcxx_limits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/limits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/ratio libcxx_ratio > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/ratio.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/type_traits libcxx_type_traits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/type_traits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/cmath libcxx_cmath > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cmath.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/cassert libcxx_cassert > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cassert.jit
                   )

add_custom_target(stringify_run DEPENDS
                  ${CMAKE_BINARY_DIR}/include/jit/types.h.jit
                  ${CMAKE_BINARY_DIR}/include/jit/types.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/bit.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/timestamps.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/fixed_point.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/durations.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/details/__config.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/limits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cfloat.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/chrono.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/ctime.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/ratio.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/type_traits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cmath.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cassert.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/version.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/__config.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/__undef_macros.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cfloat.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/chrono.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/ctime.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/limits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/ratio.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/type_traits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cmath.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cassert.jit
                  )

add_dependencies(cudf stringify_run)
