cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(NVSTRINGS_TESTS LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif()

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

set(NVSTRINGS_TEST_LIST CACHE INTERNAL "NVSTRINGS_TEST_LIST")

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                   ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread
                          rmm cudart cuda libNVCategory libNVStrings libNVText)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    set(NVSTRINGS_TEST_LIST ${NVSTRINGS_TEST_LIST} ${CMAKE_TEST_NAME} CACHE INTERNAL "NVSTRINGS_TEST_LIST")
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${GTEST_INCLUDE_DIR}"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${RMM_INCLUDE}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}"
                 "${GTEST_LIBRARY_DIR}"
                 "${RMM_LIBRARY}")

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - create tests ----------------------------------------------------------------------------------

ConfigureTest(NVSTRINGS_ARRAY_TEST   "${CMAKE_CURRENT_SOURCE_DIR}/test_array.cu")
ConfigureTest(NVSTRINGS_ATTRS_TEST   "${CMAKE_CURRENT_SOURCE_DIR}/test_attrs.cu")
ConfigureTest(NVSTRINGS_CASE_TEST    "${CMAKE_CURRENT_SOURCE_DIR}/test_case.cpp")
ConfigureTest(NVSTRINGS_COMBINE_TEST "${CMAKE_CURRENT_SOURCE_DIR}/test_combine.cpp")
ConfigureTest(NVSTRINGS_CONVERT_TEST "${CMAKE_CURRENT_SOURCE_DIR}/test_convert.cu")
# ConfigureTest(NVSTRINGS_COUNT_TEST   "${CMAKE_CURRENT_SOURCE_DIR}/test_count.cu")
ConfigureTest(NVSTRINGS_DATE_TEST    "${CMAKE_CURRENT_SOURCE_DIR}/test_datetime.cu")
ConfigureTest(NVSTRINGS_EXTRACT_TEST "${CMAKE_CURRENT_SOURCE_DIR}/test_extract.cpp")
ConfigureTest(NVSTRINGS_FIND_TEST    "${CMAKE_CURRENT_SOURCE_DIR}/test_find.cu")
ConfigureTest(NVSTRINGS_MODIFY_TEST  "${CMAKE_CURRENT_SOURCE_DIR}/test_modify.cu")
ConfigureTest(NVSTRINGS_PAD_TEST     "${CMAKE_CURRENT_SOURCE_DIR}/test_pad.cpp")
ConfigureTest(NVSTRINGS_STRIP_TEST   "${CMAKE_CURRENT_SOURCE_DIR}/test_strip.cpp")
ConfigureTest(NVSTRINGS_SPLIT_TEST   "${CMAKE_CURRENT_SOURCE_DIR}/test_split.cpp")
ConfigureTest(NVSTRINGS_REPLACE_TEST "${CMAKE_CURRENT_SOURCE_DIR}/test_replace.cpp")
ConfigureTest(NVSTRINGS_TEXT_TEST    "${CMAKE_CURRENT_SOURCE_DIR}/test_text.cu")
ConfigureTest(NVSTRINGS_URL_TEST     "${CMAKE_CURRENT_SOURCE_DIR}/test_url.cpp")
