cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDF_TESTS LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME} ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock_main gmock GTest::GTest cudart cudf arrow)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/.."
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/cub"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/moderngpu/src"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/cnmem/include")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib")

###################################################################################################
# - source paths ----------------------------------------------------------------------------------

add_subdirectory(core)
add_subdirectory(functions)
add_subdirectory(io)
add_subdirectory(rmm)

###################################################################################################
# - enable testing --------------------------------------------------------------------------------

enable_testing()








