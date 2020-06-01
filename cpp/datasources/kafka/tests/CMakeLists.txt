cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(KAFKA_DATASOURCE_TESTS LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif()

###################################################################################################
# - conda environment -----------------------------------------------------------------------------

if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{BUILD_PREFIX}/include" "$ENV{PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{BUILD_PREFIX}/lib" "$ENV{PREFIX}/lib")
    message(STATUS "Conda build detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Conda environment detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
endif("$ENV{CONDA_BUILD}" STREQUAL "1")

###################################################################################################
# - common test utils -----------------------------------------------------------------------------

# add_library(cudftestutil STATIC
#             "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/utilities/base_fixture.cpp"
#             "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/utilities/column_utilities.cu"
#             "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/utilities/scalar_utilities.cu"
#             "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/utilities/table_utilities.cu"
#             "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/strings/utilities.cu")

# set_property(TARGET cudftestutil PROPERTY POSITION_INDEPENDENT_CODE ON)

#target_link_libraries(cudftestutil cudf)

# install(TARGETS cudftestutil
#         DESTINATION lib
#         COMPONENT cudf)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

set(KAFKA_DATASOURCE_TEST_LIST CACHE INTERNAL "KAFKA_DATASOURCE_TEST_LIST")

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                    ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest cudf pthread)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    set(KAFKA_DATASOURCE_TEST_LIST ${KAFKA_DATASOURCE_TEST_LIST} ${CMAKE_TEST_NAME} CACHE INTERNAL "KAFKA_DATASOURCE_TEST_LIST")
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}../../../tests"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${GTEST_INCLUDE_DIR}")

if(CONDA_INCLUDE_DIRS)
  include_directories("${CONDA_INCLUDE_DIRS}")
endif(CONDA_INCLUDE_DIRS)

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}"
                 "${FLATBUFFERS_LIBRARY_DIR}"
                 "${GTEST_LIBRARY_DIR}"
                 "${RMM_LIBRARY}")

if(CONDA_LINK_DIRS)
    link_directories("${CONDA_LINK_DIRS}")
endif(CONDA_LINK_DIRS)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - create tests ----------------------------------------------------------------------------------

ConfigureTest(KAFKA_DATASOURCE_HOST_READ   "${CMAKE_CURRENT_SOURCE_DIR}/kafka_consumer_tests.cpp")

enable_testing()