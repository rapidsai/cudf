cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(KAFKA_TESTS LANGUAGES C CXX CUDA)

# TODO: Since we have no actual CUDA code in cudf_kafka this should be removed in the future
# in favor of using FindCUDAToolkit to get the needed CUDA include headers
if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif(NOT CMAKE_CUDA_COMPILER)


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
# - compiler function -----------------------------------------------------------------------------

set(CUDF_KAFKA_TEST_LIST CACHE INTERNAL "CUDF_KAFKA_TEST_LIST")

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                    ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gtest_main cudf pthread cudf_kafka "${RDKAFKA++_LIBRARY}")
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    set(CUDF_KAFKA_TEST_LIST ${CUDF_KAFKA_TEST_LIST} ${CMAKE_TEST_NAME} CACHE INTERNAL "CUDF_KAFKA_TEST_LIST")
endfunction(ConfigureTest)

###################################################################################################
# cudf_kafka - librdkafka -------------------------------------------------------------------------

find_path(RDKAFKA_INCLUDE "librdkafka" HINTS "$ENV{RDKAFKA_ROOT}/include")
find_library(RDKAFKA++_LIBRARY "rdkafka++" HINTS "$ENV{RDKAFKA_ROOT}/lib" "$ENV{RDKAFKA_ROOT}/build")

message(STATUS "RDKAFKA: RDKAFKA++_LIBRARY set to ${RDKAFKA++_LIBRARY}")
message(STATUS "RDKAFKA: RDKAFKA_INCLUDE set to ${RDKAFKA_INCLUDE}")

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}../../../tests"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${GTEST_INCLUDE_DIR}"
                    "${RDKAFKA_INCLUDE}")

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
                 "${RMM_LIBRARY}"
                 "${RDKAFKA++_LIBRARY}")

if(CONDA_LINK_DIRS)
    link_directories("${CONDA_LINK_DIRS}")
endif(CONDA_LINK_DIRS)

###################################################################################################
# - create tests ----------------------------------------------------------------------------------

ConfigureTest(CUDF_KAFKA_HOST_READ  "${CMAKE_CURRENT_SOURCE_DIR}/kafka_consumer_tests.cpp")

enable_testing()
