cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDF_TESTS LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME} ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread cudf)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${GTEST_INCLUDE_DIR}"
                    "${ARROW_INCLUDE_DIR}"
                    "${FLATBUFFERS_INCLUDE_DIR}"
                    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_SOURCE_DIR}/thirdparty/moderngpu/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cnmem/include")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${FLATBUFFERS_LIBRARY_DIR}"
                 "${GTEST_LIBRARY_DIR}")

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - column tests ----------------------------------------------------------------------------------

set(COLUMN_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/column/column_test.cu")

ConfigureTest(COLUMN_TEST "${COLUMN_TEST_SRC}")

###################################################################################################
# - cudf tests ------------------------------------------------------------------------------------

set(CUDF_INTERNAL_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/cudf/cudf_internal_test.cu")

ConfigureTest(CUDF_INTERNAL_TEST "${CUDF_INTERNAL_TEST_SRC}")

###################################################################################################
# - filter tests ----------------------------------------------------------------------------------

set(FILTER_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/filter/helper/utils.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/filter/helper/utils.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/filter/test_example.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/filter/test_filter_ops.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/filter/test_gpu_concat.cu")

ConfigureTest(FILTER_TEST "${FILTER_TEST_SRC}")

###################################################################################################
# - groupby tests ---------------------------------------------------------------------------------

set(GROUPBY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/groupby_test.cu")

ConfigureTest(GROUPBY_TEST "${GROUPBY_TEST_SRC}")

###################################################################################################
# - join tests ------------------------------------------------------------------------------------

set(JOIN_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/join/join_tests.cu")

ConfigureTest(JOIN_TEST "${JOIN_TEST_SRC}")

###################################################################################################
# - orderby tests ---------------------------------------------------------------------------------

set(ORDERBY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/orderby/orderby_tests.cu")

ConfigureTest(ORDERBY_TEST "${ORDERBY_TEST_SRC}")

###################################################################################################
# - bitmask tests ---------------------------------------------------------------------------------

set(BITMASK_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/valid_ops_test.cu")

ConfigureTest(BITMASK_TEST "${BITMASK_TEST_SRC}")

###################################################################################################
# - datetime tests --------------------------------------------------------------------------------

set(DATETIME_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/datetime/datetime_ops_test.cu")

ConfigureTest(DATETIME_TEST "${DATETIME_TEST_SRC}")

###################################################################################################
# - hashing tests ---------------------------------------------------------------------------------

set(HASHING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/hashing/hash_partition_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/hashing/hash_test.cu")

ConfigureTest(HASHING_TEST "${HASHING_TEST_SRC}")

###################################################################################################
# - hash_map tests --------------------------------------------------------------------------------

set(HASH_MAP_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/hash_map/map_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/hash_map/multimap_test.cu")

ConfigureTest(HASH_MAP_TEST "${HASH_MAP_TEST_SRC}")

###################################################################################################
# - quantiles tests -------------------------------------------------------------------------------

set(QUANTILES_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/quantiles/quantiles_test.cu")

ConfigureTest(QUANTILES_TEST "${QUANTILES_TEST_SRC}")

###################################################################################################
# - replace tests ---------------------------------------------------------------------------------

set(REPLACE_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/replace/replace_tests.cu")

ConfigureTest(REPLACE_TEST "${REPLACE_TEST_SRC}")

###################################################################################################
# - unary tests -----------------------------------------------------------------------------------

set(UNARY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/unary/unary_ops_test.cu")

ConfigureTest(UNARY_TEST "${UNARY_TEST_SRC}")

###################################################################################################
# - csv tests -------------------------------------------------------------------------------------

set(CSV_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/csv/csv_test.cu")

ConfigureTest(CSV_TEST "${CSV_TEST_SRC}")

###################################################################################################
# - rmm tests -------------------------------------------------------------------------------------

set(RMM_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/rmm/memory_tests.cpp")

ConfigureTest(RMM_TEST "${RMM_TEST_SRC}")

###################################################################################################
# - types tests -------------------------------------------------------------------------------------

set(TYPES_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/types/types_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/types/dispatcher_test.cu")

ConfigureTest(TYPES_TEST "${TYPES_TEST_SRC}")

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()








