cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDF_TESTS LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                   ${CMAKE_TEST_SRC}
                   "${CMAKE_CURRENT_SOURCE_DIR}/utilities/cudf_test_utils.cu")
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread cudf)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_SOURCE_DIR}/thirdparty/jitify"
                    "${CMAKE_SOURCE_DIR}/thirdparty/dlpack/include"
                    "${GTEST_INCLUDE_DIR}"
                    "${ARROW_INCLUDE_DIR}"
                    "${FLATBUFFERS_INCLUDE_DIR}"
                    "${RMM_INCLUDE}"
                    "${NVSTRINGS_INCLUDE}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${FLATBUFFERS_LIBRARY_DIR}"
                 "${GTEST_LIBRARY_DIR}"
                 "${RMM_LIBRARY}"
                 "${NVSTRINGS_LIBRARY}"
                 "${NVCATEGORY_LIBRARY}"
                 "${NVTEXT_LIBRARY}")

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

set(ERROR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/error/error_handling_test.cu")

ConfigureTest(ERROR_TEST "${ERROR_TEST_SRC}")

###################################################################################################
# - groupby tests ---------------------------------------------------------------------------------

set(GROUPBY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/groupby_test.cu")

ConfigureTest(GROUPBY_TEST "${GROUPBY_TEST_SRC}")

###################################################################################################
# - join tests ------------------------------------------------------------------------------------

set(JOIN_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities/valid_vectors.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/join/join_tests.cu")

ConfigureTest(JOIN_TEST "${JOIN_TEST_SRC}")

###################################################################################################
# - orderby tests ---------------------------------------------------------------------------------

set(ORDERBY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities/valid_vectors.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/orderby/orderby_tests.cu")

ConfigureTest(ORDERBY_TEST "${ORDERBY_TEST_SRC}")

###################################################################################################
# - bitmask tests ---------------------------------------------------------------------------------

set(BITMASK_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/bitmask_ops_test.cu")

ConfigureTest(BITMASK_TEST "${BITMASK_TEST_SRC}")

###################################################################################################
# - bit_mask tests --------------------------------------------------------------------------------

set(BIT_MASK_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/bit_mask_test.cu")

ConfigureTest(BIT_MASK_TEST "${BIT_MASK_TEST_SRC}")

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
# - reduction tests ---------------------------------------------------------------------------------

set(REDUCTION_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/reductions/reduction_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/reductions/scan_tests.cu")

ConfigureTest(REDUCTION_TEST "${REDUCTION_TEST_SRC}")

###################################################################################################
# - replace tests ---------------------------------------------------------------------------------

set(REPLACE_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/replace/replace_tests.cu")

ConfigureTest(REPLACE_TEST "${REPLACE_TEST_SRC}")

set(REPLACE_NULLS_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/replace/replace-nulls_tests.cu")

ConfigureTest(REPLACE_NULLS_TEST "${REPLACE_NULLS_TEST_SRC}")

###################################################################################################
# - unary tests -----------------------------------------------------------------------------------

set(UNARY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/unary/unary_ops_test.cu")

ConfigureTest(UNARY_TEST "${UNARY_TEST_SRC}")

###################################################################################################
# - binary tests ----------------------------------------------------------------------------------

set(BINARY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/binary/unit/binop-verify-input-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binary/integration/binary-operation-operands-null-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binary/integration/binary-operation-integration-test.cpp")

ConfigureTest(BINARY_TEST "${BINARY_TEST_SRC}")

###################################################################################################
# - io tests -------------------------------------------------------------------------------------

set(DECOMPRESSION_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/comp/decomp_test.cu")
set(CSV_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/csv/csv_test.cu")

ConfigureTest(DECOMPRESSION_TEST "${DECOMPRESSION_TEST_SRC}")
ConfigureTest(CSV_TEST "${CSV_TEST_SRC}")

###################################################################################################
# - sort tests ------------------------------------------------------------------------------------

set(SORT_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/sort/digitize_test.cu")

ConfigureTest(SORT_TEST "${SORT_TEST_SRC}")

###################################################################################################
# - types tests -----------------------------------------------------------------------------------

set(TYPES_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/types/types_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/types/dispatcher_test.cu")

ConfigureTest(TYPES_TEST "${TYPES_TEST_SRC}")

###################################################################################################
# - nvcategory tests -------------------------------------------------------------------------------

set(NVCATEGORY_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nvcategory/nvcategory_test.cu")

ConfigureTest(NVCATEGORY_TEST "${NVCATEGORY_TEST_SRC}")

###################################################################################################
# - DLPack tests -------------------------------------------------------------------------------------

set(DLPACK_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/convert/dlpack_test.cu")

ConfigureTest(DLPACK_TEST "${DLPACK_TEST_SRC}")

###################################################################################################
# - copying tests ---------------------------------------------------------------------------------

set(COPYING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/gather_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/scatter_tests.cu")

ConfigureTest(COPYING_TEST "${COPYING_TEST_SRC}")

###################################################################################################
# - utilities tests -------------------------------------------------------------------------------

set(UTILITIES_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities_tests/column_wrapper_tests.cu")

ConfigureTest(UTILITIES_TEST "${UTILITIES_TEST_SRC}")

###################################################################################################
# - device atomics tests -------------------------------------------------------------------------------

set(DEVICE_ATOMICS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/device_atomics/device_atomics_test.cu")

ConfigureTest(DEVICE_ATOMICS_TEST  "${DEVICE_ATOMICS_TEST_SRC}")

###################################################################################################
# - transpose tests -------------------------------------------------------------------------------

set(TRANSPOSE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/transpose/transpose_test.cu")

ConfigureTest(TRANSPOSE_TEST "${TRANSPOSE_TEST_SRC}")

###################################################################################################
# - table tests -----------------------------------------------------------------------------------

set(TABLE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/table/table_tests.cu")

ConfigureTest(TABLE_TEST "${TABLE_TEST_SRC}")
###################################################################################################
# - device table tests -----------------------------------------------------------------------------------

set(DEVICE_TABLE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/device_table/device_table_tests.cu")

ConfigureTest(DEVICE_TABLE_TEST "${DEVICE_TABLE_TEST_SRC}")

###################################################################################################
# - json tests -------------------------------------------------------------------------------------

set(JSON_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/json/json_test.cpp")

ConfigureTest(JSON_TEST "${JSON_TEST_SRC}")

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()








