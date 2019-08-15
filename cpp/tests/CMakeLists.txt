cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDF_TESTS LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif()

###################################################################################################
# - common test utils -----------------------------------------------------------------------------

add_library(cudftestutil STATIC
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/cudf_test_utils.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/valid_vectors.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/nvcategory_utils.cu")

target_link_libraries(cudftestutil cudf)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                   ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread cudf cudftestutil)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

option(CMAKE_ENABLE_BENCHMARKS "Enable building cuDF benchmarks" OFF)

function(ConfigureBench CMAKE_BENCH_NAME CMAKE_BENCH_SRC)
    if(CMAKE_ENABLE_BENCHMARKS)
        add_executable(${CMAKE_BENCH_NAME}
                    ${CMAKE_BENCH_SRC})
        set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(${CMAKE_BENCH_NAME} cudf cudftestutil)
        set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES
                                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench")
        add_test(NAME ${CMAKE_BENCH_NAME} COMMAND ${CMAKE_BENCH_NAME})
    endif(CMAKE_ENABLE_BENCHMARKS)
endfunction(ConfigureBench)


###################################################################################################
# - include paths ---------------------------------------------------------------------------------

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_SOURCE_DIR}/thirdparty/jitify"
                    "${CMAKE_SOURCE_DIR}/thirdparty/dlpack/include"
                    "${GTEST_INCLUDE_DIR}"
                    "${ARROW_INCLUDE_DIR}"
                    "${FLATBUFFERS_INCLUDE_DIR}"
                    "${ZLIB_INCLUDE_DIRS}"
                    "${Boost_INCLUDE_DIRS}"
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
# - legacy column tests ----------------------------------------------------------------------------------

set(LEGACY_COLUMN_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/column/legacy/column_test.cu")

ConfigureTest(LEGACY_COLUMN_TEST "${LEGACY_COLUMN_TEST_SRC}")

###################################################################################################
# - cudf tests ------------------------------------------------------------------------------------

set(ERROR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/error/error_handling_test.cu")

ConfigureTest(ERROR_TEST "${ERROR_TEST_SRC}")

###################################################################################################
# - groupby tests ---------------------------------------------------------------------------------

set(HASH_GROUPBY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/hash/single_column_count.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/hash/single_column_sum.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/hash/single_column_avg.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/hash/single_column_max.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/hash/single_column_min.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/hash/single_column_multi_agg.cu")

ConfigureTest(HASH_GROUPBY_TEST "${HASH_GROUPBY_TEST_SRC}")

set(GROUPBY_WO_AGG_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/without_agg/groupby_wo_agg_test.cu")

ConfigureTest(GROUPBY_WO_AGG_TEST "${GROUPBY_WO_AGG_TEST_SRC}")

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
# - is_sorted tests ---------------------------------------------------------------------------------

set(IS_SORTED_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/orderby/is_sorted/is_sorted_tests.cu")

ConfigureTest(IS_SORTED_TEST "${IS_SORTED_TEST_SRC}")

###################################################################################################
# - legacy bitmask tests ---------------------------------------------------------------------------------

set(LEGACY_BITMASK_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/legacy/bitmask_ops_test.cu")

ConfigureTest(LEGACY_BITMASK_TEST "${LEGACY_BITMASK_TEST_SRC}")

###################################################################################################
# - nans_to_nulls tests ---------------------------------------------------------------------------

set(NANS_TO_NULLS_TEST_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/nans_to_nulls_tests.cu")

ConfigureTest(NANS_TO_NULLS_TEST "${NANS_TO_NULLS_TEST_SRC}")


###################################################################################################
# - legacy bit_mask tests --------------------------------------------------------------------------------

set(LEGACY_BIT_MASK_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/legacy/bit_mask_test.cu")

ConfigureTest(LEGACY_BIT_MASK_TEST "${LEGACY_BIT_MASK_TEST_SRC}")

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
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/unit/binop-verify-input-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/integration/binary-operation-operands-null-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/integration/binary-operation-integration-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/integration/binary-operation-generic-ptx-test.cu")

ConfigureTest(BINARY_TEST "${BINARY_TEST_SRC}")

###################################################################################################
# - unary transform tests -------------------------------------------------------------------------

set(TRANSFORM_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/transform/integration/unary-operation-integration-test.cu")

ConfigureTest(TRANSFORM_TEST "${TRANSFORM_TEST_SRC}")

###################################################################################################
# - jit cache tests -------------------------------------------------------------------------------

set(JITCACHE_TEST_SRC 
    "${CMAKE_SOURCE_DIR}/src/jit/cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/unit/jit-cache-test.cu")

ConfigureTest(JITCACHE_TEST "${JITCACHE_TEST_SRC}")

set(JITCACHE_MULTI_TEST_SRC 
    "${CMAKE_SOURCE_DIR}/src/jit/cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/unit/jit-cache-multiprocess-test.cpp")

ConfigureTest(JITCACHE_MULTIPROC_TEST "${JITCACHE_MULTI_TEST_SRC}")

###################################################################################################
# - io tests -------------------------------------------------------------------------------------

set(DECOMPRESSION_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/comp/decomp_test.cu")
set(AVRO_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/avro/avro_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/io_test_utils.cpp")
set(CSV_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/csv/csv_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/io_test_utils.cpp")
set(JSON_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/json/json_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/io_test_utils.cpp")

ConfigureTest(DECOMPRESSION_TEST "${DECOMPRESSION_TEST_SRC}")
ConfigureTest(AVRO_TEST "${AVRO_TEST_SRC}")
ConfigureTest(CSV_TEST "${CSV_TEST_SRC}")
ConfigureTest(JSON_TEST "${JSON_TEST_SRC}")

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
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/copy_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/copy_range_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/gather_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/scatter_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/scalar_scatter_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/slice_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/split_tests.cu")

ConfigureTest(COPYING_TEST "${COPYING_TEST_SRC}")

###################################################################################################
# - nvstring scatter tests ------------------------------------------------------------------------

set(NVSTRING_SCATTER_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/nvstring_scatter_tests.cu")

  ConfigureTest(NVSTRING_SCATTER_TEST "${NVSTRING_SCATTER_TEST_SRC}")

###################################################################################################
# - utilities tests -------------------------------------------------------------------------------

set(UTILITIES_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities_tests/column_wrapper_tests.cu")

ConfigureTest(UTILITIES_TEST "${UTILITIES_TEST_SRC}")

###################################################################################################
# - iterator tests -------------------------------------------------------------------------------

set(ITERATOR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/iterator/iterator_test.cu")

ConfigureTest(ITERATOR_TEST "${ITERATOR_TEST_SRC}")

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
# - sorted-merge tests -------------------------------------------------------------------------------------

set(MERGE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/merge/merge-test.cu")

ConfigureTest(MERGE_TEST "${MERGE_TEST_SRC}")

###################################################################################################
# - stream compaction tests -----------------------------------------------------------------------

set(STREAM_COMPACTION_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/apply_boolean_mask_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/drop_nulls_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/drop_duplicates_tests.cu")

ConfigureTest(STREAM_COMPACTION_TEST "${STREAM_COMPACTION_TEST_SRC}")

###################################################################################################
# - rolling tests ---------------------------------------------------------------------------------

set(ROLLING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/rolling/rolling_test.cu")

ConfigureTest(ROLLING_TEST "${ROLLING_TEST_SRC}")

###################################################################################################
# - point in polygon test --------------------------------------------------------------------------

set(GIS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/gis/point_in_polygon_test.cu")

ConfigureTest(POINT_IN_POLYGON_TEST "${GIS_TEST_SRC}")

###################################################################################################
# - filling test --------------------------------------------------------------------------

set(FILLING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/filling/filling_tests.cu")

ConfigureTest(FILLING_TEST "${FILLING_TEST_SRC}")

###################################################################################################
# - search test -----------------------------------------------------------------------------------

set(SEARCH_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/search/search_test.cu")

ConfigureTest(SEARCH_TEST "${SEARCH_TEST_SRC}")



###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()


###################################################################################################
### benchmarks ####################################################################################
###################################################################################################

# only built if CMAKE_ENABLE_BENCHMARKS is set to ON

###################################################################################################
# - iterator benchmarks-------------------------------------------------------------------

set(ITERATOR_BENCH_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/iterator/iterator_bench.cu")

ConfigureBench(ITERATOR_BENCH "${ITERATOR_BENCH_SRC}")
