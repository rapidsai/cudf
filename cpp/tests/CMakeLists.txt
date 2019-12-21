cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDF_TESTS LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif()

###################################################################################################
# - common test utils -----------------------------------------------------------------------------

add_library(cudftestutil STATIC
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/legacy/cudf_test_utils.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/legacy/valid_vectors.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/legacy/nvcategory_utils.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/column_utilities.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/scalar_utilities.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/table_utilities.cu")

target_link_libraries(cudftestutil cudf)

install(TARGETS cudftestutil
        DESTINATION lib
        COMPONENT cudf)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

set(CUDF_TEST_LIST CACHE INTERNAL "CUDF_TEST_LIST")

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                   ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread cudf cudftestutil)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    set(CUDF_TEST_LIST ${CUDF_TEST_LIST} ${CMAKE_TEST_NAME} CACHE INTERNAL "CUDF_TEST_LIST")
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
                    "${CMAKE_SOURCE_DIR}/thirdparty/libcudacxx/include"
                    "${GTEST_INCLUDE_DIR}"
                    "${ARROW_INCLUDE_DIR}"
                    "${FLATBUFFERS_INCLUDE_DIR}"
                    "${ZLIB_INCLUDE_DIRS}"
                    "${Boost_INCLUDE_DIRS}"
                    "${RMM_INCLUDE}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}"
                 "${FLATBUFFERS_LIBRARY_DIR}"
                 "${GTEST_LIBRARY_DIR}"
                 "${RMM_LIBRARY}")

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - legacy column tests ---------------------------------------------------------------------------

set(LEGACY_COLUMN_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/column/legacy/column_test.cu")

ConfigureTest(LEGACY_COLUMN_TEST "${LEGACY_COLUMN_TEST_SRC}")

###################################################################################################
# - column tests ----------------------------------------------------------------------------------

set(COLUMN_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/column/column_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/column/column_view_test.cpp")

ConfigureTest(COLUMN_TEST "${COLUMN_TEST_SRC}")

###################################################################################################
# - scalar tests ----------------------------------------------------------------------------------

set(SCALAR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/scalar/scalar_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/scalar/scalar_device_view_test.cu")

ConfigureTest(SCALAR_TEST "${SCALAR_TEST_SRC}")

# - bool8 wrapper tests ---------------------------------------------------------------------------

set(BOOLS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/wrappers/bools_test.cpp")

ConfigureTest(BOOLS_TEST "${BOOLS_TEST_SRC}")


###################################################################################################
# - timestamps tests ----------------------------------------------------------------------------------

set(TIMESTAMPS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/wrappers/timestamps_test.cu")

ConfigureTest(TIMESTAMPS_TEST "${TIMESTAMPS_TEST_SRC}")

###################################################################################################
# - cudf tests ------------------------------------------------------------------------------------

set(ERROR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/error/error_handling_test.cu")

ConfigureTest(ERROR_TEST "${ERROR_TEST_SRC}")

###################################################################################################
# - groupby tests ---------------------------------------------------------------------------------

set(LEGACY_HASH_GROUPBY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/hash/legacy/single_column_count.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/hash/legacy/single_column_sum.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/hash/legacy/single_column_avg.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/hash/legacy/single_column_max.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/hash/legacy/single_column_min.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/hash/legacy/single_column_multi_agg.cu")

ConfigureTest(LEGACY_HASH_GROUPBY_TEST "${LEGACY_HASH_GROUPBY_TEST_SRC}")

set(LEGACY_SORT_GROUPBY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/sort/ignore_null_keys/legacy/single_column_avg.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/sort/ignore_null_keys/legacy/single_column_sum.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/sort/ignore_null_keys/legacy/single_column_count.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/sort/ignore_null_keys/legacy/single_column_min.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/sort/ignore_null_keys/legacy/single_column_max.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/sort/ignore_null_keys/legacy/single_column_quantile.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/sort/ignore_null_keys/legacy/single_column_median.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/sort/ignore_null_keys/legacy/single_column_multi_agg.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/sort/ignore_null_keys/legacy/single_column_var_std.cu"

    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/sort/not_ignore_null_keys/legacy/single_column_sum.cu"
)

ConfigureTest(LEGACY_SORT_GROUPBY_TEST "${LEGACY_SORT_GROUPBY_TEST_SRC}")

set(LEGACY_GROUPBY_WO_AGG_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/without_agg/legacy/groupby_wo_agg_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/without_agg/legacy/single_column_groupby.cu"
    )

ConfigureTest(LEGACY_GROUPBY_WO_AGG_TEST "${LEGACY_GROUPBY_WO_AGG_TEST_SRC}")

###################################################################################################
# - legacy join tests ------------------------------------------------------------------------------------

set(LEGACY_JOIN_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/join/legacy/join_tests.cu")

ConfigureTest(LEGACY_JOIN_TEST "${LEGACY_JOIN_TEST_SRC}")

###################################################################################################
# - join tests ------------------------------------------------------------------------------------

set(JOIN_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/join/join_tests.cu")

ConfigureTest(JOIN_TEST "${JOIN_TEST_SRC}")

###################################################################################################
# - orderby tests ---------------------------------------------------------------------------------

set(LEGACY_ORDERBY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/orderby/legacy/orderby_tests.cu")

ConfigureTest(LEGACY_ORDERBY_TEST "${LEGACY_ORDERBY_TEST_SRC}")

###################################################################################################
# - is_sorted tests -------------------------------------------------------------------------------

set(IS_SORTED_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/sort/is_sorted_tests.cu")

ConfigureTest(IS_SORTED_TEST "${IS_SORTED_TEST_SRC}")

set(LEGACY_IS_SORTED_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/orderby/is_sorted/legacy/is_sorted_tests.cu")

ConfigureTest(LEGACY_IS_SORTED_TEST "${LEGACY_IS_SORTED_TEST_SRC}")

###################################################################################################
# - legacy bitmask tests --------------------------------------------------------------------------

set(LEGACY_BITMASK_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/legacy/bitmask_ops_test.cu")

ConfigureTest(LEGACY_BITMASK_TEST "${LEGACY_BITMASK_TEST_SRC}")

###################################################################################################
# - legacy nans_to_nulls tests --------------------------------------------------------------------

set(LEGACY_NANS_TO_NULLS_TEST_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/legacy/nans_to_nulls_tests.cu")

ConfigureTest(LEGACY_NANS_TO_NULLS_TEST "${LEGACY_NANS_TO_NULLS_TEST_SRC}")


###################################################################################################
# - legacy bit_mask tests -------------------------------------------------------------------------

set(LEGACY_BIT_MASK_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/legacy/bit_mask_test.cu")

ConfigureTest(LEGACY_BIT_MASK_TEST "${LEGACY_BIT_MASK_TEST_SRC}")

###################################################################################################
# - datetime tests --------------------------------------------------------------------------------

set(LEGACY_DATETIME_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/datetime/legacy/datetime_ops_test.cu")

ConfigureTest(LEGACY_DATETIME_TEST "${LEGACY_DATETIME_TEST_SRC}")

set(DATETIME_OPS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/datetime/datetime_ops_test.cu")

ConfigureTest(DATETIME_OPS_TEST "${DATETIME_OPS_TEST_SRC}")

###################################################################################################
# - hashing tests ---------------------------------------------------------------------------------

set(HASHING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/hashing/hash_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hashing/hash_partition_test.cpp")

ConfigureTest(HASHING_TEST "${HASHING_TEST_SRC}")

###################################################################################################
# - legacy hashing tests ---------------------------------------------------------------------------------

set(LEGACY_HASHING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/hashing/legacy/hash_partition_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/hashing/legacy/hash_test.cu")

ConfigureTest(LEGACY_HASHING_TEST "${LEGACY_HASHING_TEST_SRC}")

###################################################################################################
# - hash_map tests --------------------------------------------------------------------------------

set(HASH_MAP_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/hash_map/map_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/hash_map/multimap_test.cu")

ConfigureTest(HASH_MAP_TEST "${HASH_MAP_TEST_SRC}")

###################################################################################################
# - quantiles tests -------------------------------------------------------------------------------

set(LEGACY_QUANTILES_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/quantiles/legacy/group_quantiles_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/quantiles/legacy/quantiles_test.cu")

ConfigureTest(LEGACY_QUANTILES_TEST "${LEGACY_QUANTILES_TEST_SRC}")

set(QUANTILES_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/quantiles/quantiles_test.cu")

ConfigureTest(QUANTILES_TEST "${QUANTILES_TEST_SRC}")

###################################################################################################
# - reduction tests -------------------------------------------------------------------------------

set(LEGACY_REDUCTION_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/reductions/legacy/group_std_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/reductions/legacy/group_var_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/reductions/legacy/reduction_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/reductions/legacy/scan_tests.cu")

ConfigureTest(LEGACY_REDUCTION_TEST "${LEGACY_REDUCTION_TEST_SRC}")

###################################################################################################
# - legacy replace tests --------------------------------------------------------------------------

set(LEGACY_REPLACE_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/replace/legacy/replace_tests.cu")

ConfigureTest(LEGACY_REPLACE_TEST "${LEGACY_REPLACE_TEST_SRC}")

set(LEGACY_REPLACE_NULLS_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/replace/legacy/replace-nulls_tests.cu")

ConfigureTest(LEGACY_REPLACE_NULLS_TEST "${LEGACY_REPLACE_NULLS_TEST_SRC}")

###################################################################################################
# - replace tests --------------------------------------------------------------------------

set(REPLACE_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/replace/replace_tests.cu")

ConfigureTest(REPLACE_TEST "${REPLACE_TEST_SRC}")

set(REPLACE_NULLS_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/replace/replace_nulls_tests.cu")

ConfigureTest(REPLACE_NULLS_TEST "${REPLACE_NULLS_TEST_SRC}")

set(NORMALIZE_REPLACE_TEST_SRC
	"${CMAKE_CURRENT_SOURCE_DIR}/replace/normalize_replace_tests.cu")
	
ConfigureTest(NORMALIZE_REPLACE_TEST "${NORMALIZE_REPLACE_TEST_SRC}")

###################################################################################################
# - unary tests -----------------------------------------------------------------------------------

set(UNARY_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/unary/unary_ops_test.cu")

ConfigureTest(UNARY_TEST "${UNARY_TEST_SRC}")

###################################################################################################
# - legacy unary tests -----------------------------------------------------------------------------------

set(LEGACY_UNARY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/unary/legacy/unary_ops_test.cu")

ConfigureTest(LEGACY_UNARY_TEST "${LEGACY_UNARY_TEST_SRC}")

###################################################################################################
# - legacy binary tests ---------------------------------------------------------------------------

set(LEGACY_BINARY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/legacy/unit/binop-verify-input-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/legacy/integration/binary-operation-operands-null-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/legacy/integration/binary-operation-integration-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/legacy/integration/binary-operation-generic-ptx-test.cu")

ConfigureTest(LEGACY_BINARY_TEST "${LEGACY_BINARY_TEST_SRC}")

###################################################################################################
# - binary tests ----------------------------------------------------------------------------------

set(BINARY_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/binop-verify-input-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/binop-null-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/binop-integration-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/binop-generic-ptx-test.cu")

ConfigureTest(BINARY_TEST "${BINARY_TEST_SRC}")

###################################################################################################
# - legacy unary transform tests ------------------------------------------------------------------

set(LEGACY_TRANSFORM_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/transform/integration/legacy/unary-operation-integration-test.cu")

ConfigureTest(LEGACY_TRANSFORM_TEST "${LEGACY_TRANSFORM_TEST_SRC}")

###################################################################################################
# - unary transform tests -------------------------------------------------------------------------

set(TRANSFORM_TEST_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/transform/integration/unary-transform-test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/transform/nans_to_null_test.cpp")

ConfigureTest(TRANSFORM_TEST "${TRANSFORM_TEST_SRC}")

###################################################################################################
# - legacy jit cache tests ------------------------------------------------------------------------

set(LEGACY_JITCACHE_TEST_SRC 
    "${CMAKE_SOURCE_DIR}/src/jit/cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/legacy/unit/jit-cache-test.cu")

ConfigureTest(LEGACY_JITCACHE_TEST "${LEGACY_JITCACHE_TEST_SRC}")

set(LEGACY_JITCACHE_MULTI_TEST_SRC 
    "${CMAKE_SOURCE_DIR}/src/jit/cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/legacy/unit/jit-cache-multiprocess-test.cpp")

ConfigureTest(LEGACY_JITCACHE_MULTIPROC_TEST "${LEGACY_JITCACHE_MULTI_TEST_SRC}")

###################################################################################################
# - io tests --------------------------------------------------------------------------------------

set(DECOMPRESSION_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/comp/decomp_test.cu")
set(LEGACY_AVRO_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/legacy/avro_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/legacy/io_test_utils.cpp")
set(LEGACY_CSV_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/legacy/csv_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/legacy/io_test_utils.cpp")
set(LEGACY_JSON_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/legacy/json_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/legacy/io_test_utils.cpp")
set(LEGACY_ORC_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/legacy/orc_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/legacy/io_test_utils.cpp")

ConfigureTest(DECOMPRESSION_TEST "${DECOMPRESSION_TEST_SRC}")
ConfigureTest(LEGACY_AVRO_TEST "${LEGACY_AVRO_TEST_SRC}")
ConfigureTest(LEGACY_CSV_TEST "${LEGACY_CSV_TEST_SRC}")
ConfigureTest(LEGACY_JSON_TEST "${LEGACY_JSON_TEST_SRC}")
ConfigureTest(LEGACY_ORC_TEST "${LEGACY_ORC_TEST_SRC}")

set(CSV_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/csv_test.cu")
set(ORC_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/orc_test.cu")
set(PARQUET_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/parquet_test.cu")

ConfigureTest(CSV_TEST "${CSV_TEST_SRC}")
ConfigureTest(ORC_TEST "${ORC_TEST_SRC}")
ConfigureTest(PARQUET_TEST "${PARQUET_TEST_SRC}")

###################################################################################################
# - sort tests ------------------------------------------------------------------------------------

set(SORT_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/sort/sort_test.cpp")

ConfigureTest(SORT_TEST "${SORT_TEST_SRC}")

###################################################################################################
# - legacy sort tests -----------------------------------------------------------------------------

set(LEGACY_SORT_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/sort/legacy/digitize_test.cu")

ConfigureTest(LEGACY_SORT_TEST "${LEGACY_SORT_TEST_SRC}")

###################################################################################################
# - types tests -----------------------------------------------------------------------------------

set(LEGACY_TYPES_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/types/legacy/types_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/types/legacy/dispatcher_test.cu")

ConfigureTest(LEGACY_TYPES_TEST "${LEGACY_TYPES_TEST_SRC}")

###################################################################################################
# - nvcategory tests ------------------------------------------------------------------------------

set(NVCATEGORY_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nvcategory/nvcategory_test.cu")

ConfigureTest(NVCATEGORY_TEST "${NVCATEGORY_TEST_SRC}")

###################################################################################################
# - DLPack tests ----------------------------------------------------------------------------------

set(DLPACK_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/dlpack/dlpack_test.cpp")

ConfigureTest(DLPACK_TEST "${DLPACK_TEST_SRC}")

###################################################################################################
# - legacy DLPack tests ----------------------------------------------------------------------------------

set(LEGACY_DLPACK_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/convert/legacy/dlpack_test.cu")

ConfigureTest(LEGACY_DLPACK_TEST "${LEGACY_DLPACK_TEST_SRC}")

###################################################################################################
# - copying tests ---------------------------------------------------------------------------------

set(COPYING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/utility_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/gather_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/scatter_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/scatter_to_tables_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/copy_range_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/slice_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/split_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/copy_tests.cu")

ConfigureTest(COPYING_TEST "${COPYING_TEST_SRC}")

###################################################################################################
# - legacy copying tests --------------------------------------------------------------------------

set(LEGACY_COPYING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/legacy/copy_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/legacy/copy_range_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/legacy/gather_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/legacy/scatter_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/legacy/scalar_scatter_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/legacy/scatter_to_tables_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/legacy/slice_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/legacy/split_tests.cu")

ConfigureTest(LEGACY_COPYING_TEST "${LEGACY_COPYING_TEST_SRC}")

###################################################################################################
# - nvstring scatter tests ------------------------------------------------------------------------

set(LEGACY_NVSTRING_SCATTER_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/legacy/nvstring_scatter_tests.cu")

  ConfigureTest(LEGACY_NVSTRING_SCATTER_TEST "${LEGACY_NVSTRING_SCATTER_TEST_SRC}")

###################################################################################################
# - utilities tests -------------------------------------------------------------------------------

set(UTILITIES_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities_tests/type_list_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities_tests/column_utilities_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities_tests/column_wrapper_tests.cu")

ConfigureTest(UTILITIES_TEST "${UTILITIES_TEST_SRC}")

###################################################################################################
# - legacy utilities tests ------------------------------------------------------------------------

set(LEGACY_UTILITIES_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities_tests/legacy/column_wrapper_tests.cu")

ConfigureTest(LEGACY_UTILITIES_TEST "${LEGACY_UTILITIES_TEST_SRC}")

###################################################################################################
# - legacy iterator tests --------------------------------------------------------------------------------

set(LEGACY_ITERATOR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/iterator/legacy/iterator_test.cu")

ConfigureTest(LEGACY_ITERATOR_TEST "${LEGACY_ITERATOR_TEST_SRC}")

###################################################################################################
# - iterator tests --------------------------------------------------------------------------------

set(ITERATOR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/iterator/iterator_test.cu")

ConfigureTest(ITERATOR_TEST "${ITERATOR_TEST_SRC}")

###################################################################################################
# - legacy device atomics tests --------------------------------------------------------------------------

set(LEGACY_DEVICE_ATOMICS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/device_atomics/legacy/device_atomics_test.cu")

ConfigureTest(LEGACY_DEVICE_ATOMICS_TEST  "${LEGACY_DEVICE_ATOMICS_TEST_SRC}")

###################################################################################################
# - device atomics tests --------------------------------------------------------------------------

set(DEVICE_ATOMICS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/device_atomics/device_atomics_test.cu")

ConfigureTest(DEVICE_ATOMICS_TEST  "${DEVICE_ATOMICS_TEST_SRC}")

###################################################################################################
# - transpose tests -------------------------------------------------------------------------------

set(TRANSPOSE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/transpose/transpose_test.cu")

ConfigureTest(TRANSPOSE_TEST "${TRANSPOSE_TEST_SRC}")

###################################################################################################
# - legacy transpose tests -------------------------------------------------------------------------------

set(LEGACY_TRANSPOSE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/transpose/legacy/transpose_test.cu")

ConfigureTest(LEGACY_TRANSPOSE_TEST "${LEGACY_TRANSPOSE_TEST_SRC}")

###################################################################################################
# - table tests -----------------------------------------------------------------------------------

set(TABLE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/table/table_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/table/table_view_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/table/row_operators_tests.cu")

ConfigureTest(TABLE_TEST "${TABLE_TEST_SRC}")
###################################################################################################
# - legacy table tests ----------------------------------------------------------------------------

set(LEGACY_TABLE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/table/legacy/table_tests.cu")

ConfigureTest(LEGACY_TABLE_TEST "${LEGACY_TABLE_TEST_SRC}")
###################################################################################################
# - device table tests ----------------------------------------------------------------------------

set(DEVICE_TABLE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/device_table/device_table_tests.cu")

ConfigureTest(DEVICE_TABLE_TEST "${DEVICE_TABLE_TEST_SRC}")

###################################################################################################
# - sorted-merge tests ----------------------------------------------------------------------------

set(LEGACY_MERGE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/merge/legacy/merge-test.cu")

ConfigureTest(LEGACY_MERGE_TEST "${LEGACY_MERGE_TEST_SRC}")

###################################################################################################
# - sorted-merge new tests ----------------------------------------------------------------------------

set(MERGE_TEST_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/merge/merge-test.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/merge/merge_string_test.cpp")

ConfigureTest(MERGE_TEST "${MERGE_TEST_SRC}")

###################################################################################################
# - stream compaction tests -----------------------------------------------------------------------

set(STREAM_COMPACTION_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/apply_boolean_mask_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/drop_nulls_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/drop_duplicates_tests.cu")

ConfigureTest(STREAM_COMPACTION_TEST "${STREAM_COMPACTION_TEST_SRC}")

###################################################################################################
# - legacy stream compaction tests -----------------------------------------------------------------------

set(LEGACY_STREAM_COMPACTION_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/legacy/apply_boolean_mask_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/legacy/drop_nulls_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/legacy/drop_duplicates_tests.cu")

ConfigureTest(LEGACY_STREAM_COMPACTION_TEST "${LEGACY_STREAM_COMPACTION_TEST_SRC}")

###################################################################################################
# - legacy rolling tests ---------------------------------------------------------------------------------

set(LEGACY_ROLLING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/rolling/legacy/rolling_test.cu")

ConfigureTest(LEGACY_ROLLING_TEST "${LEGACY_ROLLING_TEST_SRC}")

###################################################################################################
# - rolling tests ---------------------------------------------------------------------------------

set(ROLLING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/rolling/rolling_test.cpp")

ConfigureTest(ROLLING_TEST "${ROLLING_TEST_SRC}")

###################################################################################################
# - filling test ----------------------------------------------------------------------------------

set(FILLING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/filling/fill_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/filling/repeat_tests.cu")

ConfigureTest(FILLING_TEST "${FILLING_TEST_SRC}")

###################################################################################################
# - legacy filling test ---------------------------------------------------------------------------

set(LEGACY_FILLING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/filling/legacy/filling_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/filling/legacy/repeating_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/filling/legacy/tiling_tests.cu")

ConfigureTest(LEGACY_FILLING_TEST "${LEGACY_FILLING_TEST_SRC}")

###################################################################################################
# - legacy search test ----------------------------------------------------------------------------

set(LEGACY_SEARCH_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/search/legacy/search_test.cu")

ConfigureTest(LEGACY_SEARCH_TEST "${LEGACY_SEARCH_TEST_SRC}")

###################################################################################################
# - search test -----------------------------------------------------------------------------------

set(SEARCH_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/search/search_test.cpp")

ConfigureTest(SEARCH_TEST "${SEARCH_TEST_SRC}")

###################################################################################################
# - reshape test ----------------------------------------------------------------------------------

set(LEGACY_RESHAPE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/reshape/legacy/stack_test.cu")

ConfigureTest(LEGACY_RESHAPE_TEST "${LEGACY_RESHAPE_TEST_SRC}")

###################################################################################################
# - traits test -----------------------------------------------------------------------------------

set(TRAITS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/types/traits_test.cpp")

ConfigureTest(TRAITS_TEST "${TRAITS_TEST_SRC}")

###################################################################################################
# - factories test --------------------------------------------------------------------------------

set(FACTORIES_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/scalar/factories_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/column/factories_test.cpp")

ConfigureTest(FACTORIES_TEST "${FACTORIES_TEST_SRC}")

###################################################################################################
# - dispatcher test -------------------------------------------------------------------------------

set(DISPATCHER_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/types/type_dispatcher_test.cu")

ConfigureTest(DISPATCHER_TEST "${DISPATCHER_TEST_SRC}")

###################################################################################################
# - strings test ----------------------------------------------------------------------------------

set(STRINGS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/factories_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/array_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/attrs_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/booleans_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/case_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/chars_types_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/combine_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/concatenate_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/datetime_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/fill_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/find_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/find_multiple_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/floats_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/gather_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/hash_string.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/integers_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/pad_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/replace_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/split_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/substring_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/translate_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/urls_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/utilities.cu")

ConfigureTest(STRINGS_TEST "${STRINGS_TEST_SRC}")

###################################################################################################
# - bitmask tests ---------------------------------------------------------------------------------

set(BITMASK_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/valid_if_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/bitmask_tests.cu")


ConfigureTest(BITMASK_TEST "${BITMASK_TEST_SRC}")

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()

###################################################################################################
### benchmarks ####################################################################################
###################################################################################################

# only built if CMAKE_ENABLE_BENCHMARKS is set to ON

###################################################################################################
# - iterator benchmarks----------------------------------------------------------------------------

set(LEGACY_ITERATOR_BENCH_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/iterator/iterator_bench.cu")

  ConfigureBench(LEGACY_ITERATOR_BENCH "${LEGACY_ITERATOR_BENCH_SRC}")
