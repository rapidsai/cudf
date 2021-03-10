#=============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME )
    add_executable(${CMAKE_TEST_NAME} ${ARGN})
    set_target_properties(${CMAKE_TEST_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUDF_BINARY_DIR}/gtests>")
    target_link_libraries(${CMAKE_TEST_NAME} PRIVATE cudftestutil GTest::gmock_main GTest::gtest_main)
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction()

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - column tests ----------------------------------------------------------------------------------
ConfigureTest(COLUMN_TEST
    column/bit_cast_test.cpp
    column/column_test.cu
    column/column_device_view_test.cu
    column/compound_test.cu)

###################################################################################################
# - scalar tests ----------------------------------------------------------------------------------
ConfigureTest(SCALAR_TEST
    scalar/scalar_test.cpp
    scalar/scalar_device_view_test.cu)

###################################################################################################
# - timestamps tests ------------------------------------------------------------------------------
ConfigureTest(TIMESTAMPS_TEST wrappers/timestamps_test.cu)

###################################################################################################
# - cudf tests ------------------------------------------------------------------------------------
ConfigureTest(ERROR_TEST error/error_handling_test.cu)

###################################################################################################
# - groupby tests ---------------------------------------------------------------------------------
ConfigureTest(GROUPBY_TEST
    groupby/groupby_groups_test.cpp
    groupby/group_argmin_test.cpp
    groupby/group_argmax_test.cpp
    groupby/groupby_keys_test.cpp
    groupby/group_count_test.cpp
    groupby/group_sum_test.cpp
    groupby/group_min_test.cpp
    groupby/group_max_test.cpp
    groupby/group_sum_of_squares_test.cpp
    groupby/group_mean_test.cpp
    groupby/group_var_test.cpp
    groupby/group_std_test.cpp
    groupby/group_median_test.cpp
    groupby/group_quantile_test.cpp
    groupby/group_nunique_test.cpp
    groupby/group_nth_element_test.cpp
    groupby/group_collect_test.cpp)

###################################################################################################
# - join tests ------------------------------------------------------------------------------------
ConfigureTest(JOIN_TEST
    join/join_tests.cpp
    join/cross_join_tests.cpp
    join/semi_join_tests.cpp)

###################################################################################################
# - is_sorted tests -------------------------------------------------------------------------------
ConfigureTest(IS_SORTED_TEST sort/is_sorted_tests.cpp)

###################################################################################################
# - datetime tests --------------------------------------------------------------------------------
ConfigureTest(DATETIME_OPS_TEST datetime/datetime_ops_test.cpp)

###################################################################################################
# - hashing tests ---------------------------------------------------------------------------------
ConfigureTest(HASHING_TEST hashing/hash_test.cpp)

###################################################################################################
# - partitioning tests ----------------------------------------------------------------------------
ConfigureTest(PARTITIONING_TEST
    partitioning/hash_partition_test.cpp
    partitioning/round_robin_test.cpp
    partitioning/partition_test.cpp)

###################################################################################################
# - hash_map tests --------------------------------------------------------------------------------
ConfigureTest(HASH_MAP_TEST
    hash_map/map_test.cu
    hash_map/multimap_test.cu)

###################################################################################################
# - quantiles tests -------------------------------------------------------------------------------
ConfigureTest(QUANTILES_TEST
    quantiles/quantile_test.cpp
    quantiles/quantiles_test.cpp)

###################################################################################################
# - reduction tests -------------------------------------------------------------------------------
ConfigureTest(REDUCTION_TEST
    reductions/reduction_tests.cpp
    reductions/scan_tests.cpp)

###################################################################################################
# - replace tests ---------------------------------------------------------------------------------
ConfigureTest(REPLACE_TEST replace/replace_tests.cpp)

ConfigureTest(REPLACE_NULLS_TEST replace/replace_nulls_tests.cpp)

ConfigureTest(REPLACE_NANS_TEST replace/replace_nans_tests.cpp)

ConfigureTest(NORMALIZE_REPLACE_TEST replace/normalize_replace_tests.cpp)

ConfigureTest(CLAMP_TEST replace/clamp_test.cpp)

###################################################################################################
# - fixed_point tests -----------------------------------------------------------------------------
ConfigureTest(FIXED_POINT_TEST fixed_point/fixed_point_tests.cu)

###################################################################################################
# - unary tests -----------------------------------------------------------------------------------
ConfigureTest(UNARY_TEST
    unary/math_ops_test.cpp
    unary/unary_ops_test.cpp
    unary/cast_tests.cpp)

###################################################################################################
# - round tests -----------------------------------------------------------------------------------
ConfigureTest(ROUND_TEST round/round_tests.cpp)

###################################################################################################
# - binary tests ----------------------------------------------------------------------------------
ConfigureTest(BINARY_TEST
    binaryop/binop-verify-input-test.cpp
    binaryop/binop-null-test.cpp
    binaryop/binop-integration-test.cpp
    binaryop/binop-generic-ptx-test.cpp)

###################################################################################################
# - unary transform tests -------------------------------------------------------------------------
ConfigureTest(TRANSFORM_TEST
    transform/integration/unary-transform-test.cpp
    transform/nans_to_null_test.cpp
    transform/mask_to_bools_test.cpp
    transform/bools_to_mask_test.cpp)

###################################################################################################
# - interop tests -------------------------------------------------------------------------
ConfigureTest(INTEROP_TEST
    interop/to_arrow_test.cpp
    interop/from_arrow_test.cpp
    interop/dlpack_test.cpp)

###################################################################################################
# - jit cache tests -------------------------------------------------------------------------------
# ConfigureTest(JITCACHE_TEST
    # "${CUDF_SOURCE_DIR}/src/jit/cache.cpp"
    # jit/jit-cache-test.cpp)

# ConfigureTest(JITCACHE_MULTIPROC_TEST
    # "${CUDF_SOURCE_DIR}/src/jit/cache.cpp"
    # jit/jit-cache-multiprocess-test.cpp)

###################################################################################################
# - io tests --------------------------------------------------------------------------------------
ConfigureTest(DECOMPRESSION_TEST io/comp/decomp_test.cu)

ConfigureTest(CSV_TEST io/csv_test.cpp)
ConfigureTest(ORC_TEST io/orc_test.cpp)
ConfigureTest(PARQUET_TEST io/parquet_test.cpp)
ConfigureTest(JSON_TEST io/json_test.cpp)

###################################################################################################
# - sort tests ------------------------------------------------------------------------------------
ConfigureTest(SORT_TEST
    sort/segmented_sort_tests.cpp
    sort/sort_test.cpp
    sort/rank_test.cpp)

###################################################################################################
# - copying tests ---------------------------------------------------------------------------------
ConfigureTest(COPYING_TEST
    copying/utility_tests.cpp
    copying/gather_tests.cu
    copying/gather_str_tests.cu
    copying/gather_list_tests.cu
    copying/segmented_gather_list_tests.cpp
    copying/gather_struct_tests.cu
    copying/detail_gather_tests.cu
    copying/scatter_tests.cpp
    copying/scatter_list_tests.cu
    copying/copy_range_tests.cpp
    copying/slice_tests.cpp
    copying/split_tests.cpp
    copying/copy_tests.cu
    copying/shift_tests.cpp
    copying/get_value_tests.cpp
    copying/sample_tests.cpp
    copying/concatenate_tests.cu
    copying/pack_tests.cu)

###################################################################################################
# - utilities tests -------------------------------------------------------------------------------
ConfigureTest(UTILITIES_TEST
    utilities_tests/type_list_tests.cpp
    utilities_tests/column_utilities_tests.cpp
    utilities_tests/column_wrapper_tests.cpp
    utilities_tests/lists_column_wrapper_tests.cpp
    utilities_tests/default_stream_tests.cpp)

###################################################################################################
# - span tests -------------------------------------------------------------------------------
ConfigureTest(SPAN_TEST utilities_tests/span_tests.cu)

###################################################################################################
# - iterator tests --------------------------------------------------------------------------------
ConfigureTest(ITERATOR_TEST
    iterator/value_iterator_test.cu
    iterator/pair_iterator_test.cu
    iterator/scalar_iterator_test.cu)

###################################################################################################
# - device atomics tests --------------------------------------------------------------------------
ConfigureTest(DEVICE_ATOMICS_TEST device_atomics/device_atomics_test.cu)

###################################################################################################
# - transpose tests -------------------------------------------------------------------------------
ConfigureTest(TRANSPOSE_TEST transpose/transpose_test.cpp)

###################################################################################################
# - table tests -----------------------------------------------------------------------------------
ConfigureTest(TABLE_TEST
    table/table_tests.cpp
    table/table_view_tests.cu
    table/row_operators_tests.cpp)

###################################################################################################
# - sorted-merge tests ----------------------------------------------------------------------------
ConfigureTest(MERGE_TEST
    merge/merge_test.cpp
    merge/merge_dictionary_test.cpp
    merge/merge_string_test.cpp)

###################################################################################################
# - stream compaction tests -----------------------------------------------------------------------
ConfigureTest(STREAM_COMPACTION_TEST
    stream_compaction/apply_boolean_mask_tests.cpp
    stream_compaction/drop_nulls_tests.cpp
    stream_compaction/drop_nans_tests.cpp
    stream_compaction/drop_duplicates_tests.cpp)

###################################################################################################
# - rolling tests ---------------------------------------------------------------------------------
ConfigureTest(ROLLING_TEST 
    rolling/rolling_test.cpp 
    rolling/grouped_rolling_test.cpp
    rolling/lead_lag_test.cpp
    rolling/range_window_bounds_test.cpp
    rolling/range_rolling_window_test.cpp
    rolling/collect_list_test.cpp)

###################################################################################################
# - filling test ----------------------------------------------------------------------------------
ConfigureTest(FILLING_TEST
    filling/fill_tests.cpp
    filling/repeat_tests.cpp
    filling/sequence_tests.cpp)

###################################################################################################
# - search test -----------------------------------------------------------------------------------
ConfigureTest(SEARCH_TEST search/search_test.cpp)

###################################################################################################
# - reshape test ----------------------------------------------------------------------------------
ConfigureTest(RESHAPE_TEST
    reshape/byte_cast_tests.cpp
    reshape/interleave_columns_tests.cpp
    reshape/tile_tests.cpp)

###################################################################################################
# - traits test -----------------------------------------------------------------------------------
ConfigureTest(TRAITS_TEST types/traits_test.cpp)

###################################################################################################
# - factories test --------------------------------------------------------------------------------
ConfigureTest(FACTORIES_TEST
    scalar/factories_test.cpp
    column/factories_test.cpp)

###################################################################################################
# - dispatcher test -------------------------------------------------------------------------------
ConfigureTest(DISPATCHER_TEST types/type_dispatcher_test.cu)

###################################################################################################
# - strings test ----------------------------------------------------------------------------------
ConfigureTest(STRINGS_TEST
    strings/factories_test.cu
    strings/array_tests.cu
    strings/attrs_tests.cpp
    strings/booleans_tests.cpp
    strings/case_tests.cpp
    strings/chars_types_tests.cpp
    strings/combine_tests.cpp
    strings/concatenate_tests.cpp
    strings/contains_tests.cpp
    strings/datetime_tests.cpp
    strings/durations_tests.cpp
    strings/extract_tests.cpp
    strings/fill_tests.cpp
    strings/findall_tests.cpp
    strings/find_tests.cpp
    strings/find_multiple_tests.cpp
    strings/fixed_point_tests.cpp
    strings/floats_tests.cpp
    strings/hash_string.cu
    strings/integers_tests.cu
    strings/ipv4_tests.cpp
    strings/pad_tests.cpp
    strings/replace_regex_tests.cpp
    strings/replace_tests.cpp
    strings/split_tests.cpp
    strings/strip_tests.cpp
    strings/substring_tests.cpp
    strings/translate_tests.cpp
    strings/urls_tests.cpp)

###################################################################################################
# - structs test ----------------------------------------------------------------------------------
ConfigureTest(STRUCTS_TEST structs/structs_column_tests.cu)

###################################################################################################
# - nvtext test -----------------------------------------------------------------------------------
ConfigureTest(TEXT_TEST
    text/edit_distance_tests.cpp
    text/ngrams_tests.cpp
    text/ngrams_tokenize_tests.cpp
    text/normalize_tests.cpp
    text/replace_tests.cpp
    text/stemmer_tests.cpp
    text/subword_tests.cpp
    text/tokenize_tests.cpp)

###################################################################################################
# - bitmask tests ---------------------------------------------------------------------------------
ConfigureTest(BITMASK_TEST
    bitmask/valid_if_tests.cu
    bitmask/set_nullmask_tests.cu
    bitmask/bitmask_tests.cu)


###################################################################################################
# - dictionary tests ------------------------------------------------------------------------------
ConfigureTest(DICTIONARY_TEST
    dictionary/add_keys_test.cpp
    dictionary/decode_test.cpp
    dictionary/encode_test.cpp
    dictionary/factories_test.cpp
    dictionary/fill_test.cpp
    dictionary/gather_test.cpp
    dictionary/remove_keys_test.cpp
    dictionary/scatter_test.cpp
    dictionary/search_test.cpp
    dictionary/set_keys_test.cpp
    dictionary/slice_test.cpp)

###################################################################################################
# - encode tests -----------------------------------------------------------------------------------
ConfigureTest(ENCODE_TEST encode/encode_tests.cpp)

###################################################################################################
# - ast tests -------------------------------------------------------------------------------------
ConfigureTest(AST_TEST ast/transform_tests.cpp)

###################################################################################################
# - lists tests ----------------------------------------------------------------------------------
ConfigureTest(LISTS_TEST
    lists/contains_tests.cpp
    lists/count_elements_tests.cpp
    lists/explode_tests.cpp
    lists/drop_list_duplicates_tests.cpp
    lists/extract_tests.cpp
    lists/sort_lists_tests.cpp)

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()
