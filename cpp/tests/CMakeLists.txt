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

add_library(cudftestutil STATIC
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/base_fixture.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/column_utilities.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/table_utilities.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/strings/utilities.cu")

set_property(TARGET cudftestutil PROPERTY POSITION_INDEPENDENT_CODE ON)

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
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest pthread cudf cudftestutil)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    set(CUDF_TEST_LIST ${CUDF_TEST_LIST} ${CMAKE_TEST_NAME} CACHE INTERNAL "CUDF_TEST_LIST")
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${THRUST_INCLUDE_DIR}"
                    "${JITIFY_INCLUDE_DIR}"
                    "${LIBCUDACXX_INCLUDE_DIR}")

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/dlpack/include"
                    "${GTEST_INCLUDE_DIR}"
                    "${ARROW_INCLUDE_DIR}"
                    "${ZLIB_INCLUDE_DIRS}"
                    "${Boost_INCLUDE_DIRS}"
                    "${RMM_INCLUDE}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}"
                 "${GTEST_LIBRARY_DIR}"
                 "${RMM_LIBRARY}")

if(CONDA_LINK_DIRS)
    link_directories("${CONDA_LINK_DIRS}")
endif(CONDA_LINK_DIRS)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - column tests ----------------------------------------------------------------------------------

set(COLUMN_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/column/column_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/column/column_view_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/column/column_device_view_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/column/compound_test.cu")

ConfigureTest(COLUMN_TEST "${COLUMN_TEST_SRC}")

###################################################################################################
# - scalar tests ----------------------------------------------------------------------------------

set(SCALAR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/scalar/scalar_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/scalar/scalar_device_view_test.cu")

ConfigureTest(SCALAR_TEST "${SCALAR_TEST_SRC}")

###################################################################################################
# - timestamps tests ------------------------------------------------------------------------------

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

set(GROUPBY_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/groupby_groups_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_argmin_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_argmax_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/groupby_keys_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_count_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_sum_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_min_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_max_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_sum_of_squares_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_mean_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_var_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_std_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_median_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_quantile_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_nunique_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_nth_element_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_collect_test.cpp")

ConfigureTest(GROUPBY_TEST "${GROUPBY_TEST_SRC}")

###################################################################################################
# - join tests ------------------------------------------------------------------------------------

set(JOIN_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/join_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/join/cross_join_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/join/semi_join_tests.cpp")

ConfigureTest(JOIN_TEST "${JOIN_TEST_SRC}")

###################################################################################################
# - is_sorted tests -------------------------------------------------------------------------------

set(IS_SORTED_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/sort/is_sorted_tests.cpp")

ConfigureTest(IS_SORTED_TEST "${IS_SORTED_TEST_SRC}")

###################################################################################################
# - datetime tests --------------------------------------------------------------------------------

set(DATETIME_OPS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/datetime/datetime_ops_test.cpp")

ConfigureTest(DATETIME_OPS_TEST "${DATETIME_OPS_TEST_SRC}")

###################################################################################################
# - hashing tests ---------------------------------------------------------------------------------

set(HASHING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/hashing/hash_test.cpp")

ConfigureTest(HASHING_TEST "${HASHING_TEST_SRC}")

###################################################################################################
# - partitioning tests ----------------------------------------------------------------------------

set(PARTITIONING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/partitioning/hash_partition_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/partitioning/round_robin_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/partitioning/partition_test.cpp")

ConfigureTest(PARTITIONING_TEST "${PARTITIONING_TEST_SRC}")

###################################################################################################
# - hash_map tests --------------------------------------------------------------------------------

set(HASH_MAP_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/hash_map/map_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/hash_map/multimap_test.cu")

ConfigureTest(HASH_MAP_TEST "${HASH_MAP_TEST_SRC}")

###################################################################################################
# - quantiles tests -------------------------------------------------------------------------------

set(QUANTILES_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/quantiles/quantile_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/quantiles/quantiles_test.cpp")

ConfigureTest(QUANTILES_TEST "${QUANTILES_TEST_SRC}")

###################################################################################################
# - reduction tests -------------------------------------------------------------------------------

set(REDUCTION_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/reductions/reduction_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/reductions/scan_tests.cpp")

ConfigureTest(REDUCTION_TEST "${REDUCTION_TEST_SRC}")

###################################################################################################
# - replace tests ---------------------------------------------------------------------------------

set(REPLACE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/replace/replace_tests.cpp")

ConfigureTest(REPLACE_TEST "${REPLACE_TEST_SRC}")

set(REPLACE_NULLS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/replace/replace_nulls_tests.cpp")

ConfigureTest(REPLACE_NULLS_TEST "${REPLACE_NULLS_TEST_SRC}")

set(REPLACE_NANS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/replace/replace_nans_tests.cpp")

ConfigureTest(REPLACE_NANS_TEST "${REPLACE_NANS_TEST_SRC}")

set(NORMALIZE_REPLACE_TEST_SRC
	"${CMAKE_CURRENT_SOURCE_DIR}/replace/normalize_replace_tests.cpp")

ConfigureTest(NORMALIZE_REPLACE_TEST "${NORMALIZE_REPLACE_TEST_SRC}")

set(CLAMP_TEST_SRC
	"${CMAKE_CURRENT_SOURCE_DIR}/replace/clamp_test.cpp")

ConfigureTest(CLAMP_TEST "${CLAMP_TEST_SRC}")

###################################################################################################
# - fixed_point tests -----------------------------------------------------------------------------

set(FIXED_POINT_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/fixed_point/fixed_point_tests.cu")

ConfigureTest(FIXED_POINT_TEST "${FIXED_POINT_TEST_SRC}")

###################################################################################################
# - unary tests -----------------------------------------------------------------------------------

set(UNARY_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/unary/math_ops_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unary/unary_ops_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unary/cast_tests.cpp")

ConfigureTest(UNARY_TEST "${UNARY_TEST_SRC}")

###################################################################################################
# - round tests -----------------------------------------------------------------------------------

set(ROUND_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/round/round_tests.cpp")

ConfigureTest(ROUND_TEST "${ROUND_TEST_SRC}")

###################################################################################################
# - binary tests ----------------------------------------------------------------------------------

set(BINARY_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/binop-verify-input-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/binop-null-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/binop-integration-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/binop-generic-ptx-test.cpp")

ConfigureTest(BINARY_TEST "${BINARY_TEST_SRC}")

###################################################################################################
# - unary transform tests -------------------------------------------------------------------------

set(TRANSFORM_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/transform/integration/unary-transform-test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/transform/nans_to_null_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/transform/mask_to_bools_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/transform/bools_to_mask_test.cpp")

ConfigureTest(TRANSFORM_TEST "${TRANSFORM_TEST_SRC}")

###################################################################################################
# - interop tests -------------------------------------------------------------------------

set(INTEROP_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/interop/to_arrow_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interop/from_arrow_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interop/dlpack_test.cpp")

ConfigureTest(INTEROP_TEST "${INTEROP_TEST_SRC}")

###################################################################################################
# - jit cache tests -------------------------------------------------------------------------------

set(JITCACHE_TEST_SRC
    "${CMAKE_SOURCE_DIR}/src/jit/cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/jit-cache-test.cpp")

ConfigureTest(JITCACHE_TEST "${JITCACHE_TEST_SRC}")

set(JITCACHE_MULTI_TEST_SRC
    "${CMAKE_SOURCE_DIR}/src/jit/cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/jit-cache-multiprocess-test.cpp")

ConfigureTest(JITCACHE_MULTIPROC_TEST "${JITCACHE_MULTI_TEST_SRC}")

###################################################################################################
# - io tests --------------------------------------------------------------------------------------

set(DECOMPRESSION_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/comp/decomp_test.cu")

ConfigureTest(DECOMPRESSION_TEST "${DECOMPRESSION_TEST_SRC}")

set(CSV_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/csv_test.cpp")
set(ORC_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/orc_test.cpp")
set(PARQUET_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/parquet_test.cpp")
set(JSON_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/io/json_test.cpp")

ConfigureTest(CSV_TEST "${CSV_TEST_SRC}")
ConfigureTest(ORC_TEST "${ORC_TEST_SRC}")
ConfigureTest(PARQUET_TEST "${PARQUET_TEST_SRC}")
ConfigureTest(JSON_TEST "${JSON_TEST_SRC}")

###################################################################################################
# - sort tests ------------------------------------------------------------------------------------

set(SORT_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/sort/segmented_sort_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sort/sort_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sort/rank_test.cpp")

ConfigureTest(SORT_TEST "${SORT_TEST_SRC}")

###################################################################################################
# - copying tests ---------------------------------------------------------------------------------

set(COPYING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/utility_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/gather_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/gather_str_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/gather_list_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/segmented_gather_list_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/gather_struct_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/detail_gather_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/scatter_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/scatter_list_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/copy_range_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/slice_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/split_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/copy_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/shift_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/get_value_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/sample_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/concatenate_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/copying/pack_tests.cu")

ConfigureTest(COPYING_TEST "${COPYING_TEST_SRC}")

###################################################################################################
# - utilities tests -------------------------------------------------------------------------------

set(UTILITIES_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities_tests/type_list_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities_tests/column_utilities_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities_tests/column_wrapper_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities_tests/lists_column_wrapper_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities_tests/default_stream_tests.cpp")

ConfigureTest(UTILITIES_TEST "${UTILITIES_TEST_SRC}")

###################################################################################################
# - span tests -------------------------------------------------------------------------------

set(SPAN_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities_tests/span_tests.cu")

ConfigureTest(SPAN_TEST "${SPAN_TEST_SRC}")

###################################################################################################
# - iterator tests --------------------------------------------------------------------------------

set(ITERATOR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/iterator/value_iterator_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/iterator/pair_iterator_test.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/iterator/scalar_iterator_test.cu")

ConfigureTest(ITERATOR_TEST "${ITERATOR_TEST_SRC}")

###################################################################################################
# - device atomics tests --------------------------------------------------------------------------

set(DEVICE_ATOMICS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/device_atomics/device_atomics_test.cu")

ConfigureTest(DEVICE_ATOMICS_TEST  "${DEVICE_ATOMICS_TEST_SRC}")

###################################################################################################
# - transpose tests -------------------------------------------------------------------------------

set(TRANSPOSE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/transpose/transpose_test.cpp")

ConfigureTest(TRANSPOSE_TEST "${TRANSPOSE_TEST_SRC}")

###################################################################################################
# - table tests -----------------------------------------------------------------------------------

set(TABLE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/table/table_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/table/table_view_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/table/row_operators_tests.cpp")

ConfigureTest(TABLE_TEST "${TABLE_TEST_SRC}")

###################################################################################################
# - sorted-merge tests ----------------------------------------------------------------------------

set(MERGE_TEST_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/merge/merge_test.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/merge/merge_dictionary_test.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/merge/merge_string_test.cpp")

ConfigureTest(MERGE_TEST "${MERGE_TEST_SRC}")

###################################################################################################
# - stream compaction tests -----------------------------------------------------------------------

set(STREAM_COMPACTION_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/apply_boolean_mask_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/drop_nulls_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/drop_nans_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/drop_duplicates_tests.cpp")

ConfigureTest(STREAM_COMPACTION_TEST "${STREAM_COMPACTION_TEST_SRC}")

###################################################################################################
# - rolling tests ---------------------------------------------------------------------------------

set(ROLLING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/rolling/rolling_test.cpp")

ConfigureTest(ROLLING_TEST "${ROLLING_TEST_SRC}")

###################################################################################################
# - grouped rolling tests -------------------------------------------------------------------------

set(GROUPED_ROLLING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/grouped_rolling/grouped_rolling_test.cpp")

ConfigureTest(GROUPED_ROLLING_TEST "${GROUPED_ROLLING_TEST_SRC}")

###################################################################################################
# - lead/lag rolling tests ---------------------------------------------------------------------------------

set(LEAD_LAG_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/lead_lag/lead_lag_test.cpp")

ConfigureTest(LEAD_LAG_TEST "${LEAD_LAG_TEST_SRC}")

###################################################################################################
# - collect_list rolling tests ---------------------------------------------------------------------------------

set(COLLECT_LIST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/collect_list/collect_list_test.cpp")

ConfigureTest(COLLECT_LIST_TEST "${COLLECT_LIST_SRC}")

###################################################################################################
# - filling test ----------------------------------------------------------------------------------

set(FILLING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/filling/fill_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/filling/repeat_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/filling/sequence_tests.cpp")

ConfigureTest(FILLING_TEST "${FILLING_TEST_SRC}")

###################################################################################################
# - search test -----------------------------------------------------------------------------------

set(SEARCH_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/search/search_test.cpp")

ConfigureTest(SEARCH_TEST "${SEARCH_TEST_SRC}")

###################################################################################################
# - reshape test ----------------------------------------------------------------------------------

set(RESHAPE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/reshape/byte_cast_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/reshape/explode_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/reshape/interleave_columns_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/reshape/tile_tests.cpp")

ConfigureTest(RESHAPE_TEST "${RESHAPE_TEST_SRC}")

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
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/attrs_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/booleans_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/case_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/chars_types_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/combine_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/concatenate_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/contains_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/datetime_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/durations_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/extract_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/fill_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/findall_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/find_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/find_multiple_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/floats_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/hash_string.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/integers_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/ipv4_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/pad_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/replace_regex_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/replace_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/split_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/strip_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/substring_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/translate_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/urls_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/strings/json_tests.cpp")

ConfigureTest(STRINGS_TEST "${STRINGS_TEST_SRC}")

###################################################################################################
# - structs test ----------------------------------------------------------------------------------

set(STRUCTS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/structs/structs_column_tests.cu")

ConfigureTest(STRUCTS_TEST "${STRUCTS_TEST_SRC}")

###################################################################################################
# - nvtext test -----------------------------------------------------------------------------------

set(TEXT_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/text/edit_distance_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/text/ngrams_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/text/ngrams_tokenize_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/text/normalize_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/text/replace_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/text/stemmer_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/text/subword_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/text/tokenize_tests.cpp")

ConfigureTest(TEXT_TEST "${TEXT_TEST_SRC}")

###################################################################################################
# - bitmask tests ---------------------------------------------------------------------------------

set(BITMASK_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/valid_if_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/set_nullmask_tests.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmask/bitmask_tests.cu")


ConfigureTest(BITMASK_TEST "${BITMASK_TEST_SRC}")

###################################################################################################
# - dictionary tests ------------------------------------------------------------------------------

set(DICTIONARY_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/dictionary/add_keys_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dictionary/decode_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dictionary/encode_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dictionary/factories_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dictionary/fill_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dictionary/gather_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dictionary/remove_keys_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dictionary/scatter_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dictionary/search_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dictionary/set_keys_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dictionary/slice_test.cpp")

ConfigureTest(DICTIONARY_TEST "${DICTIONARY_TEST_SRC}")

###################################################################################################
# - encode tests -----------------------------------------------------------------------------------

set(ENCODE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/encode/encode_tests.cpp")

ConfigureTest(ENCODE_TEST "${ENCODE_TEST_SRC}")

###################################################################################################
# - ast tests -------------------------------------------------------------------------------------

set(AST_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/ast/transform_tests.cpp")

ConfigureTest(AST_TEST "${AST_TEST_SRC}")

###################################################################################################
# - lists tests ----------------------------------------------------------------------------------

set(LISTS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/lists/contains_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lists/count_elements_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lists/extract_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lists/sort_lists_tests.cpp")

ConfigureTest(LISTS_TEST "${LISTS_TEST_SRC}")

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()
