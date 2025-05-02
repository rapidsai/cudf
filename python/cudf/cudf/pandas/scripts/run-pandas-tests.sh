#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run Pandas unit tests with cudf.pandas.
#
# Usage:
#   run-pandas-tests.sh <pytest args> <path to pandas tests (optional)>
#
# Examples
# Run a single test
#   run-pandas-tests.sh -n auto -v tests/groupby/test_groupby_dropna.py
# Run all tests
#   run-pandas-tests.sh --tb=line --report-log=log.json
#
# This script creates a `pandas-testing` directory if it doesn't exist
#
# If running locally, it's recommended to pass '-m "not slow and not single_cpu and not db"'

set -euo pipefail

# Grab the Pandas source corresponding to the version
# of Pandas installed.
PANDAS_VERSION=$(python -c "import pandas; print(pandas.__version__)")

mkdir -p pandas-testing
cd pandas-testing

if [ ! -d "pandas" ]; then
    git clone https://github.com/pandas-dev/pandas --depth=1 -b "v${PANDAS_VERSION}" pandas
fi

if [ ! -d "pandas-tests" ]; then
    # Copy just the tests out of the Pandas source tree.
    # Not exactly sure why this is needed but Pandas
    # imports fail if we don't do this:
    mkdir -p pandas-tests
    cp -r pandas/pandas/tests pandas-tests/
    # directory layout requirement
    # conftest.py
    # pyproject.toml
    # tests/
    cp pandas/pandas/conftest.py pandas-tests/conftest.py
    # Vendored from pandas/pyproject.toml
    cat > pandas-tests/pyproject.toml << \EOF
[tool.pytest.ini_options]
xfail_strict = true
filterwarnings = [
  # Will be fixed in numba 0.56: https://github.com/numba/numba/issues/7758
  "ignore:`np.MachAr` is deprecated:DeprecationWarning:numba",
]
markers = [
  "single_cpu: tests that should run on a single cpu only",
  "slow: mark a test as slow",
  "network: mark a test as network",
  "db: tests requiring a database (mysql or postgres)",
  "clipboard: mark a pd.read_clipboard test",
  "arm_slow: mark a test as slow for arm64 architecture",
  "skip_ubsan: Tests known to fail UBSAN check",
]
EOF

    # Substitute `pandas.tests` with a relative import.
    # This will depend on the location of the test module relative to
    # the pandas-tests directory.
    for hit in $(find . -iname '*.py' | xargs grep "pandas.tests" | cut -d ":" -f 1 | sort | uniq); do
        # Get the relative path to the test module
        test_module=$(echo $hit | cut -d "/" -f 2-)
        # Get the number of directories to go up
        num_dirs=$(echo $test_module | grep -o "/" | wc -l)
        num_dots=$(($num_dirs - 2))
        # Construct the relative import
        relative_import=$(printf "%0.s." $(seq 1 $num_dots))
        # Replace the import
        sed -i "s/pandas.tests/${relative_import}/g" $hit
    done
fi

# append the contents of patch-confest.py to conftest.py
cat ../python/cudf/cudf/pandas/scripts/conftest-patch.py >> pandas-tests/conftest.py

# Run the tests
cd pandas-tests/


# TODO: Needs motoserver/moto container running on http://localhost:5000
TEST_THAT_NEED_MOTO_SERVER="not test_styler_to_s3 \
and not test_with_s3_url[None] \
and not test_with_s3_url[gzip] \
and not test_with_s3_url[bz2] \
and not test_with_s3_url[zip] \
and not test_with_s3_url[xz] \
and not test_with_s3_url[tar] \
and not test_s3_permission_output[etree] \
and not test_read_s3_jsonl \
and not test_s3_parser_consistency \
and not test_to_s3 \
and not test_parse_public_s3a_bucket \
and not test_parse_public_s3_bucket_nrows \
and not test_parse_public_s3_bucket_chunked \
and not test_parse_public_s3_bucket_chunked_python \
and not test_parse_public_s3_bucket_python \
and not test_infer_s3_compression \
and not test_parse_public_s3_bucket_nrows_python \
and not test_read_s3_fails_private \
and not test_read_csv_handles_boto_s3_object \
and not test_read_csv_chunked_download \
and not test_read_s3_with_hash_in_key \
and not test_read_feather_s3_file_path \
and not test_parse_public_s3_bucket \
and not test_parse_private_s3_bucket \
and not test_parse_public_s3n_bucket \
and not test_read_with_creds_from_pub_bucket \
and not test_read_without_creds_from_pub_bucket \
and not test_from_s3_csv \
and not test_s3_protocols[s3] \
and not test_s3_protocols[s3a] \
and not test_s3_protocols[s3n] \
and not test_s3_parquet \
and not test_s3_roundtrip_explicit_fs \
and not test_s3_roundtrip \
and not test_s3_roundtrip_for_dir[partition_col0] \
and not test_s3_roundtrip_for_dir[partition_col1] \
and not test_s3_roundtrip"

TEST_THAT_CRASH_PYTEST_WORKERS="not test_bitmasks_pyarrow \
and not test_large_string_pyarrow \
and not test_interchange_from_corrected_buffer_dtypes \
and not test_eof_states \
and not test_array_tz \
and not test_resample_empty_dataframe"


# TODO: Add reason to skip these tests
TEST_THAT_NEED_REASON_TO_SKIP="not test_groupby_raises_category_on_category \
and not test_constructor_no_pandas_array \
and not test_is_monotonic_na \
and not test_index_contains \
and not test_frame_op_subclass_nonclass_constructor \
and not test_round_trip_current \
and not test_pickle_frame_v124_unpickle_130"

# TODO: Need to fix these tests
TEST_THAT_NEED_FIX="not test_26395 \
and not test_add_2d \
and not test_add_matplotlib_datetime64 \
and not test_add_prefix \
and not test_add_suffix \
and not test_agg_both_mean_std_named_result \
and not test_agg_both_mean_sum \
and not test_agg_mixed_column_aggregation \
and not test_agg_with_lambda \
and not test_aggregate_normal \
and not test_align_copy_false \
and not test_align_frame \
and not test_align_with_series_copy_false \
and not test_all_methods_categorized \
and not test_api_execute_sql \
and not test_api_per_method \
and not test_append_len_one \
and not test_append_with_timezones \
and not test_apply_frame_concat_series \
and not test_apply_func_that_appends_group_to_list_without_copy \
and not test_apply_function_with_indexing \
and not test_apply_index_has_complex_internals \
and not test_apply_map_header_mi \
and not test_apply_multiindex_fail \
and not test_arith_frame_with_scalar \
and not test_arith_series_with_array \
and not test_array \
and not test_array_type \
and not test_arrow_array \
and not test_as_index_select_column \
and not test_asfreq_noop \
and not test_asof \
and not test_assign_drop_duplicates \
and not test_assigning_to_same_variable_removes_references \
and not test_astype \
and not test_astype_avoids_copy \
and not test_astype_dict_dtypes \
and not test_astype_different_datetime_resos \
and not test_astype_different_target_dtype \
and not test_astype_nan_to_int \
and not test_astype_object_frame \
and not test_astype_object_series \
and not test_astype_own_type \
and not test_astype_single_dtype \
and not test_astype_string_and_object \
and not test_astype_string_and_object_update_original \
and not test_axis1_numeric_only \
and not test_basic \
and not test_bins \
and not test_bins_from_interval_index \
and not test_boxplot_legacy1_series \
and not test_cache \
and not test_callable_result_dtype_frame \
and not test_categorical_index_upcast \
and not test_categorical_pivot_index_ordering \
and not test_category_order_apply \
and not test_chained_getitem_with_lists \
and not test_chunksize_with_compression \
and not test_clip \
and not test_clip_inplace \
and not test_clip_inplace_reference \
and not test_clip_inplace_reference_no_op \
and not test_clip_no_op \
and not test_closed_fixed_binary_col \
and not test_cmov_window_corner \
and not test_column_as_series \
and not test_combine_add \
and not test_combined_up_downsampling_of_irregular \
and not test_compression_roundtrip \
and not test_concat \
and not test_concat_copy_keyword \
and not test_concat_empty_series \
and not test_concat_frames \
and not test_concat_frames_chained \
and not test_concat_frames_updating_input \
and not test_concat_index_find_common \
and not test_concat_index_keep_dtype \
and not test_concat_index_keep_dtype_ea_numeric \
and not test_concat_keys_specific_levels \
and not test_concat_mixed_series_frame \
and not test_concat_multiindex_with_tz \
and not test_concat_same_index_names \
and not test_concat_series \
and not test_concat_series_axis1 \
and not test_concat_series_chained \
and not test_concat_series_updating_input \
and not test_construct_with_two_categoricalindex_series \
and not test_constructor_categorical_valid \
and not test_constructor_datetime64_tzformat \
and not test_constructor_dict_datetime64_index \
and not test_constructor_dict_timedelta64_index \
and not test_constructor_dtype \
and not test_constructor_dtype_timedelta_ns_s_astype_int64 \
and not test_constructor_for_list_with_dtypes \
and not test_constructor_fromarraylike \
and not test_constructor_unwraps_index \
and not test_context_manager \
and not test_context_manageri_user_provided \
and not test_conversion_float \
and not test_convert_dtypes \
and not test_copy \
and not test_copy_order \
and not test_copy_shares_cache \
and not test_create_index_existing_name \
and not test_crosstab_multiple \
and not test_crosstab_with_categorial_columns \
and not test_cut_bool_coercion_to_int \
and not test_cut_non_unique_labels \
and not test_cut_nullable_integer \
and not test_cut_pass_labels \
and not test_cut_pass_labels_compat \
and not test_cut_unordered_labels \
and not test_daily \
and not test_data_frame_value_counts_empty \
and not test_data_frame_value_counts_empty_normalize \
and not test_data_frame_value_counts_subset \
and not test_dataframe_add_column_from_series \
and not test_dataframe_array_ea_dtypes \
and not test_dataframe_array_string_dtype \
and not test_dataframe_compression_defaults_to_infer \
and not test_dataframe_constructor_from_dict \
and not test_dataframe_constructor_mgr_or_df \
and not test_dataframe_from_dict_of_series \
and not test_dataframe_from_dict_of_series_with_dtype \
and not test_dataframe_from_dict_of_series_with_reindex \
and not test_dataframe_from_records_with_dataframe \
and not test_dataframe_from_series \
and not test_dataframe_multiple_numpy_dtypes \
and not test_date_range_localize \
and not test_date_range_unsigned_overflow_handling \
and not test_dateindex_conversion \
and not test_datetime_like \
and not test_days \
and not test_days_neg \
and not test_del_frame \
and not test_describe_categorical_columns \
and not test_detect_chained_assignment_warning_stacklevel \
and not test_direct_arith_with_ndframe_returns_not_implemented \
and not test_drop_and_dropna_caching \
and not test_drop_on_column \
and not test_droplevel \
and not test_droplevel_multiindex_one_level \
and not test_dropna \
and not test_dt_accessor_api_for_categorical \
and not test_dt_accessor_limited_display_api \
and not test_dt_tz_localize_nonexistent \
and not test_dti_constructor_with_dtype_object_int_matches_int_dtype \
and not test_dti_timestamp_isocalendar_fields \
and not test_dtype_on_merged_different \
and not test_ellipsis_index \
and not test_empty \
and not test_empty_dataframe \
and not test_empty_datelike \
and not test_empty_frame_setitem_index_name_retained \
and not test_engine_type \
and not test_engineless_lookup \
and not test_ensure_copied_data \
and not test_equals_multi \
and not test_eval \
and not test_eval_inplace \
and not test_evenly_divisible_with_no_extra_bins2 \
and not test_excelfile_fspath \
and not test_execute_sql \
and not test_factorize_use_na_sentinel \
and not test_ffill_handles_nan_groups \
and not test_fillna \
and not test_fillna_dict \
and not test_fillna_ea_noop_shares_memory \
and not test_fillna_inplace \
and not test_fillna_inplace_ea_noop_shares_memory \
and not test_fillna_inplace_reference \
and not test_fillna_length_mismatch \
and not test_filter \
and not test_format_clear \
and not test_frame_from_dict_of_index \
and not test_frame_groupby_columns \
and not test_frame_mixed_depth_get \
and not test_frame_set_axis \
and not test_from_arrays_respects_none_names \
and not test_from_arrow_respecting_given_dtype \
and not test_from_arrow_type_error \
and not test_from_out_of_bounds_ns_datetime \
and not test_from_out_of_bounds_ns_timedelta \
and not test_get_array_masked \
and not test_get_array_numpy \
and not test_get_indexer_categorical_with_nans \
and not test_get_indexer_with_NA_values \
and not test_get_loc_nat \
and not test_getitem_boolean_array_mask \
and not test_getitem_ix_mixed_integer2 \
and not test_getitem_loc_assignment_slice_state \
and not test_getitem_mask \
and not test_getitem_midx_slice \
and not test_getitem_scalar \
and not test_getitem_series_integer_with_missing_raises \
and not test_getitem_with_datestring_with_UTC_offset \
and not test_group_on_two_row_multiindex_returns_one_tuple_key \
and not test_groupby_aggregate_directory \
and not test_groupby_column_index_name_lost \
and not test_groupby_column_index_name_lost_fill_funcs \
and not test_groupby_crash_on_nunique \
and not test_groupby_datetime64_32_bit \
and not test_groupby_groups_in_BaseGrouper \
and not test_groupby_level \
and not test_groupby_level_mapper \
and not test_groupby_preserves_metadata \
and not test_groupby_preserves_subclass \
and not test_groupby_quantile_nonmulti_levels_order \
and not test_groupby_series_with_name \
and not test_groupby_with_hier_columns \
and not test_grouper_groups \
and not test_gz_lineend \
and not test_handle_dict_return_value \
and not test_head_tail \
and not test_html_template_extends_options \
and not test_identical \
and not test_iloc_getitem_doc_issue \
and not test_in_numeric_groupby \
and not test_index_from_array \
and not test_index_putmask \
and not test_index_string_inference \
and not test_index_to_frame \
and not test_index_unique \
and not test_index_where \
and not test_indices_with_missing \
and not test_infer_objects \
and not test_infer_objects_no_reference \
and not test_infer_objects_reference \
and not test_infer_string_large_string_type \
and not test_info_memory_usage_deep_pypy \
and not test_inplace_mutation_resets_values \
and not test_insert_series \
and not test_int_dtype_different_index_not_bool \
and not test_integer_values_and_tz_interpreted_as_utc \
and not test_interp_fill_functions \
and not test_interp_fill_functions_inplace \
and not test_interpolate_cleaned_fill_method \
and not test_interpolate_downcast \
and not test_interpolate_downcast_reference_triggers_copy \
and not test_interpolate_inplace_no_reference_no_copy \
and not test_interpolate_inplace_with_refs \
and not test_interpolate_no_op \
and not test_interpolate_object_convert_copies \
and not test_interpolate_object_convert_no_op \
and not test_interpolate_triggers_copy \
and not test_intersection_non_object \
and not test_interval_range_fractional_period \
and not test_invalid_file_inputs \
and not test_is_extension_array_dtype \
and not test_iset_splits_blocks_inplace \
and not test_isetitem \
and not test_isetitem_frame \
and not test_items \
and not test_ix_categorical_index \
and not test_join \
and not test_join_empty \
and not test_join_multiple_dataframes_on_key \
and not test_join_on_key \
and not test_join_self \
and not test_json_options \
and not test_json_pandas_nulls \
and not test_kurt \
and not test_left_right_dont_share_data \
and not test_lines_with_compression \
and not test_loc_bool_multiindex \
and not test_loc_copy_vs_view \
and not test_loc_enlarging_with_dataframe \
and not test_loc_getitem_nested_indexer \
and not test_loc_getitem_uint64_scalar \
and not test_loc_internals_not_updated_correctly \
and not test_loc_named_index \
and not test_loc_setitem_frame_mixed_labels \
and not test_masking_inplace \
and not test_max \
and not test_mean \
and not test_median \
and not test_memory_usage \
and not test_merge_copy_keyword \
and not test_merge_on_index \
and not test_merge_on_index_with_more_values \
and not test_merge_on_key \
and not test_merge_on_key_enlarging_one \
and not test_merge_suffix \
and not test_metadata_propagation_indiv \
and not test_methods_copy_keyword \
and not test_min \
and not test_modf \
and not test_monthly \
and not test_multiindex_columns_empty_level \
and not test_multiindex_negative_level \
and not test_neg \
and not test_neg_freq \
and not test_no_right \
and not test_non_categorical_value_labels \
and not test_nonagg_agg \
and not test_nth_after_selection \
and not test_numeric_dtypes \
and not test_numpy_transpose \
and not test_numpy_ufuncs_basic \
and not test_observed_codes_remap \
and not test_observed_groups \
and not test_observed_groups_with_nan \
and not test_order_aggregate_multiple_funcs \
and not test_order_of_appearance_dt64tz \
and not test_override_set_noconvert_columns \
and not test_parse_dates_column_list \
and not test_pipe \
and not test_pipe_modify_df \
and not test_pivot_dtypes \
and not test_pivot_no_level_overlap \
and not test_pivot_table_categorical_observed_equal \
and not test_pivot_timegrouper \
and not test_pivot_timegrouper_double \
and not test_pivot_with_categorical \
and not test_pivot_with_tz \
and not test_pop \
and not test_pos \
and not test_prod \
and not test_putmask \
and not test_putmask_aligns_rhs_no_reference \
and not test_putmask_dont_copy_some_blocks \
and not test_putmask_no_reference \
and not test_py2_created_with_datetimez \
and not test_quantile_datetime \
and not test_quantile_ea_scalar \
and not test_quantile_empty_no_columns \
and not test_quantile_empty_no_rows_dt64 \
and not test_quantile_nat \
and not test_query_inplace \
and not test_query_multiindex_get_index_resolvers \
and not test_raises_during_exception \
and not test_range_difference \
and not test_range_tz_pytz \
and not test_read_dtype_backend_pyarrow_config \
and not test_read_empty_array \
and not test_read_timezone_information \
and not test_read_write_ea_dtypes \
and not test_register_writer \
and not test_reindex_columns \
and not test_reindex_empty \
and not test_reindex_like \
and not test_reindex_nearest_tz \
and not test_reindex_rows \
and not test_rename_axis \
and not test_rename_columns \
and not test_rename_columns_modify_parent \
and not test_reorder_levels \
and not test_replace \
and not test_replace_categorical \
and not test_replace_categorical_inplace \
and not test_replace_categorical_inplace_reference \
and not test_replace_coerce_single_column \
and not test_replace_empty_list \
and not test_replace_inplace \
and not test_replace_inplace_reference \
and not test_replace_inplace_reference_no_op \
and not test_replace_list_categorical \
and not test_replace_list_inplace_refs_categorical \
and not test_replace_list_multiple_elements_inplace \
and not test_replace_list_none \
and not test_replace_list_none_inplace_refs \
and not test_replace_listlike  \
and not test_replace_listlike_inplace \
and not test_replace_mask_all_false_second_block \
and not test_replace_object_list_inplace \
and not test_replace_regex_inplace \
and not test_replace_regex_inplace_no_op \
and not test_replace_regex_inplace_refs \
and not test_replace_to_replace_wrong_dtype \
and not test_replace_value_is_none \
and not test_resample_datetime_values \
and not test_resample_empty \
and not test_resample_group_info \
and not test_resample_how_method \
and not test_resample_median_bug_1688 \
and not test_resample_rounding \
and not test_resample_tz_localized \
and not test_resample_unsigned_int \
and not test_resample_upsampling_picked_but_not_correct \
and not test_reset_index \
and not test_right \
and not test_rolling_apply_consistency_sum \
and not test_rolling_max_gh6297 \
and not test_rolling_max_resample \
and not test_rolling_median_resample \
and not test_rolling_min_resample \
and not test_rolling_numerical_accuracy_kahan_mean \
and not test_round \
and not test_rsplit_to_multiindex_expand_no_split \
and not test_scalar_call_versus_list_call \
and not test_searchsorted \
and not test_select_dtypes \
and not test_sem \
and not test_series_constructor \
and not test_series_getitem_multiindex \
and not test_series_repr \
and not test_series_setitem \
and not test_set_column_with_array \
and not test_set_column_with_index \
and not test_set_column_with_series \
and not test_set_columns_with_dataframe \
and not test_set_frame_overwrite_object \
and not test_set_index \
and not test_set_levels_categorical \
and not test_set_value_copy_only_necessary_column \
and not test_setattr_warnings \
and not test_setitem_dont_track_unnecessary_references \
and not test_setitem_frame_2d_values \
and not test_setitem_integer_array \
and not test_setitem_loc_iloc_slice \
and not test_setitem_mask \
and not test_setitem_mask_broadcast \
and not test_setitem_sequence_broadcasts \
and not test_setitem_slice \
and not test_setitem_with_view_copies \
and not test_setitem_with_view_invalidated_does_not_copy \
and not test_shallow_copy \
and not test_shallow_copy_shares_cache \
and not test_shift_index \
and not test_shift_no_op \
and not test_shift_rows_freq \
and not test_skew \
and not test_slice_month \
and not test_sort_index_intervalindex \
and not test_sort_values \
and not test_sort_values_inplace \
and not test_squeeze \
and not test_stack \
and not test_stack_tuple_columns \
and not test_sticky_basic \
and not test_str \
and not test_str_cat_categorical \
and not test_string_inference \
and not test_sub_datetime_preserves_freq \
and not test_subdays \
and not test_subdays_neg \
and not test_subset \
and not test_subset_column_selection \
and not test_subset_column_selection_modify_parent \
and not test_subset_row_slice \
and not test_subset_set_columns \
and not test_subtype_integer \
and not test_sum \
and not test_swapaxes_noop \
and not test_swapaxes_single_block \
and not test_td64arr_floordiv_td64arr_with_nat \
and not test_timegrouper_with_reg_groups \
and not test_to_csv_from_csv1 \
and not test_to_datetime_errors_ignore_utc_true \
and not test_to_frame \
and not test_to_numpy_copy \
and not test_to_period \
and not test_to_timestamp \
and not test_to_xarray_index_types \
and not test_transform_cumcount \
and not test_transform_numeric_ret \
and not test_transform_transformation_func \
and not test_transpose \
and not test_transpose_copy_keyword \
and not test_transpose_different_dtypes \
and not test_transpose_ea_single_column \
and not test_transpose_frame \
and not test_truncate \
and not test_ufunc \
and not test_ufunc_args \
and not test_ufunc_with_out \
and not test_ufuncs \
and not test_ufuncs_single \
and not test_ufuncs_single_int \
and not test_unary_ufunc_dunder_equivalence \
and not test_unique \
and not test_unpickling_without_closed \
and not test_unstack \
and not test_unstack_bool \
and not test_unstack_categorical \
and not test_unstack_categorical_columns \
and not test_unstack_non_slice_like_blocks \
and not test_unstack_unobserved_keys \
and not test_unstack_unused_levels \
and not test_upsample_sum \
and not test_utf8_writer \
and not test_value_counts_dropna \
and not test_var_std \
and not test_view \
and not test_where_mask_noop_on_single_column \
and not test_where_upcasting \
and not test_writes_tar_gz \
and not test_xs \
and not test_xs_level_series \
and not test_xs_multiindex \
and not test_xsqlite_if_exists \
and not test_xsqlite_write_row_by_row \
and not test_yaml_dump \
and not test_zero"

PYTEST_IGNORES="--ignore=tests/io/parser/common/test_read_errors.py \
--ignore=tests/io/test_clipboard.py" # crashes pytest workers (possibly due to fixture patching clipboard functionality)


PANDAS_CI="1" timeout 90m python -m pytest -p cudf.pandas \
    --import-mode=importlib \
    -k "$TEST_THAT_NEED_MOTO_SERVER and $TEST_THAT_CRASH_PYTEST_WORKERS and $TEST_THAT_NEED_REASON_TO_SKIP and $TEST_THAT_NEED_FIX" \
    ${PYTEST_IGNORES} \
    "$@" || [ $? = 1 ]  # Exit success if exit code was 1 (permit test failures but not other errors)

mv *.json ..
cd ..
rm -rf pandas-testing/pandas-tests/
