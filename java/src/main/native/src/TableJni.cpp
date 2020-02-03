/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/groupby.hpp>
#include <cudf/hashing.hpp>
#include <cudf/io/functions.hpp>
#include <cudf/join.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>

#include "jni_utils.hpp"

namespace cudf {
namespace jni {

/**
 * Take a table returned by some operation and turn it into an array of column* so we can track them ourselves
 * in java instead of having their life tied to the table.
 * @param table_result the table to convert for return
 * @param extra_columns columns not in the table that will be added to the result at the end.
 */
static jlongArray convert_table_for_return(JNIEnv * env,
        std::unique_ptr<cudf::experimental::table> &table_result,
        std::vector<std::unique_ptr<cudf::column>> &extra_columns) {
    std::vector<std::unique_ptr<cudf::column>> ret = table_result->release();
    int table_cols = ret.size();
    int num_columns = table_cols + extra_columns.size();
    cudf::jni::native_jlongArray outcol_handles(env, num_columns);
    for (int i = 0; i < table_cols; i++) {
      outcol_handles[i] = reinterpret_cast<jlong>(ret[i].release());
    }
    for (int i = 0; i < extra_columns.size(); i++) {
      outcol_handles[i + table_cols] = reinterpret_cast<jlong>(extra_columns[i].release());
    }
    return outcol_handles.get_jArray();
}

static jlongArray convert_table_for_return(JNIEnv * env, std::unique_ptr<cudf::experimental::table> &table_result) {
    std::vector<std::unique_ptr<cudf::column>> extra;
    return convert_table_for_return(env, table_result, extra);
}

std::unique_ptr<cudf::experimental::aggregation> map_jni_aggregation(jint op) {
  // These numbers come from AggregateOp.java and must stay in sync
  switch (op) {
    case 0: //SUM
      return cudf::experimental::make_sum_aggregation();
    case 1: //MIN
      return cudf::experimental::make_min_aggregation();
    case 2: //MAX
      return cudf::experimental::make_max_aggregation();
    case 3: //COUNT
      return cudf::experimental::make_count_aggregation();
    case 4: //MEAN
      return cudf::experimental::make_mean_aggregation();
    case 5: //MEDIAN
      return cudf::experimental::make_median_aggregation();
    case 7: // ARGMAX
      return cudf::experimental::make_argmax_aggregation();
    case 8: // ARGMIN
      return cudf::experimental::make_argmin_aggregation();
    default:
      throw std::logic_error("Unsupported Aggregation Operation");
  }
}

} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_createCudfTableView(JNIEnv *env, jclass class_object,
                                                                  jlongArray j_cudf_columns) {
  JNI_NULL_CHECK(env, j_cudf_columns, "columns are null", 0);

  try {
      cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_columns(env, j_cudf_columns);

    std::vector<cudf::column_view> column_views(n_cudf_columns.size());
    for (int i = 0 ; i < n_cudf_columns.size() ; i++) {
        column_views[i] = *n_cudf_columns[i];
    }
    cudf::table_view* tv = new cudf::table_view(column_views);
    return reinterpret_cast<jlong>(tv);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_deleteCudfTable(JNIEnv *env, jclass class_object,
                                                               jlong j_cudf_table_view) {
  JNI_NULL_CHECK(env, j_cudf_table_view, "table view handle is null", );
  delete reinterpret_cast<cudf::table_view*>(j_cudf_table_view);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_orderBy(
    JNIEnv *env, jclass j_class_object, jlong j_input_table, jlongArray j_sort_keys_columns,
    jbooleanArray j_is_descending, jbooleanArray j_are_nulls_smallest) {

  // input validations & verifications
  JNI_NULL_CHECK(env, j_input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_sort_keys_columns, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_is_descending, "sort order array is null", NULL);
  JNI_NULL_CHECK(env, j_are_nulls_smallest, "null order array is null", NULL);

  try {
    cudf::jni::native_jpointerArray<cudf::column_view> n_sort_keys_columns(env, j_sort_keys_columns);
    jsize num_columns = n_sort_keys_columns.size();
    const cudf::jni::native_jbooleanArray n_is_descending(env, j_is_descending);
    jsize num_columns_is_desc = n_is_descending.size();

    if (num_columns_is_desc != num_columns) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                    "columns and is_descending lengths don't match", NULL);
    }

    const cudf::jni::native_jbooleanArray n_are_nulls_smallest(env, j_are_nulls_smallest);
    jsize num_columns_null_smallest = n_are_nulls_smallest.size();

    if (num_columns_null_smallest != num_columns) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                    "columns and areNullsSmallest lengths don't match", NULL);
    }

    std::vector<cudf::order> order(n_is_descending.size());
    for (int i = 0; i < n_is_descending.size(); i++) {
      order[i] = n_is_descending[i] ? cudf::order::DESCENDING : cudf::order::ASCENDING;
    }
    std::vector<cudf::null_order> null_order(n_are_nulls_smallest.size());
    for (int i = 0; i < n_are_nulls_smallest.size(); i++) {
      null_order[i] = n_are_nulls_smallest[i] ? cudf::null_order::BEFORE : cudf::null_order::AFTER;
    }

    std::vector<cudf::column_view> columns;
    columns.reserve(num_columns);
    for (int i = 0; i < num_columns; i++) {
      columns.push_back(*n_sort_keys_columns[i]);
    }
    cudf::table_view keys(columns);

    auto sorted_col = cudf::experimental::sorted_order(keys, order, null_order);

    cudf::table_view *input_table = reinterpret_cast<cudf::table_view *>(j_input_table);
    std::unique_ptr<cudf::experimental::table> result = cudf::experimental::gather(*input_table,
            sorted_col->view());
    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_readCSV(
    JNIEnv *env, jclass j_class_object, jobjectArray col_names, jobjectArray data_types,
    jobjectArray filter_col_names, jstring inputfilepath, jlong buffer, jlong buffer_length,
    jint header_row, jbyte delim, jbyte quote, jbyte comment, jobjectArray null_values,
    jobjectArray true_values, jobjectArray false_values) {
  JNI_NULL_CHECK(env, null_values, "null_values must be supplied, even if it is empty", NULL);

  bool read_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
    read_buffer = false;
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "cannot pass in both a buffer and an inputfilepath", NULL);
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported",
                  NULL);
  }

  try {
    cudf::jni::native_jstringArray n_col_names(env, col_names);
    cudf::jni::native_jstringArray n_data_types(env, data_types);

    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inputfilepath can't be empty",
                    NULL);
    }

    cudf::jni::native_jstringArray n_null_values(env, null_values);
    cudf::jni::native_jstringArray n_true_values(env, true_values);
    cudf::jni::native_jstringArray n_false_values(env, false_values);
    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    std::unique_ptr<cudf::experimental::io::source_info> source;
    if (read_buffer) {
      source.reset(new cudf::experimental::io::source_info(reinterpret_cast<char *>(buffer), buffer_length));
    } else {
      source.reset(new cudf::experimental::io::source_info(filename.get()));
    }

    cudf::experimental::io::read_csv_args read_arg{*source};
    read_arg.lineterminator = '\n';
    // delimiter ideally passed in
    read_arg.delimiter = delim;
    read_arg.delim_whitespace = 0;
    read_arg.skipinitialspace = 0;
    read_arg.header = header_row;

    read_arg.names = n_col_names.as_cpp_vector();
    read_arg.dtype = n_data_types.as_cpp_vector();

    read_arg.use_cols_names = n_filter_col_names.as_cpp_vector();

    read_arg.skip_blank_lines = true;

    read_arg.true_values = n_true_values.as_cpp_vector();
    read_arg.false_values = n_false_values.as_cpp_vector();

    read_arg.na_values = n_null_values.as_cpp_vector();
    read_arg.keep_default_na = false; ///< Keep the default NA values
    read_arg.na_filter = n_null_values.size() > 0;

    read_arg.mangle_dupe_cols = true;
    read_arg.dayfirst = 0;
    read_arg.compression = cudf::experimental::io::compression_type::AUTO;
    read_arg.decimal = '.';
    read_arg.quotechar = quote;
    read_arg.quoting = cudf::experimental::io::quote_style::MINIMAL;
    read_arg.doublequote = true;
    read_arg.comment = comment;

    cudf::experimental::io::table_with_metadata result = cudf::experimental::io::read_csv(read_arg);
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_readParquet(
    JNIEnv *env, jclass j_class_object, jobjectArray filter_col_names, jstring inputfilepath,
    jlong buffer, jlong buffer_length, jint unit) {
  bool read_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
    read_buffer = false;
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "cannot pass in both a buffer and an inputfilepath", NULL);
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported",
                  NULL);
  }

  try {
    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inputfilepath can't be empty",
                    NULL);
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    std::unique_ptr<cudf::experimental::io::source_info> source;
    if (read_buffer) {
      source.reset(new cudf::experimental::io::source_info(reinterpret_cast<char *>(buffer), buffer_length));
    } else {
      source.reset(new cudf::experimental::io::source_info(filename.get()));
    }

    cudf::experimental::io::read_parquet_args read_arg(*source);

    read_arg.columns = n_filter_col_names.as_cpp_vector();

    read_arg.row_group = -1;
    read_arg.skip_rows = -1;
    read_arg.num_rows = -1;
    read_arg.strings_to_categorical = false;
    read_arg.timestamp_type = cudf::data_type(static_cast<cudf::type_id>(unit));

    cudf::experimental::io::table_with_metadata result = cudf::experimental::io::read_parquet(read_arg);
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeParquet(JNIEnv* env, jclass,
    jlong j_table,
    jobjectArray j_col_names,
    jobjectArray j_metadata_keys,
    jobjectArray j_metadata_values,
    jint j_compression,
    jint j_stats_freq,
    jstring j_output_path) {
  JNI_NULL_CHECK(env, j_table, "null table", );
  JNI_NULL_CHECK(env, j_col_names, "null columns", );
  JNI_NULL_CHECK(env, j_metadata_keys, "null metadata keys", );
  JNI_NULL_CHECK(env, j_metadata_values, "null metadata values", );
  try {
    using namespace cudf::experimental::io;
    cudf::jni::native_jstringArray col_names(env, j_col_names);
    cudf::jni::native_jstringArray meta_keys(env, j_metadata_keys);
    cudf::jni::native_jstringArray meta_values(env, j_metadata_values);
    cudf::jni::native_jstring output_path(env, j_output_path);

    table_metadata metadata{col_names.as_cpp_vector()};
    for (size_t i = 0; i < meta_keys.size(); ++i) {
      metadata.user_data[meta_keys[i].get()] = meta_values[i].get();
    }

    sink_info sink{output_path.get()};
    compression_type compression{static_cast<compression_type>(j_compression)};
    statistics_freq stats{static_cast<statistics_freq>(j_stats_freq)};

    cudf::table_view *tview = reinterpret_cast<cudf::table_view *>(j_table);
    write_parquet_args args(sink, *tview, &metadata, compression, stats);
    write_parquet(args);
  } CATCH_STD(env, )
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_readORC(
    JNIEnv *env, jclass j_class_object, jobjectArray filter_col_names, jstring inputfilepath,
    jlong buffer, jlong buffer_length, jboolean usingNumPyTypes, jint unit) {
  bool read_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
    read_buffer = false;
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "cannot pass in both a buffer and an inputfilepath", NULL);
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported",
                  NULL);
  }

  try {
    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inputfilepath can't be empty",
                    NULL);
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    std::unique_ptr<cudf::experimental::io::source_info> source;
    if (read_buffer) {
      source.reset(new cudf::experimental::io::source_info(reinterpret_cast<char *>(buffer), buffer_length));
    } else {
      source.reset(new cudf::experimental::io::source_info(filename.get()));
    }

    cudf::experimental::io::read_orc_args read_arg{*source};
    read_arg.columns = n_filter_col_names.as_cpp_vector();
    read_arg.stripe = -1;
    read_arg.skip_rows = -1;
    read_arg.num_rows = -1;
    read_arg.use_index = false;
    read_arg.use_np_dtypes = static_cast<bool>(usingNumPyTypes);
    read_arg.timestamp_type = cudf::data_type(static_cast<cudf::type_id>(unit));

    cudf::experimental::io::table_with_metadata result = cudf::experimental::io::read_orc(read_arg);
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeORC(JNIEnv *env, jclass,
                                                              jint j_compression_type,
                                                              jobjectArray j_col_names,
                                                              jobjectArray j_metadata_keys,
                                                              jobjectArray j_metadata_values,
                                                              jstring outputfilepath, jlong buffer,
                                                              jlong buffer_length, jlong j_table_view) {
  bool write_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, outputfilepath, "output file or buffer must be supplied", );
    write_buffer = false;
  } else if (outputfilepath != NULL) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "cannot pass in both a buffer and an outputfilepath", );
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported", );
  }
  JNI_NULL_CHECK(env, j_col_names, "null columns", );
  JNI_NULL_CHECK(env, j_metadata_keys, "null metadata keys", );
  JNI_NULL_CHECK(env, j_metadata_values, "null metadata values", );

  try {
    cudf::jni::native_jstring filename(env, outputfilepath);
    cudf::jni::native_jstringArray meta_keys(env, j_metadata_keys);
    cudf::jni::native_jstringArray meta_values(env, j_metadata_values);
    cudf::jni::native_jstringArray col_names(env, j_col_names);
    namespace orc = cudf::experimental::io;
    if (write_buffer) {
      JNI_THROW_NEW(env, "java/lang/UnsupportedOperationException",
                        "buffers are not supported", );
    } else {
      orc::sink_info info(filename.get());
      orc::table_metadata metadata{col_names.as_cpp_vector()};
      for (size_t i = 0; i < meta_keys.size(); ++i) {
        metadata.user_data[meta_keys[i].get()] = meta_values[i].get();
      }
      orc::write_orc_args args(info, *reinterpret_cast<cudf::table_view*>(j_table_view), &metadata,
                                                static_cast<orc::compression_type>(j_compression_type));
      orc::write_orc(args);
    }
  }
  CATCH_STD(env, );
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftJoin(
    JNIEnv *env, jclass clazz, jlong left_table, jintArray left_col_join_indices, jlong right_table,
    jintArray right_col_join_indices) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, left_col_join_indices, "left_col_join_indices is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, right_col_join_indices, "right_col_join_indices is null", NULL);

  try {
    cudf::table_view *n_left_table = reinterpret_cast<cudf::table_view *>(left_table);
    cudf::table_view *n_right_table = reinterpret_cast<cudf::table_view *>(right_table);
    cudf::jni::native_jintArray left_join_cols_arr(env, left_col_join_indices);
    std::vector<cudf::size_type> left_join_cols(left_join_cols_arr.data(), left_join_cols_arr.data() + left_join_cols_arr.size());
    cudf::jni::native_jintArray right_join_cols_arr(env, right_col_join_indices);
    std::vector<cudf::size_type> right_join_cols(right_join_cols_arr.data(), right_join_cols_arr.data() + right_join_cols_arr.size());

    int dedupe_size = left_join_cols.size();
    std::vector<std::pair<cudf::size_type, cudf::size_type>> dedupe(dedupe_size);
    for (int i = 0; i < dedupe_size; i++) {
      dedupe[i].first = left_join_cols[i];
      dedupe[i].second = right_join_cols[i];
    }

    std::unique_ptr<cudf::experimental::table> result = cudf::experimental::left_join(
            *n_left_table, *n_right_table,
            left_join_cols, right_join_cols,
            dedupe);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_innerJoin(
    JNIEnv *env, jclass clazz, jlong left_table, jintArray left_col_join_indices, jlong right_table,
    jintArray right_col_join_indices) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, left_col_join_indices, "left_col_join_indices is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, right_col_join_indices, "right_col_join_indices is null", NULL);

  try {
    cudf::table_view *n_left_table = reinterpret_cast<cudf::table_view *>(left_table);
    cudf::table_view *n_right_table = reinterpret_cast<cudf::table_view *>(right_table);
    cudf::jni::native_jintArray left_join_cols_arr(env, left_col_join_indices);
    std::vector<cudf::size_type> left_join_cols(left_join_cols_arr.data(), left_join_cols_arr.data() + left_join_cols_arr.size());
    cudf::jni::native_jintArray right_join_cols_arr(env, right_col_join_indices);
    std::vector<cudf::size_type> right_join_cols(right_join_cols_arr.data(), right_join_cols_arr.data() + right_join_cols_arr.size());

    int dedupe_size = left_join_cols.size();
    std::vector<std::pair<cudf::size_type, cudf::size_type>> dedupe(dedupe_size);
    for (int i = 0; i < dedupe_size; i++) {
      dedupe[i].first = left_join_cols[i];
      dedupe[i].second = right_join_cols[i];
    }

    std::unique_ptr<cudf::experimental::table> result = cudf::experimental::inner_join(
            *n_left_table, *n_right_table,
            left_join_cols, right_join_cols,
            dedupe);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftSemiJoin(JNIEnv *env, jclass,
    jlong left_table, jintArray left_col_join_indices, jlong right_table,
    jintArray right_col_join_indices) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, left_col_join_indices, "left_col_join_indices is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, right_col_join_indices, "right_col_join_indices is null", NULL);

  try {
    cudf::table_view *n_left_table = reinterpret_cast<cudf::table_view *>(left_table);
    cudf::table_view *n_right_table = reinterpret_cast<cudf::table_view *>(right_table);
    cudf::jni::native_jintArray left_join_cols_arr(env, left_col_join_indices);
    std::vector<cudf::size_type> left_join_cols(left_join_cols_arr.data(), left_join_cols_arr.data() + left_join_cols_arr.size());
    cudf::jni::native_jintArray right_join_cols_arr(env, right_col_join_indices);
    std::vector<cudf::size_type> right_join_cols(right_join_cols_arr.data(), right_join_cols_arr.data() + right_join_cols_arr.size());
    std::vector<cudf::size_type> return_cols(n_left_table->num_columns());
    for (cudf::size_type i = 0; i < n_left_table->num_columns(); ++i) {
      return_cols[i] = i;
    }

    std::unique_ptr<cudf::experimental::table> result = cudf::experimental::left_semi_join(
            *n_left_table, *n_right_table,
            left_join_cols, right_join_cols,
            return_cols);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftAntiJoin(JNIEnv *env, jclass,
    jlong left_table, jintArray left_col_join_indices, jlong right_table,
    jintArray right_col_join_indices) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, left_col_join_indices, "left_col_join_indices is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, right_col_join_indices, "right_col_join_indices is null", NULL);

  try {
    cudf::table_view *n_left_table = reinterpret_cast<cudf::table_view *>(left_table);
    cudf::table_view *n_right_table = reinterpret_cast<cudf::table_view *>(right_table);
    cudf::jni::native_jintArray left_join_cols_arr(env, left_col_join_indices);
    std::vector<cudf::size_type> left_join_cols(left_join_cols_arr.data(), left_join_cols_arr.data() + left_join_cols_arr.size());
    cudf::jni::native_jintArray right_join_cols_arr(env, right_col_join_indices);
    std::vector<cudf::size_type> right_join_cols(right_join_cols_arr.data(), right_join_cols_arr.data() + right_join_cols_arr.size());
    std::vector<cudf::size_type> return_cols(n_left_table->num_columns());
    for (cudf::size_type i = 0; i < n_left_table->num_columns(); ++i) {
      return_cols[i] = i;
    }

    std::unique_ptr<cudf::experimental::table> result = cudf::experimental::left_anti_join(
            *n_left_table, *n_right_table,
            left_join_cols, right_join_cols,
            return_cols);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_concatenate(JNIEnv *env, jclass clazz,
                                                                   jlongArray table_handles) {
  JNI_NULL_CHECK(env, table_handles, "input tables are null", NULL);
  try {
    cudf::jni::native_jpointerArray<cudf::table_view> tables(env, table_handles);

    long unsigned int num_tables = tables.size();
    // There are some issues with table_view and std::vector. We cannot give the
    // vector a size or it will not compile.
    std::vector<cudf::table_view> to_concat;
    to_concat.reserve(num_tables);
    for (int i = 0; i < num_tables; i++) {
      JNI_NULL_CHECK(env, tables[i], "input table included a null", NULL);
      to_concat.push_back(*tables[i]);
    }
    std::unique_ptr<cudf::experimental::table> table_result = cudf::experimental::concatenate(to_concat);
    return cudf::jni::convert_table_for_return(env, table_result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_partition(
    JNIEnv *env, jclass clazz, jlong input_table, jintArray columns_to_hash,
    jint number_of_partitions, jintArray output_offsets) {

  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, columns_to_hash, "columns_to_hash is null", NULL);
  JNI_NULL_CHECK(env, output_offsets, "output_offsets is null", NULL);
  JNI_ARG_CHECK(env, number_of_partitions > 0, "number_of_partitions is zero", NULL);

  try {
    cudf::table_view *n_input_table = reinterpret_cast<cudf::table_view *>(input_table);
    cudf::jni::native_jintArray n_columns_to_hash(env, columns_to_hash);
    int n_number_of_partitions = static_cast<int>(number_of_partitions);
    cudf::jni::native_jintArray n_output_offsets(env, output_offsets);

    JNI_ARG_CHECK(env, n_columns_to_hash.size() > 0, "columns_to_hash is zero", NULL);

    std::vector<cudf::size_type> columns_to_hash_vec(n_columns_to_hash.size());
    for (int i = 0; i < n_columns_to_hash.size(); i++) {
      columns_to_hash_vec[i] = n_columns_to_hash[i];
    }

    std::pair<std::unique_ptr<cudf::experimental::table>, std::vector<cudf::size_type>> result
        = cudf::hash_partition(*n_input_table, columns_to_hash_vec, number_of_partitions);

    for (int i = 0; i < result.second.size(); i++) {
      n_output_offsets[i] = result.second[i];
    }

    return cudf::jni::convert_table_for_return(env, result.first);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_groupByAggregate(
    JNIEnv *env, jclass clazz, jlong input_table, jintArray keys,
    jintArray aggregate_column_indices, jintArray agg_types, jboolean ignore_null_keys) {
  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, keys, "input keys are null", NULL);
  JNI_NULL_CHECK(env, aggregate_column_indices, "input aggregate_column_indices are null", NULL);
  JNI_NULL_CHECK(env, agg_types, "agg_types are null", NULL);

  try {
    cudf::table_view *n_input_table = reinterpret_cast<cudf::table_view *>(input_table);
    cudf::jni::native_jintArray n_keys(env, keys);
    cudf::jni::native_jintArray n_values(env, aggregate_column_indices);
    cudf::jni::native_jintArray n_ops(env, agg_types);
    std::vector<cudf::column_view> n_keys_cols;
    n_keys_cols.reserve(n_keys.size());
    for (int i = 0; i < n_keys.size(); i++) {
      n_keys_cols.push_back(n_input_table->column(n_keys[i]));
    }

    cudf::table_view n_keys_table(n_keys_cols);
    cudf::experimental::groupby::groupby grouper(n_keys_table, ignore_null_keys);

    // Aggregates are passed in already grouped by column, so we just need to fill it in
    // as we go.
    std::vector<cudf::experimental::groupby::aggregation_request> requests;

    int previous_index = -1;
    for (int i = 0; i < n_values.size(); i++) {
      cudf::experimental::groupby::aggregation_request req;
      int col_index = n_values[i];
      if (col_index == previous_index) {
        requests.back().aggregations.push_back(cudf::jni::map_jni_aggregation(n_ops[i]));
      } else {
        req.values = n_input_table->column(col_index);
        req.aggregations.push_back(cudf::jni::map_jni_aggregation(n_ops[i]));
        requests.push_back(std::move(req));
      }
      previous_index = col_index;
    }

    std::pair<std::unique_ptr<cudf::experimental::table>, std::vector<cudf::experimental::groupby::aggregation_result>> result = grouper.aggregate(requests);

    std::vector<std::unique_ptr<cudf::column>> result_columns;
    int agg_result_size = result.second.size();
    for (int agg_result_index = 0; agg_result_index < agg_result_size; agg_result_index++) {
      int col_agg_size = result.second[agg_result_index].results.size();
      for (int col_agg_index = 0; col_agg_index < col_agg_size; col_agg_index++) {
        result_columns.push_back(std::move(result.second[agg_result_index].results[col_agg_index]));
      }
    }
    return cudf::jni::convert_table_for_return(env, result.first, result_columns);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_filter(JNIEnv *env, jclass,
                                                              jlong input_jtable,
                                                              jlong mask_jcol) {
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_NULL_CHECK(env, mask_jcol, "mask column is null", 0);
  try {
    cudf::table_view *input = reinterpret_cast<cudf::table_view *>(input_jtable);
    cudf::column_view *mask = reinterpret_cast<cudf::column_view *>(mask_jcol);
    std::unique_ptr<cudf::experimental::table> result = cudf::experimental::apply_boolean_mask(*input, *mask);
    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_bound(JNIEnv *env, jclass,
    jlong input_jtable, jlong values_jtable, jbooleanArray desc_flags, jbooleanArray are_nulls_smallest,
    jboolean is_upper_bound) {
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_NULL_CHECK(env, values_jtable, "values table is null", 0);
  using cudf::table_view;
  using cudf::column;
  try {
    table_view *input = reinterpret_cast<table_view *>(input_jtable);
    table_view *values = reinterpret_cast<table_view *>(values_jtable);
    cudf::jni::native_jbooleanArray const n_desc_flags(env, desc_flags);
    cudf::jni::native_jbooleanArray const n_are_nulls_smallest(env, are_nulls_smallest);

    std::vector<cudf::order> column_desc_flags(n_desc_flags.size());
    std::vector<cudf::null_order> column_null_orders(n_are_nulls_smallest.size());

    JNI_ARG_CHECK(env, (column_desc_flags.size() == column_null_orders.size()), "null-order and sort-order size mismatch", 0);
    uint32_t num_columns = column_null_orders.size();
    for (int i = 0 ; i < num_columns ; i++) {
      column_desc_flags[i] = n_desc_flags[i] ? cudf::order::DESCENDING : cudf::order::ASCENDING;
      column_null_orders[i] = n_are_nulls_smallest[i] ? cudf::null_order::BEFORE: cudf::null_order::AFTER;
    }

    std::unique_ptr<column> result;
    if (is_upper_bound) {
      result = std::move(cudf::experimental::upper_bound(*input, *values, column_desc_flags, column_null_orders));
    } else {
      result = std::move(cudf::experimental::lower_bound(*input, *values, column_desc_flags, column_null_orders));
    }
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jobjectArray JNICALL Java_ai_rapids_cudf_Table_contiguousSplit(JNIEnv *env, jclass clazz,
                                                             jlong input_table,
                                                             jintArray split_indices) {
  JNI_NULL_CHECK(env, input_table, "native handle is null", 0);
  JNI_NULL_CHECK(env, split_indices, "split indices are null", 0);

  try {
    cudf::table_view *n_table = reinterpret_cast<cudf::table_view *>(input_table);
    cudf::jni::native_jintArray n_split_indices(env, split_indices);

    std::vector<cudf::size_type> indices(n_split_indices.data(), n_split_indices.data() + n_split_indices.size());

    std::vector<cudf::experimental::contiguous_split_result> result = 
        cudf::experimental::contiguous_split(*n_table, indices);
    cudf::jni::native_jobjectArray<jobject> n_result = 
        cudf::jni::contiguous_table_array(env, result.size());
    for (int i = 0; i < result.size(); i++) {
      n_result.set(i, cudf::jni::contiguous_table_from(env, result[i]));
    }
    return n_result.wrapped();
  }
  CATCH_STD(env, NULL);
}

} // extern "C"
