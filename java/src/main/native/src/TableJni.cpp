/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cstring>
#include <map>

#include <unordered_set>

#include "cudf/utilities/legacy/nvcategory_util.hpp"
#include "cudf/legacy/copying.hpp"
#include "cudf/groupby.hpp"
#include "cudf/legacy/io_readers.hpp"
#include "cudf/legacy/table.hpp"
#include "cudf/stream_compaction.hpp"
#include "cudf/types.hpp"
#include "cudf/join.hpp"

#include "jni_utils.hpp"

namespace cudf {
namespace jni {

/**
 * Copy contents of a jbooleanArray into an array of int8_t pointers
 */
static jni_rmm_unique_ptr<int8_t> copy_to_device(JNIEnv *env, const native_jbooleanArray &n_arr) {
  jsize len = n_arr.size();
  size_t byte_len = len * sizeof(int8_t);
  const jboolean *tmp = n_arr.data();

  std::unique_ptr<int8_t[]> host(new int8_t[byte_len]);

  for (int i = 0; i < len; i++) {
    host[i] = static_cast<int8_t>(n_arr[i]);
  }

  auto device = jni_rmm_alloc<int8_t>(env, byte_len);
  jni_cuda_check(env, cudaMemcpy(device.get(), host.get(), byte_len, cudaMemcpyHostToDevice));
  return device;
}

} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_createCudfTable(JNIEnv *env, jclass class_object,
                                                                  jlongArray cudf_columns) {
  JNI_NULL_CHECK(env, cudf_columns, "input columns are null", 0);

  try {
    cudf::jni::native_jpointerArray<gdf_column> n_cudf_columns(env, cudf_columns);
    cudf::table *table = new cudf::table(n_cudf_columns.data(), n_cudf_columns.size());
    return reinterpret_cast<jlong>(table);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_freeCudfTable(JNIEnv *env, jclass class_object,
                                                               jlong handle) {
  cudf::table *table = reinterpret_cast<cudf::table *>(handle);
  delete table;
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfOrderBy(
    JNIEnv *env, jclass j_class_object, jlong j_input_table, jlongArray j_sort_keys_gdfcolumns,
    jbooleanArray j_is_descending, jboolean j_are_nulls_smallest) {

  // input validations & verifications
  JNI_NULL_CHECK(env, j_input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_sort_keys_gdfcolumns, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_is_descending, "sort order array is null", NULL);

  try {
    cudf::jni::native_jpointerArray<gdf_column> n_sort_keys_gdfcolumns(env, j_sort_keys_gdfcolumns);
    jsize num_columns = n_sort_keys_gdfcolumns.size();
    const cudf::jni::native_jbooleanArray n_is_descending(env, j_is_descending);
    jsize num_columns_is_desc = n_is_descending.size();

    if (num_columns_is_desc != num_columns) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                    "columns and is_descending lengths don't match", NULL);
    }

    auto is_descending = cudf::jni::copy_to_device(env, n_is_descending);

    cudf::table *input_table = reinterpret_cast<cudf::table *>(j_input_table);
    cudf::jni::output_table output(env, input_table, true);

    bool are_nulls_smallest = static_cast<bool>(j_are_nulls_smallest);

    auto col_data = cudf::jni::jni_rmm_alloc<int32_t>(
        env, n_sort_keys_gdfcolumns[0]->size * sizeof(int32_t), 0);

    gdf_column intermediate_output;
    // construct column view
    cudf::jni::jni_cudf_check(env, gdf_column_view(&intermediate_output, col_data.get(), nullptr,
                                                   n_sort_keys_gdfcolumns[0]->size,
                                                   gdf_dtype::GDF_INT32));

    gdf_context context{};
    // Most of these are probably ignored, but just to be safe
    context.flag_sorted = false;
    context.flag_method = GDF_SORT;
    context.flag_distinct = 0;
    context.flag_sort_result = 1;
    context.flag_sort_inplace = 0;
    context.flag_groupby_include_nulls = true;
    // There is also a MULTI COLUMN VERSION, that we may want to support in the
    // future.
    context.flag_null_sort_behavior =
        j_are_nulls_smallest ? GDF_NULL_AS_SMALLEST : GDF_NULL_AS_LARGEST;

    cudf::jni::jni_cudf_check(env, gdf_order_by(n_sort_keys_gdfcolumns.data(), is_descending.get(),
                                                static_cast<size_t>(num_columns),
                                                &intermediate_output, &context));

    cudf::table *cudf_table = output.get_cudf_table();

    // gather handles string categories
    gather(input_table, col_data.get(), cudf_table);

    return output.get_native_handles_and_release();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfReadCSV(
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

    std::unique_ptr<cudf::source_info> source;
    if (read_buffer) {
      source.reset(new cudf::source_info(reinterpret_cast<char *>(buffer), buffer_length));
    } else {
      source.reset(new cudf::source_info(filename.get()));
    }

    cudf::csv_read_arg read_arg{*source};
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
    read_arg.compression = "infer";
    read_arg.decimal = '.';
    read_arg.quotechar = quote;
    read_arg.quoting = cudf::csv_read_arg::QUOTE_MINIMAL;
    read_arg.doublequote = true;
    read_arg.comment = comment;

    cudf::table result = read_csv(read_arg);
    cudf::jni::native_jlongArray native_handles(env, reinterpret_cast<jlong *>(result.begin()),
                                                result.num_columns());

    return native_handles.get_jArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfReadParquet(
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

    std::unique_ptr<cudf::source_info> source;
    if (read_buffer) {
      source.reset(new cudf::source_info(reinterpret_cast<char *>(buffer), buffer_length));
    } else {
      source.reset(new cudf::source_info(filename.get()));
    }

    cudf::parquet_read_arg read_arg{*source};

    read_arg.columns = n_filter_col_names.as_cpp_vector();

    read_arg.row_group = -1;
    read_arg.skip_rows = -1;
    read_arg.num_rows = -1;
    read_arg.strings_to_categorical = false;
    read_arg.timestamp_unit = static_cast<gdf_time_unit>(unit);

    cudf::table result = read_parquet(read_arg);
    cudf::jni::native_jlongArray native_handles(env, reinterpret_cast<jlong *>(result.begin()),
                                                result.num_columns());

    return native_handles.get_jArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfReadORC(
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

    std::unique_ptr<cudf::source_info> source;
    if (read_buffer) {
      source.reset(new cudf::source_info(reinterpret_cast<char *>(buffer), buffer_length));
    } else {
      source.reset(new cudf::source_info(filename.get()));
    }

    cudf::orc_read_arg read_arg{*source};
    read_arg.columns = n_filter_col_names.as_cpp_vector();
    read_arg.stripe = -1;
    read_arg.skip_rows = -1;
    read_arg.num_rows = -1;
    read_arg.use_index = false;
    read_arg.use_np_dtypes = static_cast<bool>(usingNumPyTypes);
    read_arg.timestamp_unit = static_cast<gdf_time_unit>(unit);

    cudf::table result = read_orc(read_arg);
    cudf::jni::native_jlongArray native_handles(env, reinterpret_cast<jlong *>(result.begin()),
                                                result.num_columns());
    return native_handles.get_jArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_gdfWriteORC(JNIEnv *env, jclass,
                                                              jint compression_type,
                                                              jstring outputfilepath, jlong buffer,
                                                              jlong buffer_length, jlong table) {
  bool write_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, outputfilepath, "output file or buffer must be supplied", 0);
    write_buffer = false;
  } else if (outputfilepath != NULL) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "cannot pass in both a buffer and an outputfilepath", 0);
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported", 0);
  }

  try {
    cudf::jni::native_jstring filename(env, outputfilepath);
    namespace orc = cudf::io::orc;
    orc::compression_type n_compressionType = static_cast<orc::compression_type>(compression_type);
    if (write_buffer) {
      JNI_THROW_NEW(env, "java/lang/UnsupportedOperationException",
                        "buffers are not supported", 0);
    } else {
      cudf::sink_info info(filename.get());
      cudf::orc_write_arg args(info);
      auto writer = [&]() {

        orc::writer_options options(n_compressionType);
        return std::make_unique<orc::writer>(args.sink.filepath, options);
      }();

      args.table = *reinterpret_cast<cudf::table *>(table);
      writer->write_all(args.table);
    }
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfLeftJoin(
    JNIEnv *env, jclass clazz, jlong left_table, jintArray left_col_join_indices, jlong right_table,
    jintArray right_col_join_indices) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, left_col_join_indices, "left_col_join_indices is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, right_col_join_indices, "right_col_join_indices is null", NULL);

  try {
    cudf::table *n_left_table = reinterpret_cast<cudf::table *>(left_table);
    cudf::table *n_right_table = reinterpret_cast<cudf::table *>(right_table);
    cudf::jni::native_jintArray left_join_cols_arr(env, left_col_join_indices);
    std::vector<int> left_join_cols(left_join_cols_arr.data(), left_join_cols_arr.data() + left_join_cols_arr.size());
    cudf::jni::native_jintArray right_join_cols_arr(env, right_col_join_indices);
    std::vector<int> right_join_cols(right_join_cols_arr.data(), right_join_cols_arr.data() + right_join_cols_arr.size());

    gdf_context context{};
    context.flag_sorted = 0;
    context.flag_method = GDF_HASH;
    context.flag_distinct = 0;
    context.flag_sort_result = 1;
    context.flag_sort_inplace = 0;

    int dedupe_size = left_join_cols.size();
    std::vector<std::pair<cudf::size_type, cudf::size_type>> dedupe(dedupe_size);
    for (int i = 0; i < dedupe_size; i++) {
      dedupe[i].first = left_join_cols[i];
      dedupe[i].second = right_join_cols[i];
    }

    cudf::table result = cudf::left_join(
            *n_left_table, *n_right_table,
            left_join_cols, right_join_cols,
            dedupe,
            nullptr, &context);

    cudf::jni::native_jlongArray native_handles(env, reinterpret_cast<jlong *>(result.begin()),
                                                result.num_columns());

    return native_handles.get_jArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfInnerJoin(
    JNIEnv *env, jclass clazz, jlong left_table, jintArray left_col_join_indices, jlong right_table,
    jintArray right_col_join_indices) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, left_col_join_indices, "left_col_join_indices is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, right_col_join_indices, "right_col_join_indices is null", NULL);

  try {
    cudf::table *n_left_table = reinterpret_cast<cudf::table *>(left_table);
    cudf::table *n_right_table = reinterpret_cast<cudf::table *>(right_table);
    cudf::jni::native_jintArray left_join_cols_arr(env, left_col_join_indices);
    std::vector<int> left_join_cols(left_join_cols_arr.data(), left_join_cols_arr.data() + left_join_cols_arr.size());
    cudf::jni::native_jintArray right_join_cols_arr(env, right_col_join_indices);
    std::vector<int> right_join_cols(right_join_cols_arr.data(), right_join_cols_arr.data() + right_join_cols_arr.size());

    gdf_context context{};
    context.flag_sorted = 0;
    context.flag_method = GDF_HASH;
    context.flag_distinct = 0;
    context.flag_sort_result = 1;
    context.flag_sort_inplace = 0;

    int dedupe_size = left_join_cols.size();
    std::vector<std::pair<cudf::size_type, cudf::size_type>> dedupe(dedupe_size);
    for (int i = 0; i < dedupe_size; i++) {
      dedupe[i].first = left_join_cols[i];
      dedupe[i].second = right_join_cols[i];
    }

    cudf::table result = cudf::inner_join(
            *n_left_table, *n_right_table,
            left_join_cols, right_join_cols,
            dedupe,
            nullptr, &context);

    cudf::jni::native_jlongArray native_handles(env, reinterpret_cast<jlong *>(result.begin()),
                                                result.num_columns());

    return native_handles.get_jArray();

  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_concatenate(JNIEnv *env, jclass clazz,
                                                                   jlongArray table_handles) {
  JNI_NULL_CHECK(env, table_handles, "input tables are null", NULL);
  try {
    cudf::jni::native_jpointerArray<cudf::table> tables(env, table_handles);

    // calculate output table size and whether each column needs a validity
    // vector
    int num_columns = tables[0]->num_columns();
    std::vector<bool> need_validity(num_columns);
    size_t total_size = 0;
    for (int table_idx = 0; table_idx < tables.size(); ++table_idx) {
      total_size += tables[table_idx]->num_rows();
      for (int col_idx = 0; col_idx < num_columns; ++col_idx) {
        gdf_column const *col = tables[table_idx]->get_column(col_idx);
        // Should be checking for null_count != 0 but libcudf is checking valid
        // != nullptr
        if (col->valid != nullptr) {
          need_validity[col_idx] = true;
        }
      }
    }

    // check for overflow
    if (total_size != static_cast<cudf::size_type>(total_size)) {
      cudf::jni::throw_java_exception(env, "java/lang/IllegalArgumentException",
                                      "resulting column is too large");
    }

    std::vector<cudf::jni::gdf_column_wrapper> outcols;
    outcols.reserve(num_columns);
    std::vector<gdf_column *> outcol_ptrs(num_columns);
    std::vector<gdf_column *> concat_input_ptrs(tables.size());
    for (int col_idx = 0; col_idx < num_columns; ++col_idx) {
      outcols.emplace_back(total_size, tables[0]->get_column(col_idx)->dtype,
                           need_validity[col_idx], true);
      outcol_ptrs[col_idx] = outcols[col_idx].get();
      if (outcol_ptrs[col_idx]->dtype == GDF_TIMESTAMP) {
        outcol_ptrs[col_idx]->dtype_info.time_unit =
            tables[0]->get_column(col_idx)->dtype_info.time_unit;
      }
      for (int table_idx = 0; table_idx < tables.size(); ++table_idx) {
        concat_input_ptrs[table_idx] = tables[table_idx]->get_column(col_idx);
      }
      JNI_GDF_TRY(env, NULL,
                  gdf_column_concat(outcol_ptrs[col_idx], concat_input_ptrs.data(), tables.size()));
    }

    cudf::jni::native_jlongArray outcol_handles(env, reinterpret_cast<jlong *>(outcol_ptrs.data()),
                                                num_columns);
    jlongArray result = outcol_handles.get_jArray();
    for (int i = 0; i < num_columns; ++i) {
      outcols[i].release();
    }

    return result;
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfPartition(
    JNIEnv *env, jclass clazz, jlong input_table, jintArray columns_to_hash,
    jint cudf_hash_function, jint number_of_partitions, jintArray output_offsets) {

  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, columns_to_hash, "columns_to_hash is null", NULL);
  JNI_NULL_CHECK(env, output_offsets, "output_offsets is null", NULL);
  JNI_ARG_CHECK(env, number_of_partitions > 0, "number_of_partitions is zero", NULL);

  try {
    cudf::table *n_input_table = reinterpret_cast<cudf::table *>(input_table);
    cudf::jni::native_jintArray n_columns_to_hash(env, columns_to_hash);
    gdf_hash_func n_cudf_hash_function = static_cast<gdf_hash_func>(cudf_hash_function);
    int n_number_of_partitions = static_cast<int>(number_of_partitions);
    cudf::jni::native_jintArray n_output_offsets(env, output_offsets);

    JNI_ARG_CHECK(env, n_columns_to_hash.size() > 0, "columns_to_hash is zero", NULL);

    cudf::jni::output_table output(env, n_input_table, true);
    std::vector<gdf_column *> cols = output.get_gdf_columns();

    for (int i = 0; i < cols.size(); i++) {
      gdf_column * col = cols[i];
      if (col->dtype == GDF_STRING_CATEGORY) {
        // We need to add in the category for partition to work at all...
        NVCategory * orig = static_cast<NVCategory *>(n_input_table->get_column(i)->dtype_info.category);
        col->dtype_info.category = orig;
      }
    }

    JNI_GDF_TRY(env, NULL,
                gdf_hash_partition(n_input_table->num_columns(), n_input_table->begin(),
                                   n_columns_to_hash.data(), n_columns_to_hash.size(),
                                   n_number_of_partitions, cols.data(), n_output_offsets.data(),
                                   n_cudf_hash_function));

    // Need to gather the string categories after partitioning.
    for (int i = 0; i < cols.size(); i++) {
      gdf_column * col = cols[i];
      if (col->dtype == GDF_STRING_CATEGORY) {
        // We need to fix it up...
        NVCategory * orig = static_cast<NVCategory *>(n_input_table->get_column(i)->dtype_info.category);
        nvcategory_gather(col, orig);
      }
    }

    return output.get_native_handles_and_release();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfGroupByAggregate(
    JNIEnv *env, jclass clazz, jlong input_table, jintArray keys,
    jintArray aggregate_column_indices, jintArray agg_types, jboolean ignore_null_keys) {
  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, keys, "input keys are null", NULL);
  JNI_NULL_CHECK(env, aggregate_column_indices, "input aggregate_column_indices are null", NULL);
  JNI_NULL_CHECK(env, agg_types, "agg_types are null", NULL);

  try {
    cudf::table *n_input_table = reinterpret_cast<cudf::table *>(input_table);
    cudf::jni::native_jintArray n_keys(env, keys);
    cudf::jni::native_jintArray n_values(env, aggregate_column_indices);
    cudf::jni::native_jintArray n_ops(env, agg_types);
    std::vector<gdf_column *> n_keys_cols;
    std::vector<gdf_column *> n_values_cols;

    for (int i = 0; i < n_keys.size(); i++) {
      n_keys_cols.push_back(n_input_table->get_column(n_keys[i]));
    }

    for (int i = 0; i < n_values.size(); i++) {
      n_values_cols.push_back(n_input_table->get_column(n_values[i]));
    }

    cudf::table const n_keys_table(n_keys_cols);
    cudf::table const n_values_table(n_values_cols);

    std::vector<cudf::groupby::operators> ops;
    for (int i = 0; i < n_ops.size(); i++) {
      ops.push_back(static_cast<cudf::groupby::operators>(n_ops[i]));
    }

    std::pair<cudf::table, cudf::table> result = cudf::groupby::hash::groupby(
        n_keys_table, n_values_table, ops, cudf::groupby::hash::Options(ignore_null_keys));

    try {
      std::vector<gdf_column *> output_columns;
      output_columns.reserve(result.first.num_columns() + result.second.num_columns());
      output_columns.insert(output_columns.end(), result.first.begin(), result.first.end());
      output_columns.insert(output_columns.end(), result.second.begin(), result.second.end());
      cudf::jni::native_jlongArray native_handles(
          env, reinterpret_cast<jlong *>(output_columns.data()), output_columns.size());
      return native_handles.get_jArray();
    } catch (...) {
      result.first.destroy();
      result.second.destroy();
      throw;
    }
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfFilter(JNIEnv *env, jclass,
                                                                 jlong input_jtable,
                                                                 jlong mask_jcol) {
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_NULL_CHECK(env, mask_jcol, "mask column is null", 0);
  try {
    cudf::table *input = reinterpret_cast<cudf::table *>(input_jtable);
    gdf_column *mask = reinterpret_cast<gdf_column *>(mask_jcol);
    cudf::table result = cudf::apply_boolean_mask(*input, *mask);
    cudf::jni::native_jlongArray native_handles(env, reinterpret_cast<jlong *>(result.begin()),
                                                result.num_columns());
    return native_handles.get_jArray();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gdfReadJSON(
    JNIEnv *env, jclass clazz, jstring input_filepath, jlong buffer, jlong buffer_length,
    jlong start_range, jlong range_length, jobjectArray filter_col_names, jobjectArray col_names,
    jobjectArray data_types) {
  bool read_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, input_filepath, "input file or buffer must be supplied", NULL);
    read_buffer = false;
  } else if (input_filepath != NULL) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "cannot pass in both a buffer and an inputfilepath", NULL);
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported",
                  NULL);
  }
  try {
    cudf::jni::native_jstring filename(env, input_filepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inputfilepath can't be empty",
                    NULL);
    }

    std::unique_ptr<cudf::source_info> source;
    if (read_buffer) {
      source.reset(new cudf::source_info(reinterpret_cast<char *>(buffer), buffer_length));
    } else {
      source.reset(new cudf::source_info(filename.get()));
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    cudf::json_read_arg read_arg{*source};
    read_arg.lines = true;
    read_arg.byte_range_offset = start_range;
    read_arg.byte_range_size = range_length;
    if (data_types != 0) {
      cudf::jni::native_jstringArray n_data_types(env, data_types);
      if (col_names != nullptr) {
        cudf::jni::native_jstringArray n_col_names(env, col_names);
        JNI_ARG_CHECK(env, n_col_names.size() == n_data_types.size(),
                      "col_types and col_names should have the same size", NULL);
        std::vector<std::string> data_types_vector(n_data_types.as_cpp_vector());
        std::vector<std::string> col_names_vector(n_col_names.as_cpp_vector());
        std::transform(col_names_vector.begin(), col_names_vector.end(), data_types_vector.begin(),
                       data_types_vector.begin(),
                       [](std::string s1, std::string s2) -> std::string { return s1 + ":" + s2; });
        read_arg.dtype = data_types_vector;
      } else {
        read_arg.dtype = n_data_types.as_cpp_vector();
      }
    }
    cudf::table result = read_json(read_arg);
    try {

      if (n_filter_col_names.size() == 0) {
        cudf::jni::native_jlongArray native_handles(env, reinterpret_cast<jlong *>(result.begin()),
                                                    result.num_columns());
        return native_handles.get_jArray();
      }

      std::vector<std::string> col_vector = n_filter_col_names.as_cpp_vector();
      std::unordered_set<std::string> filter_col_lookup(col_vector.begin(), col_vector.end());
      std::map<std::string, gdf_column *> names_to_cols;
      // traverse the table and if we don't need that column free it
      // create a map from name -> col
      for (auto iter = result.begin(); iter != result.end(); iter++) {
        std::unordered_set<std::string>::const_iterator got =
            filter_col_lookup.find((*iter)->col_name);
        if (got == filter_col_lookup.end()) {
          gdf_column_free(*iter);
        } else {
          names_to_cols[*got] = *iter;
        }
      }
      // this is the list of columns that we will finally return to the client
      std::vector<gdf_column *> final_columns_to_return;
      for (auto iter = col_vector.begin(); iter != col_vector.end(); iter++) {
        final_columns_to_return.push_back(names_to_cols[*iter]);
      }
      cudf::table final_result(final_columns_to_return);
      cudf::jni::native_jlongArray native_handles(
          env, reinterpret_cast<jlong *>(final_result.begin()), final_result.num_columns());

      return native_handles.get_jArray();
    } catch (...) { result.destroy(); }
  }
  CATCH_STD(env, NULL);
}
} // extern "C"
