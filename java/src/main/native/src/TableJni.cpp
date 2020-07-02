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

#include <cudf/aggregation.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/groupby.hpp>
#include <cudf/hashing.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/functions.hpp>
#include <cudf/join.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/reshape.hpp>
#include <cudf/rolling.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>

#include "jni_utils.hpp"

namespace cudf {
namespace jni {

constexpr long MINIMUM_WRITE_BUFFER_SIZE = 10 * 1024 * 1024; // 10 MB

class jni_writer_data_sink final : public cudf::io::data_sink {
public:
  explicit jni_writer_data_sink(JNIEnv *env, jobject callback) {
    if (env->GetJavaVM(&jvm) < 0) {
      throw std::runtime_error("GetJavaVM failed");
    }

    jclass cls = env->GetObjectClass(callback);
    if (cls == nullptr) {
      throw cudf::jni::jni_exception("class not found");
    }

    handle_buffer_method =
        env->GetMethodID(cls, "handleBuffer", "(Lai/rapids/cudf/HostMemoryBuffer;J)V");
    if (handle_buffer_method == nullptr) {
      throw cudf::jni::jni_exception("handleBuffer method");
    }

    this->callback = env->NewGlobalRef(callback);
    if (this->callback == nullptr) {
      throw cudf::jni::jni_exception("global ref");
    }
  }

  virtual ~jni_writer_data_sink() {
    // This should normally be called by a JVM thread. If the JVM environment is missing then this
    // is likely being triggered by the C++ runtime during shutdown. In that case the JVM may
    // already be destroyed and this thread should not try to attach to get an environment.
    JNIEnv *env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void **>(&env), cudf::jni::MINIMUM_JNI_VERSION) == JNI_OK) {
      env->DeleteGlobalRef(callback);
      if (current_buffer != nullptr) {
        env->DeleteGlobalRef(current_buffer);
      }
    }
    callback = nullptr;
    current_buffer = nullptr;
  }

  void host_write(void const *data, size_t size) override {
    JNIEnv *env = cudf::jni::get_jni_env(jvm);
    size_t left_to_copy = size;
    const char *copy_from = static_cast<const char *>(data);
    while (left_to_copy > 0) {
      long buffer_amount_available = current_buffer_len - current_buffer_written;
      if (buffer_amount_available <= 0) {
        // should never be < 0, but just to be safe
        rotate_buffer(env);
        buffer_amount_available = current_buffer_len - current_buffer_written;
      }
      long amount_to_copy =
          left_to_copy < buffer_amount_available ? left_to_copy : buffer_amount_available;
      char *copy_to = current_buffer_data + current_buffer_written;

      std::memcpy(copy_to, copy_from, amount_to_copy);
      copy_from = copy_from + amount_to_copy;
      current_buffer_written += amount_to_copy;
      total_written += amount_to_copy;
      left_to_copy -= amount_to_copy;
    }
  }

  bool supports_device_write() const override { return true; }

  void device_write(void const *gpu_data, size_t size, cudaStream_t stream) {
    JNIEnv *env = cudf::jni::get_jni_env(jvm);
    size_t left_to_copy = size;
    const char *copy_from = static_cast<const char *>(gpu_data);
    while (left_to_copy > 0) {
      long buffer_amount_available = current_buffer_len - current_buffer_written;
      if (buffer_amount_available <= 0) {
        // should never be < 0, but just to be safe
        CUDA_TRY(cudaStreamSynchronize(stream));
        rotate_buffer(env);
        buffer_amount_available = current_buffer_len - current_buffer_written;
      }
      long amount_to_copy =
          left_to_copy < buffer_amount_available ? left_to_copy : buffer_amount_available;
      char *copy_to = current_buffer_data + current_buffer_written;

      CUDA_TRY(cudaMemcpyAsync(copy_to, copy_from, amount_to_copy, cudaMemcpyDeviceToHost, stream));

      copy_from = copy_from + amount_to_copy;
      current_buffer_written += amount_to_copy;
      total_written += amount_to_copy;
      left_to_copy -= amount_to_copy;
    }
    CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void flush() override {
    if (current_buffer_written > 0) {
      JNIEnv *env = cudf::jni::get_jni_env(jvm);
      handle_buffer(env, current_buffer, current_buffer_written);
      current_buffer = nullptr;
      current_buffer_len = 0;
      current_buffer_data = nullptr;
      current_buffer_written = 0;
    }
  }

  size_t bytes_written() override { return total_written; }

  void set_alloc_size(long size) { this->alloc_size = size; }

private:
  void rotate_buffer(JNIEnv *env) {
    if (current_buffer != nullptr) {
      handle_buffer(env, current_buffer, current_buffer_written);
      env->DeleteGlobalRef(current_buffer);
      current_buffer = nullptr;
    }
    jobject tmp_buffer = allocate_host_buffer(env, alloc_size, true);
    current_buffer = env->NewGlobalRef(tmp_buffer);
    current_buffer_len = get_host_buffer_length(env, current_buffer);
    current_buffer_data = reinterpret_cast<char *>(get_host_buffer_address(env, current_buffer));
    current_buffer_written = 0;
  }

  void handle_buffer(JNIEnv *env, jobject buffer, jlong len) {
    env->CallVoidMethod(callback, handle_buffer_method, buffer, len);
    if (env->ExceptionCheck()) {
      throw std::runtime_error("handleBuffer threw an exception");
    }
  }

  JavaVM *jvm;
  jobject callback;
  jmethodID handle_buffer_method;
  jobject current_buffer = nullptr;
  char *current_buffer_data = nullptr;
  long current_buffer_len = 0;
  long current_buffer_written = 0;
  size_t total_written = 0;
  long alloc_size = MINIMUM_WRITE_BUFFER_SIZE;
};

template <typename STATE> class jni_table_writer_handle final {
public:
  explicit jni_table_writer_handle(std::shared_ptr<STATE> &state) : state(state), sink() {}
  jni_table_writer_handle(std::shared_ptr<STATE> &state,
                          std::unique_ptr<jni_writer_data_sink> &sink)
      : state(state), sink(std::move(sink)) {}

  std::shared_ptr<STATE> state;
  std::unique_ptr<jni_writer_data_sink> sink;
};

typedef jni_table_writer_handle<cudf::io::detail::parquet::pq_chunked_state>
    native_parquet_writer_handle;
typedef jni_table_writer_handle<cudf::io::detail::orc::orc_chunked_state> native_orc_writer_handle;

/**
 * Take a table returned by some operation and turn it into an array of column* so we can track them
 * ourselves in java instead of having their life tied to the table.
 * @param table_result the table to convert for return
 * @param extra_columns columns not in the table that will be added to the result at the end.
 */
static jlongArray
convert_table_for_return(JNIEnv *env, std::unique_ptr<cudf::table> &table_result,
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

jlongArray convert_table_for_return(JNIEnv *env, std::unique_ptr<cudf::table> &table_result) {
  std::vector<std::unique_ptr<cudf::column>> extra;
  return convert_table_for_return(env, table_result, extra);
}

std::unique_ptr<cudf::aggregation> map_jni_aggregation(jint op) {
  // These numbers come from AggregateOp.java and must stay in sync
  switch (op) {
    case 0: // SUM
      return cudf::make_sum_aggregation();
    case 1: // MIN
      return cudf::make_min_aggregation();
    case 2: // MAX
      return cudf::make_max_aggregation();
    case 3: // COUNT_VALID
      return cudf::make_count_aggregation(null_policy::EXCLUDE);
    case 4: // COUNT_ALL
      return cudf::make_count_aggregation(null_policy::INCLUDE);
    case 5: // MEAN
      return cudf::make_mean_aggregation();
    case 6: // MEDIAN
      return cudf::make_median_aggregation();
    case 8: // ARGMAX
      return cudf::make_argmax_aggregation();
    case 9: // ARGMIN
      return cudf::make_argmin_aggregation();
    case 10: // PRODUCT
      return cudf::make_product_aggregation();
    case 11: // SUMOFSQUARES
      return cudf::make_sum_of_squares_aggregation();
    case 12: // VAR
      return cudf::make_variance_aggregation();
    case 13: // STD
      return cudf::make_std_aggregation();
    case 14: // ANY
      return cudf::make_any_aggregation();
    case 15: // ALL
      return cudf::make_all_aggregation();
    case 16: // FIRST_INCLUDE_NULLS
      return cudf::make_nth_element_aggregation(0, null_policy::INCLUDE);
    case 17: // FIRST_EXCLUDE_NULLS
      return cudf::make_nth_element_aggregation(0, null_policy::EXCLUDE);
    case 18: // LAST_INCLUDE_NULLS
      return cudf::make_nth_element_aggregation(-1, null_policy::INCLUDE);
    case 19: // LAST_EXCLUDE_NULLS
      return cudf::make_nth_element_aggregation(-1, null_policy::EXCLUDE);
    case 20: // ROW_NUMBER
      return cudf::make_row_number_aggregation();
    default: throw std::logic_error("Unsupported Aggregation Operation");
  }
}

namespace {
// Check that window parameters are valid.
bool valid_window_parameters(native_jintArray const &values, native_jintArray const &ops,
                             native_jintArray const &min_periods, native_jintArray const &preceding,
                             native_jintArray const &following) {
  return values.size() == ops.size() && values.size() == min_periods.size() &&
         values.size() == preceding.size() && values.size() == following.size();
}

// Check that time-range window parameters are valid.
bool valid_window_parameters(native_jintArray const &values, native_jintArray const &timestamps,
                             native_jintArray const &ops, native_jintArray const &min_periods,
                             native_jintArray const &preceding, native_jintArray const &following) {
  return values.size() == timestamps.size() &&
         valid_window_parameters(values, ops, min_periods, preceding, following);
}
} // namespace

} // namespace jni
} // namespace cudf

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_createCudfTableView(JNIEnv *env,
                                                                      jclass class_object,
                                                                      jlongArray j_cudf_columns) {
  JNI_NULL_CHECK(env, j_cudf_columns, "columns are null", 0);

  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_columns(env, j_cudf_columns);

    std::vector<cudf::column_view> column_views(n_cudf_columns.size());
    for (int i = 0; i < n_cudf_columns.size(); i++) {
      column_views[i] = *n_cudf_columns[i];
    }
    cudf::table_view *tv = new cudf::table_view(column_views);
    return reinterpret_cast<jlong>(tv);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_deleteCudfTable(JNIEnv *env, jclass class_object,
                                                                 jlong j_cudf_table_view) {
  JNI_NULL_CHECK(env, j_cudf_table_view, "table view handle is null", );
  try {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<cudf::table_view *>(j_cudf_table_view);
  }
  CATCH_STD(env, );
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_orderBy(JNIEnv *env, jclass j_class_object,
                                                               jlong j_input_table,
                                                               jlongArray j_sort_keys_columns,
                                                               jbooleanArray j_is_descending,
                                                               jbooleanArray j_are_nulls_smallest) {

  // input validations & verifications
  JNI_NULL_CHECK(env, j_input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_sort_keys_columns, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_is_descending, "sort order array is null", NULL);
  JNI_NULL_CHECK(env, j_are_nulls_smallest, "null order array is null", NULL);

  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<cudf::column_view> n_sort_keys_columns(env,
                                                                           j_sort_keys_columns);
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

    auto sorted_col = cudf::sorted_order(keys, order, null_order);

    cudf::table_view *input_table = reinterpret_cast<cudf::table_view *>(j_input_table);
    std::unique_ptr<cudf::table> result = cudf::gather(*input_table, sorted_col->view());
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
    cudf::jni::auto_set_device(env);
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

    std::unique_ptr<cudf::io::source_info> source;
    if (read_buffer) {
      source.reset(new cudf::io::source_info(reinterpret_cast<char *>(buffer), buffer_length));
    } else {
      source.reset(new cudf::io::source_info(filename.get()));
    }

    cudf::io::read_csv_args read_arg{*source};
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
    read_arg.compression = cudf::io::compression_type::AUTO;
    read_arg.decimal = '.';
    read_arg.quotechar = quote;
    read_arg.quoting = cudf::io::quote_style::MINIMAL;
    read_arg.doublequote = true;
    read_arg.comment = comment;

    cudf::io::table_with_metadata result = cudf::io::read_csv(read_arg);
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
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inputfilepath can't be empty",
                    NULL);
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    std::unique_ptr<cudf::io::source_info> source;
    if (read_buffer) {
      source.reset(new cudf::io::source_info(reinterpret_cast<char *>(buffer), buffer_length));
    } else {
      source.reset(new cudf::io::source_info(filename.get()));
    }

    cudf::io::read_parquet_args read_arg(*source);

    read_arg.columns = n_filter_col_names.as_cpp_vector();

    read_arg.row_group = -1;
    read_arg.skip_rows = -1;
    read_arg.num_rows = -1;
    read_arg.strings_to_categorical = false;
    read_arg.timestamp_type = cudf::data_type(static_cast<cudf::type_id>(unit));

    cudf::io::table_with_metadata result = cudf::io::read_parquet(read_arg);
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT long JNICALL Java_ai_rapids_cudf_Table_writeParquetBufferBegin(
    JNIEnv *env, jclass, jobjectArray j_col_names, jbooleanArray j_col_nullability,
    jobjectArray j_metadata_keys, jobjectArray j_metadata_values, jint j_compression,
    jint j_stats_freq, jobject consumer) {
  JNI_NULL_CHECK(env, j_col_names, "null columns", 0);
  JNI_NULL_CHECK(env, j_col_nullability, "null nullability", 0);
  JNI_NULL_CHECK(env, j_metadata_keys, "null metadata keys", 0);
  JNI_NULL_CHECK(env, j_metadata_values, "null metadata values", 0);
  JNI_NULL_CHECK(env, consumer, "null consumer", 0);
  try {
    cudf::jni::auto_set_device(env);
    using namespace cudf::io;
    cudf::jni::native_jstringArray col_names(env, j_col_names);
    cudf::jni::native_jbooleanArray col_nullability(env, j_col_nullability);
    cudf::jni::native_jstringArray meta_keys(env, j_metadata_keys);
    cudf::jni::native_jstringArray meta_values(env, j_metadata_values);

    auto d = col_nullability.data();
    std::vector<bool> nullability(d, d + col_nullability.size());
    table_metadata_with_nullability metadata;
    metadata.column_nullable = nullability;
    metadata.column_names = col_names.as_cpp_vector();
    for (size_t i = 0; i < meta_keys.size(); ++i) {
      metadata.user_data[meta_keys[i].get()] = meta_values[i].get();
    }

    std::unique_ptr<cudf::jni::jni_writer_data_sink> data_sink(
        new cudf::jni::jni_writer_data_sink(env, consumer));
    sink_info sink{data_sink.get()};
    compression_type compression{static_cast<compression_type>(j_compression)};
    statistics_freq stats{static_cast<statistics_freq>(j_stats_freq)};

    write_parquet_chunked_args args(sink, &metadata, compression, stats);
    std::shared_ptr<detail::parquet::pq_chunked_state> state = write_parquet_chunked_begin(args);
    cudf::jni::native_parquet_writer_handle *ret =
        new cudf::jni::native_parquet_writer_handle(state, data_sink);
    return reinterpret_cast<jlong>(ret);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT long JNICALL Java_ai_rapids_cudf_Table_writeParquetFileBegin(
    JNIEnv *env, jclass, jobjectArray j_col_names, jbooleanArray j_col_nullability,
    jobjectArray j_metadata_keys, jobjectArray j_metadata_values, jint j_compression,
    jint j_stats_freq, jstring j_output_path) {
  JNI_NULL_CHECK(env, j_col_names, "null columns", 0);
  JNI_NULL_CHECK(env, j_col_nullability, "null nullability", 0);
  JNI_NULL_CHECK(env, j_metadata_keys, "null metadata keys", 0);
  JNI_NULL_CHECK(env, j_metadata_values, "null metadata values", 0);
  JNI_NULL_CHECK(env, j_output_path, "null output path", 0);
  try {
    cudf::jni::auto_set_device(env);
    using namespace cudf::io;
    cudf::jni::native_jstringArray col_names(env, j_col_names);
    cudf::jni::native_jbooleanArray col_nullability(env, j_col_nullability);
    cudf::jni::native_jstringArray meta_keys(env, j_metadata_keys);
    cudf::jni::native_jstringArray meta_values(env, j_metadata_values);
    cudf::jni::native_jstring output_path(env, j_output_path);

    auto d = col_nullability.data();
    std::vector<bool> nullability(d, d + col_nullability.size());
    table_metadata_with_nullability metadata;
    metadata.column_nullable = nullability;
    metadata.column_names = col_names.as_cpp_vector();
    for (size_t i = 0; i < meta_keys.size(); ++i) {
      metadata.user_data[meta_keys[i].get()] = meta_values[i].get();
    }

    sink_info sink{output_path.get()};
    compression_type compression{static_cast<compression_type>(j_compression)};
    statistics_freq stats{static_cast<statistics_freq>(j_stats_freq)};

    write_parquet_chunked_args args(sink, &metadata, compression, stats);
    std::shared_ptr<detail::parquet::pq_chunked_state> state = write_parquet_chunked_begin(args);
    cudf::jni::native_parquet_writer_handle *ret =
        new cudf::jni::native_parquet_writer_handle(state);
    return reinterpret_cast<jlong>(ret);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeParquetChunk(JNIEnv *env, jclass,
                                                                   jlong j_state, jlong j_table,
                                                                   jlong mem_size) {
  JNI_NULL_CHECK(env, j_table, "null table", );
  JNI_NULL_CHECK(env, j_state, "null state", );

  using namespace cudf::io;
  cudf::table_view *tview = reinterpret_cast<cudf::table_view *>(j_table);
  cudf::jni::native_parquet_writer_handle *state =
      reinterpret_cast<cudf::jni::native_parquet_writer_handle *>(j_state);

  if (state->sink) {
    long alloc_size = std::max(cudf::jni::MINIMUM_WRITE_BUFFER_SIZE, mem_size / 2);
    state->sink->set_alloc_size(alloc_size);
  }
  try {
    cudf::jni::auto_set_device(env);
    write_parquet_chunked(*tview, state->state);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeParquetEnd(JNIEnv *env, jclass,
                                                                 jlong j_state) {
  JNI_NULL_CHECK(env, j_state, "null state", );

  using namespace cudf::io;
  cudf::jni::native_parquet_writer_handle *state =
      reinterpret_cast<cudf::jni::native_parquet_writer_handle *>(j_state);
  std::unique_ptr<cudf::jni::native_parquet_writer_handle> make_sure_we_delete(state);
  try {
    cudf::jni::auto_set_device(env);
    write_parquet_chunked_end(state->state);
  }
  CATCH_STD(env, )
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
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inputfilepath can't be empty",
                    NULL);
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    std::unique_ptr<cudf::io::source_info> source;
    if (read_buffer) {
      source.reset(new cudf::io::source_info(reinterpret_cast<char *>(buffer), buffer_length));
    } else {
      source.reset(new cudf::io::source_info(filename.get()));
    }

    cudf::io::read_orc_args read_arg{*source};
    read_arg.columns = n_filter_col_names.as_cpp_vector();
    read_arg.stripe = -1;
    read_arg.skip_rows = -1;
    read_arg.num_rows = -1;
    read_arg.use_index = false;
    read_arg.use_np_dtypes = static_cast<bool>(usingNumPyTypes);
    read_arg.timestamp_type = cudf::data_type(static_cast<cudf::type_id>(unit));

    cudf::io::table_with_metadata result = cudf::io::read_orc(read_arg);
    return cudf::jni::convert_table_for_return(env, result.tbl);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT long JNICALL Java_ai_rapids_cudf_Table_writeORCBufferBegin(
    JNIEnv *env, jclass, jobjectArray j_col_names, jbooleanArray j_col_nullability,
    jobjectArray j_metadata_keys, jobjectArray j_metadata_values, jint j_compression,
    jobject consumer) {
  JNI_NULL_CHECK(env, j_col_names, "null columns", 0);
  JNI_NULL_CHECK(env, j_col_nullability, "null nullability", 0);
  JNI_NULL_CHECK(env, j_metadata_keys, "null metadata keys", 0);
  JNI_NULL_CHECK(env, j_metadata_values, "null metadata values", 0);
  JNI_NULL_CHECK(env, consumer, "null consumer", 0);
  try {
    cudf::jni::auto_set_device(env);
    using namespace cudf::io;
    cudf::jni::native_jstringArray col_names(env, j_col_names);
    cudf::jni::native_jbooleanArray col_nullability(env, j_col_nullability);
    cudf::jni::native_jstringArray meta_keys(env, j_metadata_keys);
    cudf::jni::native_jstringArray meta_values(env, j_metadata_values);

    auto d = col_nullability.data();
    std::vector<bool> nullability(d, d + col_nullability.size());
    table_metadata_with_nullability metadata;
    metadata.column_nullable = nullability;
    metadata.column_names = col_names.as_cpp_vector();
    for (size_t i = 0; i < meta_keys.size(); ++i) {
      metadata.user_data[meta_keys[i].get()] = meta_values[i].get();
    }

    std::unique_ptr<cudf::jni::jni_writer_data_sink> data_sink(
        new cudf::jni::jni_writer_data_sink(env, consumer));
    sink_info sink{data_sink.get()};
    compression_type compression{static_cast<compression_type>(j_compression)};

    write_orc_chunked_args args(sink, &metadata, compression, true);
    std::shared_ptr<detail::orc::orc_chunked_state> state = write_orc_chunked_begin(args);
    cudf::jni::native_orc_writer_handle *ret =
        new cudf::jni::native_orc_writer_handle(state, data_sink);
    return reinterpret_cast<jlong>(ret);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT long JNICALL Java_ai_rapids_cudf_Table_writeORCFileBegin(
    JNIEnv *env, jclass, jobjectArray j_col_names, jbooleanArray j_col_nullability,
    jobjectArray j_metadata_keys, jobjectArray j_metadata_values, jint j_compression,
    jstring j_output_path) {
  JNI_NULL_CHECK(env, j_col_names, "null columns", 0);
  JNI_NULL_CHECK(env, j_col_nullability, "null nullability", 0);
  JNI_NULL_CHECK(env, j_metadata_keys, "null metadata keys", 0);
  JNI_NULL_CHECK(env, j_metadata_values, "null metadata values", 0);
  JNI_NULL_CHECK(env, j_output_path, "null output path", 0);
  try {
    cudf::jni::auto_set_device(env);
    using namespace cudf::io;
    cudf::jni::native_jstringArray col_names(env, j_col_names);
    cudf::jni::native_jbooleanArray col_nullability(env, j_col_nullability);
    cudf::jni::native_jstringArray meta_keys(env, j_metadata_keys);
    cudf::jni::native_jstringArray meta_values(env, j_metadata_values);
    cudf::jni::native_jstring output_path(env, j_output_path);

    auto d = col_nullability.data();
    std::vector<bool> nullability(d, d + col_nullability.size());
    table_metadata_with_nullability metadata;
    metadata.column_nullable = nullability;
    metadata.column_names = col_names.as_cpp_vector();
    for (size_t i = 0; i < meta_keys.size(); ++i) {
      metadata.user_data[meta_keys[i].get()] = meta_values[i].get();
    }

    sink_info sink{output_path.get()};
    compression_type compression{static_cast<compression_type>(j_compression)};

    write_orc_chunked_args args(sink, &metadata, compression, true);
    std::shared_ptr<detail::orc::orc_chunked_state> state = write_orc_chunked_begin(args);
    cudf::jni::native_orc_writer_handle *ret = new cudf::jni::native_orc_writer_handle(state);
    return reinterpret_cast<jlong>(ret);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeORCChunk(JNIEnv *env, jclass, jlong j_state,
                                                               jlong j_table, jlong mem_size) {
  JNI_NULL_CHECK(env, j_table, "null table", );
  JNI_NULL_CHECK(env, j_state, "null state", );

  using namespace cudf::io;
  cudf::table_view *tview = reinterpret_cast<cudf::table_view *>(j_table);
  cudf::jni::native_orc_writer_handle *state =
      reinterpret_cast<cudf::jni::native_orc_writer_handle *>(j_state);

  if (state->sink) {
    long alloc_size = std::max(cudf::jni::MINIMUM_WRITE_BUFFER_SIZE, mem_size / 2);
    state->sink->set_alloc_size(alloc_size);
  }
  try {
    cudf::jni::auto_set_device(env);
    write_orc_chunked(*tview, state->state);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeORCEnd(JNIEnv *env, jclass, jlong j_state) {
  JNI_NULL_CHECK(env, j_state, "null state", );

  using namespace cudf::io;
  cudf::jni::native_orc_writer_handle *state =
      reinterpret_cast<cudf::jni::native_orc_writer_handle *>(j_state);
  std::unique_ptr<cudf::jni::native_orc_writer_handle> make_sure_we_delete(state);
  try {
    cudf::jni::auto_set_device(env);
    write_orc_chunked_end(state->state);
  }
  CATCH_STD(env, )
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftJoin(JNIEnv *env, jclass clazz,
                                                                jlong left_table,
                                                                jintArray left_col_join_indices,
                                                                jlong right_table,
                                                                jintArray right_col_join_indices) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, left_col_join_indices, "left_col_join_indices is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, right_col_join_indices, "right_col_join_indices is null", NULL);

  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *n_left_table = reinterpret_cast<cudf::table_view *>(left_table);
    cudf::table_view *n_right_table = reinterpret_cast<cudf::table_view *>(right_table);
    cudf::jni::native_jintArray left_join_cols_arr(env, left_col_join_indices);
    std::vector<cudf::size_type> left_join_cols(
        left_join_cols_arr.data(), left_join_cols_arr.data() + left_join_cols_arr.size());
    cudf::jni::native_jintArray right_join_cols_arr(env, right_col_join_indices);
    std::vector<cudf::size_type> right_join_cols(
        right_join_cols_arr.data(), right_join_cols_arr.data() + right_join_cols_arr.size());

    int dedupe_size = left_join_cols.size();
    std::vector<std::pair<cudf::size_type, cudf::size_type>> dedupe(dedupe_size);
    for (int i = 0; i < dedupe_size; i++) {
      dedupe[i].first = left_join_cols[i];
      dedupe[i].second = right_join_cols[i];
    }

    std::unique_ptr<cudf::table> result =
        cudf::left_join(*n_left_table, *n_right_table, left_join_cols, right_join_cols, dedupe);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_innerJoin(JNIEnv *env, jclass clazz,
                                                                 jlong left_table,
                                                                 jintArray left_col_join_indices,
                                                                 jlong right_table,
                                                                 jintArray right_col_join_indices) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, left_col_join_indices, "left_col_join_indices is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, right_col_join_indices, "right_col_join_indices is null", NULL);

  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *n_left_table = reinterpret_cast<cudf::table_view *>(left_table);
    cudf::table_view *n_right_table = reinterpret_cast<cudf::table_view *>(right_table);
    cudf::jni::native_jintArray left_join_cols_arr(env, left_col_join_indices);
    std::vector<cudf::size_type> left_join_cols(
        left_join_cols_arr.data(), left_join_cols_arr.data() + left_join_cols_arr.size());
    cudf::jni::native_jintArray right_join_cols_arr(env, right_col_join_indices);
    std::vector<cudf::size_type> right_join_cols(
        right_join_cols_arr.data(), right_join_cols_arr.data() + right_join_cols_arr.size());

    int dedupe_size = left_join_cols.size();
    std::vector<std::pair<cudf::size_type, cudf::size_type>> dedupe(dedupe_size);
    for (int i = 0; i < dedupe_size; i++) {
      dedupe[i].first = left_join_cols[i];
      dedupe[i].second = right_join_cols[i];
    }

    std::unique_ptr<cudf::table> result =
        cudf::inner_join(*n_left_table, *n_right_table, left_join_cols, right_join_cols, dedupe);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_fullJoin(JNIEnv *env, jclass clazz,
                                                                jlong left_table,
                                                                jintArray left_col_join_indices,
                                                                jlong right_table,
                                                                jintArray right_col_join_indices) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, left_col_join_indices, "left_col_join_indices is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, right_col_join_indices, "right_col_join_indices is null", NULL);

  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *n_left_table = reinterpret_cast<cudf::table_view *>(left_table);
    cudf::table_view *n_right_table = reinterpret_cast<cudf::table_view *>(right_table);
    cudf::jni::native_jintArray left_join_cols_arr(env, left_col_join_indices);
    std::vector<cudf::size_type> left_join_cols(
        left_join_cols_arr.data(), left_join_cols_arr.data() + left_join_cols_arr.size());
    cudf::jni::native_jintArray right_join_cols_arr(env, right_col_join_indices);
    std::vector<cudf::size_type> right_join_cols(
        right_join_cols_arr.data(), right_join_cols_arr.data() + right_join_cols_arr.size());

    int dedupe_size = left_join_cols.size();
    std::vector<std::pair<cudf::size_type, cudf::size_type>> dedupe(dedupe_size);
    for (int i = 0; i < dedupe_size; i++) {
      dedupe[i].first = left_join_cols[i];
      dedupe[i].second = right_join_cols[i];
    }

    std::unique_ptr<cudf::table> result =
        cudf::full_join(*n_left_table, *n_right_table, left_join_cols, right_join_cols, dedupe);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftSemiJoin(
    JNIEnv *env, jclass, jlong left_table, jintArray left_col_join_indices, jlong right_table,
    jintArray right_col_join_indices) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, left_col_join_indices, "left_col_join_indices is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, right_col_join_indices, "right_col_join_indices is null", NULL);

  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *n_left_table = reinterpret_cast<cudf::table_view *>(left_table);
    cudf::table_view *n_right_table = reinterpret_cast<cudf::table_view *>(right_table);
    cudf::jni::native_jintArray left_join_cols_arr(env, left_col_join_indices);
    std::vector<cudf::size_type> left_join_cols(
        left_join_cols_arr.data(), left_join_cols_arr.data() + left_join_cols_arr.size());
    cudf::jni::native_jintArray right_join_cols_arr(env, right_col_join_indices);
    std::vector<cudf::size_type> right_join_cols(
        right_join_cols_arr.data(), right_join_cols_arr.data() + right_join_cols_arr.size());
    std::vector<cudf::size_type> return_cols(n_left_table->num_columns());
    for (cudf::size_type i = 0; i < n_left_table->num_columns(); ++i) {
      return_cols[i] = i;
    }

    std::unique_ptr<cudf::table> result = cudf::left_semi_join(
        *n_left_table, *n_right_table, left_join_cols, right_join_cols, return_cols);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftAntiJoin(
    JNIEnv *env, jclass, jlong left_table, jintArray left_col_join_indices, jlong right_table,
    jintArray right_col_join_indices) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, left_col_join_indices, "left_col_join_indices is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, right_col_join_indices, "right_col_join_indices is null", NULL);

  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *n_left_table = reinterpret_cast<cudf::table_view *>(left_table);
    cudf::table_view *n_right_table = reinterpret_cast<cudf::table_view *>(right_table);
    cudf::jni::native_jintArray left_join_cols_arr(env, left_col_join_indices);
    std::vector<cudf::size_type> left_join_cols(
        left_join_cols_arr.data(), left_join_cols_arr.data() + left_join_cols_arr.size());
    cudf::jni::native_jintArray right_join_cols_arr(env, right_col_join_indices);
    std::vector<cudf::size_type> right_join_cols(
        right_join_cols_arr.data(), right_join_cols_arr.data() + right_join_cols_arr.size());
    std::vector<cudf::size_type> return_cols(n_left_table->num_columns());
    for (cudf::size_type i = 0; i < n_left_table->num_columns(); ++i) {
      return_cols[i] = i;
    }

    std::unique_ptr<cudf::table> result = cudf::left_anti_join(
        *n_left_table, *n_right_table, left_join_cols, right_join_cols, return_cols);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_crossJoin(JNIEnv *env, jclass clazz,
                                                                 jlong left_table,
                                                                 jlong right_table) {
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);

  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *n_left_table = reinterpret_cast<cudf::table_view *>(left_table);
    cudf::table_view *n_right_table = reinterpret_cast<cudf::table_view *>(right_table);

    std::unique_ptr<cudf::table> result =
        cudf::cross_join(*n_left_table, *n_right_table);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_interleaveColumns(JNIEnv *env, jclass,
                                                                    jlongArray j_cudf_table_view) {

  JNI_NULL_CHECK(env, j_cudf_table_view, "table is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *table_view = reinterpret_cast<cudf::table_view *>(j_cudf_table_view);
    std::unique_ptr<cudf::column> result = cudf::interleave_columns(*table_view);
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_concatenate(JNIEnv *env, jclass clazz,
                                                                   jlongArray table_handles) {
  JNI_NULL_CHECK(env, table_handles, "input tables are null", NULL);
  try {
    cudf::jni::auto_set_device(env);
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
    std::unique_ptr<cudf::table> table_result = cudf::concatenate(to_concat);
    return cudf::jni::convert_table_for_return(env, table_result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_hashPartition(JNIEnv *env, jclass clazz,
                                                                     jlong input_table,
                                                                     jintArray columns_to_hash,
                                                                     jint number_of_partitions,
                                                                     jintArray output_offsets) {

  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, columns_to_hash, "columns_to_hash is null", NULL);
  JNI_NULL_CHECK(env, output_offsets, "output_offsets is null", NULL);
  JNI_ARG_CHECK(env, number_of_partitions > 0, "number_of_partitions is zero", NULL);

  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *n_input_table = reinterpret_cast<cudf::table_view *>(input_table);
    cudf::jni::native_jintArray n_columns_to_hash(env, columns_to_hash);
    int n_number_of_partitions = static_cast<int>(number_of_partitions);
    cudf::jni::native_jintArray n_output_offsets(env, output_offsets);

    JNI_ARG_CHECK(env, n_columns_to_hash.size() > 0, "columns_to_hash is zero", NULL);

    std::vector<cudf::size_type> columns_to_hash_vec(n_columns_to_hash.size());
    for (int i = 0; i < n_columns_to_hash.size(); i++) {
      columns_to_hash_vec[i] = n_columns_to_hash[i];
    }

    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result =
        cudf::hash_partition(*n_input_table, columns_to_hash_vec, number_of_partitions);

    for (int i = 0; i < result.second.size(); i++) {
      n_output_offsets[i] = result.second[i];
    }

    return cudf::jni::convert_table_for_return(env, result.first);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_roundRobinPartition(
    JNIEnv *env, jclass, jlong input_table, jint num_partitions, jint start_partition,
    jintArray output_offsets) {
  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, output_offsets, "output_offsets is null", NULL);
  JNI_ARG_CHECK(env, num_partitions > 0, "num_partitions <= 0", NULL);
  JNI_ARG_CHECK(env, start_partition >= 0, "start_partition is negative", NULL);

  try {
    cudf::jni::auto_set_device(env);
    auto n_input_table = reinterpret_cast<cudf::table_view *>(input_table);
    int n_num_partitions = static_cast<int>(num_partitions);
    cudf::jni::native_jintArray n_output_offsets(env, output_offsets);

    auto result = cudf::round_robin_partition(*n_input_table, num_partitions, start_partition);

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
    cudf::jni::auto_set_device(env);
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
    cudf::groupby::groupby grouper(n_keys_table, ignore_null_keys ? cudf::null_policy::EXCLUDE :
                                                                    cudf::null_policy::INCLUDE);

    // Aggregates are passed in already grouped by column, so we just need to fill it in
    // as we go.
    std::vector<cudf::groupby::aggregation_request> requests;

    int previous_index = -1;
    for (int i = 0; i < n_values.size(); i++) {
      cudf::groupby::aggregation_request req;
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

    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::groupby::aggregation_result>> result =
        grouper.aggregate(requests);

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
                                                              jlong input_jtable, jlong mask_jcol) {
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_NULL_CHECK(env, mask_jcol, "mask column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *input = reinterpret_cast<cudf::table_view *>(input_jtable);
    cudf::column_view *mask = reinterpret_cast<cudf::column_view *>(mask_jcol);
    std::unique_ptr<cudf::table> result = cudf::apply_boolean_mask(*input, *mask);
    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_bound(JNIEnv *env, jclass, jlong input_jtable,
                                                        jlong values_jtable,
                                                        jbooleanArray desc_flags,
                                                        jbooleanArray are_nulls_smallest,
                                                        jboolean is_upper_bound) {
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_NULL_CHECK(env, values_jtable, "values table is null", 0);
  using cudf::column;
  using cudf::table_view;
  try {
    cudf::jni::auto_set_device(env);
    table_view *input = reinterpret_cast<table_view *>(input_jtable);
    table_view *values = reinterpret_cast<table_view *>(values_jtable);
    cudf::jni::native_jbooleanArray const n_desc_flags(env, desc_flags);
    cudf::jni::native_jbooleanArray const n_are_nulls_smallest(env, are_nulls_smallest);

    std::vector<cudf::order> column_desc_flags(n_desc_flags.size());
    std::vector<cudf::null_order> column_null_orders(n_are_nulls_smallest.size());

    JNI_ARG_CHECK(env, (column_desc_flags.size() == column_null_orders.size()),
                  "null-order and sort-order size mismatch", 0);
    uint32_t num_columns = column_null_orders.size();
    for (int i = 0; i < num_columns; i++) {
      column_desc_flags[i] = n_desc_flags[i] ? cudf::order::DESCENDING : cudf::order::ASCENDING;
      column_null_orders[i] =
          n_are_nulls_smallest[i] ? cudf::null_order::BEFORE : cudf::null_order::AFTER;
    }

    std::unique_ptr<column> result;
    if (is_upper_bound) {
      result = std::move(cudf::upper_bound(*input, *values, column_desc_flags, column_null_orders));
    } else {
      result = std::move(cudf::lower_bound(*input, *values, column_desc_flags, column_null_orders));
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
    cudf::jni::auto_set_device(env);
    cudf::table_view *n_table = reinterpret_cast<cudf::table_view *>(input_table);
    cudf::jni::native_jintArray n_split_indices(env, split_indices);

    std::vector<cudf::size_type> indices(n_split_indices.data(),
                                         n_split_indices.data() + n_split_indices.size());

    std::vector<cudf::contiguous_split_result> result = cudf::contiguous_split(*n_table, indices);
    cudf::jni::native_jobjectArray<jobject> n_result =
        cudf::jni::contiguous_table_array(env, result.size());
    for (int i = 0; i < result.size(); i++) {
      n_result.set(i, cudf::jni::contiguous_table_from(env, result[i]));
    }
    return n_result.wrapped();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_rollingWindowAggregate(
    JNIEnv *env, jclass clazz, jlong j_input_table, jintArray j_keys,
    jintArray j_aggregate_column_indices, jintArray j_agg_types, jintArray j_min_periods,
    jintArray j_preceding, jintArray j_following, jboolean ignore_null_keys) {

  JNI_NULL_CHECK(env, j_input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_keys, "input keys are null", NULL);
  JNI_NULL_CHECK(env, j_aggregate_column_indices, "input aggregate_column_indices are null", NULL);
  JNI_NULL_CHECK(env, j_agg_types, "agg_types are null", NULL);

  try {
    cudf::jni::auto_set_device(env);

    using cudf::jni::valid_window_parameters;

    // Convert from j-types to native.
    cudf::table_view *input_table{reinterpret_cast<cudf::table_view *>(j_input_table)};
    cudf::jni::native_jintArray keys{env, j_keys};
    cudf::jni::native_jintArray values{env, j_aggregate_column_indices};
    cudf::jni::native_jintArray ops{env, j_agg_types};
    cudf::jni::native_jintArray min_periods{env, j_min_periods};
    cudf::jni::native_jintArray preceding{env, j_preceding};
    cudf::jni::native_jintArray following{env, j_following};

    if (not valid_window_parameters(values, ops, min_periods, preceding, following)) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                    "Number of aggregation columns must match number of agg ops, and window-specs",
                    nullptr);
    }

    // Extract table-view.
    cudf::table_view groupby_keys{
        input_table->select(std::vector<cudf::size_type>(keys.data(), keys.data() + keys.size()))};

    std::vector<std::unique_ptr<cudf::column>> result_columns;
    for (int i(0); i < values.size(); ++i) {
      int agg_column_index = values[i];
      result_columns.emplace_back(std::move(cudf::grouped_rolling_window(
          groupby_keys, input_table->column(agg_column_index), preceding[i], following[i],
          min_periods[i], cudf::jni::map_jni_aggregation(ops[i]))));
    }

    auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
    return cudf::jni::convert_table_for_return(env, result_table);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_timeRangeRollingWindowAggregate(
    JNIEnv *env, jclass clazz, jlong j_input_table, jintArray j_keys,
    jintArray j_timestamp_column_indices, jbooleanArray j_is_timestamp_ascending,
    jintArray j_aggregate_column_indices, jintArray j_agg_types, jintArray j_min_periods,
    jintArray j_preceding, jintArray j_following, jboolean ignore_null_keys) {

  JNI_NULL_CHECK(env, j_input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_keys, "input keys are null", NULL);
  JNI_NULL_CHECK(env, j_timestamp_column_indices, "input timestamp_column_indices are null", NULL);
  JNI_NULL_CHECK(env, j_is_timestamp_ascending, "input timestamp_ascending is null", NULL);
  JNI_NULL_CHECK(env, j_aggregate_column_indices, "input aggregate_column_indices are null", NULL);
  JNI_NULL_CHECK(env, j_agg_types, "agg_types are null", NULL);

  try {
    cudf::jni::auto_set_device(env);

    using cudf::jni::valid_window_parameters;

    // Convert from j-types to native.
    cudf::table_view *input_table{reinterpret_cast<cudf::table_view *>(j_input_table)};
    cudf::jni::native_jintArray keys{env, j_keys};
    cudf::jni::native_jintArray timestamps{env, j_timestamp_column_indices};
    cudf::jni::native_jbooleanArray timestamp_ascending{env, j_is_timestamp_ascending};
    cudf::jni::native_jintArray values{env, j_aggregate_column_indices};
    cudf::jni::native_jintArray ops{env, j_agg_types};
    cudf::jni::native_jintArray min_periods{env, j_min_periods};
    cudf::jni::native_jintArray preceding{env, j_preceding};
    cudf::jni::native_jintArray following{env, j_following};

    if (not valid_window_parameters(values, ops, min_periods, preceding, following)) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                    "Number of aggregation columns must match number of agg ops, and window-specs",
                    nullptr);
    }

    // Extract table-view.
    cudf::table_view groupby_keys{
        input_table->select(std::vector<cudf::size_type>(keys.data(), keys.data() + keys.size()))};

    std::vector<std::unique_ptr<cudf::column>> result_columns;
    for (int i(0); i < values.size(); ++i) {
      int agg_column_index = values[i];
      result_columns.emplace_back(std::move(cudf::grouped_time_range_rolling_window(
          groupby_keys, input_table->column(timestamps[i]),
          timestamp_ascending[i] ? cudf::order::ASCENDING : cudf::order::DESCENDING,
          input_table->column(agg_column_index), preceding[i], following[i], min_periods[i],
          cudf::jni::map_jni_aggregation(ops[i]))));
    }

    auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
    return cudf::jni::convert_table_for_return(env, result_table);
  }
  CATCH_STD(env, NULL);
}

} // extern "C"
