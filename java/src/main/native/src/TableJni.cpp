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

#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <cudf/aggregation.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/hashing.hpp>
#include <cudf/interop.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/join.hpp>
#include <cudf/merge.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/reshape.hpp>
#include <cudf/rolling.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>

#include "cudf_jni_apis.hpp"

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
      if (current_buffer != nullptr) {
        env->DeleteGlobalRef(current_buffer);
      }
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

typedef jni_table_writer_handle<cudf::io::pq_chunked_state>
    native_parquet_writer_handle;
typedef jni_table_writer_handle<cudf::io::orc_chunked_state> native_orc_writer_handle;

class native_arrow_ipc_writer_handle final {
public:
  explicit native_arrow_ipc_writer_handle(
          const std::vector<std::string>& col_names,
          const std::string& file_name): 
      initialized(false),
      column_names(col_names),
      file_name(file_name) {}

  explicit native_arrow_ipc_writer_handle(
          const std::vector<std::string>& col_names,
          const std::shared_ptr<arrow::io::OutputStream>& sink): 
      initialized(false),
      column_names(col_names),
      sink(sink),
      file_name("") {}

  bool initialized;
  std::vector<std::string> column_names;
  std::string file_name;
  std::shared_ptr<arrow::io::OutputStream> sink;
  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;

  void write(std::shared_ptr<arrow::Table>& arrow_tab, int64_t max_chunk) {
    if (!initialized) {
      if (!sink) {
        auto tmp_sink = arrow::io::FileOutputStream::Open(file_name);
        if (!tmp_sink.ok()) {
          throw std::runtime_error(tmp_sink.status().message());
        }
        sink = *tmp_sink;
      }

      // There is an option to have a file writer too, with metadata
      auto tmp_writer = arrow::ipc::NewStreamWriter(sink.get(), arrow_tab->schema());
      if (!tmp_writer.ok()) {
        throw std::runtime_error(tmp_writer.status().message());
      }
      writer = *tmp_writer;
      initialized = true;
    }
    writer->WriteTable(*arrow_tab, max_chunk);
  }

  void close() {
    if (initialized) {
      writer->Close();
      sink->Close();
    }
    initialized = false;
  }
};


class jni_arrow_output_stream final : public arrow::io::OutputStream {
public:
  explicit jni_arrow_output_stream(JNIEnv *env, jobject callback) {
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

  virtual ~jni_arrow_output_stream() {
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

  arrow::Status Write(const std::shared_ptr<arrow::Buffer> & data) override {
    return Write(data->data(), data->size());
  }

  arrow::Status Write(const void* data, int64_t nbytes) override {
    JNIEnv *env = cudf::jni::get_jni_env(jvm);
    int64_t left_to_copy = nbytes;
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
    return arrow::Status::OK();
  }

  arrow::Status Flush() override {
    if (current_buffer_written > 0) {
      JNIEnv *env = cudf::jni::get_jni_env(jvm);
      handle_buffer(env, current_buffer, current_buffer_written);
      if (current_buffer != nullptr) {
        env->DeleteGlobalRef(current_buffer);
      }
      current_buffer = nullptr;
      current_buffer_len = 0;
      current_buffer_data = nullptr;
      current_buffer_written = 0;
    }
    return arrow::Status::OK();
  }

  arrow::Status Close() override {
    auto ret = Flush();
    is_closed = true;
    return ret;
  }

  arrow::Status Abort() override {
    is_closed = true;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const override {
    return total_written;
  }

  bool closed() const override {
    return is_closed;
  }

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
  int64_t total_written = 0;
  long alloc_size = MINIMUM_WRITE_BUFFER_SIZE;
  bool is_closed = false;
};

class jni_arrow_input_stream final : public arrow::io::InputStream {
public:
  explicit jni_arrow_input_stream(JNIEnv *env, jobject callback) :
      mm(arrow::default_cpu_memory_manager()) {
    if (env->GetJavaVM(&jvm) < 0) {
      throw std::runtime_error("GetJavaVM failed");
    }

    jclass cls = env->GetObjectClass(callback);
    if (cls == nullptr) {
      throw cudf::jni::jni_exception("class not found");
    }

    read_into_method =
        env->GetMethodID(cls, "readInto", "(JJ)J");
    if (read_into_method == nullptr) {
      throw cudf::jni::jni_exception("readInto method");
    }

    this->callback = env->NewGlobalRef(callback);
    if (this->callback == nullptr) {
      throw cudf::jni::jni_exception("global ref");
    }
  }

  virtual ~jni_arrow_input_stream() {
    // This should normally be called by a JVM thread. If the JVM environment is missing then this
    // is likely being triggered by the C++ runtime during shutdown. In that case the JVM may
    // already be destroyed and this thread should not try to attach to get an environment.
    JNIEnv *env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void **>(&env), cudf::jni::MINIMUM_JNI_VERSION) == JNI_OK) {
      env->DeleteGlobalRef(callback);
    }
    callback = nullptr;
  }

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    JNIEnv *env = cudf::jni::get_jni_env(jvm);
    jlong ret = read_into(env, reinterpret_cast<jlong>(out), nbytes);
    total_read += ret;
    return ret;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    JNIEnv *env = cudf::jni::get_jni_env(jvm);
    arrow::Result<std::shared_ptr<arrow::ResizableBuffer>> tmp_buffer = 
        arrow::AllocateResizableBuffer(nbytes);
    if (!tmp_buffer.ok()) {
      return tmp_buffer;
    }
    jlong amount_read = read_into(env, reinterpret_cast<jlong>((*tmp_buffer)->data()), nbytes);
    arrow::Status stat = (*tmp_buffer)->Resize(amount_read);
    if (!stat.ok()) {
      return stat;
    }
    return tmp_buffer;
  }
  
  arrow::Status Close() override {
    is_closed = true;
    return arrow::Status::OK();
  }

  arrow::Status Abort() override {
    is_closed = true;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const override {
    return total_read;
  }

  bool closed() const override {
    return is_closed;
  }

private:
  jlong read_into(JNIEnv *env, jlong addr, jlong len) {
    jlong ret = env->CallLongMethod(callback, read_into_method, addr, len);
    if (env->ExceptionCheck()) {
      throw std::runtime_error("readInto threw an exception");
    }
    return ret;
  }

  JavaVM *jvm;
  jobject callback;
  jmethodID read_into_method;
  int64_t total_read = 0;
  bool is_closed = false;
  std::vector<uint8_t> tmp_buffer;
  std::shared_ptr<arrow::MemoryManager> mm;
};

class native_arrow_ipc_reader_handle final {
public:
  explicit native_arrow_ipc_reader_handle(
          const std::string& file_name) {
    auto tmp_source = arrow::io::ReadableFile::Open(file_name);
    if (!tmp_source.ok()) {
      throw std::runtime_error(tmp_source.status().message());
    }
    source = *tmp_source;
    auto tmp_reader = arrow::ipc::RecordBatchStreamReader::Open(source);
    if (!tmp_reader.ok()) {
      throw std::runtime_error(tmp_reader.status().message());
    }
    reader = *tmp_reader;
  }

  explicit native_arrow_ipc_reader_handle(
          std::shared_ptr<arrow::io::InputStream> source):
     source(source) {
    auto tmp_reader = arrow::ipc::RecordBatchStreamReader::Open(source);
    if (!tmp_reader.ok()) {
      throw std::runtime_error(tmp_reader.status().message());
    }
    reader = *tmp_reader;
  }

  std::shared_ptr<arrow::Table> next(int32_t row_target) {
    int64_t total_rows = 0;
    bool done = false;
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    while (!done) {
      arrow::Result<std::shared_ptr<arrow::RecordBatch>> batch = reader->Next();
      if (!batch.ok()) {
        throw std::runtime_error(batch.status().message());
      }
      if (!*batch) {
        done = true;
      } else {
        batches.push_back(*batch);
        total_rows += (*batch)->num_rows();
        done = (total_rows >= row_target);
      }
    }
    if (batches.empty()) {
      // EOF
      return std::unique_ptr<arrow::Table>();
    }
    arrow::Result<std::shared_ptr<arrow::Table>> tmp = 
        arrow::Table::FromRecordBatches(reader->schema(), batches);
    if (!tmp.ok()) {
      throw std::runtime_error(tmp.status().message());
    }
    return *tmp;
  }

  std::shared_ptr<arrow::io::InputStream> source;
  std::shared_ptr<arrow::ipc::RecordBatchReader> reader;

  void close() {
    source->Close();
  }
};

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

namespace {
// Check that window parameters are valid.
bool valid_window_parameters(native_jintArray const &values,
                             native_jpointerArray<cudf::aggregation> const &ops,
                             native_jintArray const &min_periods, native_jintArray const &preceding,
                             native_jintArray const &following) {
  return values.size() == ops.size() && values.size() == min_periods.size() &&
         values.size() == preceding.size() && values.size() == following.size();
}

// Check that time-range window parameters are valid.
bool valid_window_parameters(native_jintArray const &values, native_jintArray const &timestamps,
                             native_jpointerArray<cudf::aggregation> const &ops,
                             native_jintArray const &min_periods,
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

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_merge(JNIEnv *env, jclass j_class_object,
                                                             jlongArray j_table_handles,
                                                             jintArray j_sort_key_indexes,
                                                             jbooleanArray j_is_descending,
                                                             jbooleanArray j_are_nulls_smallest) {

  // input validations & verifications
  JNI_NULL_CHECK(env, j_table_handles, "input tables are null", NULL);
  JNI_NULL_CHECK(env, j_sort_key_indexes, "key indexes is null", NULL);
  JNI_NULL_CHECK(env, j_is_descending, "sort order array is null", NULL);
  JNI_NULL_CHECK(env, j_are_nulls_smallest, "null order array is null", NULL);

  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<cudf::table_view> n_table_handles(env,
                                                                      j_table_handles);

    const cudf::jni::native_jintArray n_sort_key_indexes(env, j_sort_key_indexes);
    jsize num_columns = n_sort_key_indexes.size();
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

    std::vector<int> indexes(n_sort_key_indexes.size());
    for (int i = 0; i < n_sort_key_indexes.size(); i++) {
      indexes[i] = n_sort_key_indexes[i];
    }
    std::vector<cudf::order> order(n_is_descending.size());
    for (int i = 0; i < n_is_descending.size(); i++) {
      order[i] = n_is_descending[i] ? cudf::order::DESCENDING : cudf::order::ASCENDING;
    }
    std::vector<cudf::null_order> null_order(n_are_nulls_smallest.size());
    for (int i = 0; i < n_are_nulls_smallest.size(); i++) {
      null_order[i] = n_are_nulls_smallest[i] ? cudf::null_order::BEFORE : cudf::null_order::AFTER;
    }

    jsize num_tables = n_table_handles.size();
    std::vector<cudf::table_view> tables;
    tables.reserve(num_tables);
    for (int i = 0; i < num_tables; i++) {
      tables.push_back(*n_table_handles[i]);
    }

    std::unique_ptr<cudf::table> result = cudf::merge(tables,
            indexes,
            order,
            null_order);
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

    cudf::io::csv_reader_options opts = cudf::io::csv_reader_options::builder(*source)
      .delimiter(delim)
      .header(header_row)
      .names(n_col_names.as_cpp_vector())
      .dtypes(n_data_types.as_cpp_vector())
      .use_cols_names(n_filter_col_names.as_cpp_vector())
      .true_values(n_true_values.as_cpp_vector())
      .false_values(n_false_values.as_cpp_vector())
      .na_values(n_null_values.as_cpp_vector())
      .keep_default_na(false)
      .na_filter(n_null_values.size() > 0)
      .quotechar(quote)
      .comment(comment)
      .build();
    cudf::io::table_with_metadata result = cudf::io::read_csv(opts);
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

    cudf::io::parquet_reader_options opts =
      cudf::io::parquet_reader_options::builder(*source)
        .columns(n_filter_col_names.as_cpp_vector())
        .convert_strings_to_categories(false)
        .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
        .build();
    cudf::io::table_with_metadata result = cudf::io::read_parquet(opts);
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
    chunked_parquet_writer_options opts =
      chunked_parquet_writer_options::builder(sink)
        .nullable_metadata(&metadata)
        .compression(static_cast<compression_type>(j_compression))
        .stats_level(static_cast<statistics_freq>(j_stats_freq))
        .build();
    std::shared_ptr<pq_chunked_state> state = write_parquet_chunked_begin(opts);
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
    chunked_parquet_writer_options opts =
      chunked_parquet_writer_options::builder(sink)
        .nullable_metadata(&metadata)
        .compression(static_cast<compression_type>(j_compression))
        .stats_level(static_cast<statistics_freq>(j_stats_freq))
        .build();
    std::shared_ptr<pq_chunked_state> state = write_parquet_chunked_begin(opts);
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

    cudf::io::orc_reader_options opts = cudf::io::orc_reader_options::builder(*source)
      .columns(n_filter_col_names.as_cpp_vector())
      .use_index(false)
      .use_np_dtypes(static_cast<bool>(usingNumPyTypes))
      .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
      .build();
    cudf::io::table_with_metadata result = cudf::io::read_orc(opts);
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
    chunked_orc_writer_options opts =
      chunked_orc_writer_options::builder(sink)
        .metadata(&metadata)
        .compression(static_cast<compression_type>(j_compression))
        .enable_statistics(true)
        .build();
    std::shared_ptr<orc_chunked_state> state = write_orc_chunked_begin(opts);
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
    chunked_orc_writer_options opts =
      chunked_orc_writer_options::builder(sink)
        .metadata(&metadata)
        .compression(static_cast<compression_type>(j_compression))
        .enable_statistics(true)
        .build();
    std::shared_ptr<orc_chunked_state> state = write_orc_chunked_begin(opts);
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


JNIEXPORT long JNICALL Java_ai_rapids_cudf_Table_writeArrowIPCBufferBegin(
    JNIEnv *env, jclass, jobjectArray j_col_names,
    jobject consumer) {
  JNI_NULL_CHECK(env, j_col_names, "null columns", 0);
  JNI_NULL_CHECK(env, consumer, "null consumer", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstringArray col_names(env, j_col_names);

    std::shared_ptr<cudf::jni::jni_arrow_output_stream> data_sink(
        new cudf::jni::jni_arrow_output_stream(env, consumer));

    cudf::jni::native_arrow_ipc_writer_handle *ret =
        new cudf::jni::native_arrow_ipc_writer_handle(
                col_names.as_cpp_vector(),
                data_sink);
    return reinterpret_cast<jlong>(ret);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT long JNICALL Java_ai_rapids_cudf_Table_writeArrowIPCFileBegin(
    JNIEnv *env, jclass, jobjectArray j_col_names,
    jstring j_output_path) {
  JNI_NULL_CHECK(env, j_col_names, "null columns", 0);
  JNI_NULL_CHECK(env, j_output_path, "null output path", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstringArray col_names(env, j_col_names);
    cudf::jni::native_jstring output_path(env, j_output_path);

    cudf::jni::native_arrow_ipc_writer_handle *ret =
        new cudf::jni::native_arrow_ipc_writer_handle(
                col_names.as_cpp_vector(),
                output_path.get());
    return reinterpret_cast<jlong>(ret);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_convertCudfToArrowTable(JNIEnv *env, jclass,
                                                                          jlong j_state,
                                                                          jlong j_table) {
  JNI_NULL_CHECK(env, j_table, "null table", 0);
  JNI_NULL_CHECK(env, j_state, "null state", 0);

  cudf::table_view *tview = reinterpret_cast<cudf::table_view *>(j_table);
  cudf::jni::native_arrow_ipc_writer_handle *state =
      reinterpret_cast<cudf::jni::native_arrow_ipc_writer_handle *>(j_state);

  try {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<std::shared_ptr<arrow::Table>> result(new std::shared_ptr<arrow::Table>(nullptr));
    *result = cudf::to_arrow(*tview, state->column_names);
    if (!result->get()) {
      return 0;
    }
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0)
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeArrowIPCArrowChunk(JNIEnv *env, jclass,
                                                                         jlong j_state,
                                                                         jlong arrow_table_handle,
                                                                         jlong max_chunk) {
  JNI_NULL_CHECK(env, arrow_table_handle, "null arrow table", );
  JNI_NULL_CHECK(env, j_state, "null state", );

  std::shared_ptr<arrow::Table> *handle =
      reinterpret_cast<std::shared_ptr<arrow::Table> *>(arrow_table_handle);
  cudf::jni::native_arrow_ipc_writer_handle *state =
      reinterpret_cast<cudf::jni::native_arrow_ipc_writer_handle *>(j_state);

  try {
    cudf::jni::auto_set_device(env);
    state->write(*handle, max_chunk);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeArrowIPCEnd(JNIEnv *env, jclass,
                                                                  jlong j_state) {
  JNI_NULL_CHECK(env, j_state, "null state", );

  cudf::jni::native_arrow_ipc_writer_handle *state =
      reinterpret_cast<cudf::jni::native_arrow_ipc_writer_handle *>(j_state);
  std::unique_ptr<cudf::jni::native_arrow_ipc_writer_handle> make_sure_we_delete(state);
  try {
    cudf::jni::auto_set_device(env);
    state->close();
  }
  CATCH_STD(env, )
}

JNIEXPORT long JNICALL Java_ai_rapids_cudf_Table_readArrowIPCFileBegin(JNIEnv *env, jclass,
    jstring j_input_path) {
  JNI_NULL_CHECK(env, j_input_path, "null input path", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring input_path(env, j_input_path);

    cudf::jni::native_arrow_ipc_reader_handle *ret =
        new cudf::jni::native_arrow_ipc_reader_handle(input_path.get());
    return reinterpret_cast<jlong>(ret);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT long JNICALL Java_ai_rapids_cudf_Table_readArrowIPCBufferBegin(JNIEnv *env, jclass,
    jobject provider) {
  JNI_NULL_CHECK(env, provider, "null provider", 0);
  try {
    cudf::jni::auto_set_device(env);

    std::shared_ptr<cudf::jni::jni_arrow_input_stream> data_source(
        new cudf::jni::jni_arrow_input_stream(env, provider));

    cudf::jni::native_arrow_ipc_reader_handle *ret =
        new cudf::jni::native_arrow_ipc_reader_handle(data_source);
    return reinterpret_cast<jlong>(ret);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL 
Java_ai_rapids_cudf_Table_readArrowIPCChunkToArrowTable(JNIEnv *env, jclass,
                                                        jlong j_state,
                                                        jint row_target) {
  JNI_NULL_CHECK(env, j_state, "null state", 0);

  cudf::jni::native_arrow_ipc_reader_handle *state =
      reinterpret_cast<cudf::jni::native_arrow_ipc_reader_handle *>(j_state);

  try {
    cudf::jni::auto_set_device(env);
    // This is a little odd because we have to return a pointer
    // and arrow wants to deal with shared pointers for everything.
    std::unique_ptr<std::shared_ptr<arrow::Table>> result(new std::shared_ptr<arrow::Table>(nullptr));
    *result = state->next(row_target);
    if (!result->get()) {
      return 0;
    }
    return reinterpret_cast<jlong>(result.release());
  }
  CATCH_STD(env, 0)
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_closeArrowTable(JNIEnv *env, jclass,
                                                                 jlong arrow_table_handle) {
  std::shared_ptr<arrow::Table> *handle =
      reinterpret_cast<std::shared_ptr<arrow::Table> *>(arrow_table_handle);

  try {
    cudf::jni::auto_set_device(env);
    delete handle;
  }
  CATCH_STD(env, )
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_convertArrowTableToCudf(JNIEnv *env, jclass,
                                                                               jlong arrow_table_handle) {
  JNI_NULL_CHECK(env, arrow_table_handle, "null arrow handle", 0);

  std::shared_ptr<arrow::Table> *handle =
      reinterpret_cast<std::shared_ptr<arrow::Table> *>(arrow_table_handle);

  try {
    cudf::jni::auto_set_device(env);
    std::unique_ptr<cudf::table> result = cudf::from_arrow(*(handle->get()));
    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_readArrowIPCEnd(JNIEnv *env, jclass,
                                                                  jlong j_state) {
  JNI_NULL_CHECK(env, j_state, "null state", );

  cudf::jni::native_arrow_ipc_reader_handle *state =
      reinterpret_cast<cudf::jni::native_arrow_ipc_reader_handle *>(j_state);
  std::unique_ptr<cudf::jni::native_arrow_ipc_reader_handle> make_sure_we_delete(state);
  try {
    cudf::jni::auto_set_device(env);
    state->close();
  }
  CATCH_STD(env, )
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftJoin(JNIEnv *env, jclass clazz,
                                                                jlong left_table,
                                                                jintArray left_col_join_indices,
                                                                jlong right_table,
                                                                jintArray right_col_join_indices,
                                                                jboolean compare_nulls_equal) {
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
        cudf::left_join(*n_left_table, *n_right_table, left_join_cols, right_join_cols, dedupe,
            static_cast<bool>(compare_nulls_equal)? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_innerJoin(JNIEnv *env, jclass clazz,
                                                                 jlong left_table,
                                                                 jintArray left_col_join_indices,
                                                                 jlong right_table,
                                                                 jintArray right_col_join_indices,
                                                                 jboolean compare_nulls_equal) {
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
        cudf::inner_join(*n_left_table, *n_right_table, left_join_cols, right_join_cols, dedupe,
            static_cast<bool>(compare_nulls_equal)? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_fullJoin(JNIEnv *env, jclass clazz,
                                                                jlong left_table,
                                                                jintArray left_col_join_indices,
                                                                jlong right_table,
                                                                jintArray right_col_join_indices,
                                                                jboolean compare_nulls_equal) {
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
        cudf::full_join(*n_left_table, *n_right_table, left_join_cols, right_join_cols, dedupe,
            static_cast<bool>(compare_nulls_equal)? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftSemiJoin(
    JNIEnv *env, jclass, jlong left_table, jintArray left_col_join_indices, jlong right_table,
    jintArray right_col_join_indices, jboolean compare_nulls_equal) {
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
        *n_left_table, *n_right_table, left_join_cols, right_join_cols, return_cols,
          static_cast<bool>(compare_nulls_equal)? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL);

    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftAntiJoin(
    JNIEnv *env, jclass, jlong left_table, jintArray left_col_join_indices, jlong right_table,
    jintArray right_col_join_indices, jboolean compare_nulls_equal) {
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
        *n_left_table, *n_right_table, left_join_cols, right_join_cols, return_cols,
            static_cast<bool>(compare_nulls_equal)? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL);

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
    jintArray aggregate_column_indices, jlongArray agg_instances, jboolean ignore_null_keys) {
  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, keys, "input keys are null", NULL);
  JNI_NULL_CHECK(env, aggregate_column_indices, "input aggregate_column_indices are null", NULL);
  JNI_NULL_CHECK(env, agg_instances, "agg_instances are null", NULL);

  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *n_input_table = reinterpret_cast<cudf::table_view *>(input_table);
    cudf::jni::native_jintArray n_keys(env, keys);
    cudf::jni::native_jintArray n_values(env, aggregate_column_indices);
    cudf::jni::native_jpointerArray<cudf::aggregation> n_agg_instances(env, agg_instances);
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
        requests.back().aggregations.push_back(n_agg_instances[i]->clone());
      } else {
        req.values = n_input_table->column(col_index);
        req.aggregations.push_back(n_agg_instances[i]->clone());
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

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gather(JNIEnv *env, jclass,
                                                              jlong j_input,
                                                              jlong j_map,
                                                              jboolean check_bounds) {
  JNI_NULL_CHECK(env, j_input, "input table is null", 0);
  JNI_NULL_CHECK(env, j_map, "map column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *input = reinterpret_cast<cudf::table_view *>(j_input);
    cudf::column_view *map = reinterpret_cast<cudf::column_view *>(j_map);
    std::unique_ptr<cudf::table> result = cudf::gather(*input, *map);
    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_repeatStaticCount(JNIEnv *env, jclass,
                                                                         jlong input_jtable,
                                                                         jint count) {
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *input = reinterpret_cast<cudf::table_view *>(input_jtable);
    std::unique_ptr<cudf::table> result = cudf::repeat(*input, count);
    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_repeatColumnCount(JNIEnv *env, jclass,
                                                                         jlong input_jtable,
                                                                         jlong count_jcol,
                                                                         jboolean check_count) {
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_NULL_CHECK(env, count_jcol, "count column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *input = reinterpret_cast<cudf::table_view *>(input_jtable);
    cudf::column_view *count = reinterpret_cast<cudf::column_view *>(count_jcol);
    std::unique_ptr<cudf::table> result = cudf::repeat(*input, *count, check_count);
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
    jlongArray j_default_output, 
    jintArray j_aggregate_column_indices, jlongArray j_agg_instances, 
    jintArray j_min_periods,
    jintArray j_preceding, jintArray j_following, jboolean ignore_null_keys) {

  JNI_NULL_CHECK(env, j_input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_keys, "input keys are null", NULL);
  JNI_NULL_CHECK(env, j_aggregate_column_indices, "input aggregate_column_indices are null", NULL);
  JNI_NULL_CHECK(env, j_agg_instances, "agg_instances are null", NULL);
  JNI_NULL_CHECK(env, j_default_output, "default_outputs are null", NULL);

  try {
    cudf::jni::auto_set_device(env);

    using cudf::jni::valid_window_parameters;

    // Convert from j-types to native.
    cudf::table_view *input_table{reinterpret_cast<cudf::table_view *>(j_input_table)};
    cudf::jni::native_jintArray keys{env, j_keys};
    cudf::jni::native_jintArray values{env, j_aggregate_column_indices};
    cudf::jni::native_jpointerArray<cudf::aggregation> agg_instances(env, j_agg_instances);
    cudf::jni::native_jpointerArray<cudf::column_view> default_output(env, j_default_output);
    cudf::jni::native_jintArray min_periods{env, j_min_periods};
    cudf::jni::native_jintArray preceding{env, j_preceding};
    cudf::jni::native_jintArray following{env, j_following};

    if (not valid_window_parameters(values, agg_instances, min_periods, preceding, following)) {
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
      if (default_output[i] != nullptr) {
        result_columns.emplace_back(std::move(cudf::grouped_rolling_window(
            groupby_keys, input_table->column(agg_column_index), *default_output[i],
            preceding[i], following[i],
            min_periods[i], agg_instances[i]->clone())));
      } else {
        result_columns.emplace_back(std::move(cudf::grouped_rolling_window(
            groupby_keys, input_table->column(agg_column_index),
            preceding[i], following[i],
            min_periods[i], agg_instances[i]->clone())));
      }
    }

    auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
    return cudf::jni::convert_table_for_return(env, result_table);
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_timeRangeRollingWindowAggregate(
    JNIEnv *env, jclass clazz, jlong j_input_table, jintArray j_keys,
    jintArray j_timestamp_column_indices, jbooleanArray j_is_timestamp_ascending,
    jintArray j_aggregate_column_indices, jlongArray j_agg_instances, jintArray j_min_periods,
    jintArray j_preceding, jintArray j_following, jboolean ignore_null_keys) {

  JNI_NULL_CHECK(env, j_input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_keys, "input keys are null", NULL);
  JNI_NULL_CHECK(env, j_timestamp_column_indices, "input timestamp_column_indices are null", NULL);
  JNI_NULL_CHECK(env, j_is_timestamp_ascending, "input timestamp_ascending is null", NULL);
  JNI_NULL_CHECK(env, j_aggregate_column_indices, "input aggregate_column_indices are null", NULL);
  JNI_NULL_CHECK(env, j_agg_instances, "agg_instances are null", NULL);

  try {
    cudf::jni::auto_set_device(env);

    using cudf::jni::valid_window_parameters;

    // Convert from j-types to native.
    cudf::table_view *input_table{reinterpret_cast<cudf::table_view *>(j_input_table)};
    cudf::jni::native_jintArray keys{env, j_keys};
    cudf::jni::native_jintArray timestamps{env, j_timestamp_column_indices};
    cudf::jni::native_jbooleanArray timestamp_ascending{env, j_is_timestamp_ascending};
    cudf::jni::native_jintArray values{env, j_aggregate_column_indices};
    cudf::jni::native_jpointerArray<cudf::aggregation> agg_instances(env, j_agg_instances);
    cudf::jni::native_jintArray min_periods{env, j_min_periods};
    cudf::jni::native_jintArray preceding{env, j_preceding};
    cudf::jni::native_jintArray following{env, j_following};

    if (not valid_window_parameters(values, agg_instances, min_periods, preceding, following)) {
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
          agg_instances[i]->clone())));
    }

    auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
    return cudf::jni::convert_table_for_return(env, result_table);
  }
  CATCH_STD(env, NULL);
}

} // extern "C"
