/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <memory>
#include <vector>

#include <cudf/column/column.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"

// This function is defined in `TableJni.cpp`.
jlongArray
cudf::jni::convert_table_for_return(JNIEnv *env, std::unique_ptr<cudf::table> &&table_result,
                                    std::vector<std::unique_ptr<cudf::column>> &&extra_columns);

// This file is for the code related to chunked reader (Parquet, ORC, etc.).

extern "C" {

// This function should take all the parameters that `Table.readParquet` takes,
// plus one more parameter `long chunkSizeByteLimit`.
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ParquetChunkedReader_create(
    JNIEnv *env, jclass, jlong chunk_read_limit, jlong pass_read_limit,
    jobjectArray filter_col_names, jbooleanArray j_col_binary_read, jstring inp_file_path,
    jlong buffer, jlong buffer_length, jint unit) {
  JNI_NULL_CHECK(env, j_col_binary_read, "Null col_binary_read", 0);
  bool read_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, inp_file_path, "Input file or buffer must be supplied", 0);
    read_buffer = false;
  } else if (inp_file_path != nullptr) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException",
                  "Cannot pass in both a buffer and an inp_file_path", 0);
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "An empty buffer is not supported", 0);
  }

  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring filename(env, inp_file_path);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(env, "java/lang/IllegalArgumentException", "inp_file_path cannot be empty", 0);
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    // TODO: This variable is unused now, but we still don't know what to do with it yet.
    // As such, it needs to stay here for a little more time before we decide to use it again,
    // or remove it completely.
    cudf::jni::native_jbooleanArray n_col_binary_read(env, j_col_binary_read);
    (void)n_col_binary_read;

    auto const source = read_buffer ?
                            cudf::io::source_info(reinterpret_cast<char *>(buffer),
                                                  static_cast<std::size_t>(buffer_length)) :
                            cudf::io::source_info(filename.get());

    auto opts_builder = cudf::io::parquet_reader_options::builder(source);
    if (n_filter_col_names.size() > 0) {
      opts_builder = opts_builder.columns(n_filter_col_names.as_cpp_vector());
    }
    auto const read_opts = opts_builder.convert_strings_to_categories(false)
                               .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
                               .build();

    return reinterpret_cast<jlong>(
        new cudf::io::chunked_parquet_reader(static_cast<std::size_t>(chunk_read_limit),
                                             static_cast<std::size_t>(pass_read_limit), read_opts));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ParquetChunkedReader_createWithDataSource(
    JNIEnv *env, jclass, jlong chunk_read_limit, jobjectArray filter_col_names,
    jbooleanArray j_col_binary_read, jint unit, jlong ds_handle) {
  JNI_NULL_CHECK(env, j_col_binary_read, "Null col_binary_read", 0);
  JNI_NULL_CHECK(env, ds_handle, "Null DataSouurce", 0);

  try {
    cudf::jni::auto_set_device(env);

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    // TODO: This variable is unused now, but we still don't know what to do with it yet.
    // As such, it needs to stay here for a little more time before we decide to use it again,
    // or remove it completely.
    cudf::jni::native_jbooleanArray n_col_binary_read(env, j_col_binary_read);
    (void)n_col_binary_read;

    auto ds = reinterpret_cast<cudf::io::datasource *>(ds_handle);
    cudf::io::source_info source{ds};

    auto opts_builder = cudf::io::parquet_reader_options::builder(source);
    if (n_filter_col_names.size() > 0) {
      opts_builder = opts_builder.columns(n_filter_col_names.as_cpp_vector());
    }
    auto const read_opts = opts_builder.convert_strings_to_categories(false)
                               .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
                               .build();

    return reinterpret_cast<jlong>(new cudf::io::chunked_parquet_reader(
        static_cast<std::size_t>(chunk_read_limit), read_opts));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_ParquetChunkedReader_hasNext(JNIEnv *env, jclass,
                                                                            jlong handle) {
  JNI_NULL_CHECK(env, handle, "handle is null", false);

  try {
    cudf::jni::auto_set_device(env);
    auto const reader_ptr = reinterpret_cast<cudf::io::chunked_parquet_reader *const>(handle);
    return reader_ptr->has_next();
  }
  CATCH_STD(env, false);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ParquetChunkedReader_readChunk(JNIEnv *env, jclass,
                                                                                jlong handle) {
  JNI_NULL_CHECK(env, handle, "handle is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const reader_ptr = reinterpret_cast<cudf::io::chunked_parquet_reader *const>(handle);
    auto chunk = reader_ptr->read_chunk();
    return chunk.tbl ? cudf::jni::convert_table_for_return(env, chunk.tbl) : nullptr;
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ParquetChunkedReader_close(JNIEnv *env, jclass,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "handle is null", );

  try {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<cudf::io::chunked_parquet_reader *>(handle);
  }
  CATCH_STD(env, );
}

} // extern "C"
