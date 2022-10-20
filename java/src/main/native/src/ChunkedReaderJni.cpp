/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

//===================================================================
//
// TODO: cleanup header

#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/hashing.hpp>
#include <cudf/interop.hpp>
#include <cudf/io/avro.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join.hpp>
#include <cudf/lists/explode.hpp>
#include <cudf/merge.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/replace.hpp>
#include <cudf/reshape.hpp>
#include <cudf/rolling.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <thrust/iterator/counting_iterator.h>

#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"
#include "jni_compiled_expr.hpp"
#include "jni_utils.hpp"
#include "row_conversion.hpp"

// TODO: cleanup this
namespace cudf::jni {
jlongArray convert_table_for_return(JNIEnv *env, std::unique_ptr<cudf::table> &&table_result,
                                    std::vector<std::unique_ptr<cudf::column>> &&extra_columns);
}
using cudf::jni::release_as_jlong;

// This file is for the code releated to chunked reader (Parquet, ORC, etc.).

extern "C" {

// This function should take all the parameters that `Table.readParquet` takes,
// plus one more parameter `long chunkSizeByteLimit`.
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ParquetChunkedReader_create(
    JNIEnv *env, jclass, jlong chunk_size_byte_limit, jobjectArray filter_col_names,
    jbooleanArray j_col_binary_read, jstring inputfilepath, jlong buffer, jlong buffer_length,
    jint unit) {

  JNI_NULL_CHECK(env, j_col_binary_read, "null col_binary_read", 0);
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
    cudf::jni::native_jbooleanArray n_col_binary_read(env, j_col_binary_read);

    auto const source = read_buffer ?
                            cudf::io::source_info(reinterpret_cast<char *>(buffer),
                                                  static_cast<std::size_t>(buffer_length)) :
                            cudf::io::source_info(filename.get());

    auto builder = cudf::io::chunked_parquet_reader_options::builder(source);
    if (n_filter_col_names.size() > 0) {
      builder = builder.columns(n_filter_col_names.as_cpp_vector());
    }

    auto const read_opts = builder.convert_strings_to_categories(false)
                               .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
                               .byte_limit(chunk_size_byte_limit)
                               .build();
    return reinterpret_cast<jlong>(new cudf::io::chunked_parquet_reader(read_opts));
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_ParquetChunkedReader_hasNext(JNIEnv *env, jclass,
                                                                            jlong handle) {
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);

  try {
    cudf::jni::auto_set_device(env);
    auto const reader_ptr = reinterpret_cast<cudf::io::chunked_parquet_reader *const>(handle);
    return reader_ptr->has_next();
  }
  CATCH_STD(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ParquetChunkedReader_readChunk(JNIEnv *env, jclass,
                                                                                jlong handle) {
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);

  try {
    cudf::jni::auto_set_device(env);
    auto const reader_ptr = reinterpret_cast<cudf::io::chunked_parquet_reader *const>(handle);
    auto chunk = reader_ptr->read_chunk();
    return chunk.tbl ? cudf::jni::convert_table_for_return(env, chunk.tbl) : nullptr;
  }
  CATCH_STD(env, nullptr);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ParquetChunkedReader_close(JNIEnv *env, jclass,
                                                                      jlong handle) {
  JNI_NULL_CHECK(env, handle, "handle is null", );

  try {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<cudf::io::chunked_parquet_reader *>(handle);
  }
  CATCH_STD(env, nullptr);
}

} // extern "C"
