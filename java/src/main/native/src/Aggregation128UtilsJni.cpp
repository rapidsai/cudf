/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "aggregation128_utils.hpp"
#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Aggregation128Utils_extractInt32Chunk(
  JNIEnv* env, jclass, jlong j_column_view, jint j_out_dtype, jint j_chunk_idx)
{
  JNI_NULL_CHECK(env, j_column_view, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto cview = reinterpret_cast<cudf::column_view const*>(j_column_view);
    auto dtype = cudf::jni::make_data_type(j_out_dtype, 0);
    return cudf::jni::release_as_jlong(cudf::jni::extract_chunk32(*cview, dtype, j_chunk_idx));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Aggregation128Utils_combineInt64SumChunks(
  JNIEnv* env, jclass, jlong j_table_view, jint j_dtype, jint j_scale)
{
  JNI_NULL_CHECK(env, j_table_view, "table is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto tview = reinterpret_cast<cudf::table_view const*>(j_table_view);
    std::unique_ptr<cudf::table> result =
      cudf::jni::assemble128_from_sum(*tview, cudf::jni::make_data_type(j_dtype, j_scale));
    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, 0);
}
}
