/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "cudf_jni_apis.hpp"

#include <cudf/join.hpp>

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_HashJoin_create(JNIEnv* env,
                                                            jclass,
                                                            jlong j_table,
                                                            jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_table, "table handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto tview         = reinterpret_cast<cudf::table_view const*>(j_table);
    auto nulleq        = j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    auto hash_join_ptr = new cudf::hash_join(*tview, nulleq);
    return reinterpret_cast<jlong>(hash_join_ptr);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_HashJoin_destroy(JNIEnv* env, jclass, jlong j_handle)
{
  try {
    cudf::jni::auto_set_device(env);
    auto hash_join_ptr = reinterpret_cast<cudf::hash_join*>(j_handle);
    delete hash_join_ptr;
  }
  CATCH_STD(env, );
}

}  // extern "C"
