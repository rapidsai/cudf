/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

#include <cudf/join/hash_join.hpp>

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_HashJoin_create(JNIEnv* env,
                                                            jclass,
                                                            jlong j_table,
                                                            jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_table, "table handle is null", 0);
  JNI_TRY
  {
    auto const load_factor = 0.5;
    cudf::jni::auto_set_device(env);
    auto tview         = reinterpret_cast<cudf::table_view const*>(j_table);
    auto nulleq        = j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    auto hash_join_ptr = new cudf::hash_join(*tview, cudf::nullable_join::YES, nulleq, load_factor);
    return reinterpret_cast<jlong>(hash_join_ptr);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_HashJoin_destroy(JNIEnv* env, jclass, jlong j_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto hash_join_ptr = reinterpret_cast<cudf::hash_join*>(j_handle);
    delete hash_join_ptr;
  }
  JNI_CATCH(env, );
}

}  // extern "C"
