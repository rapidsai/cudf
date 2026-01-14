/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

#include <cudf/join/join_factorizer.hpp>

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_JoinFactorizer_create(
  JNIEnv* env, jclass, jlong j_table, jboolean j_compare_nulls, jboolean j_compute_metrics)
{
  JNI_NULL_CHECK(env, j_table, "table handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto tview  = reinterpret_cast<cudf::table_view const*>(j_table);
    auto nulleq = j_compare_nulls ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    auto statistics =
      j_compute_metrics ? cudf::join_statistics::COMPUTE : cudf::join_statistics::SKIP;
    auto factorizer_ptr = new cudf::join_factorizer(*tview, nulleq, statistics);
    return reinterpret_cast<jlong>(factorizer_ptr);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_JoinFactorizer_destroy(JNIEnv* env,
                                                                  jclass,
                                                                  jlong j_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto factorizer_ptr = reinterpret_cast<cudf::join_factorizer*>(j_handle);
    delete factorizer_ptr;
  }
  JNI_CATCH(env, );
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_JoinFactorizer_getDistinctCount(JNIEnv* env,
                                                                           jclass,
                                                                           jlong j_handle)
{
  JNI_NULL_CHECK(env, j_handle, "handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto factorizer_ptr = reinterpret_cast<cudf::join_factorizer*>(j_handle);
    return static_cast<jint>(factorizer_ptr->distinct_count());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_JoinFactorizer_getMaxDuplicateCount(JNIEnv* env,
                                                                               jclass,
                                                                               jlong j_handle)
{
  JNI_NULL_CHECK(env, j_handle, "handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto factorizer_ptr = reinterpret_cast<cudf::join_factorizer*>(j_handle);
    return static_cast<jint>(factorizer_ptr->max_duplicate_count());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_JoinFactorizer_factorizeBuildKeys(JNIEnv* env,
                                                                              jclass,
                                                                              jlong j_handle)
{
  JNI_NULL_CHECK(env, j_handle, "handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto factorizer_ptr = reinterpret_cast<cudf::join_factorizer*>(j_handle);
    auto result         = factorizer_ptr->factorize_right_keys();
    return cudf::jni::release_as_jlong(result);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_JoinFactorizer_factorizeProbeKeys(JNIEnv* env,
                                                                              jclass,
                                                                              jlong j_handle,
                                                                              jlong j_keys_table)
{
  JNI_NULL_CHECK(env, j_handle, "handle is null", 0);
  JNI_NULL_CHECK(env, j_keys_table, "keys table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto factorizer_ptr = reinterpret_cast<cudf::join_factorizer*>(j_handle);
    auto keys_view      = reinterpret_cast<cudf::table_view const*>(j_keys_table);
    auto result         = factorizer_ptr->factorize_left_keys(*keys_view);
    return cudf::jni::release_as_jlong(result);
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
