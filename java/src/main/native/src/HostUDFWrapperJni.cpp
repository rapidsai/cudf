/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"

#include <cudf/aggregation/host_udf.hpp>

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_HostUDFWrapper_close(JNIEnv* env,
                                                                jclass class_object,
                                                                jlong ptr)
{
  JNI_TRY
  {
    auto to_del = reinterpret_cast<cudf::host_udf_base*>(ptr);
    delete to_del;
  }
  JNI_CATCH(env, );
}

}  // extern "C"
