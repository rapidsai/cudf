/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
