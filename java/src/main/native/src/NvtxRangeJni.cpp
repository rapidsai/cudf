/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "jni_utils.hpp"
#include "nvtx_common.hpp"

#include <nvtx3/nvtx3.hpp>

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_NvtxRange_push(JNIEnv* env,
                                                          jclass clazz,
                                                          jstring name,
                                                          jint color_bits)
{
  try {
    cudf::jni::native_jstring range_name(env, name);
    nvtx3::color range_color(static_cast<nvtx3::color::value_type>(color_bits));
    nvtx3::event_attributes attr{range_color, range_name.get()};
    nvtxDomainRangePushEx(nvtx3::domain::get<cudf::jni::java_domain>(), attr.get());
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_NvtxRange_pop(JNIEnv* env, jclass clazz)
{
  try {
    nvtxDomainRangePop(nvtx3::domain::get<cudf::jni::java_domain>());
  }
  CATCH_STD(env, );
}

}  // extern "C"
