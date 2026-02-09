/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
  JNI_TRY
  {
    cudf::jni::native_jstring range_name(env, name);
    nvtx3::color range_color(static_cast<nvtx3::color::value_type>(color_bits));
    nvtx3::event_attributes attr{range_color, range_name.get()};
    nvtxDomainRangePushEx(nvtx3::domain::get<cudf::jni::java_domain>(), attr.get());
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_NvtxRange_pop(JNIEnv* env, jclass clazz)
{
  JNI_TRY { nvtxDomainRangePop(nvtx3::domain::get<cudf::jni::java_domain>()); }
  JNI_CATCH(env, );
}

}  // extern "C"
