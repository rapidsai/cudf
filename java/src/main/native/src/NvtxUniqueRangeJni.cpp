/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jni_utils.hpp"
#include "nvtx_common.hpp"

#include <nvtx3/nvtx3.hpp>

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_NvtxUniqueRange_start(JNIEnv* env,
                                                                  jclass clazz,
                                                                  jstring name,
                                                                  jint color_bits)
{
  JNI_TRY
  {
    cudf::jni::native_jstring range_name(env, name);
    nvtx3::color range_color(static_cast<nvtx3::color::value_type>(color_bits));
    nvtx3::event_attributes attr{range_color, range_name.get()};
    auto nvtxRangeId =
      nvtxDomainRangeStartEx(nvtx3::domain::get<cudf::jni::java_domain>(), attr.get());
    return static_cast<jlong>(nvtxRangeId);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_NvtxUniqueRange_end(JNIEnv* env,
                                                               jclass clazz,
                                                               jlong nvtxRangeId)
{
  JNI_TRY
  {
    nvtxDomainRangeEnd(nvtx3::domain::get<cudf::jni::java_domain>(),
                       static_cast<nvtxRangeId_t>(nvtxRangeId));
  }
  JNI_CATCH(env, );
}

}  // extern "C"
