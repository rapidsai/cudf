/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <stddef.h>

#include <rmm/rmm.hpp>

#include "jni_utils.hpp"

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_initializeInternal(JNIEnv *env, jclass clazz,
                                                          jint allocation_mode,
                                                          jboolean enable_logging,
                                                          jlong pool_size) {
  if (rmmIsInitialized(nullptr)) {
    JNI_THROW_NEW(env, "java/lang/IllegalStateException", "RMM already initialized", );
  }
  rmmOptions_t opts;
  opts.allocation_mode = static_cast<rmmAllocationMode_t>(allocation_mode);
  opts.enable_logging = enable_logging == JNI_TRUE;
  opts.initial_pool_size = pool_size;
  JNI_RMM_TRY(env, , rmmInitialize(&opts));
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_Rmm_isInitializedInternal(JNIEnv *env, jclass clazz) {
  return rmmIsInitialized(nullptr);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_shutdown(JNIEnv *env, jclass clazz) {
  JNI_RMM_TRY(env, , rmmFinalize());
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_alloc(JNIEnv *env, jclass clazz, jlong size,
                                                      jlong stream) {
  void *ret = 0;
  cudaStream_t c_stream = reinterpret_cast<cudaStream_t>(stream);
  JNI_RMM_TRY(env, 0, RMM_ALLOC(&ret, size, c_stream));
  return (jlong)ret;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_free(JNIEnv *env, jclass clazz, jlong ptr,
                                                    jlong stream) {
  void *cptr = reinterpret_cast<void *>(ptr);
  cudaStream_t c_stream = reinterpret_cast<cudaStream_t>(stream);
  JNI_RMM_TRY(env, , RMM_FREE(cptr, c_stream));
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_freeDeviceBuffer(JNIEnv *env, jclass clazz,
                                                                jlong ptr) {
  rmm::device_buffer *cptr = reinterpret_cast<rmm::device_buffer *>(ptr);
  delete cptr;
}

JNIEXPORT jstring JNICALL Java_ai_rapids_cudf_Rmm_getLog(JNIEnv *env, jclass clazz, jlong size,
                                                      jlong stream) {
  size_t amount = rmmLogSize();
  std::unique_ptr<char> buffer(new char[amount]);
  JNI_RMM_TRY(env, nullptr, rmmGetLog(buffer.get(), amount));
  return env->NewStringUTF(buffer.get());
}

}
