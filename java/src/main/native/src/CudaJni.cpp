/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

extern "C" {

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_Cuda_memGetInfo(JNIEnv *env, jclass clazz) {
  size_t free, total;
  JNI_CUDA_TRY(env, NULL, cudaMemGetInfo(&free, &total));

  jclass info_class = env->FindClass("Lai/rapids/cudf/CudaMemInfo;");
  if (info_class == NULL) {
    return NULL;
  }

  jmethodID ctor_id = env->GetMethodID(info_class, "<init>", "(JJ)V");
  if (ctor_id == NULL) {
    return NULL;
  }

  jobject info_obj = env->NewObject(info_class, ctor_id, (jlong)free, (jlong)total);
  // No need to check for exceptions of null return value as we are just handing the object back to
  // the JVM. which will handle throwing any exceptions that happened in the constructor.
  return info_obj;
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cuda_hostAllocPinned(JNIEnv *env, jclass, jlong size) {
  void * ret = nullptr;
  JNI_CUDA_TRY(env, 0, cudaMallocHost(&ret, size));
  return reinterpret_cast<jlong>(ret);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_freePinned(JNIEnv *env, jclass, jlong ptr) {
  JNI_CUDA_TRY(env, , cudaFreeHost(reinterpret_cast<void *>(ptr)));
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_memcpy(JNIEnv *env, jclass, jlong jdst, jlong jsrc,
                                                       jlong count, jint jkind) {
  if (count == 0) {
    return;
  }
  JNI_ARG_CHECK(env, jdst != 0, "dst memory pointer is null", );
  JNI_ARG_CHECK(env, jsrc != 0, "src memory pointer is null", );
  auto dst = reinterpret_cast<void*>(jdst);
  auto src = reinterpret_cast<void*>(jsrc);
  auto kind = static_cast<cudaMemcpyKind>(jkind);
  JNI_CUDA_TRY(env, , cudaMemcpy(dst, src, count, kind));
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_memset(JNIEnv *env, jclass, jlong dst, jbyte value,
                                                       jlong count, jint kind) {
  JNI_NULL_CHECK(env, dst, "dst memory pointer is null", );
  JNI_CUDA_TRY(env, , cudaMemset((void *)dst, value, count));
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Cuda_getDevice(JNIEnv *env, jclass) {
  jint dev;
  JNI_CUDA_TRY(env, -2, cudaGetDevice(&dev));
  return dev;
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Cuda_getDeviceCount(JNIEnv *env, jclass) {
  jint count;
  JNI_CUDA_TRY(env, -2, cudaGetDeviceCount(&count));
  return count;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_setDevice(JNIEnv *env, jclass, jint dev) {
  JNI_CUDA_TRY(env, , cudaSetDevice(dev));
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_freeZero(JNIEnv *env, jclass) {
  JNI_CUDA_TRY(env, , cudaFree(0));
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cuda_createStream(JNIEnv* env, jclass,
    jboolean isNonBlocking) {
  cudaStream_t stream = nullptr;
  auto flags = isNonBlocking ? cudaStreamNonBlocking : cudaStreamDefault;
  JNI_CUDA_TRY(env, 0, cudaStreamCreateWithFlags(&stream, flags));
  return reinterpret_cast<jlong>(stream);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_destroyStream(JNIEnv* env, jclass,
    jlong jstream) {
  auto stream = reinterpret_cast<cudaStream_t>(jstream);
  JNI_CUDA_TRY(env, , cudaStreamDestroy(stream));
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_memcpyOnStream(JNIEnv* env, jclass,
    jlong jdst, jlong jsrc, jlong count, jint jkind, jlong jstream) {
  if (count == 0) {
    return;
  }
  JNI_ARG_CHECK(env, jdst != 0, "dst memory pointer is null", );
  JNI_ARG_CHECK(env, jsrc != 0, "src memory pointer is null", );
  auto dst = reinterpret_cast<void*>(jdst);
  auto src = reinterpret_cast<void*>(jsrc);
  auto kind = static_cast<cudaMemcpyKind>(jkind);
  auto stream = reinterpret_cast<cudaStream_t>(jstream);
  JNI_CUDA_TRY(env, , cudaMemcpyAsync(dst, src, count, kind, stream));
  JNI_CUDA_TRY(env, , cudaStreamSynchronize(stream));
}

} // extern "C"
