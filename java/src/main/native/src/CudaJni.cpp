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

#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#ifdef CUDF_JNI_ENABLE_PROFILING
#include <cuda_profiler_api.h>
#endif

#include "jni_utils.hpp"

namespace {

/** The CUDA device that should be used by all threads using cudf */
int Cudf_device{cudaInvalidDeviceId};

thread_local int Thread_device = cudaInvalidDeviceId;

}  // anonymous namespace

namespace cudf {
namespace jni {

/** Set the device to use for cudf */
void set_cudf_device(int device) { Cudf_device = device; }

/**
 * If a cudf device has been specified then this ensures the calling thread
 * is using the same device.
 */
void auto_set_device(JNIEnv* env)
{
  if (Cudf_device != cudaInvalidDeviceId) {
    if (Thread_device != Cudf_device) {
      cudaError_t cuda_status = cudaSetDevice(Cudf_device);
      jni_cuda_check(env, cuda_status);
      Thread_device = Cudf_device;
    }
  }
}

/** Fills all the bytes in the buffer 'buf' with 'value'. */
void device_memset_async(JNIEnv* env, rmm::device_buffer& buf, char value)
{
  cudaError_t cuda_status = cudaMemsetAsync((void*)buf.data(), value, buf.size());
  jni_cuda_check(env, cuda_status);
}

}  // namespace jni
}  // namespace cudf

extern "C" {

JNIEXPORT jobject JNICALL Java_ai_rapids_cudf_Cuda_memGetInfo(JNIEnv* env, jclass clazz)
{
  try {
    cudf::jni::auto_set_device(env);

    size_t free, total;
    CUDF_CUDA_TRY(cudaMemGetInfo(&free, &total));

    jclass info_class = env->FindClass("Lai/rapids/cudf/CudaMemInfo;");
    if (info_class == NULL) { return NULL; }

    jmethodID ctor_id = env->GetMethodID(info_class, "<init>", "(JJ)V");
    if (ctor_id == NULL) { return NULL; }

    jobject info_obj = env->NewObject(info_class, ctor_id, (jlong)free, (jlong)total);
    // No need to check for exceptions of null return value as we are just handing the object back
    // to the JVM which will handle throwing any exceptions that happened in the constructor.
    return info_obj;
  }
  CATCH_STD(env, nullptr);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cuda_hostAllocPinned(JNIEnv* env, jclass, jlong size)
{
  try {
    cudf::jni::auto_set_device(env);
    void* ret = nullptr;
    CUDF_CUDA_TRY(cudaMallocHost(&ret, size));
    return reinterpret_cast<jlong>(ret);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_freePinned(JNIEnv* env, jclass, jlong ptr)
{
  try {
    cudf::jni::auto_set_device(env);
    CUDF_CUDA_TRY(cudaFreeHost(reinterpret_cast<void*>(ptr)));
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_Cuda_memset(JNIEnv* env, jclass, jlong dst, jbyte value, jlong count, jint kind)
{
  JNI_NULL_CHECK(env, dst, "dst memory pointer is null", );
  try {
    cudf::jni::auto_set_device(env);
    CUDF_CUDA_TRY(cudaMemsetAsync((void*)dst, value, count));
    CUDF_CUDA_TRY(cudaStreamSynchronize(0));
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_asyncMemset(
  JNIEnv* env, jclass, jlong dst, jbyte value, jlong count, jint kind)
{
  JNI_NULL_CHECK(env, dst, "dst memory pointer is null", );
  try {
    cudf::jni::auto_set_device(env);
    CUDF_CUDA_TRY(cudaMemsetAsync((void*)dst, value, count));
  }
  CATCH_STD(env, );
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Cuda_getDevice(JNIEnv* env, jclass)
{
  try {
    cudf::jni::auto_set_device(env);
    jint dev;
    CUDF_CUDA_TRY(cudaGetDevice(&dev));
    return dev;
  }
  CATCH_STD(env, -2);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Cuda_getDeviceCount(JNIEnv* env, jclass)
{
  try {
    cudf::jni::auto_set_device(env);
    jint count;
    CUDF_CUDA_TRY(cudaGetDeviceCount(&count));
    return count;
  }
  CATCH_STD(env, -2);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_setDevice(JNIEnv* env, jclass, jint dev)
{
  try {
    if (Cudf_device != cudaInvalidDeviceId && dev != Cudf_device) {
      cudf::jni::throw_java_exception(
        env, cudf::jni::CUDF_ERROR_CLASS, "Cannot change device after RMM init");
    }
    CUDF_CUDA_TRY(cudaSetDevice(dev));
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_autoSetDevice(JNIEnv* env, jclass, jint dev)
{
  try {
    cudf::jni::auto_set_device(env);
  }
  CATCH_STD(env, );
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Cuda_getDriverVersion(JNIEnv* env, jclass)
{
  try {
    cudf::jni::auto_set_device(env);
    jint driver_version;
    CUDF_CUDA_TRY(cudaDriverGetVersion(&driver_version));
    return driver_version;
  }
  CATCH_STD(env, -2);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Cuda_getRuntimeVersion(JNIEnv* env, jclass)
{
  try {
    cudf::jni::auto_set_device(env);
    jint runtime_version;
    CUDF_CUDA_TRY(cudaRuntimeGetVersion(&runtime_version));
    return runtime_version;
  }
  CATCH_STD(env, -2);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Cuda_getNativeComputeMode(JNIEnv* env, jclass)
{
  try {
    cudf::jni::auto_set_device(env);
    int device;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    cudaDeviceProp device_prop;
    CUDF_CUDA_TRY(cudaGetDeviceProperties(&device_prop, device));
    return device_prop.computeMode;
  }
  CATCH_STD(env, -2);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Cuda_getComputeCapabilityMajor(JNIEnv* env, jclass)
{
  try {
    cudf::jni::auto_set_device(env);
    int device;
    CUDF_CUDA_TRY(::cudaGetDevice(&device));
    int attribute_value;
    CUDF_CUDA_TRY(
      ::cudaDeviceGetAttribute(&attribute_value, ::cudaDevAttrComputeCapabilityMajor, device));
    return attribute_value;
  }
  CATCH_STD(env, -2);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Cuda_getComputeCapabilityMinor(JNIEnv* env, jclass)
{
  try {
    cudf::jni::auto_set_device(env);
    int device;
    CUDF_CUDA_TRY(::cudaGetDevice(&device));
    int attribute_value;
    CUDF_CUDA_TRY(
      ::cudaDeviceGetAttribute(&attribute_value, ::cudaDevAttrComputeCapabilityMinor, device));
    return attribute_value;
  }
  CATCH_STD(env, -2);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_freeZero(JNIEnv* env, jclass)
{
  try {
    cudf::jni::auto_set_device(env);
    CUDF_CUDA_TRY(cudaFree(0));
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cuda_createStream(JNIEnv* env,
                                                              jclass,
                                                              jboolean isNonBlocking)
{
  try {
    cudf::jni::auto_set_device(env);
    cudaStream_t stream = nullptr;
    auto flags          = isNonBlocking ? cudaStreamNonBlocking : cudaStreamDefault;
    CUDF_CUDA_TRY(cudaStreamCreateWithFlags(&stream, flags));
    return reinterpret_cast<jlong>(stream);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_destroyStream(JNIEnv* env, jclass, jlong jstream)
{
  try {
    cudf::jni::auto_set_device(env);
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    CUDF_CUDA_TRY(cudaStreamDestroy(stream));
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_streamWaitEvent(JNIEnv* env,
                                                                jclass,
                                                                jlong jstream,
                                                                jlong jevent)
{
  try {
    cudf::jni::auto_set_device(env);
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    auto event  = reinterpret_cast<cudaEvent_t>(jevent);
    CUDF_CUDA_TRY(cudaStreamWaitEvent(stream, event, 0));
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_streamSynchronize(JNIEnv* env,
                                                                  jclass,
                                                                  jlong jstream)
{
  try {
    cudf::jni::auto_set_device(env);
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream));
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Cuda_createEvent(JNIEnv* env,
                                                             jclass,
                                                             jboolean enableTiming,
                                                             jboolean blockingSync)
{
  try {
    cudf::jni::auto_set_device(env);
    cudaEvent_t event  = nullptr;
    unsigned int flags = 0;
    if (!enableTiming) { flags = flags | cudaEventDisableTiming; }
    if (blockingSync) { flags = flags | cudaEventBlockingSync; }
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event, flags));
    return reinterpret_cast<jlong>(event);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_destroyEvent(JNIEnv* env, jclass, jlong jevent)
{
  try {
    cudf::jni::auto_set_device(env);
    auto event = reinterpret_cast<cudaEvent_t>(jevent);
    CUDF_CUDA_TRY(cudaEventDestroy(event));
  }
  CATCH_STD(env, );
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_Cuda_eventQuery(JNIEnv* env, jclass, jlong jevent)
{
  try {
    cudf::jni::auto_set_device(env);
    auto event  = reinterpret_cast<cudaEvent_t>(jevent);
    auto result = cudaEventQuery(event);
    if (result == cudaSuccess) {
      return true;
    } else if (result == cudaErrorNotReady) {
      return false;
    }  // else
    CUDF_CUDA_TRY(result);
  }
  CATCH_STD(env, false);
  return false;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_eventRecord(JNIEnv* env,
                                                            jclass,
                                                            jlong jevent,
                                                            jlong jstream)
{
  try {
    cudf::jni::auto_set_device(env);
    auto event  = reinterpret_cast<cudaEvent_t>(jevent);
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    CUDF_CUDA_TRY(cudaEventRecord(event, stream));
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_eventSynchronize(JNIEnv* env, jclass, jlong jevent)
{
  try {
    cudf::jni::auto_set_device(env);
    auto event = reinterpret_cast<cudaEvent_t>(jevent);
    CUDF_CUDA_TRY(cudaEventSynchronize(event));
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_memcpyOnStream(
  JNIEnv* env, jclass, jlong jdst, jlong jsrc, jlong count, jint jkind, jlong jstream)
{
  if (count == 0) { return; }
  JNI_ARG_CHECK(env, jdst != 0, "dst memory pointer is null", );
  JNI_ARG_CHECK(env, jsrc != 0, "src memory pointer is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto dst    = reinterpret_cast<void*>(jdst);
    auto src    = reinterpret_cast<void*>(jsrc);
    auto kind   = static_cast<cudaMemcpyKind>(jkind);
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    CUDF_CUDA_TRY(cudaMemcpyAsync(dst, src, count, kind, stream));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream));
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_asyncMemcpyOnStream(
  JNIEnv* env, jclass, jlong jdst, jlong jsrc, jlong count, jint jkind, jlong jstream)
{
  if (count == 0) { return; }
  JNI_ARG_CHECK(env, jdst != 0, "dst memory pointer is null", );
  JNI_ARG_CHECK(env, jsrc != 0, "src memory pointer is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto dst    = reinterpret_cast<void*>(jdst);
    auto src    = reinterpret_cast<void*>(jsrc);
    auto kind   = static_cast<cudaMemcpyKind>(jkind);
    auto stream = reinterpret_cast<cudaStream_t>(jstream);
    CUDF_CUDA_TRY(cudaMemcpyAsync(dst, src, count, kind, stream));
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_profilerStart(JNIEnv* env, jclass clazz)
{
#ifdef CUDF_JNI_ENABLE_PROFILING
  try {
    cudaProfilerStart();
  }
  CATCH_STD(env, );
#else
  cudf::jni::throw_java_exception(
    env, cudf::jni::CUDF_ERROR_CLASS, "This library was built without CUDA profiler support.");
#endif
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_profilerStop(JNIEnv* env, jclass clazz)
{
#ifdef CUDF_JNI_ENABLE_PROFILING
  try {
    cudaProfilerStop();
  }
  CATCH_STD(env, );
#else
  cudf::jni::throw_java_exception(
    env, cudf::jni::CUDF_ERROR_CLASS, "This library was built without CUDA profiler support.");
#endif
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Cuda_deviceSynchronize(JNIEnv* env, jclass clazz)
{
  try {
    cudf::jni::auto_set_device(env);
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
  CATCH_STD(env, );
}

}  // extern "C"
