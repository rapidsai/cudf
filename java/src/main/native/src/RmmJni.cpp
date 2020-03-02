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

#include <rmm/mr/device/default_memory_resource.hpp>

#include "jni_utils.hpp"

using rmm::mr::device_memory_resource;

namespace {

constexpr char const* RMM_EXCEPTION_CLASS = "ai/rapids/cudf/RmmException";

/**
 * @brief An RMM device memory resource adaptor that delegates to the wrapped resource
 * for most operations but will call Java to handle certain situations (e.g.: allocation faillure).
 **/
class java_event_handler_memory_resource final : public device_memory_resource {
public:
  java_event_handler_memory_resource(
      JNIEnv* env,
      jobject jhandler,
      device_memory_resource* resource_to_wrap) : resource(resource_to_wrap) {
    if (env->GetJavaVM(&jvm) < 0) {
      throw cudf::jni::jni_exception("GetJavaVM failed");
    }
    jclass cls = env->GetObjectClass(jhandler);
    if (cls == nullptr) {
      throw cudf::jni::jni_exception("class not found");
    }
    on_alloc_fail_method = env->GetMethodID(cls, "onAllocFailure", "(J)Z");
    if (on_alloc_fail_method == nullptr) {
      throw cudf::jni::jni_exception("onAllocFailure method");
    }
    handler_obj = env->NewGlobalRef(jhandler);
    if (handler_obj == nullptr) {
      throw cudf::jni::jni_exception("global ref");
    }
  }

  virtual ~java_event_handler_memory_resource() {
    JNIEnv* env = cudf::jni::get_jni_env(jvm);
    env->DeleteGlobalRef(handler_obj);
    handler_obj = nullptr;
  }

  device_memory_resource* get_wrapped_resource() {
    return resource;
  }

private:
  device_memory_resource* const resource;
  JavaVM* jvm;
  jobject handler_obj;
  jmethodID on_alloc_fail_method;

  bool on_alloc_fail(std::size_t bytes) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
      // workaround for RMM pooled mode (CNMEM backend) leaving a CUDA error pending
      if (err == cudaErrorMemoryAllocation) {
        cudaGetLastError();
      } else {
        // let this allocation fail so the application can see the CUDA error
        return false;
      }
    }

    JNIEnv* env = cudf::jni::get_jni_env(jvm);
    jboolean result = env->CallBooleanMethod(handler_obj,
                                             on_alloc_fail_method,
                                             static_cast<jlong>(bytes));
    auto throwable = env->ExceptionOccurred();
    if (throwable != NULL) {
      std::cerr << "Allocator callback threw a Java exception: ";
      env->ExceptionDescribe();
      // describe clears the exception so re-throw it
      env->Throw(throwable);
      throw std::runtime_error("Java RMM event handler threw an exception");
    }
    return result;
  }

  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    while (true) {
      try {
        return resource->allocate(bytes, stream);
      } catch (std::bad_alloc const& e) {
        if (!on_alloc_fail(bytes)) {
          throw e;
        }
      }
    }
  }

  void do_deallocate(void* p, std::size_t size, cudaStream_t stream) override {
    resource->deallocate(p, size, stream);
  }

  std::pair<size_t, size_t> do_get_mem_info(cudaStream_t stream) const override {
    return resource->get_mem_info(stream);
  }

  bool supports_streams() const noexcept override {
    return resource->supports_streams();
  }
};

std::unique_ptr<java_event_handler_memory_resource> Java_memory_resource{};

void set_java_device_memory_resource(JNIEnv* env, jobject handler_obj) {
  if (Java_memory_resource && handler_obj != nullptr) {
    JNI_THROW_NEW(env, RMM_EXCEPTION_CLASS, "Another event handler is already set", )
  }
  if (Java_memory_resource) {
    auto java_resource = Java_memory_resource.get();
    auto old_resource = rmm::mr::set_default_resource(Java_memory_resource->get_wrapped_resource());
    Java_memory_resource.reset(nullptr);
    if (old_resource != java_resource) {
      rmm::mr::set_default_resource(old_resource);
      JNI_THROW_NEW(env, RMM_EXCEPTION_CLASS,
          "Concurrent modification detected while removing memory resource", );
    }
  }
  if (handler_obj != nullptr) {
    auto resource = rmm::mr::get_default_resource();
    Java_memory_resource.reset(new java_event_handler_memory_resource(env, handler_obj, resource));
    auto replaced_resource = rmm::mr::set_default_resource(Java_memory_resource.get());
    if (resource != replaced_resource) {
      rmm::mr::set_default_resource(replaced_resource);
      Java_memory_resource.reset(nullptr);
      JNI_THROW_NEW(env, RMM_EXCEPTION_CLASS,
          "Concurrent modification detected while installing memory resource", );
    }
  }
}

} // anonymous namespace

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_initializeInternal(JNIEnv *env, jclass clazz,
                                                                  jint allocation_mode,
                                                                  jboolean enable_logging,
                                                                  jlong pool_size,
                                                                  jobject handler_obj) {
  if (rmmIsInitialized(nullptr)) {
    JNI_THROW_NEW(env, "java/lang/IllegalStateException", "RMM already initialized", );
  }
  rmmOptions_t opts;
  opts.allocation_mode = static_cast<rmmAllocationMode_t>(allocation_mode);
  opts.enable_logging = enable_logging == JNI_TRUE;
  opts.initial_pool_size = pool_size;
  JNI_RMM_TRY(env, , rmmInitialize(&opts));
  set_java_device_memory_resource(env, handler_obj);
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_Rmm_isInitializedInternal(JNIEnv *env, jclass clazz) {
  return rmmIsInitialized(nullptr);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_shutdown(JNIEnv *env, jclass clazz) {
  set_java_device_memory_resource(env, nullptr);
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

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_setEventHandlerInternal(JNIEnv* env, jclass,
                                                                       jobject handler_obj) {
  set_java_device_memory_resource(env, handler_obj);
}

JNIEXPORT jstring JNICALL Java_ai_rapids_cudf_Rmm_getLog(JNIEnv *env, jclass clazz, jlong size,
                                                      jlong stream) {
  size_t amount = rmmLogSize();
  std::unique_ptr<char> buffer(new char[amount]);
  JNI_RMM_TRY(env, nullptr, rmmGetLog(buffer.get(), amount));
  return env->NewStringUTF(buffer.get());
}

}
