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

#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <rmm/mr/device/default_memory_resource.hpp>

#include "jni_utils.hpp"

using rmm::mr::device_memory_resource;

namespace {

constexpr char const* RMM_EXCEPTION_CLASS = "ai/rapids/cudf/RmmException";

/**
 * @brief An RMM device memory resource adaptor that delegates to the wrapped resource
 * for most operations but will call Java to handle certain situations (e.g.: allocation failure).
 **/
class java_event_handler_memory_resource final : public device_memory_resource {
public:
  java_event_handler_memory_resource(
      JNIEnv* env,
      jobject jhandler,
      jlongArray jalloc_thresholds,
      jlongArray jdealloc_thresholds,
      device_memory_resource* resource_to_wrap) : resource(resource_to_wrap) {
    if (env->GetJavaVM(&jvm) < 0) {
      throw std::runtime_error("GetJavaVM failed");
    }

    jclass cls = env->GetObjectClass(jhandler);
    if (cls == nullptr) {
      throw cudf::jni::jni_exception("class not found");
    }
    on_alloc_fail_method = env->GetMethodID(cls, "onAllocFailure", "(J)Z");
    if (on_alloc_fail_method == nullptr) {
      throw cudf::jni::jni_exception("onAllocFailure method");
    }
    on_alloc_threshold_method = env->GetMethodID(cls, "onAllocThreshold", "(J)V");
    if (on_alloc_threshold_method == nullptr) {
      throw cudf::jni::jni_exception("onAllocThreshold method");
    }
    on_dealloc_threshold_method = env->GetMethodID(cls, "onDeallocThreshold", "(J)V");
    if (on_dealloc_threshold_method == nullptr) {
      throw cudf::jni::jni_exception("onDeallocThreshold method");
    }

    update_thresholds(env, alloc_thresholds, jalloc_thresholds);
    update_thresholds(env, dealloc_thresholds, jdealloc_thresholds);

    handler_obj = env->NewGlobalRef(jhandler);
    if (handler_obj == nullptr) {
      throw cudf::jni::jni_exception("global ref");
    }
  }

  virtual ~java_event_handler_memory_resource() {
    // This should normally be called by a JVM thread. If the JVM environment is missing then this
    // is likely being triggered by the C++ runtime during shutdown. In that case the JVM may
    // already be destroyed and this thread should not try to attach to get an environment.
    JNIEnv* env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) == JNI_OK) {
      env->DeleteGlobalRef(handler_obj);
    }
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
  jmethodID on_alloc_threshold_method;
  jmethodID on_dealloc_threshold_method;

  // sorted memory thresholds to trigger callbacks
  std::vector<std::size_t> alloc_thresholds{};
  std::vector<std::size_t> dealloc_thresholds{};

  std::size_t total_allocated{0};

  // map and associated lock to track memory sizes by address
  // TODO: This should be removed when rmm::alloc and rmm::free are removed and the size parameter
  //       for do_deallocate can be trusted. If map and mutex are removed then total_allocated
  //       should be updated to be atomic.
  std::unordered_map<void*, std::size_t> size_map{};
  std::mutex size_map_mutex{};


  static void update_thresholds(JNIEnv* env, std::vector<std::size_t>& thresholds, jlongArray from_java) {
    thresholds.clear();
    if (from_java != nullptr) {
      cudf::jni::native_jlongArray jvalues(env, from_java);
      thresholds.insert(thresholds.end(), jvalues.data(), jvalues.data() + jvalues.size());
    } else {
      // use a single, maximum-threshold value so we don't have to always check for the corner case.
      thresholds.push_back(std::numeric_limits<std::size_t>::max());
    }
  }

  bool on_alloc_fail(std::size_t num_bytes) {
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
                                             static_cast<jlong>(num_bytes));
    if (env->ExceptionCheck()) {
      throw std::runtime_error("onAllocFailure handler threw an exception");
    }
    return result;
  }

  void check_for_threshold_callback(
      std::size_t low,
      std::size_t high,
      std::vector<std::size_t> const& thresholds,
      jmethodID callback_method,
      char const* callback_name,
      std::size_t current_total) {
    if (high >= thresholds.front() && low < thresholds.back()) {
      // could use binary search, but assumption is threshold count is very small
      auto it = std::find_if(thresholds.begin(), thresholds.end(),
          [=](std::size_t t) -> bool { return low < t && high >= t; });
      if (it != thresholds.end()) {
        JNIEnv* env = cudf::jni::get_jni_env(jvm);
        env->CallVoidMethod(handler_obj, callback_method, current_total);
        if (env->ExceptionCheck()) {
          throw std::runtime_error("onAllocThreshold handler threw an exception");
        }
      }
    }
  }

  void* do_allocate(std::size_t num_bytes, cudaStream_t stream) override {
    void* result;
    while (true) {
      try {
        result = resource->allocate(num_bytes, stream);
        break;
      } catch (std::bad_alloc const& e) {
        if (!on_alloc_fail(num_bytes)) {
          throw e;
        }
      }
    }

    std::size_t total_before;
    std::size_t total_after;
    {
      std::lock_guard<std::mutex> lock(size_map_mutex);
      total_before = total_allocated;
      total_allocated += num_bytes;
      total_after = total_allocated;
      size_map[result] = num_bytes;
    }
    try {
      check_for_threshold_callback(total_before, total_after, alloc_thresholds,
          on_alloc_threshold_method, "onAllocThreshold", total_after);
    } catch (std::exception e) {
      // Free the allocation as app will think the exception means the memory was not allocated.
      resource->deallocate(result, num_bytes, stream);
      throw e;
    }

    return result;
  }

  void do_deallocate(void* p, std::size_t size, cudaStream_t stream) override {
    resource->deallocate(p, size, stream);

    std::size_t total_before;
    std::size_t total_after;
    {
      std::lock_guard<std::mutex> lock(size_map_mutex);
      total_before = total_allocated;
      // TODO: size can't be trusted until rmm::alloc and rmm::free are removed,
      //       see https://github.com/rapidsai/rmm/issues/302
      auto it = size_map.find(p);
      if (it != size_map.end()) {
        total_allocated -= it->second;
        size_map.erase(it);
      } else {
        // Untracked size, may be due to allocation that occurred before handler was installed.
      }
      total_after = total_allocated;
    }

    check_for_threshold_callback(total_after, total_before, dealloc_thresholds,
        on_dealloc_threshold_method, "onDeallocThreshold", total_after);
  }

  std::pair<size_t, size_t> do_get_mem_info(cudaStream_t stream) const override {
    return resource->get_mem_info(stream);
  }

  bool supports_streams() const noexcept override {
    return resource->supports_streams();
  }
};

std::unique_ptr<java_event_handler_memory_resource> Java_memory_resource{};

void set_java_device_memory_resource(
    JNIEnv* env,
    jobject handler_obj,
    jlongArray jalloc_thresholds,
    jlongArray jdealloc_thresholds) {
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
    Java_memory_resource.reset(new java_event_handler_memory_resource(
        env, handler_obj, jalloc_thresholds, jdealloc_thresholds, resource));
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
                                                                  jlong pool_size) {
  try {
    if (rmmIsInitialized(nullptr)) {
      JNI_THROW_NEW(env, "java/lang/IllegalStateException", "RMM already initialized", );
    }
    rmmOptions_t opts;
    opts.allocation_mode = static_cast<rmmAllocationMode_t>(allocation_mode);
    opts.enable_logging = enable_logging == JNI_TRUE;
    opts.initial_pool_size = pool_size;
    JNI_RMM_TRY(env, , rmmInitialize(&opts));
  } CATCH_STD(env, )
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_Rmm_isInitializedInternal(JNIEnv *env, jclass clazz) {
  try {
    return rmmIsInitialized(nullptr);
  } CATCH_STD(env, false)
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_shutdown(JNIEnv *env, jclass clazz) {
  try {
    set_java_device_memory_resource(env, nullptr, nullptr, nullptr);
    JNI_RMM_TRY(env, , rmmFinalize());
  } CATCH_STD(env, )
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_alloc(JNIEnv *env, jclass clazz, jlong size,
                                                      jlong stream) {
  try {
    void *ret = 0;
    cudaStream_t c_stream = reinterpret_cast<cudaStream_t>(stream);
    JNI_RMM_TRY(env, 0, RMM_ALLOC(&ret, size, c_stream));
    return (jlong)ret;
  } CATCH_STD(env, 0)
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_free(JNIEnv *env, jclass clazz, jlong ptr,
                                                    jlong stream) {
  try {
    void *cptr = reinterpret_cast<void *>(ptr);
    cudaStream_t c_stream = reinterpret_cast<cudaStream_t>(stream);
    JNI_RMM_TRY(env, , RMM_FREE(cptr, c_stream));
  } CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_freeDeviceBuffer(JNIEnv *env, jclass clazz,
                                                                jlong ptr) {
  rmm::device_buffer *cptr = reinterpret_cast<rmm::device_buffer *>(ptr);
  delete cptr;
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_setEventHandlerInternal(
    JNIEnv* env,
    jclass,
    jobject handler_obj,
    jlongArray jalloc_thresholds,
    jlongArray jdealloc_thresholds) {
  try {
    set_java_device_memory_resource(env, handler_obj, jalloc_thresholds, jdealloc_thresholds);
  } CATCH_STD(env, )
}

JNIEXPORT jstring JNICALL Java_ai_rapids_cudf_Rmm_getLog(JNIEnv *env, jclass clazz, jlong size,
                                                         jlong stream) {
  try {
    size_t amount = rmmLogSize();
    std::unique_ptr<char> buffer(new char[amount]);
    JNI_RMM_TRY(env, nullptr, rmmGetLog(buffer.get(), amount));
    return env->NewStringUTF(buffer.get());
  } CATCH_STD(env, nullptr)
}

}
