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

#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <unordered_map>

#include "cudf_jni_apis.hpp"

using rmm::mr::device_memory_resource;
using rmm::mr::logging_resource_adaptor;

namespace {

// Alignment to which the RMM memory resource will round allocation sizes
constexpr std::size_t RMM_ALLOC_SIZE_ALIGNMENT = 512;

constexpr char const *RMM_EXCEPTION_CLASS = "ai/rapids/cudf/RmmException";

/**
 * @brief Base class so we can template tracking_resource_adaptor but
 * still hold all instances of it without issues.
 */
class base_tracking_resource_adaptor : public device_memory_resource {
public:
  virtual std::size_t get_total_allocated() = 0;
};

/**
 * @brief An RMM device memory resource that delegates to another resource
 * while tracking the amount of memory allocated.
 *
 * @tparam Upstream Type of memory resource that will be wrapped.
 * @tparam size_align The size to which all allocation requests are
 * aligned. Must be a value >= 1.
 */
template <typename Upstream>
class tracking_resource_adaptor final : public base_tracking_resource_adaptor {
public:
  /**
   * @brief Constructs a new tracking resource adaptor that delegates to
   * `mr` for all allocation operations while tracking the amount of memory
   * allocated.
   *
   * @param mr The resource to use for memory allocation operations.
   * @param size_alignment The alignment to which the `mr` resource will
   * round up all memory allocation size requests.
   */
  tracking_resource_adaptor(Upstream *mr, std::size_t size_alignment)
      : resource{mr}, size_align{size_alignment} {}

  Upstream *get_wrapped_resource() { return resource; }

  std::size_t get_total_allocated() override {
    std::lock_guard<std::mutex> lock(size_map_mutex);
    return total_allocated;
  }

private:
  Upstream *const resource;
  std::size_t const size_align;
  std::size_t total_allocated{0};

  // map and associated lock to track memory sizes by address
  // TODO: This should be removed when rmm::alloc and rmm::free are removed and the size parameter
  //       for do_deallocate can be trusted. If map and mutex are removed then total_allocated
  //       should be updated to be atomic.
  std::unordered_map<void *, std::size_t> size_map{};
  std::mutex size_map_mutex{};

  void *do_allocate(std::size_t num_bytes, cudaStream_t stream) override {
    // adjust size of allocation based on specified size alignment
    num_bytes = (num_bytes + size_align - 1) / size_align * size_align;

    std::lock_guard<std::mutex> lock(size_map_mutex);

    auto result = resource->allocate(num_bytes, stream);
    if (result) {
      total_allocated += num_bytes;
      size_map[result] = num_bytes;
    }
    return result;
  }

  void do_deallocate(void *p, std::size_t size, cudaStream_t stream) override {
    std::lock_guard<std::mutex> lock(size_map_mutex);

    resource->deallocate(p, size, stream);

    if (p) {
      // TODO: size can't be trusted until rmm::alloc and rmm::free are removed,
      //       see https://github.com/rapidsai/rmm/issues/302
      auto it = size_map.find(p);
      if (it != size_map.end()) {
        total_allocated -= it->second;
        size_map.erase(it);
      } else {
        // Untracked size, may be an allocation from before resource was installed.
      }
    }
  }

  bool supports_get_mem_info() const noexcept override { return resource->supports_get_mem_info(); }

  std::pair<size_t, size_t> do_get_mem_info(cudaStream_t stream) const override {
    return resource->get_mem_info(stream);
  }

  bool supports_streams() const noexcept override { return resource->supports_streams(); }
};

template <typename Upstream>
tracking_resource_adaptor<Upstream> *make_tracking_adaptor(Upstream *upstream,
                                                           std::size_t size_alignment) {
  return new tracking_resource_adaptor<Upstream>{upstream, size_alignment};
}

std::unique_ptr<base_tracking_resource_adaptor> Tracking_memory_resource{};

/**
 * @brief Return the total amount of device memory allocated via RMM
 */
std::size_t get_total_bytes_allocated() {
  if (Tracking_memory_resource) {
    return Tracking_memory_resource->get_total_allocated();
  }
  return 0;
}

/**
 * @brief An RMM device memory resource adaptor that delegates to the wrapped resource
 * for most operations but will call Java to handle certain situations (e.g.: allocation failure).
 */
class java_event_handler_memory_resource final : public device_memory_resource {
public:
  java_event_handler_memory_resource(JNIEnv *env, jobject jhandler, jlongArray jalloc_thresholds,
                                     jlongArray jdealloc_thresholds,
                                     device_memory_resource *resource_to_wrap)
      : resource(resource_to_wrap) {
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
    JNIEnv *env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void **>(&env), cudf::jni::MINIMUM_JNI_VERSION) == JNI_OK) {
      env->DeleteGlobalRef(handler_obj);
    }
    handler_obj = nullptr;
  }

  device_memory_resource *get_wrapped_resource() { return resource; }

private:
  device_memory_resource *const resource;
  JavaVM *jvm;
  jobject handler_obj;
  jmethodID on_alloc_fail_method;
  jmethodID on_alloc_threshold_method;
  jmethodID on_dealloc_threshold_method;

  // sorted memory thresholds to trigger callbacks
  std::vector<std::size_t> alloc_thresholds{};
  std::vector<std::size_t> dealloc_thresholds{};

  static void update_thresholds(JNIEnv *env, std::vector<std::size_t> &thresholds,
                                jlongArray from_java) {
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

    JNIEnv *env = cudf::jni::get_jni_env(jvm);
    jboolean result =
        env->CallBooleanMethod(handler_obj, on_alloc_fail_method, static_cast<jlong>(num_bytes));
    if (env->ExceptionCheck()) {
      throw std::runtime_error("onAllocFailure handler threw an exception");
    }
    return result;
  }

  void check_for_threshold_callback(std::size_t low, std::size_t high,
                                    std::vector<std::size_t> const &thresholds,
                                    jmethodID callback_method, char const *callback_name,
                                    std::size_t current_total) {
    if (high >= thresholds.front() && low < thresholds.back()) {
      // could use binary search, but assumption is threshold count is very small
      auto it = std::find_if(thresholds.begin(), thresholds.end(),
                             [=](std::size_t t) -> bool { return low < t && high >= t; });
      if (it != thresholds.end()) {
        JNIEnv *env = cudf::jni::get_jni_env(jvm);
        env->CallVoidMethod(handler_obj, callback_method, current_total);
        if (env->ExceptionCheck()) {
          throw std::runtime_error("onAllocThreshold handler threw an exception");
        }
      }
    }
  }

  void *do_allocate(std::size_t num_bytes, cudaStream_t stream) override {
    std::size_t total_before;
    void *result;
    while (true) {
      try {
        total_before = get_total_bytes_allocated();
        result = resource->allocate(num_bytes, stream);
        break;
      } catch (std::bad_alloc const &e) {
        if (!on_alloc_fail(num_bytes)) {
          throw e;
        }
      }
    }
    auto total_after = get_total_bytes_allocated();

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

  void do_deallocate(void *p, std::size_t size, cudaStream_t stream) override {
    auto total_before = get_total_bytes_allocated();
    resource->deallocate(p, size, stream);
    auto total_after = get_total_bytes_allocated();
    check_for_threshold_callback(total_after, total_before, dealloc_thresholds,
                                 on_dealloc_threshold_method, "onDeallocThreshold", total_after);
  }

  bool supports_get_mem_info() const noexcept override { return resource->supports_get_mem_info(); }

  std::pair<size_t, size_t> do_get_mem_info(cudaStream_t stream) const override {
    return resource->get_mem_info(stream);
  }

  bool supports_streams() const noexcept override { return resource->supports_streams(); }
};

std::unique_ptr<java_event_handler_memory_resource> Java_memory_resource{};

void set_java_device_memory_resource(JNIEnv *env, jobject handler_obj, jlongArray jalloc_thresholds,
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

// Need to keep both separate so we can shut them down appropriately
std::unique_ptr<logging_resource_adaptor<base_tracking_resource_adaptor>> Logging_memory_resource{};
std::shared_ptr<device_memory_resource> Initialized_resource{};
} // anonymous namespace

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_initializeInternal(JNIEnv *env, jclass clazz,
                                                                  jint allocation_mode, jint log_to,
                                                                  jstring jpath, jlong pool_size,
                                                                  jlong max_pool_size) {
  try {
    // make sure the CUDA device is setup in the context
    cudaError_t cuda_status = cudaFree(0);
    cudf::jni::jni_cuda_check(env, cuda_status);
    int device_id;
    cuda_status = cudaGetDevice(&device_id);
    cudf::jni::jni_cuda_check(env, cuda_status);

    bool use_pool_alloc = allocation_mode & 1;
    bool use_managed_mem = allocation_mode & 2;
    if (use_pool_alloc) {
      std::size_t pool_limit = (max_pool_size > 0)
                                 ? static_cast<std::size_t>(max_pool_size)
                                 : std::numeric_limits<std::size_t>::max();
      if (use_managed_mem) {
        Initialized_resource = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
            std::make_shared<rmm::mr::managed_memory_resource>(), pool_size, pool_limit);
        auto wrapped = make_tracking_adaptor(Initialized_resource.get(), RMM_ALLOC_SIZE_ALIGNMENT);
        Tracking_memory_resource.reset(wrapped);
      } else {
        Initialized_resource = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
            std::make_shared<rmm::mr::cuda_memory_resource>(), pool_size, pool_limit);
        auto wrapped = make_tracking_adaptor(Initialized_resource.get(), RMM_ALLOC_SIZE_ALIGNMENT);
        Tracking_memory_resource.reset(wrapped);
      }
    } else if (use_managed_mem) {
      Initialized_resource = std::make_shared<rmm::mr::managed_memory_resource>();
      auto wrapped = make_tracking_adaptor(Initialized_resource.get(), RMM_ALLOC_SIZE_ALIGNMENT);
      Tracking_memory_resource.reset(wrapped);
    } else {
      Initialized_resource = std::make_shared<rmm::mr::cuda_memory_resource>();
      auto wrapped = make_tracking_adaptor(Initialized_resource.get(), RMM_ALLOC_SIZE_ALIGNMENT);
      Tracking_memory_resource.reset(wrapped);
    }
    auto resource = Tracking_memory_resource.get();
    rmm::mr::set_default_resource(resource);

    std::unique_ptr<logging_resource_adaptor<base_tracking_resource_adaptor>> log_result;
    switch (log_to) {
      case 1: // File
      {
        cudf::jni::native_jstring path(env, jpath);
        log_result.reset(new logging_resource_adaptor<base_tracking_resource_adaptor>(
            resource, path.get(), /*auto_flush=*/true));
      } break;
      case 2: // stdout
        log_result.reset(new logging_resource_adaptor<base_tracking_resource_adaptor>(
            resource, std::cout, /*auto_flush=*/true));
        break;
      case 3: // stderr
        log_result.reset(new logging_resource_adaptor<base_tracking_resource_adaptor>(
            resource, std::cerr, /*auto_flush=*/true));
        break;
    }

    if (log_result) {
      if (Logging_memory_resource) {
        JNI_THROW_NEW(env, RMM_EXCEPTION_CLASS, "Internal Error logging is double enabled", )
      }

      Logging_memory_resource = std::move(log_result);
      auto replaced_resource = rmm::mr::set_default_resource(Logging_memory_resource.get());
      if (resource != replaced_resource) {
        rmm::mr::set_default_resource(replaced_resource);
        Logging_memory_resource.reset(nullptr);
        JNI_THROW_NEW(env, RMM_EXCEPTION_CLASS,
                      "Concurrent modification detected while installing memory resource", );
      }
    }

    // Now that RMM has successfully initialized, setup all threads calling
    // cudf to use the same device RMM is using.
    cudf::jni::set_cudf_device(device_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_shutdownInternal(JNIEnv *env, jclass clazz) {
  try {
    cudf::jni::auto_set_device(env);
    set_java_device_memory_resource(env, nullptr, nullptr, nullptr);
    // Instead of trying to undo all of the adaptors that we added in reverse order
    // we just reset the base adaptor so the others will not be called any more
    // and then clean them up in really any order.  There should be no interaction with
    // RMM during this time anyways.
    Initialized_resource = std::make_shared<rmm::mr::cuda_memory_resource>();
    rmm::mr::set_default_resource(Initialized_resource.get());
    Logging_memory_resource.reset(nullptr);
    Tracking_memory_resource.reset(nullptr);
    cudf::jni::set_cudf_device(cudaInvalidDeviceId);
  }
  CATCH_STD(env, )
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_getTotalBytesAllocated(JNIEnv *env, jclass) {
  return get_total_bytes_allocated();
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_allocInternal(JNIEnv *env, jclass clazz, jlong size,
                                                              jlong stream) {
  try {
    cudf::jni::auto_set_device(env);
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource();
    cudaStream_t c_stream = reinterpret_cast<cudaStream_t>(stream);
    void *ret = mr->allocate(size, c_stream);
    return reinterpret_cast<jlong>(ret);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_free(JNIEnv *env, jclass clazz, jlong ptr,
                                                    jlong size, jlong stream) {
  try {
    cudf::jni::auto_set_device(env);
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource();
    void *cptr = reinterpret_cast<void *>(ptr);
    cudaStream_t c_stream = reinterpret_cast<cudaStream_t>(stream);
    mr->deallocate(cptr, size, c_stream);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_freeDeviceBuffer(JNIEnv *env, jclass clazz,
                                                                jlong ptr) {
  try {
    cudf::jni::auto_set_device(env);
    rmm::device_buffer *cptr = reinterpret_cast<rmm::device_buffer *>(ptr);
    delete cptr;
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_setEventHandlerInternal(
    JNIEnv *env, jclass, jobject handler_obj, jlongArray jalloc_thresholds,
    jlongArray jdealloc_thresholds) {
  try {
    set_java_device_memory_resource(env, handler_obj, jalloc_thresholds, jdealloc_thresholds);
  }
  CATCH_STD(env, )
}
}
