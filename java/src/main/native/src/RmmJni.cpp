/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "jni_rmm_resource.hpp"

#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/aligned_resource_adaptor.hpp>
#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <atomic>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace {

using cudf::jni::delete_jni_resource;
using cudf::jni::get_resource_ref;
using cudf::jni::make_jni_resource;

constexpr char const* RMM_EXCEPTION_CLASS = "ai/rapids/cudf/RmmException";

/**
 * @brief Implementation class for tracking resource adaptor.
 * This class is not copyable due to atomic/mutex members.
 * Stores upstream as device_async_resource_ref (non-owning).
 * The JNI layer ensures the upstream resource outlives this adaptor.
 */
class tracking_resource_adaptor_impl {
 public:
  tracking_resource_adaptor_impl(rmm::device_async_resource_ref upstream,
                                 std::size_t size_alignment)
    : upstream_{upstream}, size_align{size_alignment}
  {
  }

  rmm::device_async_resource_ref get_wrapped_resource() { return upstream_; }

  std::size_t get_total_allocated() { return total_allocated.load(); }

  std::size_t get_max_total_allocated() { return max_total_allocated; }

  void reset_scoped_max_total_allocated(std::size_t initial_value)
  {
    std::scoped_lock lock(max_total_allocated_mutex);
    scoped_allocated           = initial_value;
    scoped_max_total_allocated = initial_value;
  }

  std::size_t get_scoped_max_total_allocated()
  {
    std::scoped_lock lock(max_total_allocated_mutex);
    return scoped_max_total_allocated;
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t num_bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto const result = upstream_.allocate(stream, num_bytes, size_align);
    if (result) {
      total_allocated += num_bytes;
      scoped_allocated += num_bytes;
      std::scoped_lock lock(max_total_allocated_mutex);
      max_total_allocated        = std::max(total_allocated.load(), max_total_allocated);
      scoped_max_total_allocated = std::max(scoped_allocated.load(), scoped_max_total_allocated);
    }
    return result;
  }

  void deallocate(cuda::stream_ref stream,
                  void* p,
                  std::size_t size,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    upstream_.deallocate(stream, p, size, size_align);
    if (p) {
      total_allocated -= size;
      scoped_allocated -= size;
    }
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  bool operator==(tracking_resource_adaptor_impl const& other) const noexcept
  {
    return this == &other;
  }

  friend void get_property(tracking_resource_adaptor_impl const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

 private:
  rmm::device_async_resource_ref upstream_;
  std::size_t const size_align;
  std::atomic_size_t total_allocated{0};
  std::size_t max_total_allocated{0};
  std::atomic_long scoped_allocated{0};
  long scoped_max_total_allocated{0};
  std::mutex max_total_allocated_mutex;
};
static_assert(cuda::mr::resource_with<tracking_resource_adaptor_impl, cuda::mr::device_accessible>);

/**
 * @brief Tracking resource adaptor with reference-counted shared ownership.
 * This wrapper holds a shared_ptr to the impl and forwards resource operations.
 * It satisfies the CCCL resource concept and is copyable for use with device_async_resource_ref.
 */
class tracking_resource_adaptor {
 public:
  tracking_resource_adaptor(rmm::device_async_resource_ref upstream, std::size_t size_alignment)
    : impl_{std::make_shared<tracking_resource_adaptor_impl>(upstream, size_alignment)}
  {
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return impl_->allocate(stream, bytes, alignment);
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    impl_->deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return impl_->allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    impl_->deallocate_sync(ptr, bytes, alignment);
  }

  bool operator==(tracking_resource_adaptor const& other) const noexcept
  {
    return impl_ == other.impl_;
  }

  friend void get_property(tracking_resource_adaptor const&, cuda::mr::device_accessible) noexcept
  {
  }

  rmm::device_async_resource_ref get_wrapped_resource() { return impl_->get_wrapped_resource(); }
  std::size_t get_total_allocated() { return impl_->get_total_allocated(); }
  std::size_t get_max_total_allocated() { return impl_->get_max_total_allocated(); }
  void reset_scoped_max_total_allocated(std::size_t initial_value)
  {
    impl_->reset_scoped_max_total_allocated(initial_value);
  }
  std::size_t get_scoped_max_total_allocated() { return impl_->get_scoped_max_total_allocated(); }

 private:
  std::shared_ptr<tracking_resource_adaptor_impl> impl_;
};
static_assert(cuda::mr::resource_with<tracking_resource_adaptor, cuda::mr::device_accessible>);

/**
 * @brief Implementation class for java event handler memory resource.
 * This class holds all the non-copyable JNI state and is wrapped in a shared_ptr.
 */
class java_event_handler_memory_resource_impl {
 public:
  java_event_handler_memory_resource_impl(JNIEnv* env,
                                          jobject jhandler,
                                          jlongArray jalloc_thresholds,
                                          jlongArray jdealloc_thresholds,
                                          rmm::device_async_resource_ref upstream,
                                          tracking_resource_adaptor* tracker)
    : upstream_{upstream}, tracker_(tracker)
  {
    if (env->GetJavaVM(&jvm) < 0) { throw std::runtime_error("GetJavaVM failed"); }

    jclass cls = env->GetObjectClass(jhandler);
    if (cls == nullptr) { throw cudf::jni::jni_exception("class not found"); }
    on_alloc_fail_method = env->GetMethodID(cls, "onAllocFailure", "(JI)Z");
    if (on_alloc_fail_method == nullptr) {
      use_old_alloc_fail_interface = true;
      on_alloc_fail_method         = env->GetMethodID(cls, "onAllocFailure", "(J)Z");
      if (on_alloc_fail_method == nullptr) {
        throw cudf::jni::jni_exception("onAllocFailure method");
      }
    } else {
      use_old_alloc_fail_interface = false;
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

    handler_obj = cudf::jni::add_global_ref(env, jhandler);
  }

  virtual ~java_event_handler_memory_resource_impl()
  {
    JNIEnv* env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) == JNI_OK) {
      handler_obj = cudf::jni::del_global_ref(env, handler_obj);
    }
    handler_obj = nullptr;
  }

  rmm::device_async_resource_ref get_wrapped_resource() { return upstream_; }

  virtual void* allocate(cuda::stream_ref stream,
                         std::size_t num_bytes,
                         std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    std::size_t total_before;
    void* result;
    int retry_count = 0;
    while (true) {
      try {
        total_before = tracker_->get_total_allocated();
        result       = upstream_.allocate(stream, num_bytes);
        break;
      } catch (rmm::out_of_memory const& e) {
        if (!on_alloc_fail(num_bytes, retry_count++)) { throw; }
      }
    }
    auto total_after = tracker_->get_total_allocated();

    try {
      check_for_threshold_callback(total_before,
                                   total_after,
                                   alloc_thresholds,
                                   on_alloc_threshold_method,
                                   "onAllocThreshold",
                                   total_after);
    } catch (std::exception const& e) {
      upstream_.deallocate(stream, result, num_bytes);
      throw;
    }

    return result;
  }

  virtual void deallocate(cuda::stream_ref stream,
                          void* p,
                          std::size_t size,
                          std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    auto total_before = tracker_->get_total_allocated();
    upstream_.deallocate(stream, p, size);
    auto total_after = tracker_->get_total_allocated();
    check_for_threshold_callback(total_after,
                                 total_before,
                                 dealloc_thresholds,
                                 on_dealloc_threshold_method,
                                 "onDeallocThreshold",
                                 total_after);
  }

 protected:
  rmm::device_async_resource_ref upstream_;
  tracking_resource_adaptor* const tracker_;
  jmethodID on_alloc_fail_method;
  bool use_old_alloc_fail_interface;
  jmethodID on_alloc_threshold_method;
  jmethodID on_dealloc_threshold_method;
  std::vector<std::size_t> alloc_thresholds{};
  std::vector<std::size_t> dealloc_thresholds{};
  JavaVM* jvm;
  jobject handler_obj;

  static void update_thresholds(JNIEnv* env,
                                std::vector<std::size_t>& thresholds,
                                jlongArray from_java)
  {
    thresholds.clear();
    if (from_java != nullptr) {
      cudf::jni::native_jlongArray jvalues(env, from_java);
      thresholds.insert(thresholds.end(), jvalues.data(), jvalues.data() + jvalues.size());
    } else {
      thresholds.push_back(std::numeric_limits<std::size_t>::max());
    }
  }

  bool on_alloc_fail(std::size_t num_bytes, int retry_count)
  {
    JNIEnv* env     = cudf::jni::get_jni_env(jvm);
    jboolean result = false;
    if (!use_old_alloc_fail_interface) {
      result = env->CallBooleanMethod(handler_obj,
                                      on_alloc_fail_method,
                                      static_cast<jlong>(num_bytes),
                                      static_cast<jint>(retry_count));
    } else {
      result =
        env->CallBooleanMethod(handler_obj, on_alloc_fail_method, static_cast<jlong>(num_bytes));
    }
    if (env->ExceptionCheck()) {
      throw std::runtime_error("onAllocFailure handler threw an exception");
    }
    return result;
  }

  void check_for_threshold_callback(std::size_t low,
                                    std::size_t high,
                                    std::vector<std::size_t> const& thresholds,
                                    jmethodID callback_method,
                                    char const* callback_name,
                                    std::size_t current_total)
  {
    if (high >= thresholds.front() && low < thresholds.back()) {
      auto it = std::find_if(thresholds.begin(), thresholds.end(), [=](std::size_t t) -> bool {
        return low < t && high >= t;
      });
      if (it != thresholds.end()) {
        JNIEnv* env = cudf::jni::get_jni_env(jvm);
        env->CallVoidMethod(handler_obj, callback_method, current_total);
      }
    }
  }
};

/**
 * @brief Debug implementation that adds allocation/deallocation callbacks.
 */
class java_debug_event_handler_memory_resource_impl final
  : public java_event_handler_memory_resource_impl {
 public:
  java_debug_event_handler_memory_resource_impl(JNIEnv* env,
                                                jobject jhandler,
                                                jlongArray jalloc_thresholds,
                                                jlongArray jdealloc_thresholds,
                                                rmm::device_async_resource_ref upstream,
                                                tracking_resource_adaptor* tracker)
    : java_event_handler_memory_resource_impl(
        env, jhandler, jalloc_thresholds, jdealloc_thresholds, upstream, tracker)
  {
    jclass cls = env->GetObjectClass(jhandler);
    if (cls == nullptr) { throw cudf::jni::jni_exception("class not found"); }

    on_allocated_method = env->GetMethodID(cls, "onAllocated", "(J)V");
    if (on_allocated_method == nullptr) { throw cudf::jni::jni_exception("onAllocated method"); }

    on_deallocated_method = env->GetMethodID(cls, "onDeallocated", "(J)V");
    if (on_deallocated_method == nullptr) {
      throw cudf::jni::jni_exception("onDeallocated method");
    }
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t num_bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) override
  {
    void* result = java_event_handler_memory_resource_impl::allocate(stream, num_bytes, alignment);
    on_allocated_callback(num_bytes);
    return result;
  }

  void deallocate(cuda::stream_ref stream,
                  void* p,
                  std::size_t size,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept override
  {
    java_event_handler_memory_resource_impl::deallocate(stream, p, size, alignment);
    on_deallocated_callback(size);
  }

 private:
  jmethodID on_allocated_method;
  jmethodID on_deallocated_method;

  void on_allocated_callback(std::size_t num_bytes)
  {
    JNIEnv* env = cudf::jni::get_jni_env(jvm);
    env->CallVoidMethod(handler_obj, on_allocated_method, num_bytes);
    if (env->ExceptionCheck()) {
      throw std::runtime_error("onAllocated handler threw an exception");
    }
  }

  void on_deallocated_callback(std::size_t size)
  {
    JNIEnv* env = cudf::jni::get_jni_env(jvm);
    env->CallVoidMethod(handler_obj, on_deallocated_method, size);
  }
};

/**
 * @brief Copyable wrapper for java event handler that holds shared_ptr to impl.
 * Satisfies CCCL resource concept for use with device_async_resource_ref.
 */
class java_event_handler_memory_resource {
 public:
  java_event_handler_memory_resource(JNIEnv* env,
                                     jobject jhandler,
                                     jlongArray jalloc_thresholds,
                                     jlongArray jdealloc_thresholds,
                                     rmm::device_async_resource_ref upstream,
                                     tracking_resource_adaptor* tracker,
                                     bool enable_debug)
    : impl_(enable_debug
              ? std::make_shared<java_debug_event_handler_memory_resource_impl>(
                  env, jhandler, jalloc_thresholds, jdealloc_thresholds, upstream, tracker)
              : std::make_shared<java_event_handler_memory_resource_impl>(
                  env, jhandler, jalloc_thresholds, jdealloc_thresholds, upstream, tracker))
  {
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return impl_->allocate(stream, bytes, alignment);
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    impl_->deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  bool operator==(java_event_handler_memory_resource const& other) const noexcept
  {
    return impl_ == other.impl_;
  }

  friend void get_property(java_event_handler_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

  rmm::device_async_resource_ref get_wrapped_resource() { return impl_->get_wrapped_resource(); }

 private:
  std::shared_ptr<java_event_handler_memory_resource_impl> impl_;
};
static_assert(
  cuda::mr::resource_with<java_event_handler_memory_resource, cuda::mr::device_accessible>);

inline auto& prior_cudf_pinned_mr()
{
  static rmm::host_device_async_resource_ref _prior_cudf_pinned_mr =
    cudf::get_pinned_memory_resource();
  return _prior_cudf_pinned_mr;
}

/**
 * This is a pinned fallback memory resource that will try to allocate from the provided
 * `pool` resource if the requested size is less than or equal to the pool size, otherwise it
 * will fall back to the prior resource used by cuDF `prior_cudf_pinned_mr`.
 *
 * We detect whether a pointer to free is inside of the pool by checking its address (see
 * constructor).
 *
 * Most of this comes directly from `pinned_host_memory_resource` in RMM.
 */
class pinned_fallback_host_memory_resource {
 public:
  pinned_fallback_host_memory_resource(rmm::mr::pool_memory_resource* pool_) : pool{pool_}
  {
    auto pool_size = pool->pool_size();
    pool_begin     = pool->allocate_sync(pool_size);
    pool_end       = static_cast<void*>(static_cast<uint8_t*>(pool_begin) + pool_size);
    pool->deallocate_sync(pool_begin, pool_size);
  }

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    if (bytes <= pool->pool_size()) {
      try {
        return pool->allocate(stream, bytes, alignment);
      } catch (...) {
        // If the pool is exhausted, fall back to the upstream memory resource
      }
    }
    return prior_cudf_pinned_mr().allocate(stream, bytes);
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    if (bytes <= pool->pool_size() && ptr >= pool_begin && ptr < pool_end) {
      pool->deallocate(stream, ptr, bytes, alignment);
    } else {
      prior_cudf_pinned_mr().deallocate(stream, ptr, bytes);
    }
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  bool operator==(pinned_fallback_host_memory_resource const& other) const noexcept
  {
    return pool == other.pool;
  }

  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `pinned_host_memory_resource` provides device accessible memory
   */
  friend void get_property(pinned_fallback_host_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `cuda::mr::host_accessible` property
   *
   * This property declares that a `pinned_host_memory_resource` provides host accessible memory
   */
  friend void get_property(pinned_fallback_host_memory_resource const&,
                           cuda::mr::host_accessible) noexcept
  {
  }

 private:
  rmm::mr::pool_memory_resource* pool;
  void* pool_begin;
  void* pool_end;
};

// carryover from RMM pinned_host_memory_resource
static_assert(cuda::mr::resource_with<pinned_fallback_host_memory_resource,
                                      cuda::mr::device_accessible,
                                      cuda::mr::host_accessible>);

// we set this to our fallback resource if we have set it.
std::unique_ptr<pinned_fallback_host_memory_resource> pinned_fallback_mr;

}  // anonymous namespace

extern "C" {

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_initDefaultCudaDevice(JNIEnv* env, jclass clazz)
{
  // make sure the CUDA device is setup in the context
  cudaError_t cuda_status = cudaFree(0);
  cudf::jni::jni_cuda_check(env, cuda_status);
  int device_id;
  cuda_status = cudaGetDevice(&device_id);
  cudf::jni::jni_cuda_check(env, cuda_status);
  // Now that RMM has successfully initialized, setup all threads calling
  // cudf to use the same device RMM is using.
  cudf::jni::set_cudf_device(device_id);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_cleanupDefaultCudaDevice(JNIEnv* env, jclass clazz)
{
  cudf::jni::set_cudf_device(cudaInvalidDeviceId);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_allocInternal(JNIEnv* env,
                                                              jclass clazz,
                                                              jlong size,
                                                              jlong stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    auto c_stream = rmm::cuda_stream_view(reinterpret_cast<cudaStream_t>(stream));
    void* ret     = mr.allocate(c_stream, size);
    return reinterpret_cast<jlong>(ret);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_Rmm_free(JNIEnv* env, jclass clazz, jlong ptr, jlong size, jlong stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
    void* cptr                        = reinterpret_cast<void*>(ptr);
    auto c_stream = rmm::cuda_stream_view(reinterpret_cast<cudaStream_t>(stream));
    mr.deallocate(c_stream, cptr, size);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_freeDeviceBuffer(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    rmm::device_buffer* cptr = reinterpret_cast<rmm::device_buffer*>(ptr);
    delete cptr;
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_allocCudaInternal(JNIEnv* env,
                                                                  jclass clazz,
                                                                  jlong size,
                                                                  jlong stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    void* ptr{nullptr};
    RMM_CUDA_TRY_ALLOC(cudaMalloc(&ptr, size), size);
    return reinterpret_cast<jlong>(ptr);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_Rmm_freeCuda(JNIEnv* env, jclass clazz, jlong ptr, jlong size, jlong stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    void* cptr = reinterpret_cast<void*>(ptr);
    RMM_ASSERT_CUDA_SUCCESS(cudaFree(cptr));
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newCudaMemoryResource(JNIEnv* env, jclass clazz)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return make_jni_resource(rmm::mr::cuda_memory_resource{});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseCudaMemoryResource(JNIEnv* env,
                                                                         jclass clazz,
                                                                         jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newManagedMemoryResource(JNIEnv* env, jclass clazz)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return make_jni_resource(rmm::mr::managed_memory_resource{});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseManagedMemoryResource(JNIEnv* env,
                                                                            jclass clazz,
                                                                            jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newPoolMemoryResource(
  JNIEnv* env, jclass clazz, jlong child, jlong init, jlong max)
{
  JNI_NULL_CHECK(env, child, "child is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto upstream = get_resource_ref(child);
    return make_jni_resource(rmm::mr::pool_memory_resource{
      upstream, static_cast<std::size_t>(init), static_cast<std::size_t>(max)});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releasePoolMemoryResource(JNIEnv* env,
                                                                         jclass clazz,
                                                                         jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newArenaMemoryResource(
  JNIEnv* env, jclass clazz, jlong child, jlong init, jboolean dump_on_oom)
{
  JNI_NULL_CHECK(env, child, "child is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto upstream = get_resource_ref(child);
    return make_jni_resource(rmm::mr::arena_memory_resource{
      upstream, static_cast<std::size_t>(init), static_cast<bool>(dump_on_oom)});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseArenaMemoryResource(JNIEnv* env,
                                                                          jclass clazz,
                                                                          jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newCudaAsyncMemoryResource(
  JNIEnv* env, jclass clazz, jlong init, jlong release, jboolean fabric)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto handle_type =
      fabric ? std::optional{rmm::mr::cuda_async_memory_resource::allocation_handle_type::fabric}
             : std::nullopt;

    return make_jni_resource(rmm::mr::cuda_async_memory_resource{init, release, handle_type});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseCudaAsyncMemoryResource(JNIEnv* env,
                                                                              jclass clazz,
                                                                              jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newLimitingResourceAdaptor(
  JNIEnv* env, jclass clazz, jlong child, jlong limit, jlong align)
{
  JNI_NULL_CHECK(env, child, "child is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto upstream = get_resource_ref(child);
    return make_jni_resource(rmm::mr::limiting_resource_adaptor{
      upstream, static_cast<std::size_t>(limit), static_cast<std::size_t>(align)});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseLimitingResourceAdaptor(JNIEnv* env,
                                                                              jclass clazz,
                                                                              jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newLoggingResourceAdaptor(
  JNIEnv* env, jclass clazz, jlong child, jint type, jstring jpath, jboolean auto_flush)
{
  JNI_NULL_CHECK(env, child, "child is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto upstream = get_resource_ref(child);
    switch (type) {
      case 1:  // File
      {
        cudf::jni::native_jstring path(env, jpath);
        return make_jni_resource(
          rmm::mr::logging_resource_adaptor{upstream, path.get(), static_cast<bool>(auto_flush)});
      }
      case 2:  // stdout
        return make_jni_resource(
          rmm::mr::logging_resource_adaptor{upstream, std::cout, static_cast<bool>(auto_flush)});
      case 3:  // stderr
        return make_jni_resource(
          rmm::mr::logging_resource_adaptor{upstream, std::cerr, static_cast<bool>(auto_flush)});
      default: throw std::logic_error("unsupported logging location type");
    }
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseLoggingResourceAdaptor(JNIEnv* env,
                                                                             jclass clazz,
                                                                             jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

// Map to store tracking adaptors for metrics access.
// tracking_resource_adaptor is copyable (via shared_ptr to impl), so copies share state.
std::mutex tracking_adaptor_map_mutex;
std::unordered_map<jlong, tracking_resource_adaptor> tracking_adaptor_map;

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newTrackingResourceAdaptor(JNIEnv* env,
                                                                           jclass clazz,
                                                                           jlong child,
                                                                           jlong align)
{
  JNI_NULL_CHECK(env, child, "child is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto upstream = get_resource_ref(child);
    auto adaptor  = tracking_resource_adaptor(upstream, static_cast<std::size_t>(align));
    auto handle   = make_jni_resource(adaptor);
    // Store a copy in map for metrics access (copies share impl via shared_ptr)
    {
      std::lock_guard<std::mutex> lock(tracking_adaptor_map_mutex);
      tracking_adaptor_map.emplace(handle, adaptor);
    }
    return handle;
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseTrackingResourceAdaptor(JNIEnv* env,
                                                                              jclass clazz,
                                                                              jlong ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    {
      std::lock_guard<std::mutex> lock(tracking_adaptor_map_mutex);
      tracking_adaptor_map.erase(ptr);
    }
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

// Helper to get tracking adaptor from the map
inline tracking_resource_adaptor& get_tracking_adaptor(jlong handle)
{
  std::lock_guard<std::mutex> lock(tracking_adaptor_map_mutex);
  auto it = tracking_adaptor_map.find(handle);
  if (it == tracking_adaptor_map.end()) {
    throw std::runtime_error("tracking adaptor not found for handle");
  }
  return it->second;
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_nativeGetTotalBytesAllocated(JNIEnv* env,
                                                                             jclass clazz,
                                                                             jlong ptr)
{
  JNI_NULL_CHECK(env, ptr, "adaptor is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto& mr = get_tracking_adaptor(ptr);
    return mr.get_total_allocated();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_nativeGetMaxTotalBytesAllocated(JNIEnv* env,
                                                                                jclass clazz,
                                                                                jlong ptr)
{
  JNI_NULL_CHECK(env, ptr, "adaptor is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto& mr = get_tracking_adaptor(ptr);
    return mr.get_max_total_allocated();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_nativeResetScopedMaxTotalBytesAllocated(JNIEnv* env,
                                                                                       jclass clazz,
                                                                                       jlong ptr,
                                                                                       jlong init)
{
  JNI_NULL_CHECK(env, ptr, "adaptor is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto& mr = get_tracking_adaptor(ptr);
    mr.reset_scoped_max_total_allocated(init);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_nativeGetScopedMaxTotalBytesAllocated(JNIEnv* env,
                                                                                      jclass clazz,
                                                                                      jlong ptr)
{
  JNI_NULL_CHECK(env, ptr, "adaptor is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto& mr = get_tracking_adaptor(ptr);
    return mr.get_scoped_max_total_allocated();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_Rmm_newEventHandlerResourceAdaptor(JNIEnv* env,
                                                       jclass,
                                                       jlong child,
                                                       jlong tracker,
                                                       jobject handler_obj,
                                                       jlongArray jalloc_thresholds,
                                                       jlongArray jdealloc_thresholds,
                                                       jboolean enable_debug)
{
  JNI_NULL_CHECK(env, child, "child is null", 0);
  JNI_NULL_CHECK(env, tracker, "tracker is null", 0);
  JNI_TRY
  {
    auto upstream = get_resource_ref(child);
    auto& t       = get_tracking_adaptor(tracker);
    return make_jni_resource(java_event_handler_memory_resource(
      env, handler_obj, jalloc_thresholds, jdealloc_thresholds, upstream, &t, enable_debug));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releaseEventHandlerResourceAdaptor(
  JNIEnv* env, jclass clazz, jlong ptr, jboolean enable_debug)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete_jni_resource(ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_setCurrentDeviceResourceInternal(JNIEnv* env,
                                                                                jclass clazz,
                                                                                jlong new_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto ref = get_resource_ref(new_handle);
    cudf::set_current_device_resource_ref(ref);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_newPinnedPoolMemoryResource(JNIEnv* env,
                                                                            jclass clazz,
                                                                            jlong init,
                                                                            jlong max)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto pool =
      new rmm::mr::pool_memory_resource(rmm::mr::pinned_host_memory_resource{}, init, max);
    return reinterpret_cast<jlong>(pool);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_setCudfPinnedPoolMemoryResource(JNIEnv* env,
                                                                               jclass clazz,
                                                                               jlong pool_ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto pool = reinterpret_cast<rmm::mr::pool_memory_resource*>(pool_ptr);
    // create a pinned fallback pool that will allocate pinned memory
    // if the regular pinned pool is exhausted
    pinned_fallback_mr.reset(new pinned_fallback_host_memory_resource(pool));
    prior_cudf_pinned_mr() = cudf::set_pinned_memory_resource(*pinned_fallback_mr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_releasePinnedPoolMemoryResource(JNIEnv* env,
                                                                               jclass clazz,
                                                                               jlong pool_ptr)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    // set the cuio host memory resource to what it was before, or the same
    // if we didn't overwrite it with setCudfPinnedPoolMemoryResource
    cudf::set_pinned_memory_resource(prior_cudf_pinned_mr());
    pinned_fallback_mr.reset();
    delete reinterpret_cast<rmm::mr::pool_memory_resource*>(pool_ptr);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_allocFromPinnedPool(JNIEnv* env,
                                                                    jclass clazz,
                                                                    jlong pool_ptr,
                                                                    jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto pool = reinterpret_cast<rmm::mr::pool_memory_resource*>(pool_ptr);
    void* ret = pool->allocate(cudf::get_default_stream(), size, rmm::CUDA_ALLOCATION_ALIGNMENT);
    return reinterpret_cast<jlong>(ret);
  }
  JNI_CATCH_BEGIN(env, 0)
  catch (...) { return -1; }  // Catch and suppress all exceptions.
  // The return value of -1 indicates that the allocation failed.
  // This is different from the return value of 0, which indicates that the allocation succeeded
  // but the returned pointer is null (such cases can be due to allocating 0 bytes).
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_freeFromPinnedPool(
  JNIEnv* env, jclass clazz, jlong pool_ptr, jlong ptr, jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto pool  = reinterpret_cast<rmm::mr::pool_memory_resource*>(pool_ptr);
    void* cptr = reinterpret_cast<void*>(ptr);
    pool->deallocate(cudf::get_default_stream(), cptr, size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  JNI_CATCH(env, );
}

// only for tests
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Rmm_allocFromFallbackPinnedPool(JNIEnv* env,
                                                                            jclass clazz,
                                                                            jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    void* ret = cudf::get_pinned_memory_resource().allocate(cudf::get_default_stream(), size);
    return reinterpret_cast<jlong>(ret);
  }
  JNI_CATCH(env, 0);
}

// only for tests
JNIEXPORT void JNICALL Java_ai_rapids_cudf_Rmm_freeFromFallbackPinnedPool(JNIEnv* env,
                                                                          jclass clazz,
                                                                          jlong ptr,
                                                                          jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    void* cptr = reinterpret_cast<void*>(ptr);
    cudf::get_pinned_memory_resource().deallocate(cudf::get_default_stream(), cptr, size);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_Rmm_configureDefaultCudfPinnedPoolSizeImpl(
  JNIEnv* env, jclass clazz, jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return cudf::config_default_pinned_memory_resource(cudf::pinned_mr_options{size});
  }
  JNI_CATCH(env, false);
}
}
