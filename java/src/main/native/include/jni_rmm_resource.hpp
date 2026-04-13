/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <jni.h>

#include <any>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace cudf::jni {

// Type for callback that creates resource ref from raw pointer
using resource_ref_getter = std::function<rmm::device_async_resource_ref(void*)>;

// Map for tracking resource handles that can't use any_resource
// (e.g., tracking_resource_adaptor uses shared_resource which has issues with any_resource)
inline std::mutex& get_raw_resource_map_mutex()
{
  static std::mutex mtx;
  return mtx;
}

struct raw_resource_entry {
  void* resource;
  resource_ref_getter get_ref;
};

inline std::unordered_map<jlong, raw_resource_entry>& get_raw_resource_map()
{
  static std::unordered_map<jlong, raw_resource_entry> map;
  return map;
}

/**
 * @brief A type-erasing wrapper for device memory resources.
 *
 * This wrapper stores a CCCL any_resource internally, allowing the JNI layer
 * to pass resources around as opaque handles without knowing their concrete types.
 */
class jni_resource_wrapper {
 public:
  template <typename Resource>
  explicit jni_resource_wrapper(Resource&& resource)
    : resource_(std::forward<Resource>(resource))
  {
  }

  rmm::device_async_resource_ref ref() { return rmm::device_async_resource_ref{resource_}; }

  template <typename T>
  T* get_concrete()
  {
    return std::any_cast<T>(&resource_);
  }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> resource_;
};

// Helper to create a wrapper and return as jlong
template <typename Resource>
jlong make_jni_resource(Resource&& resource)
{
  auto wrapper = new jni_resource_wrapper(std::forward<Resource>(resource));
  return reinterpret_cast<jlong>(wrapper);
}

// Helper to get resource ref from jlong handle
// First checks if it's a raw resource (in the map), otherwise unwraps as jni_resource_wrapper
inline rmm::device_async_resource_ref get_resource_ref(jlong handle)
{
  {
    std::lock_guard<std::mutex> lock(get_raw_resource_map_mutex());
    auto& map = get_raw_resource_map();
    auto it = map.find(handle);
    if (it != map.end()) {
      return it->second.get_ref(it->second.resource);
    }
  }
  auto wrapper = reinterpret_cast<jni_resource_wrapper*>(handle);
  return wrapper->ref();
}

// Helper to delete wrapper
inline void delete_jni_resource(jlong handle)
{
  delete reinterpret_cast<jni_resource_wrapper*>(handle);
}

}  // namespace cudf::jni
