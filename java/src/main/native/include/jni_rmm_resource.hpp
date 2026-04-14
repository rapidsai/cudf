/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <jni.h>

namespace cudf::jni {

/**
 * @brief A type-erasing wrapper for device memory resources.
 *
 * This wrapper stores a CCCL any_resource internally, allowing the JNI layer
 * to pass resources around as opaque handles without knowing their concrete types.
 */
class jni_resource_wrapper {
 public:
  template <typename Resource>
  explicit jni_resource_wrapper(Resource&& resource) : resource_(std::forward<Resource>(resource))
  {
  }

  rmm::device_async_resource_ref ref() { return rmm::device_async_resource_ref{resource_}; }

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
inline rmm::device_async_resource_ref get_resource_ref(jlong handle)
{
  auto wrapper = reinterpret_cast<jni_resource_wrapper*>(handle);
  return wrapper->ref();
}

// Helper to delete wrapper
inline void delete_jni_resource(jlong handle)
{
  delete reinterpret_cast<jni_resource_wrapper*>(handle);
}

}  // namespace cudf::jni
