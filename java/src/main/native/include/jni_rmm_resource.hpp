/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <jni.h>

namespace cudf::jni {

template <typename Resource>
jlong make_jni_resource(Resource&& resource)
{
  return reinterpret_cast<jlong>(new any_device_resource(std::forward<Resource>(resource)));
}

inline rmm::device_async_resource_ref get_resource_ref(jlong handle)
{
  return rmm::device_async_resource_ref{*reinterpret_cast<any_device_resource*>(handle)};
}

inline void delete_jni_resource(jlong handle)
{
  delete reinterpret_cast<any_device_resource*>(handle);
}

}  // namespace cudf::jni
