/*
 * Copyright (c) 2017-2023, NVIDIA CORPORATION.
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

#pragma once

#include <new>

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

template <class T>
struct default_allocator {
  using value_type                    = T;
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

  default_allocator() = default;

  template <class U>
  constexpr default_allocator(default_allocator<U> const&) noexcept
  {
  }

  T* allocate(std::size_t n, rmm::cuda_stream_view stream = cudf::get_default_stream()) const
  {
    return static_cast<T*>(mr->allocate(n * sizeof(T), stream));
  }

  void deallocate(T* p,
                  std::size_t n,
                  rmm::cuda_stream_view stream = cudf::get_default_stream()) const
  {
    mr->deallocate(p, n * sizeof(T), stream);
  }
};

template <class T, class U>
bool operator==(default_allocator<T> const&, default_allocator<U> const&)
{
  return true;
}
template <class T, class U>
bool operator!=(default_allocator<T> const&, default_allocator<U> const&)
{
  return false;
}
