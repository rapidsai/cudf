/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.
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

#ifndef HASH_ALLOCATOR_CUH
#define HASH_ALLOCATOR_CUH

#include <new>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

template <class T>
struct managed_allocator {
  typedef T value_type;
  rmm::mr::device_memory_resource* mr = new rmm::mr::managed_memory_resource;

  managed_allocator() = default;

  template <class U>
  constexpr managed_allocator(const managed_allocator<U>&) noexcept
  {
  }

  T* allocate(std::size_t n, rmm::cuda_stream_view stream = rmm::cuda_stream_default) const
  {
    return static_cast<T*>(mr->allocate(n * sizeof(T), stream));
  }

  void deallocate(T* p,
                  std::size_t n,
                  rmm::cuda_stream_view stream = rmm::cuda_stream_default) const
  {
    mr->deallocate(p, n * sizeof(T), stream);
  }
};

template <class T, class U>
bool operator==(const managed_allocator<T>&, const managed_allocator<U>&)
{
  return true;
}
template <class T, class U>
bool operator!=(const managed_allocator<T>&, const managed_allocator<U>&)
{
  return false;
}

template <class T>
struct default_allocator {
  typedef T value_type;
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

  default_allocator() = default;

  template <class U>
  constexpr default_allocator(const default_allocator<U>&) noexcept
  {
  }

  T* allocate(std::size_t n, rmm::cuda_stream_view stream = rmm::cuda_stream_default) const
  {
    return static_cast<T*>(mr->allocate(n * sizeof(T), stream));
  }

  void deallocate(T* p,
                  std::size_t n,
                  rmm::cuda_stream_view stream = rmm::cuda_stream_default) const
  {
    mr->deallocate(p, n * sizeof(T), stream);
  }
};

template <class T, class U>
bool operator==(const default_allocator<T>&, const default_allocator<U>&)
{
  return true;
}
template <class T, class U>
bool operator!=(const default_allocator<T>&, const default_allocator<U>&)
{
  return false;
}

#endif
