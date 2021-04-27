/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuco/detail/error.hpp>

namespace cuco {

template <typename T>
class cuda_allocator {
 public:
  using value_type = T;

  cuda_allocator() = default;

  template <class U>
  cuda_allocator(cuda_allocator<U> const&) noexcept
  { }

  value_type* allocate(std::size_t n)
  {
    value_type* p;
    CUCO_CUDA_TRY(cudaMalloc(&p, sizeof(value_type) * n));
    return p;
  }

  void deallocate(value_type* p, std::size_t) { CUCO_CUDA_TRY(cudaFree(p)); }
};

template <typename T, typename U>
bool operator==(cuda_allocator<T> const&, cuda_allocator<U> const&) noexcept
{
  return true;
}

template <typename T, typename U>
bool operator!=(cuda_allocator<T> const& lhs, cuda_allocator<U> const& rhs) noexcept
{
  return not(lhs == rhs);
}

}  // namespace cuco
