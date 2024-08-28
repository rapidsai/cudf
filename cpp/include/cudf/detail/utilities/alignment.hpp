/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <memory>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Returns the aligned address for holding array of type T in pre-allocated memory.
 *
 * @tparam T The data type to align upon.
 *
 * @param destination pointer to pre-allocated contiguous storage to store type T.
 * @return Pointer of type T, aligned to alignment of type T.
 */
template <typename T>
T* align_ptr_for_type(void* destination)
{
  constexpr std::size_t bytes_needed{sizeof(T)};
  constexpr std::size_t alignment{alignof(T)};

  // pad the allocation for aligning the first pointer
  auto padded_bytes_needed = bytes_needed + (alignment - 1);
  // std::align captures last argument by reference and modifies it, but we don't want it modified
  return reinterpret_cast<T*>(
    std::align(alignment, bytes_needed, destination, padded_bytes_needed));
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
