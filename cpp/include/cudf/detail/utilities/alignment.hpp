/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
