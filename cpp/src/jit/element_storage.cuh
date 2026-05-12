/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/utilities/export.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {

struct [[nodiscard]] layout {
  std::size_t size      = 0;  //< Size in bytes of the layout
  std::size_t alignment = 1;  //< Non-zero power of 2 alignment

  constexpr layout unioned(layout const& other) const
  {
    return layout{.size      = cuda::std::max(size, other.size),
                  .alignment = cuda::std::max(alignment, other.alignment)};
  }
};

template <typename T>
inline constexpr layout layout_of = layout{.size = sizeof(T), .alignment = alignof(T)};

template <layout layout>
struct storage {
  alignas(layout.alignment) unsigned char data[layout.size];
};

template <typename Storage, typename ElementType>
inline constexpr bool storage_compatible =
  sizeof(ElementType) <= sizeof(Storage) && alignof(ElementType) <= alignof(Storage);

template <bool is_nullable, typename T>
using maybe_nullable = cuda::std::conditional_t<is_nullable, cuda::std::optional<T>, T>;

template <int max_element_size>
using nonnull_element_storage =
  storage<layout{.size = max_element_size, .alignment = alignof(cuda::std::max_align_t)}>;

template <int max_element_size>
using nullable_element_storage =
  storage<layout_of<cuda::std::optional<nonnull_element_storage<max_element_size>>>>;

template <bool null_aware, int max_element_size>
using element_storage = cuda::std::conditional_t<null_aware,
                                                 nullable_element_storage<max_element_size>,
                                                 nonnull_element_storage<max_element_size>>;

}  // namespace CUDF_EXPORT cudf
