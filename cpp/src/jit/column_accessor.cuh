
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/column_device_view_base.hpp>
#include <cudf/types.hpp>

#include <cuda/std/cstddef>

namespace cudf {
namespace jit {

template <int32_t Index,
          typename Column,
          typename Element,
          typename OptionalElement,
          bool AsScalar,
          bool MayBeNullable>
struct column_accessor {
  static constexpr int32_t index        = Index;
  using column_type                     = Column;
  using element_type                    = Element;
  using optional_element_type           = OptionalElement;
  static constexpr bool as_scalar       = AsScalar;
  static constexpr bool may_be_nullable = MayBeNullable;

  static __device__ auto& get(detail::column_device_view_base const* cols)
  {
    return reinterpret_cast<column_type const&>(cols[index]);
  }

  static __device__ element_type element(detail::column_device_view_base const* cols, size_type row)
  {
    auto& c = get(cols);

    if constexpr (AsScalar) {
      return c.template element<element_type>(0);
    } else {
      return c.template element<element_type>(row);
    }
  }

  static __device__ bool is_null(detail::column_device_view_base const* cols, size_type row)
  {
    if constexpr (!MayBeNullable) {
      return false;
    } else {
      auto& c = get(cols);

      if constexpr (AsScalar) {
        return c.is_null(0);
      } else {
        return c.is_null(row);
      }
    }
  }

  static __device__ bool is_valid(detail::column_device_view_base const* cols, size_type row)
  {
    if constexpr (!MayBeNullable) {
      return true;
    } else {
      auto& c = get(cols);

      if constexpr (AsScalar) {
        return c.is_valid(0);
      } else {
        return c.is_valid(row);
      }
    }
  }

  static __device__ optional_element_type
  nullable_element(detail::column_device_view_base const* cols, size_type row)
  {
    auto& c = get(cols);

    if constexpr (!MayBeNullable) {
      return c.template element<element_type>(row);
    } else {
      if constexpr (AsScalar) {
        return c.template nullable_element<element_type>(0);
      } else {
        return c.template nullable_element<element_type>(row);
      }
    }
  }

  static __device__ void set_null_word(detail::column_device_view_base const* cols,
                                       size_type index,
                                       bitmask_type word)
  {
    auto& c = get(cols);

    if constexpr (!MayBeNullable) {
      return;
    } else {
      auto* mask = c.null_mask();

      if (mask == nullptr) { return; }

      mask[index] = word;
    }
  }

  static __device__ void assign(detail::column_device_view_base const* cols,
                                size_type row,
                                element_type value)
    requires(!AsScalar)
  {
    get(cols).template assign<element_type>(row, value);
  }
};

}  // namespace jit
}  // namespace cudf
