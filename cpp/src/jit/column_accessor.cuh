
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/types.hpp>

#include <cuda/std/cstddef>

namespace cudf {
namespace jit {

template <int32_t Index,
          typename Column,
          typename Element,
          typename OptionalElement,
          bool AsScalar,
          bool MayBeNullable,
          bool IsStringsOutput>
struct column_accessor {
  static constexpr int32_t index          = Index;
  using column_type                       = Column;
  using element_type                      = Element;
  using optional_element_type             = OptionalElement;
  static constexpr bool as_scalar         = AsScalar;
  static constexpr bool may_be_nullable   = MayBeNullable;
  static constexpr bool is_strings_output = IsStringsOutput;

  static __device__ constexpr size_type map_index(size_type row)
  {
    if constexpr (as_scalar) {
      return 0;
    } else {
      return row;
    }
  }

  template <typename T>
  static __device__ auto& column(T const* cols)
    requires(sizeof(T) == sizeof(column_type))
  {
    return reinterpret_cast<column_type const&>(cols[index]);
  }

  static __device__ element_type element(auto const* cols, size_type row)
  {
    return column(cols).template element<element_type>(map_index(row));
  }

  static __device__ bool is_null(auto const* cols, size_type row)
  {
    if constexpr (!may_be_nullable) {
      return false;
    } else {
      return column(cols).is_null(map_index(row));
    }
  }

  static __device__ bool is_valid(auto const* cols, size_type row)
  {
    if constexpr (!may_be_nullable) {
      return true;
    } else {
      return column(cols).is_valid(map_index(row));
    }
  }

  static __device__ optional_element_type nullable_element(auto const* cols, size_type row)
  {
    auto& c = column(cols);

    if constexpr (!may_be_nullable) {
      return c.template element<element_type>(map_index(row));
    } else {
      return c.template nullable_element<element_type>(map_index(row));
    }
  }

  static __device__ void set_null_mask_word(auto const* cols,
                                            size_type word_index,
                                            bitmask_type word)
    requires(!as_scalar)
  {
    if constexpr (!may_be_nullable) {
      return;
    } else {
      auto* mask = column(cols).null_mask();

      if (mask == nullptr) { return; }

      mask[word_index] = word;
    }
  }

  static __device__ void assign(auto const* cols, size_type row, element_type value)
    requires(!as_scalar)
  {
    column(cols).template assign<element_type>(row, value);
  }

  static __device__ element_type output_arg(auto const* cols, size_type row)
    requires(!as_scalar)
  {
    if constexpr (is_strings_output) {
      return element(cols, row);
    } else {
      return {};
    }
  }

  static __device__ optional_element_type null_output_arg(auto const* cols, size_type row)
    requires(!as_scalar)
  {
    if constexpr (is_strings_output) {
      return element(cols, row);
    } else {
      return {};
    }
  }
};

}  // namespace jit
}  // namespace cudf
