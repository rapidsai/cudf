/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/jit/lto/types.cuh>

// TODO: scope variables should be aligned to avoid uncoalesced reads/writes

namespace CUDF_LTO_EXPORT cudf {
namespace lto {
namespace scope {

using args = void* const __restrict__* __restrict__;

template <int ScopeIndex,
          typename ColumnType /* = column_device_view_core, mutable_column_device_view_core, span,
                                 optional_span ... */
          ,
          typename T /* = int, float, fixed_point, string_view ... */,
          bool IsScalar,
          bool IsNullable>
struct column {
  static constexpr bool IS_SCALAR   = IsScalar;
  static constexpr bool IS_NULLABLE = IsNullable;

  using Type = T;
  using Arg  = ColumnType const* __restrict__;

  static __device__ decltype(auto) element(args scope, size_type i)
  {
    auto p     = static_cast<Arg>(scope[ScopeIndex]);
    auto index = IsScalar ? 0 : i;

    if constexpr (!IsNullable) {
      return p->template element<T>(index);
    } else {
      return p->template nullable_element<T>(index);
    }
  }

  static __device__ void assign(args scope, size_type i, T value)
  {
    auto p     = static_cast<Arg>(scope[ScopeIndex]);
    auto index = IsScalar ? 0 : i;

    p->template assign<T>(index, value);
  }

  static __device__ auto* null_mask(args scope)
  {
    auto p = static_cast<Arg>(scope[ScopeIndex]);
    return p->null_mask();
  }

  static __device__ bool is_null(args scope, size_type i)
  {
    if constexpr (!IsNullable) { return false; }

    auto p     = static_cast<Arg>(scope[ScopeIndex]);
    auto index = IsScalar ? 0 : i;

    return p->is_null(index);
  }

  static __device__ bool is_valid(args scope, size_type i) { return !is_null(scope, i); }
};

template <int ScopeIndex>
struct user_data {
  using Arg = void* __restrict__;

  static __device__ decltype(auto) element(args scope, [[maybe_unused]] size_type i)
  {
    return static_cast<Arg>(scope[ScopeIndex]);
  }
};

}  // namespace scope
}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
