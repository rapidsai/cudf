/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/jit/lto/types.cuh>

// TODO: cuda std optional?
// TODO: how to convert to our optional?
// TODO: handle scalar
// TODO: scope variables should be aligned to avoid uncoalesced reads/writes
// TODO: how will expression evaluation work to handle multiple outputs?
// TODO: make fixed_point be able to assign values to columns
// TODO: is_null and is_valid

namespace cudf {
namespace lto {

using scope_type = void* const __restrict__* __restrict__;

template <size_type ScopeIndex,
          typename ColumnType /* = column_device_view_core/mutable_column_device_view_core */,
          typename T,
          bool IsScalar,
          bool IsNullable>
struct column {
  static constexpr bool IS_SCALAR   = IsScalar;
  static constexpr bool IS_NULLABLE = IsNullable;

  using Type = T;
  using Arg  = ColumnType const* __restrict__;

  static __device__ auto get(scope_type scope, size_type i)
  {
    auto p     = static_cast<Arg>(scope[ScopeIndex]);
    auto index = IsScalar ? 0 : i;

    if constexpr (!IsNullable) {
      return p->template element<T>(index);
    } else {
      return p->template nullable_element<T>(index);
    }
  }

  static __device__ void assign(scope_type scope, size_type i, auto const& value)
  {
    auto p     = static_cast<Arg>(scope[ScopeIndex]);
    auto index = IsScalar ? 0 : i;

    p->template assign<T>(index, value);
  }

  static __device__ auto* null_mask(scope_type scope)
  {
    auto p = static_cast<Arg>(scope[ScopeIndex]);
    return p->null_mask();
  }
};

template <size_type ScopeIndex,
          typename SpanType /* = device_optional_span */,
          typename T,
          bool IsScalar,
          bool IsNullable>
struct span {
  static constexpr bool IS_SCALAR   = IsScalar;
  static constexpr bool IS_NULLABLE = IsNullable;

  using Type = T;
  using Arg  = SpanType const* __restrict__;

  static __device__ auto get(scope_type scope, size_type i)
  {
    auto p     = static_cast<Arg>(scope[ScopeIndex]);
    auto index = IsScalar ? 0 : i;

    if constexpr (!IsNullable) {
      return p->template element<T>(index);
    } else {
      return p->template nullable_element<T>(index);
    }
  }

  static __device__ void assign(scope_type scope, size_type i, auto const& value)
  {
    auto p     = static_cast<Arg>(scope[ScopeIndex]);
    auto index = IsScalar ? 0 : i;

    p->template assign<T>(index, value);
  }

  static __device__ auto* null_mask(scope_type scope)
  {
    auto p = static_cast<Arg>(scope[ScopeIndex]);
    return p->null_mask();
  }
};

}  // namespace lto
}  // namespace cudf

template <int NumInputs,
          int NumOutputs,
          int UserDataIndex,
          typename InputGetters,
          typename OutputSetters>
struct element_operation {
  template <typename Operator>
  static __device__ void evaluate(scope_type scope, cudf::size_type i, Operator&& op)
  {
    if constexpr (UserDataIndex >= 0) {
      auto output_args;
      GENERIC_TRANSFORM_OP(user_data, i, &res, In::element(inputs, i)...);
    } else {
      GENERIC_TRANSFORM_OP(&res, In::element(inputs, i)...);
    }
  }
};
