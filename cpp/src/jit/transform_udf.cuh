
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <cstddef>

namespace cudf {
namespace jit {

/**
 * @tparam is_null_aware = whether the UDF is null-aware or not. If `YES`, the UDF will receive
 * nullable elements for each input column, and the UDF will determine the nullability of the output
 * columns. If `NO`, the UDF will receive elements for each input column, and the UDF will not
 * handle null masks for the output columns.
 * @tparam has_stencil whether the operation has stencil values or not. If true, the kernel will
 * receive a bitmask pointer for the stencil and will skip rows where the stencil bit is not set.
 * @tparam has_user_data whether the UDF has user data or not. If true, the UDF will receive a
 * pointer to the user data as the first argument, followed by the row index.
 * @tparam Ins = type_list<column_accessor,  ...>
 * @tparam Outs = type_list<column_accessor, ...>
 */
template <null_aware is_null_aware,
          bool has_stencil,
          bool has_user_data,
          typename Ins,
          typename Outs>
struct transform_udf {
  template <typename Fn>
  static __device__ void call(Fn&& udf,
                              size_type index,
                              void* user_data,
                              column_device_view_core const* in_cols,
                              bitmask_type const* stencil,
                              mutable_column_device_view_core const* out_cols,
                              [[maybe_unused]] bool* is_valid)
    requires(is_null_aware == null_aware::NO)
  {
    if constexpr (has_stencil) {
      if (stencil != nullptr && !bit_is_set(stencil, index)) { return; }
    }

    auto outs =
      Outs::map([]<typename... A>() { return cuda::std::tuple{typename A::element_type{}...}; });

    auto out_ptrs =
      cuda::std::apply([&](auto&&... args) { return cuda::std::tuple{&args...}; }, outs);

    auto inputs =
      Ins::map([&]<typename... A>() { return cuda::std::tuple{A::element(in_cols, index)...}; });

    if constexpr (has_user_data) {
      auto args = cuda::std::tuple_cat(cuda::std::tuple{user_data, index}, out_ptrs, inputs);
      cuda::std::apply(udf, args);

    } else {
      auto args = cuda::std::tuple_cat(out_ptrs, inputs);
      cuda::std::apply(udf, args);
    }

    [&]<int... I>(cuda::std::integer_sequence<int, I...>) {
      (Outs::at<I>::assign(out_cols, index, cuda::std::get<I>(outs)), ...);
    }(Outs::indexed);
  }

  template <typename Fn>
  static __device__ void call(Fn&& udf,
                              size_type index,
                              void* user_data,
                              column_device_view_core const* in_cols,
                              [[maybe_unused]] bitmask_type const* stencil,
                              mutable_column_device_view_core const* out_cols,
                              bool* is_valid)
    requires(is_null_aware == null_aware::YES)
  {
    auto outs = Outs::map(
      []<typename... A>() { return cuda::std::tuple{typename A::optional_element_type{}...}; });

    auto out_ptrs =
      cuda::std::apply([&](auto&&... args) { return cuda::std::tuple{&args...}; }, outs);

    auto inputs = Ins::map(
      [&]<typename... A>() { return cuda::std::tuple{A::nullable_element(in_cols, index)...}; });

    if constexpr (has_user_data) {
      auto args = cuda::std::tuple_cat(cuda::std::tuple{user_data, index}, out_ptrs, inputs);
      cuda::std::apply(udf, args);

    } else {
      auto args = cuda::std::tuple_cat(out_ptrs, inputs);
      cuda::std::apply(udf, args);
    }

    [&]<int... I>(cuda::std::integer_sequence<int, I...>) {
      (Outs::at<I>::assign(out_cols, index, cuda::std::get<I>(outs)), ...);
      ((is_valid[I] = cuda::std::get<I>(outs).has_value()), ...);
    }(Ins::indexed);
  }
};

}  // namespace jit
}  // namespace cudf
