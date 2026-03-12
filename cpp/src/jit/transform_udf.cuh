
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
                              bitmask_type const* stencil,
                              void* user_data,
                              column_device_view_core const* incols,
                              mutable_column_device_view_core const* outcols,
                              [[maybe_unused]] bool* is_valid,
                              size_type element_idx)
    requires(is_null_aware == null_aware::NO)
  {
    if constexpr (has_stencil) {
      if (stencil != nullptr) {
        if (!bit_is_set(stencil, element_idx)) { return; }
      }
    }

    auto outs = Outs::map(
      [&]<typename... A>() { return cuda::std::tuple{A::output_arg(outcols, element_idx)...}; });

    auto out_ptrs =
      cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

    auto inputs = Ins::map(
      [&]<typename... A>() { return cuda::std::tuple{A::element(incols, element_idx)...}; });

    if constexpr (has_user_data) {
      auto args = cuda::std::tuple_cat(cuda::std::tuple{user_data, element_idx}, out_ptrs, inputs);
      cuda::std::apply(udf, args);

    } else {
      auto args = cuda::std::tuple_cat(out_ptrs, inputs);
      cuda::std::apply(udf, args);
    }

    Outs::map([&]<typename... A>() {
      (A::assign(outcols, element_idx, cuda::std::get<A::index>(outs)), ...);
    });
  }

  template <typename Fn>
  static __device__ void call(Fn&& udf,
                              [[maybe_unused]] bitmask_type const* stencil,
                              void* user_data,
                              column_device_view_core const* incols,
                              mutable_column_device_view_core const* outcols,
                              bool* is_valid,
                              size_type element_idx)
    requires(is_null_aware == null_aware::YES)
  {
    auto outs = Outs::map([&]<typename... A>() {
      return cuda::std::tuple{A::null_output_arg(outcols, element_idx)...};
    });

    auto out_ptrs =
      cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

    auto inputs = Ins::map([&]<typename... A>() {
      return cuda::std::tuple{A::nullable_element(incols, element_idx)...};
    });

    if constexpr (has_user_data) {
      auto args = cuda::std::tuple_cat(cuda::std::tuple{user_data, element_idx}, out_ptrs, inputs);
      cuda::std::apply(udf, args);

    } else {
      auto args = cuda::std::tuple_cat(out_ptrs, inputs);
      cuda::std::apply(udf, args);
    }

    Outs::map([&]<typename... A>() {
      (A::assign(outcols, element_idx, *cuda::std::get<A::index>(outs)), ...);
      ((is_valid[A::index] = cuda::std::get<A::index>(outs).has_value()), ...);
    });
  }
};

}  // namespace jit
}  // namespace cudf
