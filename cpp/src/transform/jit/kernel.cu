/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/ast/detail/operator_functor.cuh>
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <jit/column_accessor.cuh>
#include <jit/column_device_view_wrappers.cuh>
#include <jit/sync.cuh>
#include <jit/type_list.cuh>

#pragma nv_hdrstop  // The above headers are used by the kernel below and need to be included before
                    // it. Each UDF will have a different operation-udf.hpp generated for it, so we
                    // need to put this pragma before including it to avoid PCH mismatch.

// clang-format off
// This header is an inlined header that defines the GENERIC_TRANSFORM_OP function. It is placed here
// so the symbols in the headers above can be used by it.
#include <cudf/detail/kernel-instance.hpp>
#include <cudf/detail/operation-udf.hpp>
// clang-format on

namespace cudf {
namespace jit {

/// @brief The generic transform kernel. Supports all types and nullability combinations.
template <null_aware is_null_aware,
          bool has_user_data,
          typename InputAccessors,
          typename OutputAccessors>
CUDF_KERNEL void transform_kernel(size_type row_size,
                                  bitmask_type const* __restrict__ stencil,
                                  bool stencil_has_nulls,
                                  void* __restrict__ user_data,
                                  column_device_view_core const* __restrict__ input_cols,
                                  mutable_column_device_view_core const* __restrict__ output_cols)
{
  auto start  = detail::grid_1d::global_thread_id();
  auto stride = detail::grid_1d::grid_stride();

  for (auto element_idx = start; element_idx < row_size; element_idx += stride) {
    if constexpr (is_null_aware == null_aware::NO) {
      if (stencil_has_nulls && !bit_is_set(stencil, element_idx)) { continue; }

      auto outs = OutputAccessors::map([&]<typename... A>() {
        return cuda::std::tuple{A::output_arg(output_cols, element_idx)...};
      });

      auto out_ptrs =
        cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

      auto inputs = InputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::element(input_cols, element_idx)...}; });

      if constexpr (has_user_data) {
        auto args =
          cuda::std::tuple_cat(cuda::std::tuple{user_data, element_idx}, out_ptrs, inputs);
        cuda::std::apply([](auto&&... a) { GENERIC_TRANSFORM_OP(a...); }, args);

      } else {
        // TODO: static assert invocable
        auto args = cuda::std::tuple_cat(out_ptrs, inputs);
        cuda::std::apply([](auto&&... a) { GENERIC_TRANSFORM_OP(a...); }, args);
      }

      OutputAccessors::map([&]<typename... A>() {
        (A::assign(output_cols, element_idx, cuda::std::get<A::index>(outs)), ...);
      });
    } else {
      bool is_valid[OutputAccessors::size];

      auto outs = OutputAccessors::map([&]<typename... A>() {
        return cuda::std::tuple{A::null_output_arg(output_cols, element_idx)...};
      });

      auto out_ptrs =
        cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

      auto inputs = InputAccessors::map([&]<typename... A>() {
        return cuda::std::tuple{A::nullable_element(input_cols, element_idx)...};
      });

      if constexpr (has_user_data) {
        auto args =
          cuda::std::tuple_cat(cuda::std::tuple{user_data, element_idx}, out_ptrs, inputs);
        cuda::std::apply([](auto&&... a) { GENERIC_TRANSFORM_OP(a...); }, args);

      } else {
        auto args = cuda::std::tuple_cat(out_ptrs, inputs);
        cuda::std::apply([](auto&&... a) { GENERIC_TRANSFORM_OP(a...); }, args);
      }

      OutputAccessors::map([&]<typename... A>() {
        (A::assign(output_cols, element_idx, *cuda::std::get<A::index>(outs)), ...);
        ((is_valid[A::index] = cuda::std::get<A::index>(outs).has_value()), ...);
      });

      OutputAccessors::map([&]<typename... A>() {
        auto active_mask = __ballot_sync(0xFFFF'FFFFU, element_idx < row_size);
        (warp_compact_validity<A>(active_mask, output_cols, element_idx, is_valid[A::index]), ...);
      });
    }
  }
}

}  // namespace jit
}  // namespace cudf

extern "C" __global__ void kernel(
  cudf::size_type row_size,
  cudf::bitmask_type const* __restrict__ stencil,
  bool stencil_has_nulls,
  void* __restrict__ user_data,
  cudf::column_device_view_core const* __restrict__ input_cols,
  cudf::mutable_column_device_view_core const* __restrict__ output_cols)
{
  KERNEL_INSTANCE(row_size, stencil, stencil_has_nulls, user_data, input_cols, output_cols);
}
