/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/row_ir/opcode.hpp>
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
                                  void* __restrict__ user_data,
                                  column_device_view_core const* __restrict__ input_cols,
                                  mutable_column_device_view_core const* __restrict__ output_cols,
                                  int32_t* __restrict__ max_error,
                                  errc* __restrict__ row_errors)
{
  // TODO: ensure block size is a multiple of warp size for correct warp-synchronous behavior
  auto start        = detail::grid_1d::global_thread_id();
  auto stride       = detail::grid_1d::grid_stride();
  auto thread_error = errc::SUCCESS;

  for (auto row = start; row < row_size; row += stride) {
    auto operation = [&]<typename Args>(Args args) {
      // TODO: static assert invocable
      auto func = [&](auto... a) {
        if constexpr (!cuda::std::is_void_v<decltype(GENERIC_TRANSFORM_OP(a...))>) {
          return GENERIC_TRANSFORM_OP(a...);
        } else {
          GENERIC_TRANSFORM_OP(a...);
          return errc::SUCCESS;
        }
      };

      if constexpr (has_user_data) {
        return cuda::std::apply(func, cuda::std::tuple_cat(cuda::std::tuple{user_data, row}, args));
      } else {
        return cuda::std::apply(func, args);
      }
    };

    if constexpr (is_null_aware == null_aware::NO) {
      if (stencil != nullptr && !bit_is_set(stencil, row)) {
        if (row_errors != nullptr) { row_errors[row] = errc::SUCCESS; }
        continue;
      }

      auto ins = InputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::element(input_cols, row)...}; });

      auto outs = OutputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::output_arg(output_cols, row)...}; });

      auto out_ptrs =
        cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

      auto row_error = operation(cuda::std::tuple_cat(out_ptrs, ins));

      OutputAccessors::map([&]<typename... A>() {
        (A::assign(output_cols, row, cuda::std::get<A::index>(outs)), ...);
      });

      if (row_errors != nullptr) { row_errors[row] = row_error; }

      thread_error = cuda::std::max(thread_error, row_error);
    } else {
      auto active_mask = __ballot_sync(0xFFFF'FFFFU, row < row_size);

      auto ins = InputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::nullable_element(input_cols, row)...}; });

      auto outs = OutputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::null_output_arg(output_cols, row)...}; });

      auto out_ptrs =
        cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

      auto row_error = operation(cuda::std::tuple_cat(out_ptrs, ins));

      OutputAccessors::map([&]<typename... A>() {
        (A::assign(output_cols, row, *cuda::std::get<A::index>(outs)), ...);
        (warp_compact_validity<A>(
           active_mask, output_cols, row, cuda::std::get<A::index>(outs).has_value()),
         ...);
      });

      if (row_errors != nullptr) { row_errors[row] = row_error; }

      thread_error = cuda::std::max(thread_error, row_error);
    }
  }

  atomicMax(max_error, static_cast<int32_t>(thread_error));
}

}  // namespace jit
}  // namespace cudf
