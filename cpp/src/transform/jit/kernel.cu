/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/ast/detail/operator_functor.cuh>
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/cstddef>

#include <jit/accessors.cuh>
#include <jit/span.cuh>
#include <jit/type_list.cuh>
#include <jit/udf_invoker.cuh>

#pragma nv_hdrstop  // The above headers are used by the kernel below and need to be included before
                    // it. Each UDF will have a different operation-udf.hpp generated for it, so we
                    // need to put this pragma before including it to avoid PCH mismatch.

// clang-format off
// This header is an inlined header that defines the GENERIC_FILTER_OP function. It is placed here
// so the symbols in the headers above can be used by it.
#include <cudf/detail/operation-udf.hpp>
// clang-format on

namespace cudf {
namespace jit {
namespace {
template <typename Out>
__device__ void warp_compact_validity(mutable_column_device_view_core const* outputs,
                                      size_type row,
                                      bool is_valid)
{
  if constexpr (!Out::may_be_nullable) {
    return;
  } else {
    auto null_word = __ballot_sync(0xFFFF'FFFFU, is_valid);
    if ((threadIdx.x % 32) == 0) { Out::set_null_word(outputs, row / 32, null_word); }
  }
}
}  // namespace

template <null_aware is_null_aware,
          bool has_stencil,
          bool has_user_data,
          typename Ins,
          typename Outs>
CUDF_KERNEL void transform_kernel(size_type num_rows,
                                  void* user_data,
                                  column_device_view_core const* inputs,
                                  bitmask_type const* stencil,
                                  mutable_column_device_view_core const* outputs)
{
  auto const start  = detail::grid_1d::global_thread_id();
  auto const stride = detail::grid_1d::grid_stride();

  for (auto row = start; row < num_rows; row += stride) {
    bool is_valid[Outs::size];

    transform_udf<is_null_aware, has_stencil, has_user_data, Ins, Outs>::call(
      GENERIC_TRANSFORM_OP, row, user_data, inputs, stencil, outputs, is_valid);

    Outs::map([&]<typename... Out> {
      (warp_compact_validity<Out>(outputs, row, is_valid[Out::index]), ...);
    });
  }
}

}  // namespace jit
}  // namespace cudf
