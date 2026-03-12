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

#include <jit/column_accessor.cuh>
#include <jit/column_device_view_wrappers.cuh>
#include <jit/sync.cuh>
#include <jit/transform_udf.cuh>
#include <jit/type_list.cuh>

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

template <null_aware is_null_aware,
          bool has_stencil,
          bool has_user_data,
          typename Ins,
          typename Outs>
CUDF_KERNEL void transform_kernel(size_type row_size,
                                  bitmask_type const* stencil,
                                  void* user_data,
                                  column_device_view_core const* incols,
                                  mutable_column_device_view_core const* outcols)
{
  // ensure block size is a multiple of warp size for correct warp-synchronous behavior
  assert((blockDim.x & 31) == 0);
  auto start  = detail::grid_1d::global_thread_id();
  auto stride = detail::grid_1d::grid_stride();

  for (auto i = start; i < row_size; i += stride) {
    bool is_valid[Outs::size] = {};

    transform_udf<is_null_aware, has_stencil, has_user_data, Ins, Outs>::call(
      GENERIC_TRANSFORM_OP, stencil, user_data, incols, outcols, is_valid, i);

    if constexpr (is_null_aware == null_aware::YES) {
      Outs::map(
        [&]<typename... A>() { (warp_compact_validity<A>(outcols, i, is_valid[A::index]), ...); });
    }
  }
}

}  // namespace jit
}  // namespace cudf
