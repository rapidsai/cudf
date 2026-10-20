/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/row_abi.cuh>
#include <cudf/utilities/bit.hpp>

#include <cuda/atomic>
extern "C" __global__ void cudf_kernel_entry(cudf::size_type rows,
                                             cudf::bitmask_type const* stencil,
                                             void* user_data,
                                             cudf::column_device_view_core const* inputs,
                                             cudf::mutable_column_device_view_core const* outputs,
                                             int32_t* error)
{
  auto const start     = cudf::detail::grid_1d::global_thread_id();
  auto const stride    = cudf::detail::grid_1d::grid_stride();
  int32_t thread_error = 0;
  for (auto index = start; index < rows; index += stride) {
    auto const row = static_cast<cudf::size_type>(index);
    if (stencil != nullptr && !cudf::bit_is_set(stencil, row)) { continue; }
    auto const result = cudf::row_abi::transform_operator(inputs, outputs, user_data, row);
    thread_error      = thread_error > result ? thread_error : result;
  }
  if (thread_error != 0) {
    cuda::atomic_ref<int32_t, cuda::thread_scope_device>{*error}.fetch_max(
      thread_error, cuda::memory_order_relaxed);
  }
}
