/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/element.cuh"
#include "jit/element_storage.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/jit/transform_operation.cuh>
#include <cudf/jit/type_tags.cuh>
#include <cudf/types.hpp>

#include <jit/column_device_view_wrappers.cuh>
#include <jit/element_storage.cuh>
#include <jit/sync.cuh>
#include <jit/type_list.cuh>

#pragma nv_hdrstop  // The above headers are used by the kernel below and need to be included before
                    // it. Each UDF will have a different operation-udf.hpp generated for it, so we
                    // need to put this pragma before including it to avoid PCH mismatch.

// clang-format off
#include <cudf/detail/kernel-instance.hpp>
// clang-format on

namespace cudf {

/**
 * @brief An untyped Generic Transform kernel. This is a catch-all kernel that can be used for any
 * transform operation, but is not expected to be the most performant. The intent is that this
 * kernel can be used for any operator and type combination with minimal specialization, but that
 * more specialized kernels can be generated for common cases (e.g. binary operations on
 * fixed-width types) that will be more performant.
 *
 */
template <bool null_aware, int max_element_size>
__device__ void untyped_lto_transform_kernel_shmem(
  size_type row_size,
  bitmask_type const* __restrict__ stencil,
  void* __restrict__ user_data,
  column_device_view_core const* __restrict__ input_cols,
  mutable_column_device_view_core const* __restrict__ output_cols,
  size_type num_inputs,
  size_type num_outputs,
  size_type const* __restrict__ input_strides)
{
  using storage_t = element_storage<null_aware, max_element_size>;

  extern __shared__ char shmem[];

  auto per_thread_bytes = sizeof(storage_t) * num_inputs + sizeof(storage_t) * num_outputs;
  auto shmem_iter       = shmem + per_thread_bytes * threadIdx.x;

  auto* __restrict__ input_storage = reinterpret_cast<storage_t*>(shmem_iter);
  shmem_iter += sizeof(storage_t) * num_inputs;
  auto* __restrict__ output_storage = reinterpret_cast<storage_t*>(shmem_iter);

  auto start  = detail::grid_1d::global_thread_id();
  auto stride = detail::grid_1d::grid_stride();

  for (auto i = start; i < row_size; i += stride) {
    if constexpr (!null_aware) {
      if (stencil != nullptr && !bit_is_set(stencil, i)) { continue; }
    }

    // used only for null-aware
    auto active_mask = null_aware ? __ballot_sync(0xFFFF'FFFFU, i < row_size) : 0xFFFF'FFFFU;

    for (size_type c = 0; c < num_inputs; c++) {
      load_element<null_aware, storage_t>(input_cols + c, i * input_strides[c], input_storage + c);
    }

    cudf_transform_operation(
      user_data, i, input_storage, sizeof(storage_t), output_storage, sizeof(storage_t));

    for (size_type c = 0; c < num_outputs; c++) {
      store_element<null_aware, storage_t>(output_cols + c, output_storage + c, i, active_mask);
    }
  }
}

template <bool null_aware, int max_element_size, int max_elements>
__device__ void untyped_lto_transform_kernel_stack(
  size_type row_size,
  bitmask_type const* __restrict__ stencil,
  void* __restrict__ user_data,
  column_device_view_core const* __restrict__ input_cols,
  mutable_column_device_view_core const* __restrict__ output_cols,
  size_type num_inputs,
  size_type num_outputs,
  size_type const* __restrict__ input_strides)
{
  using storage_t = element_storage<null_aware, max_element_size>;

  storage_t storage[max_elements];
  auto* __restrict__ input_storage  = storage;
  auto* __restrict__ output_storage = storage + num_inputs;

  auto start  = detail::grid_1d::global_thread_id();
  auto stride = detail::grid_1d::grid_stride();

  for (auto i = start; i < row_size; i += stride) {
    if constexpr (!null_aware) {
      if (stencil != nullptr && !bit_is_set(stencil, i)) { continue; }
    }

    // used only for null-aware
    auto active_mask = null_aware ? __ballot_sync(0xFFFF'FFFFU, i < row_size) : 0xFFFF'FFFFU;

    for (size_type c = 0; c < num_inputs; c++) {
      load_element<null_aware, storage_t>(input_cols + c, i * input_strides[c], input_storage + c);
    }

    cudf_transform_operation(
      user_data, i, input_storage, sizeof(storage_t), output_storage, sizeof(storage_t));

    for (size_type c = 0; c < num_outputs; c++) {
      store_element<null_aware, storage_t>(output_cols + c, output_storage + c, i, active_mask);
    }
  }
}

}  // namespace cudf

extern "C" __global__ void cudf_kernel_entry(
  cudf::size_type row_size,
  cudf::bitmask_type const* __restrict__ stencil,
  void* __restrict__ user_data,
  cudf::column_device_view_core const* __restrict__ input_cols,
  cudf::mutable_column_device_view_core const* __restrict__ output_cols,
  cudf::size_type num_inputs,
  cudf::size_type num_outputs,
  cudf::size_type const* __restrict__ input_strides)
{
  CUDF_KERNEL_INSTANCE(
    row_size, stencil, user_data, input_cols, output_cols, num_inputs, num_outputs, input_strides);
}
