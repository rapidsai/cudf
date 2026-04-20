/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/element.cuh"
#include "jit/element_storage.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>

#include <jit/column_device_view_wrappers.cuh>
#include <jit/sync.cuh>
#include <jit/type_list.cuh>

namespace cudf {

/// @brief The compute operation to perform on each element
/// @param user_data Pointer to user data passed to the kernel
/// @param element_index The index of the element to compute
/// @param inputs Pointer to the input elements for this operation; the caller guarantees the memory
/// layout and type of these elements based on the input column device views and input strides
/// @param input_stride The stride (in bytes) between consecutive input elements for a given input
/// column
/// @param outputs Pointer to the output elements for this operation; the caller guarantees the
/// memory layout and type of these elements based on the output column device views
/// @param output_stride The stride (in bytes) between consecutive output elements for a given
/// output column
/// @return An integer status code; the meaning of this code is defined by the caller and operator
/// implementation
extern "C" __device__ int operation(void* user_data,
                                    long int element_index,
                                    void const* inputs,
                                    int input_stride,
                                    void* outputs,
                                    int output_stride);

template <bool has_nulls, int max_element_size>
__device__ void compute_columns_kernel(size_type row_size,
                                       void* __restrict__ user_data,
                                       column_device_view const* __restrict__ input_cols,
                                       size_type num_inputs,
                                       mutable_column_device_view const* __restrict__ output_cols,
                                       size_type num_outputs,
                                       size_type const* __restrict__ input_strides)
{
  // 255 i32 registers per thread for GH200
  // 227 SHMEM bytes per thread for GH200

  using input_storage_t  = element_storage<has_nulls, max_element_size>;
  using output_storage_t = element_storage<has_nulls, max_element_size>;

  extern __shared__ char shmem[];

  auto shmem_iter                = shmem;
  input_storage_t* input_storage = reinterpret_cast<input_storage_t*>(shmem_iter);
  shmem_iter += sizeof(input_storage_t) * num_inputs;
  output_storage_t* output_storage = reinterpret_cast<output_storage_t*>(shmem_iter);

  auto start  = detail::grid_1d::global_thread_id();
  auto stride = detail::grid_1d::grid_stride();

  for (auto element_idx = start; element_idx < row_size; element_idx += stride) {
    auto active_mask = [&] __device__() {
      if constexpr (has_nulls) {
        return __ballot_sync(0xFFFF'FFFFU, element_idx < row_size);
      } else {
        return 0xFFFF'FFFFU;
      }
    }();

    for (int i = 0; i < num_inputs; i++) {
      load_element<has_nulls, input_storage_t>(
        input_cols + i, element_idx * input_strides[i], input_storage + i);
    }

    operation(user_data,
              element_idx,
              input_storage,
              sizeof(input_storage_t),
              output_storage,
              sizeof(output_storage_t));

    for (int i = 0; i < num_outputs; i++) {
      store_element<has_nulls, output_storage_t>(
        output_cols + i, output_storage + i, element_idx, active_mask);
    }
  }
}

}  // namespace cudf

extern "C" __global__ void kernel(cudf::size_type row_size,
                                  void* __restrict__ user_data,
                                  cudf::column_device_view const* __restrict__ input_cols,
                                  cudf::size_type num_inputs,
                                  cudf::mutable_column_device_view const* __restrict__ output_cols,
                                  cudf::size_type num_outputs,
                                  cudf::size_type const* __restrict__ input_strides)
{
  compute_columns_kernel<COMPUTE_COLUMNS_HAS_NULLS, COMPUTE_COLUMNS_MAX_ELEMENT_SIZE>(
    row_size, user_data, input_cols, num_inputs, output_cols, num_outputs, input_strides);
}
