/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>

namespace cudf {
namespace detail {

// Compute the count of elements that pass the mask within each block
template <typename Filter, int block_size>
CUDF_KERNEL void compute_block_counts(cudf::size_type* __restrict__ block_counts,
                                      cudf::size_type size,
                                      cudf::size_type per_thread,
                                      Filter filter)
{
  int tid   = threadIdx.x + per_thread * block_size * blockIdx.x;
  int count = 0;

  for (int i = 0; i < per_thread; i++) {
    bool mask_true = (tid < size) && filter(tid);
    count += __syncthreads_count(mask_true);
    tid += block_size;
  }

  if (threadIdx.x == 0) block_counts[blockIdx.x] = count;
}

// Compute the exclusive prefix sum of each thread's mask value within each block
template <int block_size>
__device__ cudf::size_type block_scan_mask(bool mask_true, cudf::size_type& block_sum)
{
  int offset = 0;

  using BlockScan = cub::BlockScan<cudf::size_type, block_size>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  BlockScan(temp_storage).ExclusiveSum(mask_true, offset, block_sum);

  return offset;
}

// This kernel scatters data and validity mask of a column based on the
// scan of the boolean mask. The block offsets for the scan are already computed.
// Just compute the scan of the mask in each block and add it to the block's
// output offset. This is the output index of each element. Scattering
// the valid mask is not as easy, because each thread is only responsible for
// one bit. Warp-level processing (ballot) makes this simpler.
// To make scattering efficient, we "coalesce" the block's scattered data and
// valids in shared memory, and then write from shared memory to global memory
// in a contiguous manner.
// The has_validity template parameter specializes this kernel for the
// non-nullable case for performance without writing another kernel.
//
// Note: `filter` is not run on indices larger than the input column size
template <typename T, typename Filter, int block_size, bool has_validity>
__launch_bounds__(block_size) CUDF_KERNEL
  void scatter_kernel(cudf::mutable_column_device_view output_view,
                      cudf::size_type* output_null_count,
                      cudf::column_device_view input_view,
                      cudf::size_type const* __restrict__ block_offsets,
                      cudf::size_type size,
                      cudf::size_type per_thread,
                      Filter filter)
{
  T* __restrict__ output_data                   = output_view.data<T>();
  cudf::bitmask_type* __restrict__ output_valid = output_view.null_mask();
  static_assert(block_size <= 1024, "Maximum thread block size exceeded");

  int tid                      = threadIdx.x + per_thread * block_size * blockIdx.x;
  cudf::size_type block_offset = block_offsets[blockIdx.x];

  // one extra warp worth in case the block is not aligned
  __shared__ bool temp_valids[has_validity ? block_size + cudf::detail::warp_size : 1];
  __shared__ T temp_data[block_size];

  cudf::size_type warp_valid_counts{0};  // total valid sum over the `per_thread` loop below
  cudf::size_type block_sum = 0;         // count passing filter over the `per_thread` loop below

  // Note that since the maximum gridDim.x on all supported GPUs is as big as
  // cudf::size_type, this loop is sufficient to cover our maximum column size
  // regardless of the value of block_size and per_thread.
  for (int i = 0; i < per_thread; i++) {
    bool mask_true = (tid < size) && filter(tid);

    cudf::size_type tmp_block_sum = 0;
    // get output location using a scan of the mask result
    cudf::size_type const local_index = block_scan_mask<block_size>(mask_true, tmp_block_sum);
    block_sum += tmp_block_sum;

    if (has_validity) {
      temp_valids[threadIdx.x] = false;  // init shared memory
      if (threadIdx.x < cudf::detail::warp_size) temp_valids[block_size + threadIdx.x] = false;
      __syncthreads();  // wait for init
    }

    if (mask_true) {
      temp_data[local_index] = input_view.data<T>()[tid];  // scatter data to shared

      // scatter validity mask to shared memory
      if (has_validity and input_view.is_valid(tid)) {
        // determine aligned offset for this warp's output
        cudf::size_type const aligned_offset      = block_offset % cudf::detail::warp_size;
        temp_valids[local_index + aligned_offset] = true;
      }
    }

    __syncthreads();  // wait for shared data and validity mask to be complete

    // Copy output data coalesced from shared to global
    if (threadIdx.x < tmp_block_sum)
      output_data[block_offset + threadIdx.x] = temp_data[threadIdx.x];

    if (has_validity) {
      // Since the valid bools are contiguous in shared memory now, we can use
      // __popc to combine them into a single mask element.
      // Then, most mask elements can be directly copied from shared to global
      // memory. Only the first and last 32-bit mask elements of each block must
      // use an atomicOr, because these are where other blocks may overlap.

      constexpr int num_warps = block_size / cudf::detail::warp_size;
      // account for partial blocks with non-warp-aligned offsets
      int const last_index = tmp_block_sum + (block_offset % cudf::detail::warp_size) - 1;
      int const last_warp  = min(num_warps, last_index / cudf::detail::warp_size);
      int const wid        = threadIdx.x / cudf::detail::warp_size;
      int const lane       = threadIdx.x % cudf::detail::warp_size;

      cudf::size_type tmp_warp_valid_counts{0};

      if (tmp_block_sum > 0 && wid <= last_warp) {
        int valid_index = (block_offset / cudf::detail::warp_size) + wid;

        // compute the valid mask for this warp
        uint32_t valid_warp = __ballot_sync(0xffff'ffffu, temp_valids[threadIdx.x]);

        // Note the atomicOr's below assume that output_valid has been set to
        // all zero before the kernel
        if (lane == 0 && valid_warp != 0) {
          tmp_warp_valid_counts = __popc(valid_warp);
          if (wid > 0 && wid < last_warp)
            output_valid[valid_index] = valid_warp;
          else {
            cuda::atomic_ref<cudf::bitmask_type, cuda::thread_scope_device> ref{
              output_valid[valid_index]};
            ref.fetch_or(valid_warp, cuda::std::memory_order_relaxed);
          }
        }

        // if the block is full and not aligned then we have one more warp to cover
        if ((wid == 0) && (last_warp == num_warps)) {
          uint32_t valid_warp = __ballot_sync(0xffff'ffffu, temp_valids[block_size + threadIdx.x]);
          if (lane == 0 && valid_warp != 0) {
            tmp_warp_valid_counts += __popc(valid_warp);
            cuda::atomic_ref<cudf::bitmask_type, cuda::thread_scope_device> ref{
              output_valid[valid_index + num_warps]};
            ref.fetch_or(valid_warp, cuda::std::memory_order_relaxed);
          }
        }
      }
      warp_valid_counts += tmp_warp_valid_counts;
    }

    block_offset += tmp_block_sum;
    tid += block_size;
  }
  // Compute total null_count for this block and add it to global count
  constexpr cudf::size_type leader_lane{0};
  cudf::size_type block_valid_count =
    cudf::detail::single_lane_block_sum_reduce<block_size, leader_lane>(warp_valid_counts);

  if (threadIdx.x == 0) {  // one thread computes and adds to null count
    cuda::atomic_ref<size_type, cuda::thread_scope_device> ref{*output_null_count};
    ref.fetch_add(block_sum - block_valid_count, cuda::std::memory_order_relaxed);
  }
}

template <typename T, typename Enable = void>
struct DeviceType {
  using type = T;
};

template <typename T>
struct DeviceType<T, std::enable_if_t<cudf::is_timestamp<T>()>> {
  using type = typename T::rep;
};

template <typename T>
struct DeviceType<T, std::enable_if_t<cudf::is_fixed_point<T>()>> {
  using type = typename cudf::device_storage_type_t<T>;
};

// Dispatch functor which performs the scatter for fixed column types and gather for other
template <typename Filter, int block_size>
struct scatter_gather_functor {
  template <typename T, std::enable_if_t<cudf::is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::size_type const& output_size,
                                           cudf::size_type const* block_offsets,
                                           Filter filter,
                                           cudf::size_type per_thread,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    auto output_column =
      cudf::allocate_like(input, output_size, cudf::mask_allocation_policy::RETAIN, stream, mr);
    auto output = output_column->mutable_view();

    bool has_valid = input.nullable();

    using Type = typename DeviceType<T>::type;

    auto scatter = (has_valid) ? scatter_kernel<Type, Filter, block_size, true>
                               : scatter_kernel<Type, Filter, block_size, false>;

    cudf::detail::grid_1d grid{input.size(), block_size, per_thread};

    cudf::detail::device_scalar<cudf::size_type> null_count{0, stream};
    if (output.nullable()) {
      // Have to initialize the output mask to all zeros because we may update
      // it with atomicOr().
      CUDF_CUDA_TRY(cudaMemsetAsync(static_cast<void*>(output.null_mask()),
                                    0,
                                    cudf::bitmask_allocation_size_bytes(output.size()),
                                    stream.value()));
    }

    auto output_device_view = cudf::mutable_column_device_view::create(output, stream);
    auto input_device_view  = cudf::column_device_view::create(input, stream);
    scatter<<<grid.num_blocks, block_size, 0, stream.value()>>>(*output_device_view,
                                                                null_count.data(),
                                                                *input_device_view,
                                                                block_offsets,
                                                                input.size(),
                                                                per_thread,
                                                                filter);

    if (has_valid) { output_column->set_null_count(null_count.value(stream)); }
    return output_column;
  }

  template <typename T,
            std::enable_if_t<!cudf::is_fixed_width<T>() and !cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cudf::size_type const& output_size,
                                           cudf::size_type const*,
                                           Filter filter,
                                           cudf::size_type,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    rmm::device_uvector<cudf::size_type> indices(output_size, stream);

    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::counting_iterator<cudf::size_type>(0),
                    thrust::counting_iterator<cudf::size_type>(input.size()),
                    indices.begin(),
                    filter);

    auto output_table = cudf::detail::gather(cudf::table_view{{input}},
                                             indices,
                                             cudf::out_of_bounds_policy::DONT_CHECK,
                                             cudf::detail::negative_index_policy::NOT_ALLOWED,
                                             stream,
                                             mr);

    // There will be only one column
    return std::make_unique<cudf::column>(std::move(output_table->get_column(0)));
  }
};

/**
 * @brief Filters `input` using a Filter function object
 *
 * @p filter must be a functor or lambda with the following signature:
 * __device__ bool operator()(cudf::size_type i);
 * It will return true if element i of @p input should be copied,
 * false otherwise.
 *
 * @tparam Filter the filter functor type
 * @param[in] input The table_view to filter
 * @param[in] filter A function object that takes an index and returns a bool
 * @return unique_ptr<table> The table generated from filtered `input`.
 */
template <typename Filter>
std::unique_ptr<table> copy_if(table_view const& input,
                               Filter filter,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (0 == input.num_rows() || 0 == input.num_columns()) { return empty_like(input); }

  constexpr int block_size = 256;
  cudf::size_type per_thread =
    elements_per_thread(compute_block_counts<Filter, block_size>, input.num_rows(), block_size);
  cudf::detail::grid_1d grid{input.num_rows(), block_size, per_thread};

  // temp storage for block counts and offsets
  rmm::device_uvector<cudf::size_type> block_counts(grid.num_blocks, stream);
  rmm::device_uvector<cudf::size_type> block_offsets(grid.num_blocks + 1, stream);

  // 1. Find the count of elements in each block that "pass" the mask
  compute_block_counts<Filter, block_size><<<grid.num_blocks, block_size, 0, stream.value()>>>(
    block_counts.begin(), input.num_rows(), per_thread, filter);

  // initialize just the first element of block_offsets to 0 since the InclusiveSum below
  // starts at the second element.
  CUDF_CUDA_TRY(cudaMemsetAsync(block_offsets.begin(), 0, sizeof(cudf::size_type), stream.value()));

  // 2. Find the offset for each block's output using a scan of block counts
  if (grid.num_blocks > 1) {
    // Determine and allocate temporary device storage
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr,
                                  temp_storage_bytes,
                                  block_counts.begin(),
                                  block_offsets.begin() + 1,
                                  grid.num_blocks,
                                  stream.value());
    rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);

    // Run exclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  block_counts.begin(),
                                  block_offsets.begin() + 1,
                                  grid.num_blocks,
                                  stream.value());
  }

  // As it is InclusiveSum, last value in block_offsets will be output_size
  // unless num_blocks == 1, in which case output_size is just block_counts[0]
  cudf::size_type output_size{0};
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    &output_size,
    grid.num_blocks > 1 ? block_offsets.begin() + grid.num_blocks : block_counts.begin(),
    sizeof(cudf::size_type),
    cudaMemcpyDefault,
    stream.value()));

  stream.synchronize();

  if (output_size == input.num_rows()) {
    return std::make_unique<table>(input, stream, mr);
  } else if (output_size > 0) {
    std::vector<std::unique_ptr<column>> out_columns(input.num_columns());
    std::transform(input.begin(), input.end(), out_columns.begin(), [&](auto col_view) {
      return cudf::type_dispatcher(col_view.type(),
                                   scatter_gather_functor<Filter, block_size>{},
                                   col_view,
                                   output_size,
                                   block_offsets.begin(),
                                   filter,
                                   per_thread,
                                   stream,
                                   mr);
    });

    return std::make_unique<table>(std::move(out_columns));
  } else {
    return empty_like(input);
  }
}

}  // namespace detail
}  // namespace cudf
