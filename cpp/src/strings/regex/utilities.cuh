/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <strings/regex/regex.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

namespace cudf {
namespace strings {
namespace detail {

constexpr auto regex_launch_kernel_block_size = 256;

template <typename ForEachFunction>
__global__ void for_each_kernel(ForEachFunction fn, reprog_device const d_prog, size_type size)
{
  extern __shared__ u_char shmem[];
  if (threadIdx.x == 0) { d_prog.store(shmem); }
  __syncthreads();
  auto const s_prog = reprog_device::load(d_prog, shmem);

  auto const thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  auto const stride     = s_prog.thread_count();
  for (auto idx = thread_idx; idx < size; idx += stride) {
    fn(idx, s_prog, thread_idx);
  }
}

template <typename ForEachFunction>
void launch_for_each_kernel(ForEachFunction fn,
                            reprog_device& d_prog,
                            size_type size,
                            rmm::cuda_stream_view stream)
{
  auto [buffer_size, thread_count] = d_prog.compute_strided_working_memory(size);

  auto d_buffer = rmm::device_buffer(buffer_size, stream);
  d_prog.set_working_memory(d_buffer.data(), thread_count);

  auto const shmem_size = d_prog.compute_shared_memory_size();
  cudf::detail::grid_1d grid{thread_count, regex_launch_kernel_block_size};
  for_each_kernel<<<grid.num_blocks, grid.num_threads_per_block, shmem_size, stream.value()>>>(
    fn, d_prog, size);
}

template <typename TransformFunction, typename OutputType>
__global__ void transform_kernel(TransformFunction fn,
                                 reprog_device const d_prog,
                                 OutputType* d_output,
                                 size_type size)
{
  extern __shared__ u_char shmem[];
  if (threadIdx.x == 0) { d_prog.store(shmem); }
  __syncthreads();
  auto const s_prog = reprog_device::load(d_prog, shmem);

  auto const thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  auto const stride     = s_prog.thread_count();
  for (auto idx = thread_idx; idx < size; idx += stride) {
    d_output[idx] = fn(idx, s_prog, thread_idx);
  }
}

template <typename TransformFunction, typename OutputType>
void launch_transform_kernel(TransformFunction fn,
                             reprog_device& d_prog,
                             OutputType* d_output,
                             size_type size,
                             rmm::cuda_stream_view stream)
{
  auto [buffer_size, thread_count] = d_prog.compute_strided_working_memory(size);

  auto d_buffer = rmm::device_buffer(buffer_size, stream);
  d_prog.set_working_memory(d_buffer.data(), thread_count);

  auto const shmem_size = d_prog.compute_shared_memory_size();
  cudf::detail::grid_1d grid{thread_count, regex_launch_kernel_block_size};
  transform_kernel<<<grid.num_blocks, grid.num_threads_per_block, shmem_size, stream.value()>>>(
    fn, d_prog, d_output, size);
}

template <typename SizeAndExecuteFunction>
auto make_strings_children(SizeAndExecuteFunction size_and_exec_fn,
                           reprog_device& d_prog,
                           size_type strings_count,
                           rmm::cuda_stream_view stream,
                           rmm::mr::device_memory_resource* mr)
{
  auto offsets = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto d_offsets             = offsets->mutable_view().template data<int32_t>();
  size_and_exec_fn.d_offsets = d_offsets;

  auto [buffer_size, thread_count] = d_prog.compute_strided_working_memory(strings_count);

  auto d_buffer = rmm::device_buffer(buffer_size, stream);
  d_prog.set_working_memory(d_buffer.data(), thread_count);
  auto const shmem_size = d_prog.compute_shared_memory_size();
  cudf::detail::grid_1d grid{thread_count, 256};

  // Compute the output size for each row
  if (strings_count > 0) {
    for_each_kernel<<<grid.num_blocks, grid.num_threads_per_block, shmem_size, stream.value()>>>(
      size_and_exec_fn, d_prog, strings_count);
  }

  // Convert sizes to offsets
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_offsets, d_offsets + strings_count + 1, d_offsets);

  // Now build the chars column
  auto const char_bytes = cudf::detail::get_value<int32_t>(offsets->view(), strings_count, stream);
  std::unique_ptr<column> chars = create_chars_child_column(char_bytes, stream, mr);
  if (char_bytes > 0) {
    size_and_exec_fn.d_chars = chars->mutable_view().template data<char>();
    for_each_kernel<<<grid.num_blocks, grid.num_threads_per_block, shmem_size, stream.value()>>>(
      size_and_exec_fn, d_prog, strings_count);
  }

  return std::make_pair(std::move(offsets), std::move(chars));
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
