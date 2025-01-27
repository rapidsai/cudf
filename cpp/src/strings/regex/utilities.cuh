/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "strings/regex/regex.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

#include <stdexcept>

namespace cudf {
namespace strings {
namespace detail {

constexpr auto regex_launch_kernel_block_size = 256;

template <typename ForEachFunction>
CUDF_KERNEL void for_each_kernel(ForEachFunction fn, reprog_device const d_prog, size_type size)
{
  extern __shared__ u_char shmem[];
  if (threadIdx.x == 0) { d_prog.store(shmem); }
  __syncthreads();
  auto const s_prog = reprog_device::load(d_prog, shmem);

  auto const thread_idx = cudf::detail::grid_1d::global_thread_id();
  auto const stride     = s_prog.thread_count();
  if (thread_idx < stride) {
    for (auto idx = thread_idx; idx < size; idx += stride) {
      fn(idx, s_prog, thread_idx);
    }
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
CUDF_KERNEL void transform_kernel(TransformFunction fn,
                                  reprog_device const d_prog,
                                  OutputType* d_output,
                                  size_type size)
{
  extern __shared__ u_char shmem[];
  if (threadIdx.x == 0) { d_prog.store(shmem); }
  __syncthreads();
  auto const s_prog = reprog_device::load(d_prog, shmem);

  auto const thread_idx = cudf::detail::grid_1d::global_thread_id();
  auto const stride     = s_prog.thread_count();
  if (thread_idx < stride) {
    for (auto idx = thread_idx; idx < size; idx += stride) {
      d_output[idx] = fn(idx, s_prog, thread_idx);
    }
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
                           rmm::device_async_resource_ref mr)
{
  auto output_sizes        = rmm::device_uvector<size_type>(strings_count, stream);
  size_and_exec_fn.d_sizes = output_sizes.data();

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
  // Convert the sizes to offsets
  auto [offsets, char_bytes] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);
  size_and_exec_fn.d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // Now build the chars column
  rmm::device_uvector<char> chars(char_bytes, stream, mr);
  if (char_bytes > 0) {
    size_and_exec_fn.d_chars = chars.data();
    for_each_kernel<<<grid.num_blocks, grid.num_threads_per_block, shmem_size, stream.value()>>>(
      size_and_exec_fn, d_prog, strings_count);
  }

  return std::make_pair(std::move(offsets), std::move(chars));
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
