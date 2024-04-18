/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/strings/detail/strings_children.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace strings {
namespace detail {
namespace experimental {

/**
 * @brief Creates child offsets and chars data by applying the template function that
 * can be used for computing the output size of each string as well as create the output
 *
 * The `size_and_exec_fn` is expected to be functor with 3 settable member variables.
 * @code{.cpp}
 * struct size_and_exec_fn {
 *   size_type* d_sizes;
 *   char* d_chars{};
 *   cudf::detail::input_offsetalator d_offsets;
 *   __device__ void operator()(size_type thread_idx) {
 *      // functor-specific logic to resolve out_idx from thread_idx
 *      if( !d_chars ) {
 *         d_sizes[out_idx] = output_size;
 *      } else {
 *         auto d_output = d_chars + d_offsets[out_idx];
 *         // write characters to d_output
 *      }
 *   }
 * };
 * @endcode
 *
 * @tparam SizeAndExecuteFunction Function must accept a row index.
 *         It must also have member `d_sizes` to hold computed row output sizes on the 1st pass
 *         and members `d_offsets` and `d_chars` for the 2nd pass to resolve the output memory
 *         location for each row.
 *
 * @param size_and_exec_fn This is called twice. Once for the output size of each string
 *        and once again to fill in the memory pointed to by d_chars.
 * @param exec_size Number of threads for executing the `size_and_exec_fn` function
 * @param strings_count Number of strings
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return Offsets child column and chars vector for creating a strings column
 */
template <typename SizeAndExecuteFunction>
auto make_strings_children(SizeAndExecuteFunction size_and_exec_fn,
                           size_type exec_size,
                           size_type strings_count,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto output_sizes        = rmm::device_uvector<size_type>(strings_count, stream);
  size_and_exec_fn.d_sizes = output_sizes.data();

  // This is called twice -- once for computing sizes and once for writing chars.
  // Reducing the number of places size_and_exec_fn is inlined speeds up compile time.
  auto for_each_fn = [exec_size, stream](SizeAndExecuteFunction& size_and_exec_fn) {
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       exec_size,
                       size_and_exec_fn);
  };

  // Compute the output sizes
  for_each_fn(size_and_exec_fn);

  // Convert the sizes to offsets
  auto [offsets_column, bytes] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);
  size_and_exec_fn.d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());

  // Now build the chars column
  rmm::device_uvector<char> chars(bytes, stream, mr);

  // Execute the function fn again to fill in the chars data.
  if (bytes > 0) {
    size_and_exec_fn.d_chars = chars.data();
    for_each_fn(size_and_exec_fn);
  }

  return std::pair(std::move(offsets_column), std::move(chars));
}

/**
 * @brief Creates child offsets and chars columns by applying the template function that
 * can be used for computing the output size of each string as well as create the output
 *
 * The `size_and_exec_fn` is expected to be functor with 3 settable member variables.
 * @code{.cpp}
 * struct size_and_exec_fn {
 *   size_type* d_sizes;
 *   char* d_chars{};
 *   cudf::detail::input_offsetalator d_offsets;
 *   __device__ void operator()(size_type idx) {
 *      if( !d_chars ) {
 *         d_sizes[idx] = output_size;
 *      } else {
 *         auto d_output = d_chars + d_offsets[idx];
 *         // write characters to d_output
 *      }
 *   }
 * };
 * @endcode
 *
 * @tparam SizeAndExecuteFunction Function must accept a row index.
 *         It must also have member `d_sizes` to hold computed row output sizes on the 1st pass
 *         and members `d_offsets` and `d_chars` for the 2nd pass to resolve the output memory
 *         location for each row.
 *
 * @param size_and_exec_fn This is called twice. Once for the output size of each string
 *        and once again to fill in the memory pointed to by d_chars.
 * @param strings_count Number of strings
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return Offsets child column and chars vector for creating a strings column
 */
template <typename SizeAndExecuteFunction>
auto make_strings_children(SizeAndExecuteFunction size_and_exec_fn,
                           size_type strings_count,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  return make_strings_children(size_and_exec_fn, strings_count, strings_count, stream, mr);
}

}  // namespace experimental
}  // namespace detail
}  // namespace strings
}  // namespace cudf
