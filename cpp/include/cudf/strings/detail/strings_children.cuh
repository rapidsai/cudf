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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <stdexcept>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Create an offsets column to be a child of a compound column
 *
 * This function sets the offsets values by executing scan over the sizes in the provided
 * Iterator.
 *
 * The return also includes the total number of elements -- the last element value from the
 * scan.
 *
 * @tparam InputIterator Used as input to scan to set the offset values
 * @param begin The beginning of the input sequence
 * @param end The end of the input sequence
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Offsets column and total elements
 */
template <typename InputIterator>
std::pair<std::unique_ptr<column>, int64_t> make_offsets_child_column(
  InputIterator begin,
  InputIterator end,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto constexpr size_type_max = static_cast<int64_t>(std::numeric_limits<size_type>::max());
  auto const lcount            = static_cast<int64_t>(std::distance(begin, end));
  CUDF_EXPECTS(
    lcount <= size_type_max, "Size of output exceeds the column size limit", std::overflow_error);
  auto const strings_count = static_cast<size_type>(lcount);
  auto offsets_column      = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto d_offsets = offsets_column->mutable_view().template data<int32_t>();

  // The number of offsets is strings_count+1 so to build the offsets from the sizes
  // using exclusive-scan technically requires strings_count+1 input values even though
  // the final input value is never used.
  // The input iterator is wrapped here to allow the 'last value' to be safely read.
  auto map_fn = cuda::proclaim_return_type<size_type>(
    [begin, strings_count] __device__(size_type idx) -> size_type {
      return idx < strings_count ? static_cast<size_type>(begin[idx]) : size_type{0};
    });
  auto input_itr = cudf::detail::make_counting_transform_iterator(0, map_fn);
  // Use the sizes-to-offsets iterator to compute the total number of elements
  auto const total_bytes =
    cudf::detail::sizes_to_offsets(input_itr, input_itr + strings_count + 1, d_offsets, stream);

  auto const threshold = get_offset64_threshold();
  CUDF_EXPECTS(is_large_strings_enabled() || (total_bytes < threshold),
               "Size of output exceeds the column size limit",
               std::overflow_error);
  if (total_bytes >= get_offset64_threshold()) {
    // recompute as int64 offsets when above the threshold
    offsets_column = make_numeric_column(
      data_type{type_id::INT64}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
    auto d_offsets64 = offsets_column->mutable_view().template data<int64_t>();
    cudf::detail::sizes_to_offsets(input_itr, input_itr + strings_count + 1, d_offsets64, stream);
  }

  return std::pair(std::move(offsets_column), total_bytes);
}

/**
 * @brief Kernel used by make_strings_children for calling the given functor
 *
 * @tparam SizeAndExecuteFunction Functor type to call in each thread
 *
 * @param fn Functor to call in each thread
 * @param exec_size Total number of threads to be processed by this kernel
 */
template <typename SizeAndExecuteFunction>
CUDF_KERNEL void strings_children_kernel(SizeAndExecuteFunction fn, size_type exec_size)
{
  auto tid = cudf::detail::grid_1d::global_thread_id();
  if (tid < exec_size) { fn(tid); }
}

/**
 * @brief Creates child offsets and chars data by applying the template function that
 * can be used for computing the output size of each string as well as create the output
 *
 * The `size_and_exec_fn` is expected declare an operator() function with a size_type parameter
 * and 3 member variables:
 * - `d_sizes`: output size in bytes of each output row for the 1st pass call
 * - `d_chars`: output buffer for new string data for the 2nd pass call
 * - `d_offsets`: used for addressing the specific output row data in `d_chars`
 *
 * The 1st pass call computes the output sizes and is identified by `d_chars==nullptr`.
 * Null rows should be set with an output size of 0.
 *
 * @code{.cpp}
 * struct size_and_exec_fn {
 *  size_type* d_sizes;
 *  char* d_chars;
 *  input_offsetalator d_offsets;
 *
 *   __device__ void operator()(size_type thread_idx)
 *   {
 *     // functor-specific logic to resolve out_idx from thread_idx
 *     if( !d_chars ) {
 *       d_sizes[out_idx] = output_size;
 *     } else {
 *       auto d_output = d_chars + d_offsets[out_idx];
 *       // write characters to d_output
 *     }
 *   }
 * };
 * @endcode
 *
 * @tparam SizeAndExecuteFunction Functor type with an operator() function accepting
 *         an index parameter and three member variables: `size_type* d_sizes`
 *         `char* d_chars`, and `input_offsetalator d_offsets`.
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
  // This is called twice -- once for computing sizes and once for writing chars.
  // Reducing the number of places size_and_exec_fn is inlined speeds up compile time.
  auto for_each_fn = [exec_size, stream](SizeAndExecuteFunction& size_and_exec_fn) {
    auto constexpr block_size = 256;
    auto grid                 = cudf::detail::grid_1d{exec_size, block_size};
    strings_children_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(size_and_exec_fn,
                                                                                exec_size);
  };

  // Compute the output sizes
  auto output_sizes        = rmm::device_uvector<size_type>(strings_count, stream);
  size_and_exec_fn.d_sizes = output_sizes.data();
  size_and_exec_fn.d_chars = nullptr;
  for_each_fn(size_and_exec_fn);

  // Convert the sizes to offsets
  auto [offsets_column, bytes] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);
  size_and_exec_fn.d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());

  // Now build the chars column
  rmm::device_uvector<char> chars(bytes, stream, mr);
  size_and_exec_fn.d_chars = chars.data();

  // Execute the function fn again to fill in the chars data.
  if (bytes > 0) { for_each_fn(size_and_exec_fn); }

  return std::pair(std::move(offsets_column), std::move(chars));
}

/**
 * @brief Creates child offsets and chars columns by applying the template function that
 * can be used for computing the output size of each string as well as create the output
 *
 * The `size_and_exec_fn` is expected declare an operator() function with a size_type parameter
 * and 3 member variables:
 * - `d_sizes`: output size in bytes of each output row for the 1st pass call
 * - `d_chars`: output buffer for new string data for the 2nd pass call
 * - `d_offsets`: used for addressing the specific output row data in `d_chars`
 *
 * The 1st pass call computes the output sizes and is identified by `d_chars==nullptr`.
 * Null rows should be set with an output size of 0.
 *
 * @code{.cpp}
 * struct size_and_exec_fn {
 *  size_type* d_sizes;
 *  char* d_chars;
 *  input_offsetalator d_offsets;
 *
 *   __device__ void operator()(size_type idx)
 *   {
 *     if( !d_chars ) {
 *       d_sizes[idx] = output_size;
 *     } else {
 *       auto d_output = d_chars + d_offsets[idx];
 *       // write characters to d_output
 *     }
 *   }
 * };
 * @endcode
 *
 * @tparam SizeAndExecuteFunction Functor type with an operator() function accepting
 *         an index parameter and three member variables: `size_type* d_sizes`
 *         `char* d_chars`, and `input_offsetalator d_offsets`.
 *
 * @param size_and_exec_fn This is called twice. Once for the output size of each string
 *        and once again to fill in the memory pointed to by `d_chars`.
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

}  // namespace detail
}  // namespace strings
}  // namespace cudf
