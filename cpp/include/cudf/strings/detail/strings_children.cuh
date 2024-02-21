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
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <stdexcept>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Creates child offsets and chars columns by applying the template function that
 * can be used for computing the output size of each string as well as create the output
 *
 * @throws std::overflow_error if the output strings column exceeds the column size limit
 *
 * @tparam SizeAndExecuteFunction Function must accept an index and return a size.
 *         It must also have members d_offsets and d_chars which are set to
 *         memory containing the offsets and chars columns during write.
 *
 * @param size_and_exec_fn This is called twice. Once for the output size of each string
 *        and once again to fill in the memory pointed to by d_chars.
 * @param exec_size Number of rows for executing the `size_and_exec_fn` function.
 * @param strings_count Number of strings.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned columns' device memory.
 * @return offsets child column and chars child column for a strings column
 */
template <typename SizeAndExecuteFunction>
auto make_strings_children(SizeAndExecuteFunction size_and_exec_fn,
                           size_type exec_size,
                           size_type strings_count,
                           rmm::cuda_stream_view stream,
                           rmm::mr::device_memory_resource* mr)
{
  auto offsets_column = make_numeric_column(
    data_type{type_to_id<size_type>()}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view          = offsets_column->mutable_view();
  auto d_offsets             = offsets_view.template data<int32_t>();
  size_and_exec_fn.d_offsets = d_offsets;

  // This is called twice -- once for offsets and once for chars.
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
  auto const bytes =
    cudf::detail::sizes_to_offsets(d_offsets, d_offsets + strings_count + 1, d_offsets, stream);
  CUDF_EXPECTS(bytes <= std::numeric_limits<size_type>::max(),
               "Size of output exceeds the column size limit",
               std::overflow_error);

  // Now build the chars column
  std::unique_ptr<column> chars_column =
    create_chars_child_column(static_cast<size_type>(bytes), stream, mr);

  // Execute the function fn again to fill the chars column.
  // Note that if the output chars column has zero size, the function fn should not be called to
  // avoid accidentally overwriting the offsets.
  if (bytes > 0) {
    size_and_exec_fn.d_chars = chars_column->mutable_view().template data<char>();
    for_each_fn(size_and_exec_fn);
  }

  return std::pair(std::move(offsets_column), std::move(chars_column));
}

/**
 * @brief Creates child offsets and chars columns by applying the template function that
 * can be used for computing the output size of each string as well as create the output.
 *
 * @tparam SizeAndExecuteFunction Function must accept an index and return a size.
 *         It must also have members d_offsets and d_chars which are set to
 *         memory containing the offsets and chars columns during write.
 *
 * @param size_and_exec_fn This is called twice. Once for the output size of each string
 *        and once again to fill in the memory pointed to by d_chars.
 * @param strings_count Number of strings.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned columns' device memory.
 * @return offsets child column and chars child column for a strings column
 */
template <typename SizeAndExecuteFunction>
auto make_strings_children(SizeAndExecuteFunction size_and_exec_fn,
                           size_type strings_count,
                           rmm::cuda_stream_view stream,
                           rmm::mr::device_memory_resource* mr)
{
  return make_strings_children(size_and_exec_fn, strings_count, strings_count, stream, mr);
}

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
  rmm::mr::device_memory_resource* mr)
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
  auto const total_elements =
    sizes_to_offsets(input_itr, input_itr + strings_count + 1, d_offsets, stream);

  // TODO: replace exception with if-statement when enabling creating INT64 offsets
  CUDF_EXPECTS(total_elements <= size_type_max,
               "Size of output exceeds the character size limit",
               std::overflow_error);
  // if (total_elements >= get_offset64_threshold()) {
  //   // recompute as int64 offsets when above the threshold
  //   offsets_column = make_numeric_column(
  //     data_type{type_id::INT64}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  //   auto d_offsets64 = offsets_column->mutable_view().template data<int64_t>();
  //   sizes_to_offsets(input_itr, input_itr + strings_count + 1, d_offsets64, stream);
  // }

  return std::pair(std::move(offsets_column), total_elements);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
