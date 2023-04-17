/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
  CUDF_EXPECTS(bytes <= static_cast<int64_t>(std::numeric_limits<size_type>::max()),
               "Size of output exceeds column size limit",
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

}  // namespace detail
}  // namespace strings
}  // namespace cudf
