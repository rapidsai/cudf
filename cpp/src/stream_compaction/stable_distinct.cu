/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/scatter.h>
#include <thrust/uninitialized_fill.h>

namespace cudf {
namespace detail {

std::unique_ptr<table> stable_distinct(table_view const& input,
                                       std::vector<size_type> const& keys,
                                       duplicate_keep_option keep,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  if (input.num_rows() == 0 or input.num_columns() == 0 or keys.empty()) {
    return empty_like(input);
  }

  auto const distinct_indices = detail::distinct_indices(input.select(keys),
                                                         keep,
                                                         nulls_equal,
                                                         nans_equal,
                                                         stream,
                                                         rmm::mr::get_current_device_resource());

  // The only difference between this implementation and the unstable version
  // is that the stable implementation must retain the input order. The
  // distinct indices are not sorted, so we cannot simply copy the rows in the
  // order of the distinct indices and retain the input order. Instead, we use
  // a boolean mask to indicate which rows to copy to the output. This avoids
  // the need to sort the distinct indices, which is slower.

  auto const output_markers = [&] {
    auto markers = rmm::device_uvector<bool>(input.num_rows(), stream);
    thrust::uninitialized_fill(rmm::exec_policy(stream), markers.begin(), markers.end(), false);
    thrust::scatter(
      rmm::exec_policy(stream),
      thrust::constant_iterator<bool>(true, 0),
      thrust::constant_iterator<bool>(true, static_cast<size_type>(distinct_indices.size())),
      distinct_indices.begin(),
      markers.begin());
    return markers;
  }();

  return cudf::detail::apply_boolean_mask(
    input, cudf::device_span<bool const>(output_markers), stream, mr);
}

}  // namespace detail

std::unique_ptr<table> stable_distinct(table_view const& input,
                                       std::vector<size_type> const& keys,
                                       duplicate_keep_option keep,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_distinct(
    input, keys, keep, nulls_equal, nans_equal, cudf::get_default_stream(), mr);
}

}  // namespace cudf
