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

#include <lists/utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.cuh>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/lists/detail/combine.hpp>
#include <cudf/lists/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/experimental/row_operators.cuh>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/static_map.cuh>
#include <cuco/static_multimap.cuh>

namespace cudf::lists {
namespace detail {

std::unique_ptr<column> distinct_by_labels(size_type n_lists,
                                           column_view const& child_labels,
                                           column_view const& child,
                                           rmm::device_buffer&& null_mask,
                                           size_type null_count,
                                           null_equality nulls_equal,
                                           nan_equality nans_equal,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  // Algorithm:
  // - Get indices of distinct rows of the table {labels, child}.
  // - Scatter these indices into a marker array that marks if a row will be copied to the output.
  // - Collect output rows (with order preserved) using the marker array and build the output
  //   offsets column.

  auto const input_table = table_view{{child_labels, child}};

  auto const distinct_indices = cudf::detail::get_distinct_indices(
    input_table, duplicate_keep_option::KEEP_ANY, nulls_equal, nans_equal, stream);

  auto const index_markers = [&] {
    auto markers = rmm::device_uvector<bool>(child.size(), stream);
    thrust::uninitialized_fill(rmm::exec_policy(stream), markers.begin(), markers.end(), false);
    thrust::scatter(
      rmm::exec_policy(stream),
      thrust::constant_iterator<size_type>(true, 0),
      thrust::constant_iterator<size_type>(true, static_cast<size_type>(distinct_indices.size())),
      distinct_indices.begin(),
      markers.begin());
    return markers;
  }();

  auto const out_table = cudf::detail::copy_if(
    input_table,
    [index_markers = index_markers.begin()] __device__(auto const idx) {
      return index_markers[idx];
    },
    stream,
    mr);
  auto out_offsets = reconstruct_offsets(out_table->get_column(0).view(), n_lists, stream, mr);

  return make_lists_column(n_lists,
                           std::move(out_offsets),
                           std::move(out_table->release().back()),
                           null_count,
                           std::move(null_mask),
                           stream,
                           mr);
}

std::unique_ptr<column> distinct(lists_column_view const& input,
                                 null_equality nulls_equal,
                                 nan_equality nans_equal,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  // Algorithm:
  // - Generate labels for the child elements.
  // - Get indices of distinct rows of the table {labels, child}.
  // - Scatter these indices into a marker array that marks if a row will be copied to the output.
  // - Collect output rows (with order preserved) using the marker array and build the output
  //   lists column.

  auto const child  = input.get_sliced_child(stream);
  auto const labels = generate_labels(input, child.size(), stream);

  return distinct_by_labels(input.size(),
                            labels->view(),
                            child,
                            cudf::detail::copy_bitmask(input.parent(), stream, mr),
                            input.null_count(),
                            nulls_equal,
                            nans_equal,
                            stream,
                            mr);
}

}  // namespace detail

std::unique_ptr<column> distinct(lists_column_view const& input,
                                 null_equality nulls_equal,
                                 nan_equality nans_equal,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::distinct(input, nulls_equal, nans_equal, rmm::cuda_stream_default, mr);
}

}  // namespace cudf::lists
