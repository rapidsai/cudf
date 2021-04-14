/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/detail/sorting.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {
namespace {

}  // anonymous namespace

/**
 * @copydoc cudf::lists::detail::interleave_columns
 *
 */
std::unique_ptr<column> interleave_columns(table_view const& lists_columns,
                                           bool create_mask,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  auto const first_col_entry_type_id =
    lists_column_view(*lists_columns.begin()).child().type().id();
  for (auto const& col : lists_columns) {
    CUDF_EXPECTS(col.type().id() == type_id::LIST,
                 "All columns of the input table must be of lists column type.");

    auto const child_col = lists_column_view(col).child();
    CUDF_EXPECTS(not cudf::is_nested(child_col.type()), "Nested types are not supported.");
    CUDF_EXPECTS(first_col_entry_type_id == child_col.type().id(),
                 "The types of entries in the input columns must be the same.");
  }

  // Single column returns a copy
  if (lists_columns.num_columns() == 1) {
    return std::make_unique<column>(*(lists_columns.begin()), stream, mr);
  }

  auto const num_rows = lists_columns.num_rows();
  if (num_rows == 0) { return cudf::empty_like(lists_columns.column(0)); }

  static_assert(sizeof(offset_type) == sizeof(int32_t));
  auto lists_offsets = make_numeric_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);
  auto const output_offsets_ptr = lists_offsets->mutable_view().begin<offset_type>();

  auto lists_entries = std::make_unique<column>();

  static constexpr auto invalid_size = std::numeric_limits<size_type>::lowest();
  auto const count_it                = thrust::make_counting_iterator<size_type>(0);
  auto [null_mask, null_count]       = cudf::detail::valid_if(
    count_it,
    count_it + num_rows,
    [str_sizes = output_offsets_ptr + 1] __device__(size_type idx) {
      return str_sizes[idx] != invalid_size;
    },
    stream,
    mr);

  return make_lists_column(num_rows,
                           std::move(lists_offsets),
                           std::move(lists_entries),
                           null_count,
                           null_count ? std::move(null_mask) : rmm::device_buffer{},
                           stream,
                           mr);
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
