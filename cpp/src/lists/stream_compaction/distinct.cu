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

#include "lists/utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <utility>

namespace cudf::lists {
namespace detail {

std::unique_ptr<column> distinct(lists_column_view const& input,
                                 null_equality nulls_equal,
                                 nan_equality nans_equal,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  // Algorithm:
  // - Generate labels for the child elements.
  // - Get distinct rows of the table {labels, child} using `stable_distinct`.
  // - Build the output lists column from the output distinct rows above.

  if (input.is_empty()) { return empty_like(input.parent()); }

  auto const child = input.get_sliced_child(stream);
  auto const labels =
    generate_labels(input, child.size(), stream, rmm::mr::get_current_device_resource());

  auto const distinct_table =
    cudf::detail::stable_distinct(table_view{{labels->view(), child}},  // input table
                                  std::vector<size_type>{0, 1},         // keys
                                  duplicate_keep_option::KEEP_ANY,
                                  nulls_equal,
                                  nans_equal,
                                  stream,
                                  mr);

  auto out_offsets =
    reconstruct_offsets(distinct_table->get_column(0).view(), input.size(), stream, mr);

  return make_lists_column(input.size(),
                           std::move(out_offsets),
                           std::move(distinct_table->release().back()),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);
}

}  // namespace detail

std::unique_ptr<column> distinct(lists_column_view const& input,
                                 null_equality nulls_equal,
                                 nan_equality nans_equal,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::distinct(input, nulls_equal, nans_equal, stream, mr);
}

}  // namespace cudf::lists
