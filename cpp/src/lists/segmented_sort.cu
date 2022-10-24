/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {

namespace {

/**
 * @brief Create output offsets for segmented sort
 *
 * This creates a normalized set of offsets from the offsets child column of the input.
 */
std::unique_ptr<column> build_output_offsets(lists_column_view const& input,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  auto output_offset = make_numeric_column(
    input.offsets().type(), input.size() + 1, mask_state::UNALLOCATED, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    input.offsets_begin(),
                    input.offsets_end(),
                    output_offset->mutable_view().begin<size_type>(),
                    [first = input.offsets_begin()] __device__(auto offset_index) {
                      return offset_index - *first;
                    });
  return output_offset;
}

}  // namespace

std::unique_ptr<column> sort_lists(lists_column_view const& input,
                                   order column_order,
                                   null_order null_precedence,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) return empty_like(input.parent());

  auto output_offset = build_output_offsets(input, stream, mr);
  auto const child   = input.get_sliced_child(stream);

  auto const sorted_child_table = segmented_sort_by_key(table_view{{child}},
                                                        table_view{{child}},
                                                        output_offset->view(),
                                                        {column_order},
                                                        {null_precedence},
                                                        stream,
                                                        mr);

  return make_lists_column(input.size(),
                           std::move(output_offset),
                           std::move(sorted_child_table->release().front()),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);
}

std::unique_ptr<column> stable_sort_lists(lists_column_view const& input,
                                          order column_order,
                                          null_order null_precedence,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) { return empty_like(input.parent()); }

  auto output_offset = build_output_offsets(input, stream, mr);
  auto const child   = input.get_sliced_child(stream);

  auto const sorted_child_table = stable_segmented_sort_by_key(table_view{{child}},
                                                               table_view{{child}},
                                                               output_offset->view(),
                                                               {column_order},
                                                               {null_precedence},
                                                               stream,
                                                               mr);

  return make_lists_column(input.size(),
                           std::move(output_offset),
                           std::move(sorted_child_table->release().front()),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);
}
}  // namespace detail

std::unique_ptr<column> sort_lists(lists_column_view const& input,
                                   order column_order,
                                   null_order null_precedence,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort_lists(input, column_order, null_precedence, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> stable_sort_lists(lists_column_view const& input,
                                          order column_order,
                                          null_order null_precedence,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_sort_lists(
    input, column_order, null_precedence, cudf::get_default_stream(), mr);
}

}  // namespace lists
}  // namespace cudf
