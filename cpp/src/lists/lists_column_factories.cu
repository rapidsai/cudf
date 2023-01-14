/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

namespace cudf {
namespace lists {
namespace detail {

std::unique_ptr<cudf::column> make_lists_column_from_scalar(list_scalar const& value,
                                                            size_type size,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::mr::device_memory_resource* mr)
{
  if (size == 0) {
    return make_lists_column(0,
                             make_empty_column(type_to_id<offset_type>()),
                             empty_like(value.view()),
                             0,
                             cudf::detail::create_null_mask(0, mask_state::UNALLOCATED, stream, mr),
                             stream,
                             mr);
  }
  auto mr_final = size == 1 ? mr : rmm::mr::get_current_device_resource();

  // Handcraft a 1-row column
  auto offsets = make_numeric_column(
    data_type{type_to_id<offset_type>()}, 2, mask_state::UNALLOCATED, stream, mr_final);
  auto m_offsets = offsets->mutable_view();
  thrust::sequence(rmm::exec_policy(stream),
                   m_offsets.begin<size_type>(),
                   m_offsets.end<size_type>(),
                   0,
                   value.view().size());
  size_type null_count = value.is_valid(stream) ? 0 : 1;
  auto null_mask_state = null_count ? mask_state::ALL_NULL : mask_state::UNALLOCATED;
  auto null_mask       = cudf::detail::create_null_mask(1, null_mask_state, stream, mr_final);

  if (size == 1) {
    auto child = std::make_unique<column>(value.view(), stream, mr_final);
    return make_lists_column(
      1, std::move(offsets), std::move(child), null_count, std::move(null_mask), stream, mr_final);
  }

  auto children_views   = std::vector<column_view>{offsets->view(), value.view()};
  auto one_row_col_view = column_view(data_type{type_id::LIST},
                                      1,
                                      nullptr,
                                      static_cast<bitmask_type const*>(null_mask.data()),
                                      null_count,
                                      0,
                                      children_views);

  auto begin = thrust::make_constant_iterator(0);
  auto res   = cudf::detail::gather(table_view({one_row_col_view}),
                                  begin,
                                  begin + size,
                                  out_of_bounds_policy::DONT_CHECK,
                                  stream,
                                  mr_final);
  return std::move(res->release()[0]);
}

}  // namespace detail
}  // namespace lists

/**
 * @copydoc cudf::make_lists_column
 */
std::unique_ptr<column> make_lists_column(size_type num_rows,
                                          std::unique_ptr<column> offsets_column,
                                          std::unique_ptr<column> child_column,
                                          size_type null_count,
                                          rmm::device_buffer&& null_mask,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  if (null_count > 0) { CUDF_EXPECTS(null_mask.size() > 0, "Column with nulls must be nullable."); }
  CUDF_EXPECTS(
    (num_rows == 0 && offsets_column->size() == 0) || num_rows == offsets_column->size() - 1,
    "Invalid offsets column size for lists column.");
  CUDF_EXPECTS(offsets_column->null_count() == 0, "Offsets column should not contain nulls");
  CUDF_EXPECTS(child_column != nullptr, "Must pass a valid child column");

  // Save type_id of the child column for later use.
  auto const child_type_id = child_column->type().id();

  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(offsets_column));
  children.emplace_back(std::move(child_column));

  auto output = std::make_unique<column>(cudf::data_type{type_id::LIST},
                                         num_rows,
                                         rmm::device_buffer{},
                                         std::move(null_mask),
                                         null_count,
                                         std::move(children));

  // We need to enforce all null lists to be empty.
  // `has_nonempty_nulls` is less expensive than `purge_nonempty_nulls` and can save some
  // run time if we don't have any non-empty nulls.
  if (auto const output_cv = output->view(); detail::has_nonempty_nulls(output_cv, stream)) {
    return detail::purge_nonempty_nulls(output_cv, stream, mr);
  }

  return output;
}

}  // namespace cudf
