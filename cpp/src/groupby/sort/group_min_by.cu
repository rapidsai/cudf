/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_single_pass_reduction_util.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace groupby {
namespace detail {

std::unique_ptr<column> group_min_by(column_view const& values,
                                     size_type num_groups,
                                     cudf::device_span<size_type const> group_labels,
                                     column_view const& key_sort_order,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(values.type().id() == type_id::STRUCT,
               "MIN_BY aggregation expects a struct column as input");

  auto const struct_view = structs_column_view(values);
  CUDF_EXPECTS(struct_view.num_children() == 2,
               "MIN_BY aggregation expects a struct column with exactly 2 children");

  // First child is the ordering column, second child is the value column
  auto const& ordering_column = struct_view.get_sliced_child(0, stream);
  auto const& value_column    = struct_view.get_sliced_child(1, stream);

  // Find argmin of the ordering column
  auto arg_indices = type_dispatcher(ordering_column.type(),
                                     group_reduction_dispatcher<aggregation::ARGMIN>{},
                                     ordering_column,
                                     num_groups,
                                     group_labels,
                                     stream,
                                     mr);

  // Map indices from sorted to original order
  auto indices_view = arg_indices->view();
  auto mapped_indices =
    rmm::device_uvector<size_type>(indices_view.size(), stream, mr);
  thrust::gather(rmm::exec_policy_nosync(stream),
                 indices_view.begin<size_type>(),
                 indices_view.end<size_type>(),
                 key_sort_order.begin<size_type>(),
                 mapped_indices.data());

  // Create a column view from the mapped indices
  auto const indices_column = column_view(
    data_type{type_id::INT32}, mapped_indices.size(), mapped_indices.data());

  // Gather the value column using the mapped indices
  auto gathered_table = cudf::detail::gather(table_view{{value_column}},
                                             indices_column,
                                             cudf::out_of_bounds_policy::NULLIFY,
                                             cudf::detail::negative_index_policy::NOT_ALLOWED,
                                             stream,
                                             mr);

  // Preserve null mask from argmin results
  auto result = std::move(gathered_table->release().front());
  if (arg_indices->nullable()) {
    result->set_null_mask(rmm::device_buffer{*arg_indices->view().null_mask(),
                                            cudf::bitmask_allocation_size_bytes(result->size()),
                                            stream,
                                            mr},
                         arg_indices->null_count());
  }

  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
