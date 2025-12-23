/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf::reduction::detail {

std::unique_ptr<scalar> max_by(column_view const& input,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == type_id::STRUCT,
               "MAX_BY reduction expects a struct column as input");

  auto const struct_view = structs_column_view(input);
  CUDF_EXPECTS(struct_view.num_children() == 2,
               "MAX_BY reduction expects a struct column with exactly 2 children");

  // First child is the ordering column, second child is the value column
  auto const& ordering_column = struct_view.get_sliced_child(0, stream);
  auto const& value_column    = struct_view.get_sliced_child(1, stream);

  // Handle empty input
  if (input.size() == 0 || input.null_count() == input.size()) {
    return make_default_constructed_scalar(value_column.type(), stream, mr);
  }

  // Find argmax of the ordering column
  auto const argmax_result = argmax(ordering_column, stream, mr);
  
  // Check if argmax result is valid
  if (!argmax_result->is_valid(stream)) {
    return make_default_constructed_scalar(value_column.type(), stream, mr);
  }

  // Extract the index
  auto const max_index = static_cast<cudf::numeric_scalar<size_type> const&>(*argmax_result).value(stream);

  // Gather the value at that index
  auto const indices = rmm::device_uvector<size_type>(1, stream, mr);
  cudaMemcpyAsync(const_cast<size_type*>(indices.data()),
                  &max_index,
                  sizeof(size_type),
                  cudaMemcpyHostToDevice,
                  stream.value());

  auto const indices_column = column_view(data_type{type_id::INT32}, 1, indices.data());
  auto gathered_table = cudf::detail::gather(table_view{{value_column}},
                                             indices_column,
                                             cudf::out_of_bounds_policy::NULLIFY,
                                             cudf::detail::negative_index_policy::NOT_ALLOWED,
                                             stream,
                                             mr);

  // Extract scalar from the gathered result
  return cudf::get_element(gathered_table->get_column(0).view(), 0, stream, mr);
}

}  // namespace cudf::reduction::detail
