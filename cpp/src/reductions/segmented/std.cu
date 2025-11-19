/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compound.cuh"

#include <cudf/reduction/detail/segmented_reduction_functions.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace reduction {
namespace detail {

std::unique_ptr<cudf::column> segmented_standard_deviation(column_view const& col,
                                                           device_span<size_type const> offsets,
                                                           cudf::data_type const output_dtype,
                                                           null_policy null_handling,
                                                           size_type ddof,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr)
{
  using reducer = compound::detail::compound_segmented_dispatcher<op::standard_deviation>;
  return cudf::type_dispatcher(
    col.type(), reducer(), col, offsets, output_dtype, null_handling, ddof, stream, mr);
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
