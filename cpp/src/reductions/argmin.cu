/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "extrema_utils.cuh"

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf::reduction::detail {

std::unique_ptr<scalar> argmin(column_view const& input,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  auto const dispatch_type =
    is_dictionary(input.type()) ? dictionary_column_view(input).indices().type() : input.type();
  return type_dispatcher(
    dispatch_type, simple::detail::arg_minmax_dispatcher<aggregation::ARGMIN>{}, input, stream, mr);
}

}  // namespace cudf::reduction::detail
