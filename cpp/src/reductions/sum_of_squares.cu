/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "simple.cuh"

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace reduction {
namespace detail {

std::unique_ptr<cudf::scalar> sum_of_squares(column_view const& col,
                                             cudf::data_type const output_dtype,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(
    cudf::is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type(),
    simple::detail::element_type_dispatcher<op::sum_of_squares>{},
    col,
    output_dtype,
    std::nullopt,
    stream,
    mr);
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
