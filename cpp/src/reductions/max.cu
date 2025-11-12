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

std::unique_ptr<cudf::scalar> max(column_view const& col,
                                  cudf::data_type const output_dtype,
                                  std::optional<std::reference_wrapper<scalar const>> init,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  auto const input_type =
    cudf::is_dictionary(col.type()) ? cudf::dictionary_column_view(col).keys().type() : col.type();
  CUDF_EXPECTS(input_type == output_dtype, "max() operation requires matching output type");
  auto const dispatch_type = cudf::is_dictionary(col.type())
                               ? cudf::dictionary_column_view(col).indices().type()
                               : col.type();

  using reducer = simple::detail::same_element_type_dispatcher<op::max>;
  return cudf::type_dispatcher(dispatch_type, reducer{}, col, init, stream, mr);
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
