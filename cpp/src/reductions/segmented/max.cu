/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "simple.cuh"

#include <cudf/reduction/detail/segmented_reduction_functions.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace reduction {
namespace detail {

std::unique_ptr<cudf::column> segmented_max(
  column_view const& col,
  device_span<size_type const> offsets,
  cudf::data_type const output_dtype,
  null_policy null_handling,
  std::optional<std::reference_wrapper<scalar const>> init,
  rmm::cuda_stream_view stream,
  cudf::memory_resources resources)
{
  CUDF_EXPECTS(col.type() == output_dtype,
               "segmented_max() operation requires matching output type");
  using reducer = simple::detail::same_column_type_dispatcher<op::max>;
  return cudf::type_dispatcher(
    col.type(), reducer{}, col, offsets, null_handling, init, stream, resources);
}
}  // namespace detail
}  // namespace reduction
}  // namespace cudf
