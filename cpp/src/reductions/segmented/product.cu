/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "simple.cuh"

#include <cudf/reduction/detail/segmented_reduction_functions.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace reduction {
namespace detail {
std::unique_ptr<cudf::column> segmented_product(
  column_view const& col,
  device_span<size_type const> offsets,
  cudf::data_type const output_dtype,
  null_policy null_handling,
  std::optional<std::reference_wrapper<scalar const>> init,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  using reducer = simple::detail::column_type_dispatcher<op::product>;
  return cudf::type_dispatcher(
    col.type(), reducer{}, col, offsets, output_dtype, null_handling, init, stream, mr);
}
}  // namespace detail
}  // namespace reduction
}  // namespace cudf
