/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/detail/calendrical_month_sequence.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace detail {
std::unique_ptr<cudf::column> calendrical_month_sequence(size_type size,
                                                         scalar const& init,
                                                         size_type months,
                                                         rmm::cuda_stream_view stream,
                                                         cudf::memory_resources resources)
{
  return type_dispatcher(
    init.type(), calendrical_month_sequence_functor{}, size, init, months, stream, resources);
}
}  // namespace detail

std::unique_ptr<cudf::column> calendrical_month_sequence(size_type size,
                                                         scalar const& init,
                                                         size_type months,
                                                         rmm::cuda_stream_view stream,
                                                         cudf::memory_resources resources)
{
  CUDF_FUNC_RANGE();
  return detail::calendrical_month_sequence(size, init, months, stream, resources);
}

}  // namespace cudf
