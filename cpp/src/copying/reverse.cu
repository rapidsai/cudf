/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scan.h>

namespace cudf {
namespace detail {
std::unique_ptr<table> reverse(table_view const& source_table,
                               rmm::cuda_stream_view stream,
                               cudf::memory_resources resources)
{
  size_type num_rows = source_table.num_rows();
  auto elements      = make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_type>([num_rows] __device__(auto i) {
      return num_rows - i - 1;
    }));
  auto elements_end = elements + source_table.num_rows();

  return gather(
    source_table, elements, elements_end, out_of_bounds_policy::DONT_CHECK, stream, resources);
}

std::unique_ptr<column> reverse(column_view const& source_column,
                                rmm::cuda_stream_view stream,
                                cudf::memory_resources resources)
{
  return std::move(
    cudf::detail::reverse(table_view({source_column}), stream, resources)->release().front());
}
}  // namespace detail

std::unique_ptr<table> reverse(table_view const& source_table,
                               rmm::cuda_stream_view stream,
                               cudf::memory_resources resources)
{
  CUDF_FUNC_RANGE();
  return detail::reverse(source_table, stream, resources);
}

std::unique_ptr<column> reverse(column_view const& source_column,
                                rmm::cuda_stream_view stream,
                                cudf::memory_resources resources)
{
  CUDF_FUNC_RANGE();
  return detail::reverse(source_column, stream, resources);
}
}  // namespace cudf
