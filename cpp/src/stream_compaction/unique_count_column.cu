/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/row_operators.cuh>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace detail {
cudf::size_type unique_count(column_view const& input,
                             null_policy null_handling,
                             nan_policy nan_handling,
                             rmm::cuda_stream_view stream)
{
  auto const num_rows = input.size();

  if (num_rows == 0 or num_rows == input.null_count()) { return 0; }

  auto input_table_view = table_view{{input}};

  // Use the new row::equality::self_comparator with NaN treated as null and nulls equal
  auto const comparator = cudf::detail::row::equality::self_comparator{input_table_view, stream};
  auto const comp       = comparator.equal_to<false>(
    nullate::DYNAMIC{cudf::has_nulls(input_table_view)},
    null_equality::EQUAL,
    cudf::detail::row::equality::nan_equal_physical_equality_comparator{});

  return thrust::count_if(rmm::exec_policy(stream),
                          thrust::counting_iterator<cudf::size_type>(0),
                          thrust::counting_iterator<cudf::size_type>(num_rows),
                          [comp] __device__(cudf::size_type i) {
                            if (i == 0) { return true; }
                            return not comp(i, i - 1);
                          });
}
}  // namespace detail

cudf::size_type unique_count(column_view const& input,
                             null_policy null_handling,
                             nan_policy nan_handling,
                             rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::unique_count(input, null_handling, nan_handling, stream);
}

}  // namespace cudf
