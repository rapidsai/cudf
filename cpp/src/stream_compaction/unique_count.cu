/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {

cudf::size_type unique_count(table_view const& keys,
                             null_equality nulls_equal,
                             rmm::cuda_stream_view stream)
{
  auto const row_comp = cudf::experimental::row::equality::self_comparator(keys, stream);
  if (cudf::detail::has_nested_columns(keys)) {
    auto const comp =
      row_comp.equal_to<true>(nullate::DYNAMIC{has_nested_nulls(keys)}, nulls_equal);
    // Using a temporary buffer for intermediate transform results from the lambda containing
    // the comparator speeds up compile-time significantly without much degradation in
    // runtime performance over using the comparator directly in thrust::count_if.
    auto d_results = rmm::device_uvector<bool>(keys.num_rows(), stream);
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(keys.num_rows()),
                      d_results.begin(),
                      [comp] __device__(auto i) { return (i == 0 or not comp(i, i - 1)); });

    return static_cast<size_type>(
      thrust::count(rmm::exec_policy(stream), d_results.begin(), d_results.end(), true));
  } else {
    auto const comp =
      row_comp.equal_to<false>(nullate::DYNAMIC{has_nested_nulls(keys)}, nulls_equal);
    // Using thrust::copy_if with the comparator directly will compile more slowly but
    // improves runtime by up to 2x over the transform/count approach above.
    return thrust::count_if(
      rmm::exec_policy(stream),
      thrust::counting_iterator<cudf::size_type>(0),
      thrust::counting_iterator<cudf::size_type>(keys.num_rows()),
      [comp] __device__(cudf::size_type i) { return (i == 0 or not comp(i, i - 1)); });
  }
}

}  // namespace detail

cudf::size_type unique_count(table_view const& input, null_equality nulls_equal)
{
  CUDF_FUNC_RANGE();
  return detail::unique_count(input, nulls_equal, cudf::get_default_stream());
}

}  // namespace cudf
