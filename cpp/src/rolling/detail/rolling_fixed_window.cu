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

#include "rolling.cuh"

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cudf_test/column_utilities.hpp>

#include <thrust/extrema.h>

// TODO: DELETEME!
namespace
{
  using namespace cudf;
  template <typename Calculator>
  std::unique_ptr<column> expand_to_column(Calculator const& calc,
                                           size_type const& num_rows,
                                           rmm::cuda_stream_view stream)
  {
    auto window_column = cudf::make_numeric_column(
        cudf::data_type{type_to_id<size_type>()}, num_rows, cudf::mask_state::UNALLOCATED, stream);

    auto begin = cudf::detail::make_counting_transform_iterator(0, calc);

    thrust::copy_n(
        rmm::exec_policy(stream), begin, num_rows, window_column->mutable_view().data<size_type>());

    return window_column;
  }
}

namespace cudf::detail {

// Applies a fixed-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& default_outputs,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (input.is_empty()) { return cudf::detail::empty_output_for_rolling_aggregation(input, agg); }

  CUDF_EXPECTS((min_periods >= 0), "min_periods must be non-negative");

  CUDF_EXPECTS((default_outputs.is_empty() || default_outputs.size() == input.size()),
               "Defaults column must be either empty or have as many rows as the input column.");

  CUDF_EXPECTS(-(preceding_window - 1) <= following_window,
               "Preceding window bounds must precede the following window bounds.");

  if (agg.kind == aggregation::CUDA || agg.kind == aggregation::PTX) {
    // TODO: In future, might need to clamp preceding/following to column boundaries.
    return cudf::detail::rolling_window_udf(input,
                                            preceding_window,
                                            "cudf::size_type",
                                            following_window,
                                            "cudf::size_type",
                                            min_periods,
                                            agg,
                                            stream,
                                            mr);
  } else {
    // Clamp preceding/following to column boundaries.
    // E.g. If preceding_window == 2, then for a column of 5 elements, preceding_window will be:
    //      [1, 2, 2, 2, 1]
    // TODO: Handle capping preceding/following for negative values.
    auto const preceding_window_begin = cudf::detail::make_counting_transform_iterator(
      0,
      [preceding_window] __device__(size_type i) { return thrust::min(i + 1, preceding_window); });
    auto const following_window_begin = cudf::detail::make_counting_transform_iterator(
      0, [col_size = input.size(), following_window] __device__(size_type i) {
        return thrust::min(col_size - i - 1, following_window);
      });

    // TODO: DELETEME
    auto const preceding_calc =
        [preceding_window] __device__(size_type i) { return thrust::min(i + 1, preceding_window); };
    auto const tmp_preceding = expand_to_column(preceding_calc, input.size(), stream);
    std::cout << "Ungrouped: Preceding: " << std::endl;
    cudf::test::print(*tmp_preceding);

    auto const following_calc =
        [col_size = input.size(), following_window] __device__(size_type i) {
          return thrust::min(col_size - i - 1, following_window);
        };
    auto const tmp_following = expand_to_column(following_calc, input.size(), stream);
    std::cout << "Ungrouped: Following: " << std::endl;
    cudf::test::print(*tmp_following);

    return cudf::detail::rolling_window(input,
                                        default_outputs,
                                        preceding_window_begin,
                                        following_window_begin,
                                        min_periods,
                                        agg,
                                        stream,
                                        mr);
  }
}
}  // namespace cudf::detail
