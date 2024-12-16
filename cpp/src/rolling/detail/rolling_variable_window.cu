/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/functional>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>

namespace cudf::detail {

// Applies a variable-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (preceding_window.is_empty() || following_window.is_empty() || input.is_empty()) {
    return cudf::detail::empty_output_for_rolling_aggregation(input, agg);
  }

  CUDF_EXPECTS(preceding_window.type().id() == type_id::INT32 &&
                 following_window.type().id() == type_id::INT32,
               "preceding_window/following_window must have type_id::INT32 type");

  CUDF_EXPECTS(preceding_window.size() == input.size() && following_window.size() == input.size(),
               "preceding_window/following_window size must match input size");

  if (agg.kind == aggregation::CUDA || agg.kind == aggregation::PTX) {
    // TODO: In future, might need to clamp preceding/following to column boundaries.
    return cudf::detail::rolling_window_udf(input,
                                            preceding_window.begin<size_type>(),
                                            "cudf::size_type*",
                                            following_window.begin<size_type>(),
                                            "cudf::size_type*",
                                            min_periods,
                                            agg,
                                            stream,
                                            mr);
  } else {
    auto defaults_col =
      cudf::is_dictionary(input.type()) ? dictionary_column_view(input).indices() : input;
    // Clamp preceding/following to column boundaries.
    // E.g. If preceding_window == [2, 2, 2, 2, 2] for a column of 5 elements, the new
    // preceding_window will be: [1, 2, 2, 2, 1]
    auto const preceding_window_begin = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<size_type>(
        [preceding = preceding_window.begin<size_type>()] __device__(size_type i) {
          return thrust::min(i + 1, preceding[i]);
        }));
    auto const following_window_begin = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<size_type>(
        [col_size = input.size(), following = following_window.begin<size_type>()] __device__(
          size_type i) { return thrust::min(col_size - i - 1, following[i]); }));
    return cudf::detail::rolling_window(input,
                                        empty_like(defaults_col)->view(),
                                        preceding_window_begin,
                                        following_window_begin,
                                        min_periods,
                                        agg,
                                        stream,
                                        mr);
  }
}

}  // namespace cudf::detail
