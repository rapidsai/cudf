/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/utilities/bit.hpp>

#include <limits>
#include <rmm/exec_policy.hpp>

namespace cudf::detail::rolling {

auto constexpr NULL_INDEX = std::numeric_limits<size_type>::min();  // For nullifying with gather.

template <null_policy null_handling, typename PrecedingIter, typename FollowingIter>
std::unique_ptr<column> nth_element(size_type n,
                                    column_view const& input,
                                    PrecedingIter preceding,
                                    FollowingIter following,
                                    size_type min_periods,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  auto const gather_iter = cudf::detail::make_counting_transform_iterator(
    0,
    [exclude_nulls = null_handling == null_policy::EXCLUDE and input.nullable(),
     preceding,
     following,
     min_periods,
     n,
     input_nullmask = input.null_mask()] __device__(size_type i) {
      // preceding[i] includes the current row.
      auto const window_size = preceding[i] + following[i];
      if (min_periods > window_size) { return NULL_INDEX; }

      auto const wrapped_n = n >= 0 ? n : (window_size + n);
      if (wrapped_n < 0 || wrapped_n > (window_size - 1)) {
        return NULL_INDEX;  // n lies outside the window.
      }

      auto const window_start = i - preceding[i] + 1;

      if (not exclude_nulls) { return window_start + wrapped_n; }

      // Must exclude nulls, and n is in range [-window_size, window_size-1].
      // Depending on n >= 0, count forwards from window_start, or backwards from window_end.
      auto const window_end = window_start + window_size;

      auto reqd_valid_count = n >= 0 ? n : (-n - 1);
      auto const nth_valid  = [&reqd_valid_count, input_nullmask](size_type j) {
        return cudf::bit_is_set(input_nullmask, j) && reqd_valid_count-- == 0;
      };

      if (n >= 0) {  // Search forwards from window_start.
        auto const begin = thrust::make_counting_iterator(window_start);
        auto const end   = begin + window_size;
        auto const found = thrust::find_if(thrust::seq, begin, end, nth_valid);
        return found == end ? NULL_INDEX : *found;
      } else {  // Search backwards from window-end.
        auto const begin =
          thrust::make_reverse_iterator(thrust::make_counting_iterator(window_end));
        auto const end   = begin + window_size;
        auto const found = thrust::find_if(thrust::seq, begin, end, nth_valid);
        return found == end ? NULL_INDEX : *found;
      }
    });

  auto gathered = cudf::detail::gather(table_view{{input}},
                                       gather_iter,
                                       gather_iter + input.size(),
                                       cudf::out_of_bounds_policy::NULLIFY,
                                       stream,
                                       mr)
                    ->release();
  return std::move(gathered[0]);
}

}  // namespace cudf::detail::rolling