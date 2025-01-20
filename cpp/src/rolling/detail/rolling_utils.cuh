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

#pragma once

#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>

#include <cuda/functional>

namespace CUDF_EXPORT cudf {

namespace detail::rolling {

/**
 * @brief A group descriptor for an ungrouped rolling window.
 *
 * @param num_rows The number of rows to be rolled over.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped iterator
 * construction.
 */
struct ungrouped {
  cudf::size_type num_rows;

  [[nodiscard]] __device__ constexpr cudf::size_type label(cudf::size_type) const noexcept
  {
    return 0;
  }
  [[nodiscard]] __device__ constexpr cudf::size_type start(cudf::size_type) const noexcept
  {
    return 0;
  }
  [[nodiscard]] __device__ constexpr cudf::size_type end(cudf::size_type) const noexcept
  {
    return num_rows;
  }
};

/**
 * @brief A group descriptor for a grouped rolling window.
 *
 * @param labels The group labels, mapping from input rows to group.
 * @param offsets The group offsets providing the endpoints of each group.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped iterator
 * construction.
 */
struct grouped {
  // Taking raw pointers here to avoid stealing two registers for the sizes which are never needed.
  cudf::size_type const* labels;
  cudf::size_type const* offsets;

  [[nodiscard]] __device__ constexpr cudf::size_type label(cudf::size_type i) const noexcept
  {
    return labels[i];
  }
  [[nodiscard]] __device__ constexpr cudf::size_type start(cudf::size_type label) const noexcept
  {
    return offsets[label];
  }
  [[nodiscard]] __device__ constexpr cudf::size_type end(cudf::size_type label) const noexcept
  {
    return offsets[label + 1];
  }
};

enum class direction : bool {
  PRECEDING,
  FOLLOWING,
};

template <direction Direction, typename Grouping>
struct fixed_window_clamper {
  Grouping groups;
  cudf::size_type delta;
  [[nodiscard]] __device__ constexpr cudf::size_type operator()(cudf::size_type i) const
  {
    auto const label = groups.label(i);
    // i is contained in [start, end)
    auto const start = groups.start(label);
    auto const end   = groups.end(label);
    if constexpr (Direction == direction::PRECEDING) {
      return cuda::std::min(i + 1 - start, cuda::std::max(delta, i + 1 - end));
    } else {
      return cuda::std::max(start - i - 1, cuda::std::min(delta, end - i - 1));
    }
  }
};

/**
 * @brief Construct a clamped counting iterator for a row-based window offset
 *
 * @tparam Direction the direction of the window `PRECEDING` or `FOLLOWING`.
 * @tparam Grouping the group specification.
 * @param delta the window offset.
 * @param grouper the grouping object.
 *
 * @return An iterator suitable for passing to `cudf::detail::rolling_window`
 */
template <direction Direction, typename Grouping>
[[nodiscard]] auto inline make_clamped_window_iterator(cudf::size_type delta, Grouping grouper)
{
  return cudf::detail::make_counting_transform_iterator(
    cudf::size_type{0}, fixed_window_clamper<Direction, Grouping>{grouper, delta});
}
}  // namespace detail::rolling
}  // namespace CUDF_EXPORT cudf
