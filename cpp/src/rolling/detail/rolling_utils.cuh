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
#include <cudf/detail/rolling.hpp>
#include <cudf/types.hpp>

#include <cuda/functional>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {

namespace detail::rolling {

/**
 * @brief Information about group bounds of the current row's group.
 */
struct range_group_info {
  size_type const group_start;
  size_type const group_end;
  size_type const null_start;
  size_type const null_end;
  size_type const non_null_start;
  size_type const non_null_end;
};

/**
 * @brief A group descriptor for an ungrouped rolling window.
 *
 * @param num_rows The number of rows to be rolled over.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped iterator
 * construction.
 */
struct ungrouped {
  cudf::size_type const num_rows;

  static constexpr bool has_nulls{false};

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

  /**
   * @brief Return information about the current row.
   *
   * @param i The row
   * @returns `range_group_info` with the information about the row.
   */
  [[nodiscard]] __device__ constexpr range_group_info row_info(size_type i) const noexcept
  {
    return {0, num_rows, 0, 0, 0, num_rows};
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

  static constexpr bool has_nulls{false};

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

  /**
   * @copydoc ungrouped::row_info
   */
  [[nodiscard]] __device__ constexpr range_group_info row_info(size_type i) const noexcept
  {
    auto const label       = labels[i];
    auto const group_start = offsets[label];
    auto const group_end   = offsets[label + 1];
    return {group_start, group_end, group_start, group_start, group_start, group_end};
  }
};

/**
 * @brief A group descriptor for an ungrouped rolling window with nulls
 *
 * @param nulls_at_start Are the nulls at the start or end?
 * @param num_rows The number of rows to be rolled over.
 * @param null_count The number of nulls.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped
 * iterator construction.
 */
struct ungrouped_with_nulls {
  bool const nulls_at_start;
  cudf::size_type const num_rows;
  cudf::size_type const null_count;

  static constexpr bool has_nulls{true};
  /**
   * @copydoc ungrouped::row_info
   */
  [[nodiscard]] __device__ constexpr range_group_info row_info(size_type i) const noexcept
  {
    if (nulls_at_start) {
      return {0, num_rows, 0, null_count, null_count, num_rows};
    } else {
      return {num_rows, null_count, num_rows - null_count, num_rows, 0, num_rows - null_count};
    }
  }
};

/**
 * @brief A group descriptor for a grouped rolling window with nulls
 *
 * @param nulls_at_start Are the nulls at the start of each group?
 * @param labels The group labels, mapping from input rows to group.
 * @param offsets The group offsets providing the endpoints of each group.
 * @param null_counts The null counts per group.
 * @param orderby The orderby column, sorted groupwise.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped
 * iterator construction.
 */
struct grouped_with_nulls {
  bool const nulls_at_start;
  // Taking raw pointers here to avoid stealing three registers for the sizes which are never
  // needed.
  cudf::size_type const* labels;
  cudf::size_type const* offsets;
  cudf::size_type const* null_counts;

  static constexpr bool has_nulls{true};
  /**
   * @copydoc ungrouped::row_info
   */
  [[nodiscard]] __device__ constexpr range_group_info row_info(size_type i) const noexcept
  {
    auto const label       = labels[i];
    auto const null_count  = null_counts[label];
    auto const group_start = offsets[label];
    auto const group_end   = offsets[label + 1];
    if (nulls_at_start) {
      return {group_start,
              group_end,
              group_start,
              group_start + null_count,
              group_start + null_count,
              group_end};
    } else {
      return {group_start,
              group_end,
              group_end - null_count,
              group_end,
              group_start,
              group_end - null_count};
    }
  }
};

template <direction Direction, typename Grouping>
struct fixed_window_clamper {
  Grouping groups;
  cudf::size_type delta;
  static_assert(cuda::std::is_same_v<Grouping, ungrouped> ||
                  cuda::std::is_same_v<Grouping, grouped>,
                "Invalid grouping descriptor");

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
