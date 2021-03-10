/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/scalar/scalar.hpp>

namespace cudf {

/**
 * @brief Abstraction for window boundary sizes, to be used with
 *        `grouped_range_rolling_window()`.
 *
 * Similar to `window_bounds` in `grouped_rolling_window()`, `range_window_bounds`
 * represents window boundaries for use with `grouped_range_rolling_window()`.
 * A window may be specified as either of the following:
 *   1. A fixed-width numeric scalar value. E.g.
 *      a) A `duration_D` scalar, for use with a `TIMESTAMP_DAYS` orderby column
 *      b) An `INT32` scalar, for use with an `INT32` orderby column
 *   2. "unbounded", indicating that the bounds stretch to the first/last
 *      row in the group.
 */
struct range_window_bounds {
 public:
  /**
   * @brief Factory method to construct a bounded window boundary.
   *
   * @param value Finite window boundary
   *
   */
  static range_window_bounds get(std::unique_ptr<scalar>&& scalar_)
  {
    return range_window_bounds(false, std::move(scalar_));
  }

  /**
   * @brief Factory method to construct an unbounded window boundary.
   *
   * @param @type The datatype of the window boundary
   */
  static range_window_bounds unbounded(data_type type);

  /**
   * @brief Whether or not the window is unbounded
   *
   * @return true If window is unbounded
   * @return false If window is of finite bounds
   */
  bool is_unbounded() const { return _is_unbounded; }

  /**
   * @brief Returns the underlying scalar value for the bounds
   */
  scalar const& range_scalar() const { return *_range_scalar; }

  /**
   * @brief Rescale underlying scalar to the specified target type.
   *
   * A range_window_bounds is used in conjunction with the orderby column
   * in `grouped_range_rolling_window()`. Its scalar value is compared against
   * the rows in the orderby column to determine the width of the window.
   *
   * For instance, if the orderby column is integral (INT32), the range_window_bounds
   * must also be integral (INT32). No scaling is required for comparison.
   *
   * However, if the orderby column is in TIMESTAMP_SECONDS, the range_window_bounds
   * value must be specified as a comparable duration (between timestamp rows).
   * The duration may be of similar precision (DURATION_SECONDS) or lower (DURATION_DAYS).
   *
   * `scale_to()` scales the bounds scalar from its original granularity (e.g. DURATION_DAYS)
   * to the orderby column's granularity (DURATION_SECONDS), before comparions are made.
   *
   * @param target_type The type to which the range_window_bounds scalar must be scaled
   * @param stream The CUDA stream to use for device memory operations
   * @param mr Device memory resource used to allocate the scalar
   */
  void scale_to(data_type target_type,
                rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

 private:
  const bool _is_unbounded{false};
  std::unique_ptr<scalar> _range_scalar{nullptr};  // Required: Reseated in `scale_to()`.
                                                   // Allocates new scalar.

  range_window_bounds(bool is_unbounded_, std::unique_ptr<scalar>&& range_scalar_)
    : _is_unbounded{is_unbounded_}, _range_scalar{std::move(range_scalar_)}
  {
    assert_invariants();
  }

  void assert_invariants() const;
};

}  // namespace cudf
