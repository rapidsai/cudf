/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
 *      a) A `DURATION_DAYS` scalar, for use with a `TIMESTAMP_DAYS` orderby column
 *      b) An `INT32` scalar, for use with an `INT32` orderby column
 *   2. "unbounded", indicating that the bounds stretch to the first/last
 *      row in the group.
 */
struct range_window_bounds {
 public:
  /**
   * @brief Factory method to construct a bounded window boundary.
   *
   * @param boundary Finite window boundary
   * @return A bounded window boundary object
   */
  static range_window_bounds get(scalar const& boundary);

  /**
   * @brief Factory method to construct an unbounded window boundary.
   *
   * @param type type The datatype of the window boundary
   * @return  An unbounded window boundary object
   */
  static range_window_bounds unbounded(data_type type);

  /**
   * @brief Whether or not the window is unbounded
   *
   * @return true If window is unbounded
   * @return false If window is of finite bounds
   */
  [[nodiscard]] bool is_unbounded() const { return _is_unbounded; }

  /**
   * @brief Returns the underlying scalar value for the bounds
   *
   * @return  The underlying scalar value for the bounds
   */
  [[nodiscard]] scalar const& range_scalar() const { return *_range_scalar; }

  range_window_bounds(range_window_bounds const&) = default;  ///< Copy constructor
  range_window_bounds() = default;  // Required for use as return types from dispatch functors.

 private:
  const bool _is_unbounded{true};
  std::shared_ptr<scalar> _range_scalar{nullptr};  // To enable copy construction/assignment.

  range_window_bounds(bool is_unbounded_, std::unique_ptr<scalar> range_scalar_);
};

}  // namespace cudf
