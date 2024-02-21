/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

namespace cudf {

namespace detail {

template <class T>
T minimum(T a, T b)
{
  return b < a ? b : a;
}

struct preceding_window_wrapper {
  cudf::size_type const* d_group_offsets;
  cudf::size_type const* d_group_labels;
  cudf::size_type preceding_window;

  cudf::size_type operator[](cudf::size_type idx)
  {
    auto group_label = d_group_labels[idx];
    auto group_start = d_group_offsets[group_label];
    return minimum(preceding_window, idx - group_start + 1);  // Preceding includes current row.
  }
};

struct following_window_wrapper {
  cudf::size_type const* d_group_offsets;
  cudf::size_type const* d_group_labels;
  cudf::size_type following_window;

  cudf::size_type operator[](cudf::size_type idx)
  {
    auto group_label = d_group_labels[idx];
    auto group_end =
      d_group_offsets[group_label +
                      1];  // Cannot fall off the end, since offsets is capped with `input.size()`.
    return minimum(following_window, (group_end - 1) - idx);
  }
};

}  // namespace detail

}  // namespace cudf
