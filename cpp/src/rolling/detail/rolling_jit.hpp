/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
