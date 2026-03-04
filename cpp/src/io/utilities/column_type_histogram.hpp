/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

namespace cudf {
namespace io {

/**
 * @brief Per-column histogram struct containing detected occurrences of each dtype
 */
struct column_type_histogram {
  cudf::size_type null_count{};
  cudf::size_type float_count{};
  cudf::size_type datetime_count{};
  cudf::size_type string_count{};
  cudf::size_type negative_small_int_count{};
  cudf::size_type positive_small_int_count{};
  cudf::size_type big_int_count{};
  cudf::size_type bool_count{};
  auto total_count() const
  {
    return null_count + float_count + datetime_count + string_count + negative_small_int_count +
           positive_small_int_count + big_int_count + bool_count;
  }
};

}  // namespace io
}  // namespace cudf
