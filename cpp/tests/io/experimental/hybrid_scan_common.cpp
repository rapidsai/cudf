/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_common.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <format>

cudf::test::strings_column_wrapper constant_strings(cudf::size_type value)
{
  CUDF_EXPECTS(value >= 0 && value <= 9999, "String value must be between 0000 and 9999");

  auto elements = thrust::make_transform_iterator(thrust::make_constant_iterator(value),
                                                  [](auto i) { return std::format("{:04d}", i); });
  return cudf::test::strings_column_wrapper(elements, elements + num_ordered_rows);
}
