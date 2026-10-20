/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <string>
namespace cudf::jit {
/** Canonical representation metadata, excluding allocation addresses, lengths and slice offsets.
 * Child types encode list/string offset widths and dictionary index/key representations.
 * This is part of generated-source identity, hence the compilation/link cache key.
 */
inline std::string row_schema(column_view column)
{
  auto result = std::to_string(static_cast<int>(column.type().id())) + ":" +
                std::to_string(
                  is_fixed_point(column.type()) ? static_cast<int32_t>(column.type().scale()) : 0) +
                ":" + (column.nullable() ? "nullable" : "valid") + "[";
  for (size_type i = 0; i < column.num_children(); ++i) {
    result += row_schema(column.child(i)) + ";";
  }
  return result + "]";
}
}  // namespace cudf::jit
