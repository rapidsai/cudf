/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "core/exceptions.hpp"

#include <cudf/column.h>
#include <cudf/column/column.hpp>
#include <cudf/table.h>
#include <cudf/table/table.hpp>

#include <cstdint>
#include <memory>

namespace {
std::unique_ptr<cudf::table>& get_table_owner(cudfTable_t table)
{
  return *reinterpret_cast<std::unique_ptr<cudf::table>*>(table->addr);
}

cudf::table& get_table(cudfTable_t table) { return *get_table_owner(table); }
}  // namespace

extern "C" cudfError_t cudfTableDestroy(cudfTable_t table)
{
  return cudf::c::translate_exceptions([=] {
    delete reinterpret_cast<std::unique_ptr<cudf::table>*>(table->addr);
    delete table;
  });
}

extern "C" cudfError_t cudfTableGetNumColumns(cudfTable_t table, int32_t* num_cols)
{
  return cudf::c::translate_exceptions(
    [=] { *num_cols = static_cast<int32_t>(get_table(table).num_columns()); });
}

extern "C" cudfError_t cudfTableGetNumRows(cudfTable_t table, int64_t* num_rows)
{
  return cudf::c::translate_exceptions(
    [=] { *num_rows = static_cast<int64_t>(get_table(table).num_rows()); });
}

extern "C" cudfError_t cudfTableGetColumn(cudfTable_t table, int32_t index, cudfColumn_t* col)
{
  return cudf::c::translate_exceptions([=] {
    auto const& source_column = get_table(table).get_column(index);
    auto column_copy          = std::make_unique<cudf::column>(source_column);
    auto owner                = new std::unique_ptr<cudf::column>(std::move(column_copy));
    *col                      = new cudfColumn{reinterpret_cast<uintptr_t>(owner)};
  });
}

extern "C" cudfError_t cudfTableCreate(cudfTable_t* table)
{
  return cudf::c::translate_exceptions([=] {
    auto owner = new std::unique_ptr<cudf::table>(
      std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{}));
    *table = new cudfTable{reinterpret_cast<uintptr_t>(owner)};
  });
}

// NOTE: This file must be added to c/CMakeLists.txt's add_library sources list.
// Add: src/column.cpp src/table.cpp to the cudf_c library target.
