/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "core/exceptions.hpp"

#include <cudf/column.h>
#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <memory>

namespace {
cudfDataType_t to_cudf_data_type(cudf::data_type type)
{
  return cudfDataType_t{static_cast<cudfTypeId_t>(static_cast<int32_t>(type.id())), type.scale()};
}

std::unique_ptr<cudf::column>& get_column_owner(cudfColumn_t col)
{
  return *reinterpret_cast<std::unique_ptr<cudf::column>*>(col->addr);
}
}  // namespace

extern "C" cudfError_t cudfColumnDestroy(cudfColumn_t col)
{
  return cudf::c::translate_exceptions([=] {
    delete reinterpret_cast<std::unique_ptr<cudf::column>*>(col->addr);
    delete col;
  });
}

extern "C" cudfError_t cudfColumnCreate(cudfColumn_t* col)
{
  return cudf::c::translate_exceptions([=] {
    auto owner = new std::unique_ptr<cudf::column>(std::make_unique<cudf::column>());
    *col       = new cudfColumn{reinterpret_cast<uintptr_t>(owner)};
  });
}

extern "C" cudfError_t cudfColumnGetType(cudfColumn_t col, cudfDataType_t* type)
{
  return cudf::c::translate_exceptions(
    [=] { *type = to_cudf_data_type(get_column_owner(col)->type()); });
}

extern "C" cudfError_t cudfColumnGetSize(cudfColumn_t col, int64_t* size)
{
  return cudf::c::translate_exceptions(
    [=] { *size = static_cast<int64_t>(get_column_owner(col)->size()); });
}

extern "C" cudfError_t cudfColumnGetNullCount(cudfColumn_t col, int64_t* null_count)
{
  return cudf::c::translate_exceptions(
    [=] { *null_count = static_cast<int64_t>(get_column_owner(col)->null_count()); });
}

// NOTE: This file must be added to c/CMakeLists.txt's add_library sources list.
// Add: src/column.cpp src/table.cpp to the cudf_c library target.
