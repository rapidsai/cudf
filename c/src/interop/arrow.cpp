/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../core/exceptions.hpp"

#include <cudf/column/column.hpp>
#include <cudf/interop.hpp>
#include <cudf/interop/arrow.h>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow_device.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {
std::unique_ptr<cudf::column>& get_column_owner(cudfColumn_t col)
{
  return *reinterpret_cast<std::unique_ptr<cudf::column>*>(col->addr);
}

cudf::column& get_column(cudfColumn_t col) { return *get_column_owner(col); }

std::unique_ptr<cudf::table>& get_table_owner(cudfTable_t table)
{
  return *reinterpret_cast<std::unique_ptr<cudf::table>*>(table->addr);
}

cudf::table& get_table(cudfTable_t table) { return *get_table_owner(table); }

bool has_dictionary(ArrowSchema const* schema)
{
  if (schema == nullptr) { return false; }
  if (schema->dictionary != nullptr) { return true; }
  for (int64_t i = 0; i < schema->n_children; ++i) {
    if (has_dictionary(schema->children[i])) { return true; }
  }
  return false;
}

bool has_dictionary(cudf::column_view const& col)
{
  if (col.type().id() == cudf::type_id::DICTIONARY32) { return true; }
  for (auto child = col.child_begin(); child != col.child_end(); ++child) {
    if (has_dictionary(*child)) { return true; }
  }
  return false;
}

bool has_dictionary(cudf::table_view const& table)
{
  for (auto const& col : table) {
    if (has_dictionary(col)) { return true; }
  }
  return false;
}

void expect_no_dictionary(ArrowSchema const* schema)
{
  CUDF_EXPECTS(!has_dictionary(schema), "dictionary-encoded Arrow arrays are not supported");
}

void expect_no_dictionary(cudf::column_view const& col)
{
  CUDF_EXPECTS(!has_dictionary(col), "dictionary-encoded Arrow arrays are not supported");
}

void expect_no_dictionary(cudf::table_view const& table)
{
  CUDF_EXPECTS(!has_dictionary(table), "dictionary-encoded Arrow arrays are not supported");
}

void populate_table_schema(cudf::table_view const& table, ArrowSchema* schema)
{
  auto metadata = cudf::interop::get_table_metadata(table);
  auto result   = cudf::to_arrow_schema(table, metadata);
  ArrowSchemaMove(result.get(), schema);
}

void populate_column_schema(cudf::column_view const& col, ArrowSchema* schema)
{
  auto table    = cudf::table_view{{col}};
  auto metadata = std::vector<cudf::column_metadata>{cudf::interop::get_column_metadata(col)};
  auto result   = cudf::to_arrow_schema(table, metadata);
  ArrowSchemaMove(result->children[0], schema);
}

ArrowDeviceArray make_cpu_device_array(ArrowArray* input)
{
  ArrowDeviceArray result{};
  result.array       = *input;
  result.device_id   = -1;
  result.device_type = ARROW_DEVICE_CPU;
  result.sync_event  = nullptr;
  return result;
}
}  // namespace

extern "C" cudfError_t cudfTableFromArrow(ArrowSchema* schema,
                                          ArrowDeviceArray* input,
                                          cudaStream_t stream,
                                          cudfTable_t* table)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(table != nullptr, "output cudfTable_t pointer cannot be null");
    CUDF_EXPECTS(schema != nullptr, "ArrowSchema pointer cannot be null");
    CUDF_EXPECTS(input != nullptr, "ArrowDeviceArray pointer cannot be null");
    expect_no_dictionary(schema);

    auto view  = cudf::from_arrow_device(schema, input, rmm::cuda_stream_view{stream});
    auto owner = new std::unique_ptr<cudf::table>(std::make_unique<cudf::table>(*view));
    *table     = new cudfTable{reinterpret_cast<uintptr_t>(owner)};
  });
}

extern "C" cudfError_t cudfTableToArrow(cudfTable_t table,
                                        ArrowSchema* schema,
                                        ArrowDeviceArray* output,
                                        cudaStream_t stream)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(schema != nullptr, "ArrowSchema pointer cannot be null");
    CUDF_EXPECTS(output != nullptr, "ArrowDeviceArray pointer cannot be null");

    auto table_view = get_table(table).view();
    expect_no_dictionary(table_view);
    populate_table_schema(table_view, schema);

    auto result = cudf::to_arrow_device(table_view, rmm::cuda_stream_view{stream});
    ArrowDeviceArrayMove(result.get(), output);
  });
}

extern "C" cudfError_t cudfColumnFromArrow(ArrowSchema* schema,
                                           ArrowDeviceArray* input,
                                           cudaStream_t stream,
                                           cudfColumn_t* col)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(col != nullptr, "output cudfColumn_t pointer cannot be null");
    CUDF_EXPECTS(schema != nullptr, "ArrowSchema pointer cannot be null");
    CUDF_EXPECTS(input != nullptr, "ArrowDeviceArray pointer cannot be null");
    expect_no_dictionary(schema);

    auto view  = cudf::from_arrow_device_column(schema, input, rmm::cuda_stream_view{stream});
    auto owner = new std::unique_ptr<cudf::column>(std::make_unique<cudf::column>(*view));
    *col       = new cudfColumn{reinterpret_cast<uintptr_t>(owner)};
  });
}

extern "C" cudfError_t cudfColumnToArrow(cudfColumn_t col,
                                         ArrowSchema* schema,
                                         ArrowDeviceArray* output,
                                         cudaStream_t stream)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(schema != nullptr, "ArrowSchema pointer cannot be null");
    CUDF_EXPECTS(output != nullptr, "ArrowDeviceArray pointer cannot be null");

    auto column_view = get_column(col).view();
    expect_no_dictionary(column_view);
    populate_column_schema(column_view, schema);

    auto result = cudf::to_arrow_device(column_view, rmm::cuda_stream_view{stream});
    ArrowDeviceArrayMove(result.get(), output);
  });
}

extern "C" cudfError_t cudfTableFromArrowHost(ArrowSchema* schema,
                                              ArrowArray* input,
                                              cudaStream_t stream,
                                              cudfTable_t* table)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(table != nullptr, "output cudfTable_t pointer cannot be null");
    CUDF_EXPECTS(schema != nullptr, "ArrowSchema pointer cannot be null");
    CUDF_EXPECTS(input != nullptr, "ArrowArray pointer cannot be null");
    expect_no_dictionary(schema);

    auto device_input = make_cpu_device_array(input);
    auto result       = cudf::from_arrow_host(schema, &device_input, rmm::cuda_stream_view{stream});
    auto owner        = new std::unique_ptr<cudf::table>(std::move(result));
    *table            = new cudfTable{reinterpret_cast<uintptr_t>(owner)};
  });
}

extern "C" cudfError_t cudfTableToArrowHost(cudfTable_t table,
                                            ArrowSchema* schema,
                                            ArrowArray* output,
                                            cudaStream_t stream)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(schema != nullptr, "ArrowSchema pointer cannot be null");
    CUDF_EXPECTS(output != nullptr, "ArrowArray pointer cannot be null");

    auto table_view = get_table(table).view();
    expect_no_dictionary(table_view);
    populate_table_schema(table_view, schema);

    auto result = cudf::to_arrow_host(table_view, rmm::cuda_stream_view{stream});
    ArrowArrayMove(&result->array, output);
  });
}
