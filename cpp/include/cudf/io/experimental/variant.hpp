/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string_view>

/**
 * @file
 * @brief Utilities for Parquet VARIANT columns read as `struct` of `list<uint8>` children.
 */

namespace CUDF_EXPORT cudf {
namespace io::parquet::experimental {
/**
 * @addtogroup io_readers
 * @{
 */

/**
 * @brief Extract the raw VARIANT-encoded bytes of a nested field by JSONPath-like path.
 *
 * Path grammar:
 *   path  := "$"? first_step step*
 *   first := name | "." name | "[" index "]"
 *   step  := "." name | "[" index "]"
 *   name  := any sequence of bytes other than '.' or '['
 *   index := non-negative base-10 integer (leading zeros are allowed, e.g. "[01]" == "[1]")
 *
 * Examples:
 *   "x"            -> top-level field "x" (leading $ optional)
 *   "$.foo"        -> top-level field "foo"
 *   "$.foo.bar"    -> object descent foo -> bar
 *   "$[0]"         -> first element of a top-level array
 *   "$.a[0].b"     -> object key "a" -> first array element -> object key "b"
 *
 * @param variant_column Struct column (VARIANT materialization) with `list<uint8>` children
 *                       (`metadata`, `value`), plus optional shredded siblings
 * @param path JSONPath-like path string identifying the target field
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @param status_out If non-null, receives a `UINT8` column of `variant_operation_status` values
 *        (one per row, aligned with the returned value column). SQL-null input rows produce a null
 *        status entry. All other rows receive a valid status: `success`, `missing_path`,
 *        `variant_null`, or `malformed_variant`. Missing-path and malformed rows produce a SQL-null
 *        output regardless; `success` and `variant_null` rows produce non-null output with the
 *        resolved bytes. When `nullptr` (the default), no status column is produced.
 * @return `list<uint8>` column with the extracted value's encoded bytes. A row is null when the
 *         input row is null, a name is absent, an index is out of bounds, a step does not match
 *         the current value, or bytes are malformed. Encoded VARIANT-null terminal values are
 *         always returned as the raw VARIANT-null bytes (non-null output); callers can detect them
 *         via the status column or by inspecting the returned bytes.
 *
 * @throws std::invalid_argument on empty path or malformed syntax (`[*]` wildcards, negative
 *         indices, out-of-range indices, and quoted names inside `[...]` are not supported)
 */
[[nodiscard]] std::unique_ptr<column> get_variant_field(
  column_view const& variant_column,
  std::string_view path,
  std::unique_ptr<column>* status_out = nullptr,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr   = cudf::get_current_device_resource_ref());

/**
 * @brief Decode a VARIANT value column's blobs into a typed cuDF column.
 *
 * A null value is produced when the input row is null or the encoded type does not match
 * `desired_type`.
 *
 * @param values `list<uint8>` column of VARIANT-encoded value bytes
 * @param desired_type Target cuDF type (`STRING`, `INT8`/`INT16`/`INT32`/`INT64`,
 *        `FLOAT32`/`FLOAT64`, or `BOOL8`)
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @param incoming_status Optional status column from a prior `get_variant_field` call. When
 *        non-null, rows whose status is null remain null in both the output and the new status
 *        column, and rows with a non-`success` status are propagated unchanged (output is null,
 *        status is preserved). Only `success` rows are decoded.
 * @param status_out If non-null, receives a `UINT8` column of `variant_operation_status` values
 *        aligned with the output column. SQL-null input rows (or rows whose incoming status is
 *        null) produce null status entries.
 * @return Typed column decoded from the VARIANT value blobs
 *
 * @throws std::invalid_argument if `values` is not a `list<uint8>` column, or if `desired_type`
 *         is not one of the supported types (`STRING`, `INT8`/`INT16`/`INT32`/`INT64`,
 *         `FLOAT32`/`FLOAT64`, or `BOOL8`)
 */
[[nodiscard]] std::unique_ptr<column> cast_variant(
  column_view const& values,
  data_type desired_type,
  column_view const* incoming_status  = nullptr,
  std::unique_ptr<column>* status_out = nullptr,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr   = cudf::get_current_device_resource_ref());

/**
 * @brief Convenience wrapper: extract a nested object value by path and decode into a typed column.
 *
 * Semantically equivalent to extracting the field with `get_variant_field` and then decoding
 * the extracted `list<uint8>` values with `cast_variant`.
 *
 * @param variant_column Struct column (VARIANT materialization)
 * @param path JSONPath-like path string (see `get_variant_field` for syntax)
 * @param desired_type Target type: `STRING`, `INT8`/`INT16`/`INT32`/`INT64`,
 *        `FLOAT32`/`FLOAT64`, or `BOOL8`
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @param status_out If non-null, receives a `UINT8` column of `variant_operation_status` values
 *        aligned with the output column, combining extraction and decode outcomes. SQL-null input
 *        rows produce null status entries.
 * @return Column of `desired_type`
 *
 * @throws std::invalid_argument on empty path or malformed syntax
 */
[[nodiscard]] std::unique_ptr<column> extract_variant_field(
  column_view const& variant_column,
  std::string_view path,
  data_type desired_type,
  std::unique_ptr<column>* status_out = nullptr,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr   = cudf::get_current_device_resource_ref());

/** @} */
}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
