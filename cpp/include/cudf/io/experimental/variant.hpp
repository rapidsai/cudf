/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>
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
 * @brief Extract the raw VARIANT-encoded bytes of a nested object field by JSONPath-like path.
 *
 * Walks `path` step by step, descending into object values (`basic_type == 2`) at each name step.
 * Returns a `list<uint8>` column containing the raw encoded bytes of the value at the end of
 * the path for each row.
 *
 * Null is produced when the struct row is null, a name step's key is absent from the dictionary,
 * or the current value is not an object (`basic_type != 2`).
 *
 * Path grammar:
 *   path  := "$"? first_step ("." name)*
 *   first := name | "." name
 *   name  := [^.\[]+   // any byte except '.' (step separator) and '[' (reserved)
 *
 * Examples:
 *   "x"          -> top-level field "x" (leading $ optional)
 *   "$.foo"      -> top-level field "foo"
 *   "$.foo.bar"  -> object descent foo -> bar
 *
 * @param variant_column Struct column (VARIANT materialization) with `list<uint8>` children
 *                       (`metadata`, `value`), plus optional shredded siblings
 * @param path JSONPath-like path string identifying the target object field
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return `list<uint8>` column with the extracted field's encoded bytes
 *
 * @throws std::invalid_argument on empty path or malformed syntax (including bracket steps,
 *         which require array-indexing support that is not yet implemented)
 */
[[nodiscard]] std::unique_ptr<column> get_variant_field(
  column_view const& variant_column,
  std::string_view path,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Decode a VARIANT value column's blobs into a typed cuDF column.
 *
 * A null value is produced when the input row is null or the encoded type does not match
 * `desired_type`.
 *
 * @param values `list<uint8>` column of VARIANT-encoded value bytes
 * @param desired_type Target cuDF type (`STRING` or `INT8`/`INT16`/`INT32`/`INT64`)
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Typed column decoded from the VARIANT value blobs
 *
 * @throws std::invalid_argument if `values` is not a `list<uint8>` column, or if `desired_type`
 *         is not one of the supported types (`STRING` or `INT8`/`INT16`/`INT32`/`INT64`)
 */
[[nodiscard]] std::unique_ptr<column> cast_variant(
  column_view const& values,
  data_type desired_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Convenience wrapper: extract a nested object value by path and decode into a typed column.
 *
 * Semantically equivalent to extracting the field with `get_variant_field` and then decoding
 * the extracted `list<uint8>` values with `cast_variant`.
 *
 * @param variant_column Struct column (VARIANT materialization)
 * @param path JSONPath-like path string (see `get_variant_field` for syntax)
 * @param desired_type Target type: `STRING` or `INT8`/`INT16`/`INT32`/`INT64`
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Column of `desired_type`
 *
 * @throws std::invalid_argument on empty path or malformed syntax
 */
[[nodiscard]] std::unique_ptr<column> extract_variant_field(
  column_view const& variant_column,
  std::string_view path,
  data_type desired_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Encode a STRING column as a scalar VARIANT column.
 *
 * Each non-null string is encoded as a VARIANT scalar value: a short_string blob (1 header byte +
 * payload) for strings shorter than 64 bytes, or a long_string primitive blob (1 header + 4-byte
 * LE length + payload) for longer strings. The metadata blob for every row is the minimal
 * empty-dictionary encoding `{0x01, 0x00, 0x00}` (version 1, 0 keys). Null input rows produce
 * null VARIANT struct rows.
 *
 * @param strings STRING column to encode
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return VARIANT column: `STRUCT<list<uint8>, list<uint8>>` (metadata child, value child)
 *
 * @throws std::invalid_argument if `strings` is not a STRING column
 */
[[nodiscard]] std::unique_ptr<column> encode_strings_to_variant(
  column_view const& strings,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Encode a table as a VARIANT object column, one row per VARIANT.
 *
 * Each table row is encoded as a VARIANT object with one field per column. The shared metadata
 * blob stores column names as a sorted UTF-8 key dictionary (version 1, offset_size chosen to
 * fit the total key-string length). Null column values are encoded as VARIANT null primitives.
 *
 * Supported column types: `INT8`, `INT16`, `INT32`, `INT64`, `STRING`, and columns whose type is
 * `EMPTY` (treated as all-null). Other types throw. Tables must have fewer than 256 columns.
 * Total encoded output must be < 2 GiB.
 *
 * @param input Table to encode (typically produced by `cudf::io::read_json`)
 * @param column_names Column name for each column in `input`; must satisfy
 *                     `column_names.size() == input.num_columns()`
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return VARIANT column: `STRUCT<list<uint8>, list<uint8>>` (metadata child, value child)
 *
 * @throws std::invalid_argument if any column has an unsupported type, if the table has ≥ 256
 *         columns, or if `column_names.size() != input.num_columns()`
 */
[[nodiscard]] std::unique_ptr<column> encode_variant(
  cudf::table_view const& input,
  cudf::host_span<std::string const> column_names,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */
}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
