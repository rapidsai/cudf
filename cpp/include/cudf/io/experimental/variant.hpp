/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <string_view>

namespace CUDF_EXPORT cudf {
namespace io::parquet::experimental {
/**
 * @addtogroup io_readers
 * @{
 * @file
 * @brief Utilities for Parquet VARIANT columns read as `struct` of `list<uint8>` children.
 */

/**
 * @brief Extract the raw VARIANT-encoded bytes of a nested object field by JSONPath-like path.
 *
 * Walks `path` step by step, descending into object values (`basic_type == 2`) at each name step.
 * Returns a new VARIANT struct column (`struct<list<uint8>, list<uint8>>`) where child 0 is the
 * original `metadata` (copied once) and child 1 contains the raw encoded bytes of the value at
 * the end of the path for each row. The metadata dictionary is shared across all nesting levels
 * in the Variant spec, so passing it through preserves validity for further operations on the
 * extracted values.
 *
 * Null is produced when the struct row is null, a name step's key is absent from the dictionary,
 * or the current value is not an object (`basic_type != 2`).
 *
 * Path grammar:
 *   path  := "$"? first_step ("." name)*
 *   first := name | "." name
 *   name  := [A-Za-z_][A-Za-z0-9_]*
 *
 * Examples:
 *   "x"          -> top-level field "x" (leading $ optional)
 *   "$.foo"      -> top-level field "foo"
 *   "$.foo.bar"  -> object descent foo -> bar
 *
 * @param variant_column Struct column (VARIANT materialization) with `list<uint8>` children
 * @param path JSONPath-like path string identifying the target object field
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return VARIANT struct column with the extracted field's encoded bytes
 *
 * @throws std::invalid_argument on empty path or malformed syntax (including bracket steps,
 *         which are reserved for a later phase)
 */
[[nodiscard]] std::unique_ptr<column> get_variant_field(
  column_view const& variant_column,
  std::string_view path,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Decode a VARIANT struct column's `value` blobs into a typed cuDF column.
 *
 * Each row's `value` child (child 1) is interpreted as a Variant-encoded primitive and decoded
 * into `desired_type`. Supported targets: `INT8`, `INT16`, `INT32`, `INT64`, `STRING`. Type
 * matching is strict: an INT16-encoded value cast to `INT32` produces null. Null is produced
 * when the struct row is null or the encoded type does not match `desired_type`.
 *
 * @param variant_column Struct column (VARIANT materialization) with `list<uint8>` children
 * @param desired_type Target cuDF type (`STRING` or `INT8`/`INT16`/`INT32`/`INT64`)
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Typed column decoded from the VARIANT value blobs
 */
[[nodiscard]] std::unique_ptr<column> cast_variant(
  column_view const& variant_column,
  data_type desired_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Convenience wrapper: extract a nested object value by path and decode into a typed column.
 *
 * Equivalent to `cast_variant(get_variant_field(variant_column, path), desired_type)`.
 *
 * @param variant_column Struct column (VARIANT materialization)
 * @param path JSONPath-like path string (see `get_variant_field` for syntax)
 * @param desired_type Target type: `STRING` or `INT8`/`INT16`/`INT32`/`INT64`
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Column of `desired_type` with one row per struct row; null where the struct row is null,
 *         any step along the path misses, or the final encoded value does not match `desired_type`
 *
 * @throws std::invalid_argument on empty path or malformed syntax
 */
[[nodiscard]] std::unique_ptr<column> extract_variant_field(
  column_view const& variant_column,
  std::string_view path,
  data_type desired_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */
}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
