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
namespace io::parquet {
/**
 * @addtogroup io_readers
 * @{
 * @file
 * @brief Utilities for Parquet VARIANT columns read as `struct` of `list<uint8>` children.
 */

/**
 * @brief Extract the raw VARIANT-encoded bytes of a nested field by JSONPath-like path string.
 *
 * Walks `path` step by step, descending into object values (`basic_type == 2`) at name steps and
 * into array values (`basic_type == 3`) at integer index steps. Returns a new VARIANT struct column
 * (`struct<list<uint8>, list<uint8>>`) where child 0 is the original `metadata` (copied once) and
 * child 1 contains the raw encoded bytes of the value at the end of the path for each row. The
 * metadata dictionary is shared across all nesting levels in the Variant spec, so passing it
 * through preserves validity for further operations on the extracted values.
 *
 * Null is produced when the struct row is null, a name step's key is absent from the dictionary,
 * an index step is out of bounds, or a step's kind does not match the current value's type (index
 * step on an object or name step on an array).
 *
 * Path grammar (subset of JSONPath; no filters or expressions):
 *   path   := "$"? first_step step*
 *   step   := "." name | "[" index "]" | "[" quoted "]"
 *   name   := [A-Za-z_][A-Za-z0-9_]*   (first step may also be a bare name)
 *   quoted := "'...'" | "\"...\""   (no escape handling; may not contain the wrapping quote char)
 *   index  := non-negative base-10 integer
 *
 * Examples:
 *   "x"                         -> top-level field "x" (leading $ optional)
 *   "$.foo.bar"                 -> object descent foo -> bar
 *   "$.foo[0].bar"              -> object -> array[0] -> object
 *   "$[0].item002[0].item003"   -> array[0] -> obj.item002 -> array[0] -> obj.item003
 *   "$['weird.key'][2]"         -> object key literally named "weird.key", then array[2]
 *
 * @param variant_column Struct column (VARIANT materialization) with `list<uint8>` children
 * @param path JSONPath-like path string identifying the target value
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return VARIANT struct column with the extracted field's encoded bytes
 *
 * @throws std::invalid_argument on empty path, `[*]` wildcard, negative index, or malformed syntax
 */
std::unique_ptr<column> get_variant_field(
  column_view const& variant_column,
  std::string_view path,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Decode a VARIANT struct column's `value` blobs into a typed cuDF column.
 *
 * Each row's `value` child (child 1) is interpreted as a Variant-encoded primitive and decoded
 * into `desired_type`. Only `INT32` and `STRING` are currently supported. Null is produced when
 * the struct row is null or the encoded type does not match `desired_type`.
 *
 * @param variant_column Struct column (VARIANT materialization) with `list<uint8>` children
 * @param desired_type Target cuDF type (`STRING` or `INT32`)
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Typed column decoded from the VARIANT value blobs
 */
std::unique_ptr<column> cast_variant(
  column_view const& variant_column,
  data_type desired_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Convenience wrapper: extract a nested value by path and decode into a typed column.
 *
 * Equivalent to `cast_variant(get_variant_field(variant_column, path), desired_type)`.
 *
 * @param variant_column Struct column (VARIANT materialization)
 * @param path JSONPath-like path string (see `get_variant_field` for syntax)
 * @param desired_type Target type: `STRING` or `INT32` only
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Column of `desired_type` with one row per struct row; null where the struct row is null,
 *         any step along the path misses, or the final encoded value does not match `desired_type`
 *
 * @throws std::invalid_argument on empty path, `[*]` wildcard, negative index, or malformed syntax
 */
std::unique_ptr<column> extract_variant_field(
  column_view const& variant_column,
  std::string_view path,
  data_type desired_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */
}  // namespace io::parquet
}  // namespace CUDF_EXPORT cudf
