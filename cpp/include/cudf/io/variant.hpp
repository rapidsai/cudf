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

#include <string>

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup io_readers
 * @{
 * @file
 * @brief Utilities for Parquet VARIANT columns read as `struct` of `list<uint8>` children.
 */

/**
 * @brief Extract the raw VARIANT-encoded bytes of a top-level field by name.
 *
 * Returns a new VARIANT struct column (`struct<list<uint8>, list<uint8>>`) where child 0 is the
 * original `metadata` (copied) and child 1 contains the raw encoded bytes of the named field's
 * value for each row. The metadata dictionary is shared across all nesting levels in the Variant
 * spec, so passing it through preserves validity for further operations on nested extracted values.
 *
 * Null is produced when the struct row is null, the field is absent, or the value blob is not an
 * object. Only top-level keys are supported (no nested paths).
 *
 * @param variant_column Struct column (VARIANT materialization) with `list<uint8>` children
 * @param field_name UTF-8 field name (case-sensitive)
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return VARIANT struct column with the extracted field's encoded bytes
 */
std::unique_ptr<column> get_variant_field(
  column_view const& variant_column,
  std::string const& field_name,
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
 * @brief Convenience wrapper: extract a top-level field and decode it into a typed column.
 *
 * Equivalent to `cast_variant(get_variant_field(variant_column, field_name), desired_type)`.
 *
 * @param variant_column Struct column (VARIANT materialization)
 * @param field_name UTF-8 field name (case-sensitive)
 * @param desired_type Target type: `STRING` or `INT32` only
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Column of `desired_type` with one row per struct row; null where the struct row is null,
 *         the field is missing, or the encoded value does not match `desired_type`
 */
std::unique_ptr<column> extract_variant_field(
  column_view const& variant_column,
  std::string const& field_name,
  data_type desired_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */
}  // namespace CUDF_EXPORT cudf
