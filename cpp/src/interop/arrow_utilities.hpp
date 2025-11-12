/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <nanoarrow/nanoarrow.h>

namespace cudf {
namespace detail {

/**
 * @brief constants for buffer indexes of Arrow arrays
 *
 */
static constexpr int validity_buffer_idx         = 0;
static constexpr int fixed_width_data_buffer_idx = 1;

/**
 * @brief Map ArrowType id to cudf column type id
 *
 * @param arrow_view SchemaView to pull the logical and storage types from
 * @return Column type id
 */
data_type arrow_to_cudf_type(ArrowSchemaView const* arrow_view);

/**
 * @brief Map cudf column type id to ArrowType id
 *
 * @param id Column type id
 * @return ArrowType id
 */
ArrowType id_to_arrow_type(cudf::type_id id);

/**
 * @brief Map cudf column type id to the storage type for Arrow
 *
 * Specifically this is for handling the underlying storage type of
 * timestamps and durations.
 *
 * @param id column type id
 * @return ArrowType storage type
 */
ArrowType id_to_arrow_storage_type(cudf::type_id id);

/**
 * @brief Helper to initialize ArrowArray struct
 *
 * @param arr Pointer to ArrowArray to initialize
 * @param storage_type The type to initialize with
 * @param column view for column to get the length and null count from
 * @return nanoarrow status code, should be NANOARROW_OK if there are no errors
 */
int initialize_array(ArrowArray* arr, ArrowType storage_type, cudf::column_view column);

}  // namespace detail
}  // namespace cudf
