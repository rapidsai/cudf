/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file arrow_schema_writer.hpp
 * @brief Arrow IPC schema writer implementation
 */

#pragma once

#include <cudf/detail/utilities/linked_column.hpp>
#include <cudf/io/detail/utils.hpp>
#include <cudf/io/types.hpp>
#include <cudf/strings/detail/utilities.hpp>

namespace cudf::io::parquet::detail {

/**
 * @brief Construct and return arrow schema from input parquet schema
 *
 * Recursively traverses through parquet schema to construct the arrow schema tree.
 * Serializes the arrow schema tree and stores it as the header (or metadata) of
 * an otherwise empty ipc message using flatbuffers. The ipc message is then prepended
 * with header size (padded for 16 byte alignment) and a continuation string. The final
 * string is base64 encoded and returned.
 *
 * @param linked_columns Vector of table column views
 * @param metadata Metadata of the columns of the table
 * @param write_mode Flag to indicate that we are guaranteeing a single table write
 * @param utc_timestamps Flag to indicate if timestamps are UTC
 *
 * @return The constructed arrow ipc message string
 */
std::string construct_arrow_schema_ipc_message(cudf::detail::LinkedColVector const& linked_columns,
                                               table_input_metadata const& metadata,
                                               cudf::io::detail::single_write_mode const write_mode,
                                               bool const utc_timestamps);

}  // namespace cudf::io::parquet::detail
