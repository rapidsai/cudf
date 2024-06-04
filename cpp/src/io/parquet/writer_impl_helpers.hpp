/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file writer_impl_helpers.hpp
 * @brief Helper function implementation for Parquet writer
 */

#pragma once
#include "io/comp/nvcomp_adapter.hpp"
#include "parquet_common.hpp"

#include <cudf/detail/utilities/linked_column.hpp>
#include <cudf/io/detail/parquet.hpp>

namespace cudf::io::parquet::detail {

using namespace cudf::io::detail;

/**
 * @brief Function that translates GDF compression to parquet compression.
 *
 * @param compression The compression type
 * @return The supported Parquet compression
 */
Compression to_parquet_compression(compression_type compression);

nvcomp::compression_type to_nvcomp_compression_type(Compression codec);

uint32_t page_alignment(Compression codec);

size_t max_compression_output_size(Compression codec, uint32_t compression_blocksize);

/**
 * @brief Fill the table metadata with default column names.
 *
 * @param table_meta The table metadata to fill
 */
void fill_table_meta(std::unique_ptr<table_input_metadata> const& table_meta);

/**
 * @brief Returns ``true`` if the column is nullable or if the write mode is not
 *        set to write the table all at once instead of chunked
 *
 * @param column A view of the column
 * @param column_metadata Metadata of the column
 * @param write_mode Flag to indicate that we are guaranteeing a single table write
 *
 * @return Whether the column is nullable.
 */
[[nodiscard]] bool is_col_nullable(cudf::detail::LinkedColPtr const& column,
                                   column_in_metadata const& column_metadata,
                                   single_write_mode write_mode);
/**
 * @brief Returns ``true`` if the given column has a fixed size.
 *
 * This doesn't check every row, so assumes string and list columns are not fixed, even
 * if each row is the same width.
 * TODO: update this if FIXED_LEN_BYTE_ARRAY is ever supported for writes.
 *
 * @param column A view of the column
 *
 * @return Whether the column has a fixed size
 */
[[nodiscard]] bool is_col_fixed_width(column_view const& column);

}  // namespace cudf::io::parquet::detail
