/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "hybrid_scan_helpers.hpp"

#include "io/parquet/compact_protocol_reader.hpp"
#include "io/parquet/reader_impl_helpers.hpp"
#include "io/utilities/row_selection.hpp"

#include <cudf/logger.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>

namespace cudf::io::parquet::experimental::detail {

using aggregate_reader_metadata_base = parquet::detail::aggregate_reader_metadata;
using metadata_base                  = parquet::detail::metadata;

using parquet::detail::CompactProtocolReader;
using parquet::detail::equality_literals_collector;
using parquet::detail::input_column_info;
using parquet::detail::row_group_info;

metadata::metadata(cudf::host_span<uint8_t const> footer_bytes)
{
  CompactProtocolReader cp(footer_bytes.data(), footer_bytes.size());
  cp.read(this);
  CUDF_EXPECTS(cp.InitSchema(this), "Cannot initialize schema");
  sanitize_schema();
}

aggregate_reader_metadata::aggregate_reader_metadata(cudf::host_span<uint8_t const> footer_bytes,
                                                     bool use_arrow_schema,
                                                     bool has_cols_from_mismatched_srcs)
  : aggregate_reader_metadata_base({}, false, false)
{
  // Re-initialize internal variables here as base class was initialized without a source
  per_file_metadata = std::vector<metadata_base>{metadata{footer_bytes}.get_file_metadata()};
  keyval_maps       = collect_keyval_metadata();
  schema_idx_maps   = init_schema_idx_maps(has_cols_from_mismatched_srcs);
  num_rows          = calc_num_rows();
  num_row_groups    = calc_num_row_groups();

  // Force all columns to be nullable
  auto& schema = per_file_metadata.front().schema;
  std::for_each(schema.begin(), schema.end(), [](auto& col) {
    col.repetition_type = FieldRepetitionType::OPTIONAL;
  });

  // Collect and apply arrow:schema from Parquet's key value metadata section
  if (use_arrow_schema) {
    apply_arrow_schema();

    // Erase ARROW_SCHEMA_KEY from the output pfm if exists
    std::for_each(keyval_maps.begin(), keyval_maps.end(), [](auto& pfm) {
      pfm.erase(cudf::io::parquet::detail::ARROW_SCHEMA_KEY);
    });
  }
}

cudf::io::text::byte_range_info aggregate_reader_metadata::get_page_index_bytes() const
{
  auto& schema     = per_file_metadata.front();
  auto& row_groups = schema.row_groups;

  if (row_groups.size() and row_groups.front().columns.size()) {
    int64_t const min_offset = schema.row_groups.front().columns.front().column_index_offset;
    auto const& last_col     = schema.row_groups.back().columns.back();
    int64_t const max_offset = last_col.offset_index_offset + last_col.offset_index_length;
    return {min_offset, (max_offset - min_offset)};
  }

  return {};
}

FileMetaData const& aggregate_reader_metadata::get_parquet_metadata() const
{
  return per_file_metadata.front();
}

void aggregate_reader_metadata::setup_page_index(cudf::host_span<uint8_t const> page_index_bytes)
{
  // Return early if empty page index buffer span
  if (not page_index_bytes.size()) {
    CUDF_LOG_WARN("Hybrid scan reader encountered empty page index buffer");
    return;
  }

  auto& schema     = per_file_metadata.front();
  auto& row_groups = schema.row_groups;

  CUDF_EXPECTS(row_groups.size() and row_groups.front().columns.size(),
               "No column chunks in Parquet schema to read page index for");

  CompactProtocolReader cp(page_index_bytes.data(), page_index_bytes.size());

  // Set the first ColumnChunk's offset of ColumnIndex as the adjusted zero offset
  int64_t const min_offset = row_groups.front().columns.front().column_index_offset;
  // now loop over row groups
  for (auto& rg : row_groups) {
    for (auto& col : rg.columns) {
      // Read the ColumnIndex for this ColumnChunk
      if (col.column_index_length > 0 && col.column_index_offset > 0) {
        int64_t const offset = col.column_index_offset - min_offset;
        cp.init(page_index_bytes.data() + offset, col.column_index_length);
        ColumnIndex ci;
        cp.read(&ci);
        col.column_index = std::move(ci);
      }
      // Read the OffsetIndex for this ColumnChunk
      if (col.offset_index_length > 0 && col.offset_index_offset > 0) {
        int64_t const offset = col.offset_index_offset - min_offset;
        cp.init(page_index_bytes.data() + offset, col.offset_index_length);
        OffsetIndex oi;
        cp.read(&oi);
        col.offset_index = std::move(oi);
      }
    }
  }
}

}  // namespace cudf::io::parquet::experimental::detail
