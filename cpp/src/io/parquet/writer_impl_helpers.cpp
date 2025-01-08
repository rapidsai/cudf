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
 * @file writer_impl_helpers.cpp
 * @brief Helper function implementation for Parquet writer
 */

#include "writer_impl_helpers.hpp"

#include "io/comp/nvcomp_adapter.hpp"

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>

namespace cudf::io::parquet::detail {

using namespace cudf::io::detail;

Compression to_parquet_compression(compression_type compression)
{
  switch (compression) {
    case compression_type::AUTO:
    case compression_type::SNAPPY: return Compression::SNAPPY;
    case compression_type::ZSTD: return Compression::ZSTD;
    case compression_type::LZ4:
      // Parquet refers to LZ4 as "LZ4_RAW"; Parquet's "LZ4" is not standard LZ4
      return Compression::LZ4_RAW;
    case compression_type::NONE: return Compression::UNCOMPRESSED;
    default: CUDF_FAIL("Unsupported compression type");
  }
}

nvcomp::compression_type to_nvcomp_compression_type(Compression codec)
{
  switch (codec) {
    case Compression::SNAPPY: return nvcomp::compression_type::SNAPPY;
    case Compression::ZSTD: return nvcomp::compression_type::ZSTD;
    // Parquet refers to LZ4 as "LZ4_RAW"; Parquet's "LZ4" is not standard LZ4
    case Compression::LZ4_RAW: return nvcomp::compression_type::LZ4;
    default: CUDF_FAIL("Unsupported compression type");
  }
}

uint32_t page_alignment(Compression codec)
{
  if (codec == Compression::UNCOMPRESSED or
      nvcomp::is_compression_disabled(to_nvcomp_compression_type(codec))) {
    return 1u;
  }

  return nvcomp::required_alignment(to_nvcomp_compression_type(codec));
}

size_t max_compression_output_size(Compression codec, uint32_t compression_blocksize)
{
  if (codec == Compression::UNCOMPRESSED) return 0;

  return compress_max_output_chunk_size(to_nvcomp_compression_type(codec), compression_blocksize);
}

void fill_table_meta(table_input_metadata& table_meta)
{
  // Fill unnamed columns' names in table_meta
  std::function<void(column_in_metadata&, std::string)> add_default_name =
    [&](column_in_metadata& col_meta, std::string default_name) {
      if (col_meta.get_name().empty()) col_meta.set_name(default_name);
      for (size_type i = 0; i < col_meta.num_children(); ++i) {
        add_default_name(col_meta.child(i), col_meta.get_name() + "_" + std::to_string(i));
      }
    };
  for (size_t i = 0; i < table_meta.column_metadata.size(); ++i) {
    add_default_name(table_meta.column_metadata[i], "_col" + std::to_string(i));
  }
}

[[nodiscard]] size_t column_size(column_view const& column, rmm::cuda_stream_view stream)
{
  if (column.is_empty()) { return 0; }

  if (is_fixed_width(column.type())) {
    return size_of(column.type()) * column.size();
  } else if (column.type().id() == type_id::STRING) {
    auto const scol = strings_column_view(column);
    return cudf::strings::detail::get_offset_value(
             scol.offsets(), column.size() + column.offset(), stream) -
           cudf::strings::detail::get_offset_value(scol.offsets(), column.offset(), stream);
  } else if (column.type().id() == type_id::STRUCT) {
    auto const scol = structs_column_view(column);
    size_t ret      = 0;
    for (int i = 0; i < scol.num_children(); i++) {
      ret += column_size(scol.get_sliced_child(i, stream), stream);
    }
    return ret;
  } else if (column.type().id() == type_id::LIST) {
    auto const lcol = lists_column_view(column);
    return column_size(lcol.get_sliced_child(stream), stream);
  }

  CUDF_FAIL("Unexpected compound type");
}

[[nodiscard]] bool is_output_column_nullable(cudf::detail::LinkedColPtr const& column,
                                             column_in_metadata const& column_metadata,
                                             single_write_mode write_mode)
{
  if (column_metadata.is_nullability_defined()) {
    CUDF_EXPECTS(column_metadata.nullable() or column->null_count() == 0,
                 "Mismatch in metadata prescribed nullability and input column. "
                 "Metadata for input column with nulls cannot prescribe nullability = false");
    return column_metadata.nullable();
  }
  // For chunked write, when not provided nullability, we assume the worst case scenario
  // that all columns are nullable.
  return write_mode == single_write_mode::NO or column->nullable();
}

}  // namespace cudf::io::parquet::detail
