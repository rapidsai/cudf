/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "reader_impl_helpers.hpp"

#include "compact_protocol_reader.hpp"
#include "io/parquet/parquet.hpp"
#include "io/utilities/base64_utilities.hpp"
#include "io/utilities/row_selection.hpp"
#include "ipc/Message_generated.h"
#include "ipc/Schema_generated.h"

#include <cudf/detail/utilities/logger.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <functional>
#include <numeric>
#include <regex>

namespace cudf::io::parquet::detail {

namespace flatbuf = cudf::io::parquet::flatbuf;

namespace {

std::optional<LogicalType> converted_to_logical_type(SchemaElement const& schema)
{
  if (schema.converted_type.has_value()) {
    switch (schema.converted_type.value()) {
      case ENUM:  // treat ENUM as UTF8 string
      case UTF8: return LogicalType{LogicalType::STRING};
      case MAP: return LogicalType{LogicalType::MAP};
      case LIST: return LogicalType{LogicalType::LIST};
      case DECIMAL: return LogicalType{DecimalType{schema.decimal_scale, schema.decimal_precision}};
      case DATE: return LogicalType{LogicalType::DATE};
      case TIME_MILLIS: return LogicalType{TimeType{true, {TimeUnit::MILLIS}}};
      case TIME_MICROS: return LogicalType{TimeType{true, {TimeUnit::MICROS}}};
      case TIMESTAMP_MILLIS: return LogicalType{TimestampType{true, {TimeUnit::MILLIS}}};
      case TIMESTAMP_MICROS: return LogicalType{TimestampType{true, {TimeUnit::MICROS}}};
      case UINT_8: return LogicalType{IntType{8, false}};
      case UINT_16: return LogicalType{IntType{16, false}};
      case UINT_32: return LogicalType{IntType{32, false}};
      case UINT_64: return LogicalType{IntType{64, false}};
      case INT_8: return LogicalType{IntType{8, true}};
      case INT_16: return LogicalType{IntType{16, true}};
      case INT_32: return LogicalType{IntType{32, true}};
      case INT_64: return LogicalType{IntType{64, true}};
      case JSON: return LogicalType{LogicalType::JSON};
      case BSON: return LogicalType{LogicalType::BSON};
      case INTERVAL:  // there is no logical type for INTERVAL yet
      default: return LogicalType{LogicalType::UNDEFINED};
    }
  }
  return std::nullopt;
}

}  // namespace

/**
 * @brief Function that translates Parquet datatype to cuDF type enum
 */
type_id to_type_id(SchemaElement const& schema,
                   bool strings_to_categorical,
                   type_id timestamp_type_id)
{
  auto const physical_type = schema.type;
  auto const arrow_type    = schema.arrow_type;
  auto logical_type        = schema.logical_type;

  // sanity check, but not worth failing over
  if (schema.converted_type.has_value() and not logical_type.has_value()) {
    CUDF_LOG_WARN("ConvertedType is specified but not LogicalType");
    logical_type = converted_to_logical_type(schema);
  }

  // check if have set the type through arrow schema?
  if (arrow_type.has_value()) {
    // is it duration type? i.e. phyical_type == INT64 and no logical/converted types
    if (physical_type == Type::INT64 and not logical_type.has_value()) {
      return arrow_type.value();
    }
    // should warn but not fail.
    CUDF_LOG_WARN("Indeterminable arrow type encountered");
  }

  if (logical_type.has_value()) {
    switch (logical_type->type) {
      case LogicalType::INTEGER: {
        auto const is_signed = logical_type->is_signed();
        switch (logical_type->bit_width()) {
          case 8: return is_signed ? type_id::INT8 : type_id::UINT8;
          case 16: return is_signed ? type_id::INT16 : type_id::UINT16;
          case 32: return is_signed ? type_id::INT32 : type_id::UINT32;
          case 64: return is_signed ? type_id::INT64 : type_id::UINT64;
          default: CUDF_FAIL("Invalid integer bitwidth");
        }
      } break;

      case LogicalType::DATE: return type_id::TIMESTAMP_DAYS;

      case LogicalType::TIME:
        if (logical_type->is_time_millis()) {
          return type_id::DURATION_MILLISECONDS;
        } else if (logical_type->is_time_micros()) {
          return type_id::DURATION_MICROSECONDS;
        } else if (logical_type->is_time_nanos()) {
          return type_id::DURATION_NANOSECONDS;
        }
        break;

      case LogicalType::TIMESTAMP:
        if (timestamp_type_id != type_id::EMPTY) {
          return timestamp_type_id;
        } else if (logical_type->is_timestamp_millis()) {
          return type_id::TIMESTAMP_MILLISECONDS;
        } else if (logical_type->is_timestamp_micros()) {
          return type_id::TIMESTAMP_MICROSECONDS;
        } else if (logical_type->is_timestamp_nanos()) {
          return type_id::TIMESTAMP_NANOSECONDS;
        }

      case LogicalType::DECIMAL: {
        int32_t const decimal_precision = logical_type->precision();
        if (physical_type == INT32) {
          return type_id::DECIMAL32;
        } else if (physical_type == INT64) {
          return type_id::DECIMAL64;
        } else if (physical_type == FIXED_LEN_BYTE_ARRAY) {
          if (schema.type_length <= static_cast<int32_t>(sizeof(int32_t))) {
            return type_id::DECIMAL32;
          } else if (schema.type_length <= static_cast<int32_t>(sizeof(int64_t))) {
            return type_id::DECIMAL64;
          } else if (schema.type_length <= static_cast<int32_t>(sizeof(__int128_t))) {
            return type_id::DECIMAL128;
          }
        } else if (physical_type == BYTE_ARRAY) {
          CUDF_EXPECTS(decimal_precision <= MAX_DECIMAL128_PRECISION, "Invalid decimal precision");
          if (decimal_precision <= MAX_DECIMAL32_PRECISION) {
            return type_id::DECIMAL32;
          } else if (decimal_precision <= MAX_DECIMAL64_PRECISION) {
            return type_id::DECIMAL64;
          } else {
            return type_id::DECIMAL128;
          }
        } else {
          CUDF_FAIL("Invalid representation of decimal type");
        }
      } break;

      // maps are just List<Struct<>>.
      case LogicalType::MAP:
      case LogicalType::LIST: return type_id::LIST;

      // All null column that can't have its type deduced.
      // Note: originally LogicalType::UNKNOWN was converted to ConvertedType::NA, and
      // NA then became type_id::STRING, but with the following TODO:
      // return type_id::EMPTY; //TODO(kn): enable after Null/Empty column support
      case LogicalType::UNKNOWN: return type_id::STRING;

      default: break;
    }
  }

  // is it simply a struct?
  if (schema.is_struct()) { return type_id::STRUCT; }

  // Physical storage type supported by Parquet; controls the on-disk storage
  // format in combination with the encoding type.
  switch (physical_type) {
    case BOOLEAN: return type_id::BOOL8;
    case INT32: return type_id::INT32;
    case INT64: return type_id::INT64;
    case FLOAT: return type_id::FLOAT32;
    case DOUBLE: return type_id::FLOAT64;
    case BYTE_ARRAY:
      // strings can be mapped to a 32-bit hash
      if (strings_to_categorical) { return type_id::INT32; }
      [[fallthrough]];
    case FIXED_LEN_BYTE_ARRAY: return type_id::STRING;
    case INT96:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_NANOSECONDS;
    default: break;
  }

  return type_id::EMPTY;
}

void metadata::sanitize_schema()
{
  // Parquet isn't very strict about incoming metadata. Lots of things can and should be inferred.
  // There are also a lot of rules that simply aren't followed and are expected to be worked around.
  // This step sanitizes the metadata to something that isn't ambiguous.
  //
  // Take, for example, the following schema:
  //
  //  required group field_id=-1 user {
  //    required int32 field_id=-1 id;
  //    optional group field_id=-1 phoneNumbers {
  //      repeated group field_id=-1 phone {
  //        required int64 field_id=-1 number;
  //        optional binary field_id=-1 kind (String);
  //      }
  //    }
  //  }
  //
  // This real-world example has no annotations telling us what is a list or a struct. On the
  // surface this looks like a column of id's and a column of list<struct<int64, string>>, but this
  // actually should be interpreted as a struct<list<struct<int64, string>>>. The phoneNumbers field
  // has to be a struct because it is a group with no repeated tag and we have no annotation. The
  // repeated group is actually BOTH a struct due to the multiple children and a list due to
  // repeated.
  //
  // This code attempts to make this less messy for the code that follows.

  std::function<void(size_t)> process = [&](size_t schema_idx) -> void {
    auto& schema_elem = schema[schema_idx];
    if (schema_idx != 0 && schema_elem.type == UNDEFINED_TYPE) {
      auto const parent_type = schema[schema_elem.parent_idx].converted_type;
      if (schema_elem.repetition_type == REPEATED && schema_elem.num_children > 1 &&
          parent_type != LIST && parent_type != MAP) {
        // This is a list of structs, so we need to mark this as a list, but also
        // add a struct child and move this element's children to the struct
        schema_elem.converted_type  = LIST;
        schema_elem.logical_type    = LogicalType::LIST;
        schema_elem.repetition_type = OPTIONAL;
        auto const struct_node_idx  = static_cast<size_type>(schema.size());

        SchemaElement struct_elem;
        struct_elem.name            = "struct_node";
        struct_elem.repetition_type = REQUIRED;
        struct_elem.num_children    = schema_elem.num_children;
        struct_elem.type            = UNDEFINED_TYPE;
        struct_elem.converted_type  = std::nullopt;

        // swap children
        struct_elem.children_idx = std::move(schema_elem.children_idx);
        schema_elem.children_idx = {struct_node_idx};
        schema_elem.num_children = 1;

        struct_elem.max_definition_level = schema_elem.max_definition_level;
        struct_elem.max_repetition_level = schema_elem.max_repetition_level;
        schema_elem.max_definition_level--;
        schema_elem.max_repetition_level = schema[schema_elem.parent_idx].max_repetition_level;

        // change parent index on new node and on children
        struct_elem.parent_idx = schema_idx;
        for (auto& child_idx : struct_elem.children_idx) {
          schema[child_idx].parent_idx = struct_node_idx;
        }
        // add our struct
        schema.push_back(struct_elem);
      }
    }

    // convert ConvertedType to LogicalType for older files
    if (schema_elem.converted_type.has_value() and not schema_elem.logical_type.has_value()) {
      schema_elem.logical_type = converted_to_logical_type(schema_elem);
    }

    for (auto& child_idx : schema_elem.children_idx) {
      process(child_idx);
    }
  };

  process(0);
}

metadata::metadata(datasource* source)
{
  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len  = sizeof(file_ender_s);

  auto const len           = source->size();
  auto const header_buffer = source->host_read(0, header_len);
  auto const header        = reinterpret_cast<file_header_s const*>(header_buffer->data());
  auto const ender_buffer  = source->host_read(len - ender_len, ender_len);
  auto const ender         = reinterpret_cast<file_ender_s const*>(ender_buffer->data());
  CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
  CUDF_EXPECTS(header->magic == parquet_magic && ender->magic == parquet_magic,
               "Corrupted header or footer");
  CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
               "Incorrect footer length");

  auto const buffer = source->host_read(len - ender->footer_len - ender_len, ender->footer_len);
  CompactProtocolReader cp(buffer->data(), ender->footer_len);
  cp.read(this);
  CUDF_EXPECTS(cp.InitSchema(this), "Cannot initialize schema");

  // Reading the page indexes is somewhat expensive, so skip if there are no byte array columns.
  // Currently the indexes are only used for the string size calculations.
  // Could also just read indexes for string columns, but that would require changes elsewhere
  // where we're trying to determine if we have the indexes or not.
  // Note: This will have to be modified if there are other uses in the future (e.g. calculating
  // chunk/pass boundaries).
  auto const has_strings = std::any_of(
    schema.begin(), schema.end(), [](auto const& elem) { return elem.type == BYTE_ARRAY; });

  if (has_strings and not row_groups.empty() and not row_groups.front().columns.empty()) {
    // column index and offset index are encoded back to back.
    // the first column of the first row group will have the first column index, the last
    // column of the last row group will have the final offset index.
    int64_t const min_offset = row_groups.front().columns.front().column_index_offset;
    auto const& last_col     = row_groups.back().columns.back();
    int64_t const max_offset = last_col.offset_index_offset + last_col.offset_index_length;

    if (max_offset > 0) {
      int64_t const length = max_offset - min_offset;
      auto const idx_buf   = source->host_read(min_offset, length);

      // now loop over row groups
      for (auto& rg : row_groups) {
        for (auto& col : rg.columns) {
          if (col.column_index_length > 0 && col.column_index_offset > 0) {
            int64_t const offset = col.column_index_offset - min_offset;
            cp.init(idx_buf->data() + offset, col.column_index_length);
            ColumnIndex ci;
            cp.read(&ci);
            col.column_index = std::move(ci);
          }
          if (col.offset_index_length > 0 && col.offset_index_offset > 0) {
            int64_t const offset = col.offset_index_offset - min_offset;
            cp.init(idx_buf->data() + offset, col.offset_index_length);
            OffsetIndex oi;
            cp.read(&oi);
            col.offset_index = std::move(oi);
          }
        }
      }
    }
  }

  sanitize_schema();
}

std::vector<metadata> aggregate_reader_metadata::metadatas_from_sources(
  host_span<std::unique_ptr<datasource> const> sources)
{
  std::vector<metadata> metadatas;
  std::transform(
    sources.begin(), sources.end(), std::back_inserter(metadatas), [](auto const& source) {
      return metadata(source.get());
    });
  return metadatas;
}

std::vector<std::unordered_map<std::string, std::string>>
aggregate_reader_metadata::collect_keyval_metadata() const
{
  std::vector<std::unordered_map<std::string, std::string>> kv_maps;
  std::transform(per_file_metadata.cbegin(),
                 per_file_metadata.cend(),
                 std::back_inserter(kv_maps),
                 [](auto const& pfm) {
                   std::unordered_map<std::string, std::string> kv_map;
                   std::transform(pfm.key_value_metadata.cbegin(),
                                  pfm.key_value_metadata.cend(),
                                  std::inserter(kv_map, kv_map.end()),
                                  [](auto const& kv) {
                                    return std::pair{kv.key, kv.value};
                                  });
                   return kv_map;
                 });

  return kv_maps;
}

std::vector<std::unordered_map<int32_t, int32_t>> aggregate_reader_metadata::init_schema_idx_maps(
  bool const has_cols_from_mismatched_srcs) const
{
  // Only initialize if more than 1 data sources and has select columns from mismatched data sources
  if (has_cols_from_mismatched_srcs and per_file_metadata.size() > 1) {
    return std::vector<std::unordered_map<int32_t, int32_t>>{per_file_metadata.size() - 1};
  }

  return {};
}

int64_t aggregate_reader_metadata::calc_num_rows() const
{
  return std::accumulate(
    per_file_metadata.cbegin(), per_file_metadata.cend(), 0l, [](auto& sum, auto& pfm) {
      auto const rowgroup_rows = std::accumulate(
        pfm.row_groups.cbegin(), pfm.row_groups.cend(), 0l, [](auto& rg_sum, auto& rg) {
          return rg_sum + rg.num_rows;
        });
      CUDF_EXPECTS(pfm.num_rows == 0 || pfm.num_rows == rowgroup_rows,
                   "Header and row groups disagree about number of rows in file!");
      return sum + (pfm.num_rows == 0 && rowgroup_rows > 0 ? rowgroup_rows : pfm.num_rows);
    });
}

size_type aggregate_reader_metadata::calc_num_row_groups() const
{
  return std::accumulate(
    per_file_metadata.cbegin(), per_file_metadata.cend(), 0, [](auto& sum, auto& pfm) {
      return sum + pfm.row_groups.size();
    });
}

// Copies info from the column and offset indexes into the passed in row_group_info.
void aggregate_reader_metadata::column_info_for_row_group(row_group_info& rg_info,
                                                          size_type chunk_start_row) const
{
  auto const& fmd = per_file_metadata[rg_info.source_index];
  auto const& rg  = fmd.row_groups[rg_info.index];

  std::vector<column_chunk_info> chunks(rg.columns.size());

  for (size_t col_idx = 0; col_idx < rg.columns.size(); col_idx++) {
    auto const& col_chunk = rg.columns[col_idx];
    auto const is_schema_idx_mapped =
      is_schema_index_mapped(col_chunk.schema_idx, rg_info.source_index);
    auto const mapped_schema_idx = is_schema_idx_mapped
                                     ? map_schema_index(col_chunk.schema_idx, rg_info.source_index)
                                     : col_chunk.schema_idx;
    auto& schema = get_schema(mapped_schema_idx, is_schema_idx_mapped ? rg_info.source_index : 0);
    auto const max_def_level = schema.max_definition_level;
    auto const max_rep_level = schema.max_repetition_level;

    // If any columns lack the page indexes then just return without modifying the
    // row_group_info.
    if (not col_chunk.offset_index.has_value() or not col_chunk.column_index.has_value()) {
      return;
    }

    auto const& offset_index = col_chunk.offset_index.value();
    auto const& column_index = col_chunk.column_index.value();

    auto& chunk_info     = chunks[col_idx];
    auto const num_pages = offset_index.page_locations.size();

    // There is a bug in older versions of parquet-mr where the first data page offset
    // really points to the dictionary page. The first possible offset in a file is 4 (after
    // the "PAR1" header), so check to see if the dictionary_page_offset is > 0. If it is, then
    // we haven't encountered the bug.
    if (col_chunk.meta_data.dictionary_page_offset > 0) {
      chunk_info.dictionary_offset = col_chunk.meta_data.dictionary_page_offset;
      chunk_info.dictionary_size =
        col_chunk.meta_data.data_page_offset - chunk_info.dictionary_offset.value();
    } else {
      // dictionary_page_offset is 0, so check to see if the data_page_offset does not match
      // the first offset in the offset index.  If they don't match, then data_page_offset points
      // to the dictionary page.
      if (num_pages > 0 &&
          col_chunk.meta_data.data_page_offset < offset_index.page_locations[0].offset) {
        chunk_info.dictionary_offset = col_chunk.meta_data.data_page_offset;
        chunk_info.dictionary_size =
          offset_index.page_locations[0].offset - col_chunk.meta_data.data_page_offset;
      }
    }

    // Use the definition_level_histogram to get num_valid and num_null. For now, these are
    // only ever used for byte array columns. The repetition_level_histogram might be
    // necessary to determine the total number of values in the page if the
    // definition_level_histogram is absent.
    //
    // In the future we might want the full histograms saved in the `column_info` struct.
    int64_t const* const def_hist = column_index.definition_level_histogram.has_value()
                                      ? column_index.definition_level_histogram.value().data()
                                      : nullptr;
    int64_t const* const rep_hist = column_index.repetition_level_histogram.has_value()
                                      ? column_index.repetition_level_histogram.value().data()
                                      : nullptr;

    for (size_t pg_idx = 0; pg_idx < num_pages; pg_idx++) {
      auto const& page_loc = offset_index.page_locations[pg_idx];
      // translate chunk-relative row nums to absolute within the file
      auto const pg_start_row = chunk_start_row + page_loc.first_row_index;
      auto const pg_end_row =
        chunk_start_row + (pg_idx == (num_pages - 1)
                             ? rg.num_rows
                             : offset_index.page_locations[pg_idx + 1].first_row_index);

      auto const num_rows = pg_end_row - pg_start_row;
      page_info pg_info{page_loc, num_rows};

      // check to see if we already have null counts for each page
      if (column_index.null_counts.has_value()) {
        pg_info.num_nulls = column_index.null_counts.value()[pg_idx];
      }

      // save variable length byte info if present
      if (offset_index.unencoded_byte_array_data_bytes.has_value()) {
        pg_info.var_bytes_size = offset_index.unencoded_byte_array_data_bytes.value()[pg_idx];
      }

      // if def histogram is present, then use it to calculate num_valid and num_nulls
      if (def_hist != nullptr) {
        auto const h      = &def_hist[pg_idx * (max_def_level + 1)];
        pg_info.num_valid = h[max_def_level];

        // calculate num_nulls if not available from column index
        if (not pg_info.num_nulls.has_value()) {
          pg_info.num_nulls = std::reduce(h, h + max_def_level);
        }
      }
      // there is no def histogram.
      // if there is no repetition (no lists), then num_values == num_rows, and num_nulls can be
      // obtained from the column index
      else if (max_rep_level == 0) {
        // if we already have num_nulls from column index
        if (pg_info.num_nulls.has_value()) {
          pg_info.num_valid = pg_info.num_rows - pg_info.num_nulls.value();
        }
        // if max_def is 0, there are no nulls
        else if (max_def_level == 0) {
          pg_info.num_nulls = 0;
          pg_info.num_valid = pg_info.num_rows;
        }
      }
      // if the rep level histogram is present, we can get the total number of values
      // from that
      else if (rep_hist != nullptr) {
        if (pg_info.num_nulls.has_value()) {
          auto const h          = &rep_hist[pg_idx * (max_rep_level + 1)];
          auto const num_values = std::reduce(h, h + max_rep_level + 1);
          pg_info.num_valid     = num_values - pg_info.num_nulls.value();
        }
      }

      // If none of the ifs above triggered, then we have neither histogram (likely the writer
      // doesn't produce them, the r:0 d:1 case should have been handled above). The column index
      // doesn't give us value counts, so we'll have to rely on the page headers. If the histogram
      // info is missing or insufficient, then just return without modifying the row_group_info.
      if (not pg_info.num_nulls.has_value() or not pg_info.num_valid.has_value()) { return; }

      // Like above, if using older page indexes that lack size info, then return without modifying
      // the row_group_info.
      // TODO: cudf will still set the per-page var_bytes to '0' even for all null pages. Need to
      // check the behavior of other implementations (once there are some). Some may not set the
      // var bytes for all null pages, so check the `null_pages` field on the column index.
      if (schema.type == BYTE_ARRAY and not pg_info.var_bytes_size.has_value()) { return; }

      chunk_info.pages.push_back(std::move(pg_info));
    }
  }

  rg_info.column_chunks = std::move(chunks);
}

aggregate_reader_metadata::aggregate_reader_metadata(
  host_span<std::unique_ptr<datasource> const> sources,
  bool use_arrow_schema,
  bool has_cols_from_mismatched_srcs)
  : per_file_metadata(metadatas_from_sources(sources)),
    keyval_maps(collect_keyval_metadata()),
    schema_idx_maps(init_schema_idx_maps(has_cols_from_mismatched_srcs)),
    num_rows(calc_num_rows()),
    num_row_groups(calc_num_row_groups())
{
  if (per_file_metadata.size() > 1) {
    auto& first_meta = per_file_metadata.front();
    auto const num_cols =
      first_meta.row_groups.size() > 0 ? first_meta.row_groups.front().columns.size() : 0;
    auto& schema = first_meta.schema;

    // Validate that all sources have the same schema unless we are reading select columns
    // from mismatched sources, in which case, we will only check the projected columns later.
    if (not has_cols_from_mismatched_srcs) {
      // Verify that the input files have matching numbers of columns and schema.
      for (auto const& pfm : per_file_metadata) {
        if (pfm.row_groups.size() > 0) {
          CUDF_EXPECTS(num_cols == pfm.row_groups.front().columns.size(),
                       "All sources must have the same number of columns");
        }
        CUDF_EXPECTS(schema == pfm.schema, "All sources must have the same schema");
      }
    }

    // Mark the column schema in the first (default) source as nullable if it is nullable in any of
    // the input sources. This avoids recomputing this within build_column() and
    // populate_metadata().
    std::for_each(
      thrust::make_counting_iterator(static_cast<size_t>(1)),
      thrust::make_counting_iterator(schema.size()),
      [&](auto const schema_idx) {
        if (schema[schema_idx].repetition_type == REQUIRED and
            std::any_of(
              per_file_metadata.begin() + 1, per_file_metadata.end(), [&](auto const& pfm) {
                return pfm.schema[schema_idx].repetition_type != REQUIRED;
              })) {
          schema[schema_idx].repetition_type = OPTIONAL;
        }
      });
  }

  // Collect and apply arrow:schema from Parquet's key value metadata section
  if (use_arrow_schema) { apply_arrow_schema(); }

  // Erase ARROW_SCHEMA_KEY from the output pfm if exists
  std::for_each(
    keyval_maps.begin(), keyval_maps.end(), [](auto& pfm) { pfm.erase(ARROW_SCHEMA_KEY); });
}

arrow_schema_data_types aggregate_reader_metadata::collect_arrow_schema() const
{
  // Check the key_value metadata for arrow schema, decode and walk it
  // Function to convert from flatbuf::duration type to cudf::type_id
  auto const duration_from_flatbuffer = [](flatbuf::Duration const* duration) {
    // TODO: we only need this for arrow::DurationType for now. Else, we can take in a
    // void ptr and typecast it to the corresponding type based on the type_id.
    auto fb_unit = duration->unit();
    switch (fb_unit) {
      case flatbuf::TimeUnit::TimeUnit_SECOND:
        return cudf::data_type{cudf::type_id::DURATION_SECONDS};
      case flatbuf::TimeUnit::TimeUnit_MILLISECOND:
        return cudf::data_type{cudf::type_id::DURATION_MILLISECONDS};
      case flatbuf::TimeUnit::TimeUnit_MICROSECOND:
        return cudf::data_type{cudf::type_id::DURATION_MICROSECONDS};
      case flatbuf::TimeUnit::TimeUnit_NANOSECOND:
        return cudf::data_type{cudf::type_id::DURATION_NANOSECONDS};
      default: return cudf::data_type{};
    }
  };

  // variable that tracks if an arrow_type specific column is seen
  // in the walk
  bool arrow_type_col_seen = false;

  // Lambda function to walk a field and its children in DFS manner and
  // return boolean walk success status
  std::function<bool(flatbuf::Field const* const, arrow_schema_data_types&)> walk_field =
    [&walk_field, &duration_from_flatbuffer, &arrow_type_col_seen](
      flatbuf::Field const* const field, arrow_schema_data_types& schema_elem) {
      // DFS: recursively walk over the children first
      auto const field_children = field->children();

      if (field_children != nullptr) {
        auto schema_children = std::vector<arrow_schema_data_types>(field->children()->size());

        if (not std::all_of(
              thrust::make_counting_iterator(0),
              thrust::make_counting_iterator(static_cast<int32_t>(field_children->size())),
              [&](auto const& idx) {
                return walk_field((*field_children)[idx], schema_children[idx]);
              })) {
          return false;
        }
        // arrow and parquet schemas are structured slightly differently for list type fields. list
        // type fields in arrow are structured as: "field:list<element>" vs structured as:
        // "field:list.element" in Parquet. To handle this, whenever we encounter a list type field,
        // we add a dummy node "field.list" to the end of current children and move the current
        // children (".element") to it.
        switch (field->type_type()) {
          case flatbuf::Type::Type_List:
          case flatbuf::Type::Type_LargeList:
          case flatbuf::Type::Type_FixedSizeList:
            schema_elem.children.emplace_back(arrow_schema_data_types{std::move(schema_children)});
            break;
          default: schema_elem.children = std::move(schema_children); break;
        }
      }

      // Walk the field itself
      if (field->type_type() == flatbuf::Type::Type_Duration) {
        auto type_data = field->type_as_Duration();
        if (type_data != nullptr) {
          auto name = field->name() ? field->name()->str() : "";
          // set the schema_elem type to duration type
          schema_elem.type = duration_from_flatbuffer(type_data);
          arrow_type_col_seen |= (schema_elem.type.id() != type_id::EMPTY);
        } else {
          CUDF_LOG_ERROR("Parquet reader encountered an invalid type_data pointer.",
                         "arrow:schema not processed.");
          return false;
        }
      }
      return true;
    };

  auto const it = keyval_maps[0].find(ARROW_SCHEMA_KEY);
  if (it == keyval_maps[0].end()) { return {}; }

  // Decode the base64 encoded ipc message string
  // Note: Store the output from base64_decode in the lvalue here and then pass
  // it to decode_ipc_message. Directly passing rvalue from base64_decode to
  // decode_ipc_message can lead to unintended nullptr dereferences.
  auto const decoded_message = cudf::io::detail::base64_decode(it->second);

  // Decode the ipc message to get an optional string_view of the ipc:Message flatbuffer
  auto const metadata_buf = decode_ipc_message(decoded_message);

  // Check if the string_view exists
  if (not metadata_buf.has_value()) {
    // No need to re-log error here as already logged inside decode_ipc_message
    return {};
  }

  // Check if the decoded Message flatbuffer is valid
  if (flatbuf::GetMessage(metadata_buf.value().data()) == nullptr) {
    CUDF_LOG_ERROR("Parquet reader encountered an invalid ipc:Message flatbuffer pointer.",
                   "arrow:schema not processed.");
    return {};
  }

  // Check if the Message flatbuffer has a valid arrow:schema in its header
  if (flatbuf::GetMessage(metadata_buf.value().data())->header_as_Schema() == nullptr) {
    CUDF_LOG_ERROR("Parquet reader encountered an invalid arrow:schema flatbuffer pointer.",
                   "arrow:schema not processed.");
    return {};
  }

  // Get the vector of fields from arrow:schema flatbuffer object
  auto const fields =
    flatbuf::GetMessage(metadata_buf.value().data())->header_as_Schema()->fields();
  if (fields == nullptr) {
    CUDF_LOG_ERROR("Parquet reader encountered an invalid fields pointer.",
                   "arrow:schema not processed.");
    return {};
  }

  // arrow schema structure to return
  arrow_schema_data_types schema;

  // Recursively walk the arrow schema and set cudf::data_type for all duration columns
  if (fields->size() > 0) {
    schema.children = std::vector<arrow_schema_data_types>(fields->size());

    if (not std::all_of(
          thrust::make_counting_iterator(0),
          thrust::make_counting_iterator(static_cast<int32_t>(fields->size())),
          [&](auto const& idx) { return walk_field((*fields)[idx], schema.children[idx]); })) {
      return {};
    }

    // if no arrow type column seen, return nullopt.
    if (not arrow_type_col_seen) { return {}; }
  }

  return schema;
}

void aggregate_reader_metadata::apply_arrow_schema()
{
  // Collect the arrow schema from the key value section of Parquet metadata
  auto arrow_schema_root = collect_arrow_schema();

  // Check if empty arrow schema collected
  if (arrow_schema_root.type.id() == type_id::EMPTY and arrow_schema_root.children.size() == 0) {
    return;
  }

  // Function to verify equal num_children at each level in Parquet and arrow schemas.
  std::function<bool(arrow_schema_data_types const&, int const)> validate_schemas =
    [&](arrow_schema_data_types const& arrow_schema, int const schema_idx) {
      auto& pq_schema_elem = per_file_metadata[0].schema[schema_idx];

      // ensure equal number of children first to avoid any segfaults in children
      if (pq_schema_elem.num_children == static_cast<int32_t>(arrow_schema.children.size())) {
        // true if and only if true for all children as well
        return std::all_of(thrust::make_zip_iterator(thrust::make_tuple(
                             arrow_schema.children.begin(), pq_schema_elem.children_idx.begin())),
                           thrust::make_zip_iterator(thrust::make_tuple(
                             arrow_schema.children.end(), pq_schema_elem.children_idx.end())),
                           [&](auto const& elem) {
                             return validate_schemas(thrust::get<0>(elem), thrust::get<1>(elem));
                           });
      } else {
        return false;
      }
    };

  // Function to co-walk arrow and parquet schemas
  std::function<void(arrow_schema_data_types const&, int const)> co_walk_schemas =
    [&](arrow_schema_data_types const& arrow_schema, int const schema_idx) {
      auto& pq_schema_elem = per_file_metadata[0].schema[schema_idx];
      std::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(arrow_schema.children.begin(), pq_schema_elem.children_idx.begin())),
        thrust::make_zip_iterator(
          thrust::make_tuple(arrow_schema.children.end(), pq_schema_elem.children_idx.end())),
        [&](auto const& elem) { co_walk_schemas(thrust::get<0>(elem), thrust::get<1>(elem)); });

      // true for DurationType columns only for now.
      if (arrow_schema.type.id() != type_id::EMPTY) {
        pq_schema_elem.arrow_type = arrow_schema.type.id();
      }
    };

  // Get Parquet schema root
  auto pq_schema_root = get_schema(0);

  // verify equal number of children for both schemas at root level
  if (pq_schema_root.num_children != static_cast<int32_t>(arrow_schema_root.children.size())) {
    CUDF_LOG_ERROR("Parquet reader encountered a mismatch between Parquet and arrow schema.",
                   "arrow:schema not processed.");
    return;
  }

  // zip iterator to validate and co-walk the two schemas
  auto schemas = thrust::make_zip_iterator(
    thrust::make_tuple(arrow_schema_root.children.begin(), pq_schema_root.children_idx.begin()));

  // Verify equal number of children at all sub-levels
  if (not std::all_of(schemas, schemas + pq_schema_root.num_children, [&](auto const& elem) {
        return validate_schemas(thrust::get<0>(elem), thrust::get<1>(elem));
      })) {
    CUDF_LOG_ERROR("Parquet reader encountered a mismatch between Parquet and arrow schema.",
                   "arrow:schema not processed.");
    return;
  }

  // All good, now co-walk schemas
  std::for_each(schemas, schemas + pq_schema_root.num_children, [&](auto const& elem) {
    co_walk_schemas(thrust::get<0>(elem), thrust::get<1>(elem));
  });
}

std::optional<std::string_view> aggregate_reader_metadata::decode_ipc_message(
  std::string_view const serialized_message) const
{
  // message buffer
  auto message_buf = serialized_message.data();
  // current message (buffer) size
  auto message_size = static_cast<int32_t>(serialized_message.size());

  // Lambda function to read and return 4 bytes as int32_t from the ipc message buffer and update
  // buffer pointer and size
  auto read_int32_from_ipc_message = [&]() {
    int32_t bytes;
    std::memcpy(&bytes, message_buf, sizeof(int32_t));
    // Offset the message buf and reduce remaining size
    message_buf += sizeof(int32_t);
    message_size -= sizeof(int32_t);
    return bytes;
  };

  // Check for empty message
  if (message_size == 0) {
    CUDF_LOG_ERROR("Parquet reader encountered zero length arrow:schema.",
                   "arrow:schema not processed.");
    return std::nullopt;
  }

  // Check for improper message size.
  if (message_size < MESSAGE_DECODER_NEXT_REQUIRED_SIZE_INITIAL) {
    CUDF_LOG_ERROR("Parquet reader encountered unexpected arrow:schema message length.",
                   "arrow:schema not processed.");
    return std::nullopt;
  }

  // Get the first 4 bytes (continuation) of the ipc message
  // and check if it matches the expected token
  if (read_int32_from_ipc_message() != IPC_CONTINUATION_TOKEN) {
    CUDF_LOG_ERROR("Parquet reader encountered unexpected IPC continuation token.",
                   "arrow:schema not processed.");
    return std::nullopt;
  }

  // Check for improper message size after the continuation bytes.
  if (message_size < MESSAGE_DECODER_NEXT_REQUIRED_SIZE_METADATA_LENGTH) {
    CUDF_LOG_ERROR("Parquet reader encountered unexpected arrow:schema message length.",
                   "arrow:schema not processed.");
    return std::nullopt;
  }

  // Get the next 4 bytes (metadata_len) of the ipc message
  // and check if invalid metadata length read
  auto const metadata_len = read_int32_from_ipc_message();

  // Check if the read metadata (header) length is > zero
  if (metadata_len <= 0) {
    CUDF_LOG_ERROR("Parquet reader encountered unexpected metadata length.",
                   "arrow:schema not processed.");
    return std::nullopt;
  }

  // Check if the remaining message size is smaller than the expected metadata length
  // TODO: Since the arrow:schema message doesn't have a body,
  // the following check may be made tighter from < to ==
  if (message_size < metadata_len) {
    CUDF_LOG_ERROR("Parquet reader encountered unexpected arrow:schema message length.",
                   "arrow:schema not processed.");
    return std::nullopt;
  }

  // All good, return the current message_buf as string_view
  return std::string_view{message_buf,
                          static_cast<std::basic_string_view<char>::size_type>(message_size)};
}

RowGroup const& aggregate_reader_metadata::get_row_group(size_type row_group_index,
                                                         size_type src_idx) const
{
  CUDF_EXPECTS(src_idx >= 0 && src_idx < static_cast<size_type>(per_file_metadata.size()),
               "invalid source index");
  return per_file_metadata[src_idx].row_groups[row_group_index];
}

ColumnChunkMetaData const& aggregate_reader_metadata::get_column_metadata(size_type row_group_index,
                                                                          size_type src_idx,
                                                                          int schema_idx) const
{
  // Map schema index to the provided source file index
  schema_idx = map_schema_index(schema_idx, src_idx);

  auto col =
    std::find_if(per_file_metadata[src_idx].row_groups[row_group_index].columns.begin(),
                 per_file_metadata[src_idx].row_groups[row_group_index].columns.end(),
                 [schema_idx](ColumnChunk const& col) { return col.schema_idx == schema_idx; });
  CUDF_EXPECTS(col != std::end(per_file_metadata[src_idx].row_groups[row_group_index].columns),
               "Found no metadata for schema index",
               std::range_error);
  return col->meta_data;
}

std::vector<std::unordered_map<std::string, int64_t>>
aggregate_reader_metadata::get_rowgroup_metadata() const
{
  std::vector<std::unordered_map<std::string, int64_t>> rg_metadata;

  std::for_each(
    per_file_metadata.cbegin(), per_file_metadata.cend(), [&rg_metadata](auto const& pfm) {
      std::transform(pfm.row_groups.cbegin(),
                     pfm.row_groups.cend(),
                     std::back_inserter(rg_metadata),
                     [](auto const& rg) {
                       std::unordered_map<std::string, int64_t> rg_meta_map;
                       rg_meta_map["num_rows"]        = rg.num_rows;
                       rg_meta_map["total_byte_size"] = rg.total_byte_size;
                       return rg_meta_map;
                     });
    });
  return rg_metadata;
}

bool aggregate_reader_metadata::is_schema_index_mapped(int schema_idx, int pfm_idx) const
{
  // Check if schema_idx or pfm_idx is invalid
  CUDF_EXPECTS(
    schema_idx >= 0 and pfm_idx >= 0 and pfm_idx < static_cast<int>(per_file_metadata.size()),
    "Parquet reader encountered an invalid schema_idx or pfm_idx",
    std::out_of_range);

  // True if root index requested or zeroth file index or schema_idx maps doesn't exist. (i.e.
  // schemas are identical).
  if (schema_idx == 0 or pfm_idx == 0 or schema_idx_maps.empty()) { return true; }

  // Check if mapped
  auto const& schema_idx_map = schema_idx_maps[pfm_idx - 1];
  return schema_idx_map.find(schema_idx) != schema_idx_map.end();
}

int aggregate_reader_metadata::map_schema_index(int schema_idx, int pfm_idx) const
{
  // Check if schema_idx or pfm_idx is invalid
  CUDF_EXPECTS(
    schema_idx >= 0 and pfm_idx >= 0 and pfm_idx < static_cast<int>(per_file_metadata.size()),
    "Parquet reader encountered an invalid schema_idx or pfm_idx",
    std::out_of_range);

  // Check if pfm_idx is zero or root index requested or schema_idx_maps doesn't exist (i.e.
  // schemas are identical).
  if (schema_idx == 0 or pfm_idx == 0 or schema_idx_maps.empty()) { return schema_idx; }

  // schema_idx_maps will only have > 0 size when we are reading matching column projection from
  // mismatched Parquet sources.
  auto const& schema_idx_map = schema_idx_maps[pfm_idx - 1];
  CUDF_EXPECTS(schema_idx_map.find(schema_idx) != schema_idx_map.end(),
               "Unmapped schema index encountered in the specified source tree",
               std::out_of_range);

  // Return the mapped schema idx.
  return schema_idx_map.at(schema_idx);
}

std::string aggregate_reader_metadata::get_pandas_index() const
{
  // Assumes that all input files have the same metadata
  // TODO: verify this assumption
  auto it = keyval_maps[0].find("pandas");
  if (it != keyval_maps[0].end()) {
    // Captures a list of quoted strings found inside square brackets after `"index_columns":`
    // Inside quotes supports newlines, brackets, escaped quotes, etc.
    // One-liner regex:
    // "index_columns"\s*:\s*\[\s*((?:"(?:|(?:.*?(?![^\\]")).?)[^\\]?",?\s*)*)\]
    // Documented below.
    std::regex index_columns_expr{
      R"("index_columns"\s*:\s*\[\s*)"  // match preamble, opening square bracket, whitespace
      R"(()"                            // Open first capturing group
      R"((?:")"                         // Open non-capturing group match opening quote
      R"((?:|(?:.*?(?![^\\]")).?))"     // match empty string or anything between quotes
      R"([^\\]?")"                      // Match closing non-escaped quote
      R"(,?\s*)"                        // Match optional comma and whitespace
      R"()*)"                           // Close non-capturing group and repeat 0 or more times
      R"())"                            // Close first capturing group
      R"(\])"                           // Match closing square brackets
    };
    std::smatch sm;
    if (std::regex_search(it->second, sm, index_columns_expr)) { return sm[1].str(); }
  }
  return "";
}

std::vector<std::string> aggregate_reader_metadata::get_pandas_index_names() const
{
  std::vector<std::string> names;
  auto str = get_pandas_index();
  if (str.length() != 0) {
    std::regex index_name_expr{R"(\"((?:\\.|[^\"])*)\")"};
    std::smatch sm;
    while (std::regex_search(str, sm, index_name_expr)) {
      if (sm.size() == 2) {  // 2 = whole match, first item
        if (std::find(names.begin(), names.end(), sm[1].str()) == names.end()) {
          std::regex esc_quote{R"(\\")"};
          names.emplace_back(std::regex_replace(sm[1].str(), esc_quote, R"(")"));
        }
      }
      str = sm.suffix();
    }
  }
  return names;
}

std::tuple<int64_t, size_type, std::vector<row_group_info>, std::vector<size_t>>
aggregate_reader_metadata::select_row_groups(
  host_span<std::vector<size_type> const> row_group_indices,
  int64_t skip_rows_opt,
  std::optional<size_type> const& num_rows_opt,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter,
  rmm::cuda_stream_view stream) const
{
  std::optional<std::vector<std::vector<size_type>>> filtered_row_group_indices;
  // if filter is not empty, then gather row groups to read after predicate pushdown
  if (filter.has_value()) {
    filtered_row_group_indices = filter_row_groups(
      row_group_indices, output_dtypes, output_column_schemas, filter.value(), stream);
    if (filtered_row_group_indices.has_value()) {
      row_group_indices =
        host_span<std::vector<size_type> const>(filtered_row_group_indices.value());
    }
  }
  std::vector<row_group_info> selection;
  auto [rows_to_skip, rows_to_read] = [&]() {
    if (not row_group_indices.empty()) { return std::pair<int64_t, size_type>{}; }
    auto const from_opts = cudf::io::detail::skip_rows_num_rows_from_options(
      skip_rows_opt, num_rows_opt, get_num_rows());
    CUDF_EXPECTS(from_opts.second <= static_cast<int64_t>(std::numeric_limits<size_type>::max()),
                 "Number of reading rows exceeds cudf's column size limit.");
    return std::pair{static_cast<int64_t>(from_opts.first),
                     static_cast<size_type>(from_opts.second)};
  }();

  // Get number of rows in each data source
  std::vector<size_t> num_rows_per_source(per_file_metadata.size(), 0);

  if (!row_group_indices.empty()) {
    CUDF_EXPECTS(row_group_indices.size() == per_file_metadata.size(),
                 "Must specify row groups for each source");

    for (size_t src_idx = 0; src_idx < row_group_indices.size(); ++src_idx) {
      auto const& fmd = per_file_metadata[src_idx];
      for (auto const& rowgroup_idx : row_group_indices[src_idx]) {
        CUDF_EXPECTS(
          rowgroup_idx >= 0 && rowgroup_idx < static_cast<size_type>(fmd.row_groups.size()),
          "Invalid rowgroup index");
        selection.emplace_back(rowgroup_idx, rows_to_read, src_idx);
        // if page-level indexes are present, then collect extra chunk and page info.
        column_info_for_row_group(selection.back(), 0);
        auto const rows_this_rg = get_row_group(rowgroup_idx, src_idx).num_rows;
        rows_to_read += rows_this_rg;
        num_rows_per_source[src_idx] += rows_this_rg;
      }
    }
  } else {
    size_type count = 0;
    for (size_t src_idx = 0; src_idx < per_file_metadata.size(); ++src_idx) {
      auto const& fmd = per_file_metadata[src_idx];
      for (size_t rg_idx = 0;
           rg_idx < fmd.row_groups.size() and count < rows_to_skip + rows_to_read;
           ++rg_idx) {
        auto const& rg             = fmd.row_groups[rg_idx];
        auto const chunk_start_row = count;
        count += rg.num_rows;
        if (count > rows_to_skip || count == 0) {
          // start row of this row group adjusted with rows_to_skip
          num_rows_per_source[src_idx] += count;
          num_rows_per_source[src_idx] -=
            (chunk_start_row <= rows_to_skip) ? rows_to_skip : chunk_start_row;

          // We need the unadjusted start index of this row group to correctly initialize
          // ColumnChunkDesc for this row group in create_global_chunk_info() and calculate
          // the row offset for the first pass in compute_input_passes().
          selection.emplace_back(rg_idx, chunk_start_row, src_idx);

          // If page-level indexes are present, then collect extra chunk and page info.
          // The page indexes rely on absolute row numbers, not adjusted for skip_rows.
          column_info_for_row_group(selection.back(), chunk_start_row);
        }
        // Adjust the number of rows for the last source file.
        if (count >= rows_to_skip + rows_to_read) {
          num_rows_per_source[src_idx] -= count - rows_to_skip - rows_to_read;
        }
      }
    }
  }

  return {rows_to_skip, rows_to_read, std::move(selection), std::move(num_rows_per_source)};
}

std::tuple<std::vector<input_column_info>,
           std::vector<cudf::io::detail::inline_column_buffer>,
           std::vector<size_type>>
aggregate_reader_metadata::select_columns(
  std::optional<std::vector<std::string>> const& use_names,
  std::optional<std::vector<std::string>> const& filter_columns_names,
  bool include_index,
  bool strings_to_categorical,
  type_id timestamp_type_id)
{
  auto const find_schema_child =
    [&](SchemaElement const& schema_elem, std::string const& name, int const pfm_idx = 0) {
      auto const& col_schema_idx = std::find_if(
        schema_elem.children_idx.cbegin(),
        schema_elem.children_idx.cend(),
        [&](size_t col_schema_idx) { return get_schema(col_schema_idx, pfm_idx).name == name; });

      return (col_schema_idx != schema_elem.children_idx.end())
               ? static_cast<size_type>(*col_schema_idx)
               : -1;
    };

  std::vector<cudf::io::detail::inline_column_buffer> output_columns;
  std::vector<input_column_info> input_columns;
  std::vector<int> nesting;

  // Return true if column path is valid. e.g. if the path is {"struct1", "child1"}, then it is
  // valid if "struct1.child1" exists in this file's schema. If "struct1" exists but "child1" is
  // not a child of "struct1" then the function will return false for "struct1"
  std::function<bool(
    column_name_info const*, int, std::vector<cudf::io::detail::inline_column_buffer>&, bool)>
    build_column = [&](column_name_info const* col_name_info,
                       int schema_idx,
                       std::vector<cudf::io::detail::inline_column_buffer>& out_col_array,
                       bool has_list_parent) {
      auto const& schema_elem = get_schema(schema_idx);

      // if schema_elem is a stub then it does not exist in the column_name_info and column_buffer
      // hierarchy. So continue on
      if (schema_elem.is_stub()) {
        // is this legit?
        CUDF_EXPECTS(schema_elem.num_children == 1, "Unexpected number of children for stub");
        auto const child_col_name_info = col_name_info ? &col_name_info->children[0] : nullptr;
        return build_column(
          child_col_name_info, schema_elem.children_idx[0], out_col_array, has_list_parent);
      }

      auto const one_level_list = schema_elem.is_one_level_list(get_schema(schema_elem.parent_idx));

      // if we're at the root, this is a new output column
      auto const col_type = one_level_list
                              ? type_id::LIST
                              : to_type_id(schema_elem, strings_to_categorical, timestamp_type_id);
      auto const dtype    = to_data_type(col_type, schema_elem);

      cudf::io::detail::inline_column_buffer output_col(dtype,
                                                        schema_elem.repetition_type == OPTIONAL);
      if (has_list_parent) { output_col.user_data |= PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT; }
      // store the index of this element if inserted in out_col_array
      nesting.push_back(static_cast<int>(out_col_array.size()));
      output_col.name = schema_elem.name;

      // build each child
      bool path_is_valid = false;
      if (col_name_info == nullptr or col_name_info->children.empty()) {
        // add all children of schema_elem.
        // At this point, we can no longer pass a col_name_info to build_column
        for (int idx = 0; idx < schema_elem.num_children; idx++) {
          path_is_valid |= build_column(nullptr,
                                        schema_elem.children_idx[idx],
                                        output_col.children,
                                        has_list_parent || col_type == type_id::LIST);
        }
      } else {
        for (const auto& idx : col_name_info->children) {
          path_is_valid |= build_column(&idx,
                                        find_schema_child(schema_elem, idx.name),
                                        output_col.children,
                                        has_list_parent || col_type == type_id::LIST);
        }
      }

      // if I have no children, we're at a leaf and I'm an input column (that is, one with actual
      // data stored) so add me to the list.
      if (schema_elem.num_children == 0) {
        input_column_info& input_col = input_columns.emplace_back(
          schema_idx, schema_elem.name, schema_elem.max_repetition_level > 0);

        // set up child output column for one-level encoding list
        if (one_level_list) {
          // determine the element data type
          auto const element_type =
            to_type_id(schema_elem, strings_to_categorical, timestamp_type_id);
          auto const element_dtype = to_data_type(element_type, schema_elem);

          cudf::io::detail::inline_column_buffer element_col(
            element_dtype, schema_elem.repetition_type == OPTIONAL);
          if (has_list_parent || col_type == type_id::LIST) {
            element_col.user_data |= PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT;
          }
          // store the index of this element
          nesting.push_back(static_cast<int>(output_col.children.size()));
          // TODO: not sure if we should assign a name or leave it blank
          element_col.name = "element";

          output_col.children.push_back(std::move(element_col));
        }

        std::copy(nesting.cbegin(), nesting.cend(), std::back_inserter(input_col.nesting));

        // pop off the extra nesting element.
        if (one_level_list) { nesting.pop_back(); }

        path_is_valid = true;  // If we're able to reach leaf then path is valid
      }

      if (path_is_valid) { out_col_array.push_back(std::move(output_col)); }

      nesting.pop_back();
      return path_is_valid;
    };

  // Compares two schema elements to be equal except their number of children
  auto const equal_to_except_num_children = [](SchemaElement const& lhs, SchemaElement const& rhs) {
    return lhs.type == rhs.type and lhs.converted_type == rhs.converted_type and
           lhs.type_length == rhs.type_length and lhs.name == rhs.name and
           lhs.decimal_scale == rhs.decimal_scale and
           lhs.decimal_precision == rhs.decimal_precision and lhs.field_id == rhs.field_id;
  };

  // Maps a projected column's schema_idx in the zeroth per_file_metadata (source) to the
  // corresponding schema_idx in pfm_idx'th per_file_metadata (destination). The projected
  // column's path must match across sources, else an appropriate exception is thrown.
  std::function<void(column_name_info const*, int const, int const, int const)> map_column =
    [&](column_name_info const* col_name_info,
        int const src_schema_idx,
        int const dst_schema_idx,
        int const pfm_idx) {
      auto const& src_schema_elem = get_schema(src_schema_idx);
      auto const& dst_schema_elem = get_schema(dst_schema_idx, pfm_idx);

      // Check the schema elements to be equal except their number of children as we only care about
      // the specific column paths in the schema trees. Raise an invalid_argument error if the
      // schema elements don't match.
      CUDF_EXPECTS(equal_to_except_num_children(src_schema_elem, dst_schema_elem),
                   "Encountered mismatching SchemaElement properties for a column in "
                   "the selected path",
                   std::invalid_argument);

      // Get the schema_idx_map for this data source (pfm)
      auto& schema_idx_map = schema_idx_maps[pfm_idx - 1];
      // Map the schema index from 0th tree (src) to the one in the current (dst) tree.
      schema_idx_map[src_schema_idx] = dst_schema_idx;

      // If src_schema_elem is a stub, it does not exist in the column_name_info and column_buffer
      // hierarchy. So continue on with mapping.
      if (src_schema_elem.is_stub()) {
        // Check if dst_schema_elem is also a stub i.e. has num_children == 1 that we didn't
        // previously check. Raise an invalid_argument error if dst_schema_elem is not a stub.
        CUDF_EXPECTS(dst_schema_elem.is_stub(),
                     "Encountered mismatching schemas for stub.",
                     std::invalid_argument);
        auto const child_col_name_info = col_name_info ? &col_name_info->children[0] : nullptr;
        return map_column(child_col_name_info,
                          src_schema_elem.children_idx[0],
                          dst_schema_elem.children_idx[0],
                          pfm_idx);
      }

      // The path ends here. If this is a list/struct col (has children), then map all its children
      // which must be identical.
      if (col_name_info == nullptr or col_name_info->children.empty()) {
        // Check the number of children to be equal to be mapped. An out_of_range error if the
        // number of children isn't equal.
        CUDF_EXPECTS(src_schema_elem.num_children == dst_schema_elem.num_children,
                     "Encountered mismatching number of children for a "
                     "column in the selected path",
                     std::out_of_range);

        std::for_each(thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(src_schema_elem.num_children),
                      [&](auto const child_idx) {
                        map_column(nullptr,
                                   src_schema_elem.children_idx[child_idx],
                                   dst_schema_elem.children_idx[child_idx],
                                   pfm_idx);
                      });
      }
      // The path goes further down to specific child(ren) of this column so map only those
      // children.
      else {
        std::for_each(
          col_name_info->children.cbegin(),
          col_name_info->children.cend(),
          [&](auto const& child_col_name_info) {
            // Ensure that each named child column exists in the destination schema tree for the
            // paths to align up. An out_of_range error otherwise.
            CUDF_EXPECTS(
              find_schema_child(dst_schema_elem, child_col_name_info.name, pfm_idx) != -1,
              "Encountered mismatching schema tree depths across data sources",
              std::out_of_range);
            map_column(&child_col_name_info,
                       find_schema_child(src_schema_elem, child_col_name_info.name),
                       find_schema_child(dst_schema_elem, child_col_name_info.name, pfm_idx),
                       pfm_idx);
          });
      }
    };

  std::vector<int> output_column_schemas;

  //
  // there is not necessarily a 1:1 mapping between input columns and output columns.
  // For example, parquet does not explicitly store a ColumnChunkDesc for struct columns.
  // The "structiness" is simply implied by the schema.  For example, this schema:
  //  required group field_id=1 name {
  //    required binary field_id=2 firstname (String);
  //    required binary field_id=3 middlename (String);
  //    required binary field_id=4 lastname (String);
  // }
  // will only contain 3 internal columns of data (firstname, middlename, lastname).  But of
  // course "name" is ultimately the struct column we want to return.
  //
  // "firstname", "middlename" and "lastname" represent the input columns in the file that we
  // process to produce the final cudf "name" column.
  //
  // A user can ask for a single field out of the struct e.g. firstname.
  // In this case they'll pass a fully qualified name to the schema element like
  // ["name", "firstname"]
  //
  auto const& root = get_schema(0);
  if (not use_names.has_value()) {
    for (auto const& schema_idx : root.children_idx) {
      build_column(nullptr, schema_idx, output_columns, false);
      output_column_schemas.push_back(schema_idx);
    }
  } else {
    struct path_info {
      std::string full_path;
      int schema_idx;
    };

    // Convert schema into a vector of every possible path
    std::vector<path_info> all_paths;
    std::function<void(std::string, int)> add_path = [&](std::string path_till_now,
                                                         int schema_idx) {
      auto const& schema_elem = get_schema(schema_idx);
      std::string curr_path   = path_till_now + schema_elem.name;
      all_paths.push_back({curr_path, schema_idx});
      for (auto const& child_idx : schema_elem.children_idx) {
        add_path(curr_path + ".", child_idx);
      }
    };
    for (auto const& child_idx : get_schema(0).children_idx) {
      add_path("", child_idx);
    }

    // Find which of the selected paths are valid and get their schema index
    std::vector<path_info> valid_selected_paths;
    // vector reference pushback (*use_names). If filter names passed.
    std::vector<std::reference_wrapper<std::vector<std::string> const>> column_names{
      *use_names, *filter_columns_names};
    for (auto const& used_column_names : column_names) {
      for (auto const& selected_path : used_column_names.get()) {
        auto found_path =
          std::find_if(all_paths.begin(), all_paths.end(), [&](path_info& valid_path) {
            return valid_path.full_path == selected_path;
          });
        if (found_path != all_paths.end()) {
          valid_selected_paths.push_back({selected_path, found_path->schema_idx});
        }
      }
    }

    // Now construct paths as vector of strings for further consumption
    std::vector<std::vector<std::string>> use_names3;
    std::transform(valid_selected_paths.cbegin(),
                   valid_selected_paths.cend(),
                   std::back_inserter(use_names3),
                   [&](path_info const& valid_path) {
                     auto schema_idx = valid_path.schema_idx;
                     std::vector<std::string> result_path;
                     do {
                       SchemaElement const& elem = get_schema(schema_idx);
                       result_path.push_back(elem.name);
                       schema_idx = elem.parent_idx;
                     } while (schema_idx > 0);
                     return std::vector<std::string>(result_path.rbegin(), result_path.rend());
                   });

    std::vector<column_name_info> selected_columns;
    if (include_index) {
      std::vector<std::string> index_names = get_pandas_index_names();
      std::transform(index_names.cbegin(),
                     index_names.cend(),
                     std::back_inserter(selected_columns),
                     [](std::string const& name) { return column_name_info(name); });
    }
    // Merge the vector use_names into a set of hierarchical column_name_info objects
    /* This is because if we have columns like this:
     *     col1
     *      / \
     *    s3   f4
     *   / \
     * f5   f6
     *
     * there may be common paths in use_names like:
     * {"col1", "s3", "f5"}, {"col1", "f4"}
     * which means we want the output to contain
     *     col1
     *      / \
     *    s3   f4
     *   /
     * f5
     *
     * rather than
     *  col1   col1
     *   |      |
     *   s3     f4
     *   |
     *   f5
     */
    for (auto const& path : use_names3) {
      auto array_to_find_in = &selected_columns;
      for (auto const& name_to_find : path) {
        // Check if the path exists in our selected_columns and if not, add it.
        auto found_col = std::find_if(
          array_to_find_in->begin(),
          array_to_find_in->end(),
          [&name_to_find](column_name_info const& col) { return col.name == name_to_find; });
        if (found_col == array_to_find_in->end()) {
          auto& col        = array_to_find_in->emplace_back(name_to_find);
          array_to_find_in = &col.children;
        } else {
          // Path exists. go down further.
          array_to_find_in = &found_col->children;
        }
      }
    }
    for (auto& col : selected_columns) {
      auto const& top_level_col_schema_idx = find_schema_child(root, col.name);
      bool valid_column = build_column(&col, top_level_col_schema_idx, output_columns, false);
      if (valid_column) {
        output_column_schemas.push_back(top_level_col_schema_idx);

        // Map the column's schema_idx across the rest of the data sources if required.
        if (per_file_metadata.size() > 1 and not schema_idx_maps.empty()) {
          std::for_each(thrust::make_counting_iterator(static_cast<size_t>(1)),
                        thrust::make_counting_iterator(per_file_metadata.size()),
                        [&](auto const pfm_idx) {
                          auto const& dst_root = get_schema(0, pfm_idx);
                          // Ensure that each top level column exists in the destination schema
                          // tree. An out_of_range error is thrown otherwise.
                          CUDF_EXPECTS(
                            find_schema_child(dst_root, col.name, pfm_idx) != -1,
                            "Encountered mismatching schema tree depths across data sources",
                            std::out_of_range);
                          map_column(&col,
                                     top_level_col_schema_idx,
                                     find_schema_child(dst_root, col.name, pfm_idx),
                                     pfm_idx);
                        });
        }
      }
    }
  }

  return std::make_tuple(
    std::move(input_columns), std::move(output_columns), std::move(output_column_schemas));
}

}  // namespace cudf::io::parquet::detail
