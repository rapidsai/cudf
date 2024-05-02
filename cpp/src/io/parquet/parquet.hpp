/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#pragma once

#include "parquet_common.hpp"

#include <cudf/types.hpp>

#include <thrust/optional.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cudf::io::parquet::detail {

constexpr uint32_t parquet_magic = (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));

/**
 * @brief Struct that describes the Parquet file data header
 */
struct file_header_s {
  uint32_t magic;
};

/**
 * @brief Struct that describes the Parquet file data postscript
 */
struct file_ender_s {
  uint32_t footer_len;
  uint32_t magic;
};

// thrift inspired code simplified.
struct DecimalType {
  int32_t scale     = 0;
  int32_t precision = 0;
};

struct TimeUnit {
  enum Type { UNDEFINED, MILLIS, MICROS, NANOS };
  Type type;
};

struct TimeType {
  // Default to true because the timestamps are implicitly in UTC
  // Writer option overrides this default
  bool isAdjustedToUTC = true;
  TimeUnit unit        = {TimeUnit::MILLIS};
};

struct TimestampType {
  // Default to true because the timestamps are implicitly in UTC
  // Writer option overrides this default
  bool isAdjustedToUTC = true;
  TimeUnit unit        = {TimeUnit::MILLIS};
};

struct IntType {
  int8_t bitWidth = 0;
  bool isSigned   = false;
};

struct LogicalType {
  enum Type {
    UNDEFINED,
    STRING,
    MAP,
    LIST,
    ENUM,
    DECIMAL,
    DATE,
    TIME,
    TIMESTAMP,
    // 9 is reserved
    INTEGER = 10,
    UNKNOWN,
    JSON,
    BSON
  };
  Type type;
  thrust::optional<DecimalType> decimal_type;
  thrust::optional<TimeType> time_type;
  thrust::optional<TimestampType> timestamp_type;
  thrust::optional<IntType> int_type;

  LogicalType(Type tp = UNDEFINED) : type(tp) {}
  LogicalType(DecimalType&& dt) : type(DECIMAL), decimal_type(dt) {}
  LogicalType(TimeType&& tt) : type(TIME), time_type(tt) {}
  LogicalType(TimestampType&& tst) : type(TIMESTAMP), timestamp_type(tst) {}
  LogicalType(IntType&& it) : type(INTEGER), int_type(it) {}

  constexpr bool is_time_millis() const
  {
    return type == TIME and time_type->unit.type == TimeUnit::MILLIS;
  }

  constexpr bool is_time_micros() const
  {
    return type == TIME and time_type->unit.type == TimeUnit::MICROS;
  }

  constexpr bool is_time_nanos() const
  {
    return type == TIME and time_type->unit.type == TimeUnit::NANOS;
  }

  constexpr bool is_timestamp_millis() const
  {
    return type == TIMESTAMP and timestamp_type->unit.type == TimeUnit::MILLIS;
  }

  constexpr bool is_timestamp_micros() const
  {
    return type == TIMESTAMP and timestamp_type->unit.type == TimeUnit::MICROS;
  }

  constexpr bool is_timestamp_nanos() const
  {
    return type == TIMESTAMP and timestamp_type->unit.type == TimeUnit::NANOS;
  }

  constexpr int8_t bit_width() const { return type == INTEGER ? int_type->bitWidth : -1; }

  constexpr bool is_signed() const { return type == INTEGER and int_type->isSigned; }

  constexpr int32_t scale() const { return type == DECIMAL ? decimal_type->scale : -1; }

  constexpr int32_t precision() const { return type == DECIMAL ? decimal_type->precision : -1; }
};

/**
 * Union to specify the order used for the min_value and max_value fields for a column.
 */
struct ColumnOrder {
  enum Type { UNDEFINED, TYPE_ORDER };
  Type type;
};

/**
 * @brief Struct for describing an element/field in the Parquet format schema
 *
 * Parquet is a strongly-typed format so the file layout can be interpreted as
 * as a schema tree.
 */
struct SchemaElement {
  // 1: parquet physical type for output
  Type type = UNDEFINED_TYPE;
  // 2: byte length of FIXED_LENGTH_BYTE_ARRAY elements, or maximum bit length for other types
  int32_t type_length = 0;
  // 3: repetition of the field
  FieldRepetitionType repetition_type = REQUIRED;
  // 4: name of the field
  std::string name = "";
  // 5: nested fields
  int32_t num_children = 0;
  // 6: DEPRECATED: record the original type before conversion to parquet type
  thrust::optional<ConvertedType> converted_type;
  // 7: DEPRECATED: record the scale for DECIMAL converted type
  int32_t decimal_scale = 0;
  // 8: DEPRECATED: record the precision for DECIMAL converted type
  int32_t decimal_precision = 0;
  // 9: save field_id from original schema
  thrust::optional<int32_t> field_id;
  // 10: replaces converted type
  thrust::optional<LogicalType> logical_type;

  // extra cudf specific fields
  bool output_as_byte_array = false;

  // cudf type determined from arrow:schema
  thrust::optional<type_id> arrow_type;

  // The following fields are filled in later during schema initialization
  int max_definition_level = 0;
  int max_repetition_level = 0;
  size_type parent_idx     = 0;
  std::vector<size_type> children_idx;

  bool operator==(SchemaElement const& other) const
  {
    return type == other.type && converted_type == other.converted_type &&
           type_length == other.type_length && repetition_type == other.repetition_type &&
           name == other.name && num_children == other.num_children &&
           decimal_scale == other.decimal_scale && decimal_precision == other.decimal_precision &&
           field_id == other.field_id;
  }

  // the parquet format is a little squishy when it comes to interpreting
  // repeated fields. sometimes repeated fields act as "stubs" in the schema
  // that don't represent a true nesting level.
  //
  // this is the case with plain lists:
  //
  // optional group my_list (LIST) {
  //   repeated group element {        <-- not part of the output hierarchy
  //     required binary str (UTF8);
  //   };
  // }
  //
  // However, for backwards compatibility reasons, there are a few special cases, namely
  // List<Struct<>> (which also corresponds to how the map type is specified), where
  // this does not hold true
  //
  // optional group my_list (LIST) {
  //   repeated group element {        <-- part of the hierarchy because it represents a struct
  //     required binary str (UTF8);
  //     required int32 num;
  //  };
  // }
  [[nodiscard]] bool is_stub() const { return repetition_type == REPEATED && num_children == 1; }

  // https://github.com/apache/parquet-cpp/blob/642da05/src/parquet/schema.h#L49-L50
  // One-level LIST encoding: Only allows required lists with required cells:
  //   repeated value_type name
  [[nodiscard]] bool is_one_level_list(SchemaElement const& parent) const
  {
    return repetition_type == REPEATED and num_children == 0 and not parent.is_list();
  }

  // returns true if the element is a list
  [[nodiscard]] bool is_list() const { return converted_type == LIST; }

  // in parquet terms, a group is a level of nesting in the schema. a group
  // can be a struct or a list
  [[nodiscard]] bool is_struct() const
  {
    return type == UNDEFINED_TYPE &&
           // this assumption might be a little weak.
           ((repetition_type != REPEATED) || (repetition_type == REPEATED && num_children > 1));
  }
};

/**
 * @brief Thrift-derived struct describing column chunk statistics
 */
struct Statistics {
  // deprecated max value in signed comparison order
  thrust::optional<std::vector<uint8_t>> max;
  // deprecated min value in signed comparison order
  thrust::optional<std::vector<uint8_t>> min;
  // count of null values in the column
  thrust::optional<int64_t> null_count;
  // count of distinct values occurring
  thrust::optional<int64_t> distinct_count;
  // max value for column determined by ColumnOrder
  thrust::optional<std::vector<uint8_t>> max_value;
  // min value for column determined by ColumnOrder
  thrust::optional<std::vector<uint8_t>> min_value;
  // If true, max_value is the actual maximum value for a column
  thrust::optional<bool> is_max_value_exact;
  // If true, min_value is the actual minimum value for a column
  thrust::optional<bool> is_min_value_exact;
};

/**
 * @brief Thrift-derived struct containing statistics used to estimate page and column chunk sizes
 */
struct SizeStatistics {
  // Number of variable-width bytes stored for the page/chunk. Should not be set for anything
  // but the BYTE_ARRAY physical type.
  thrust::optional<int64_t> unencoded_byte_array_data_bytes;
  /**
   * When present, there is expected to be one element corresponding to each
   * repetition (i.e. size=max repetition_level+1) where each element
   * represents the number of times the repetition level was observed in the
   * data.
   *
   * This value should not be written if max_repetition_level is 0.
   */
  thrust::optional<std::vector<int64_t>> repetition_level_histogram;

  /**
   * Same as repetition_level_histogram except for definition levels.
   *
   * This value should not be written if max_definition_level is 0 or 1.
   */
  thrust::optional<std::vector<int64_t>> definition_level_histogram;
};

/**
 * @brief Thrift-derived struct describing page location information stored
 * in the offsets index.
 */
struct PageLocation {
  int64_t offset;                // Offset of the page in the file
  int32_t compressed_page_size;  // Compressed page size in bytes plus the heeader length
  int64_t first_row_index;  // Index within the column chunk of the first row of the page. reset to
                            // 0 at the beginning of each column chunk
};

/**
 * @brief Thrift-derived struct describing the offset index.
 */
struct OffsetIndex {
  std::vector<PageLocation> page_locations;
  // per-page size info. see description of the same field in SizeStatistics. only present for
  // columns with a BYTE_ARRAY physical type.
  thrust::optional<std::vector<int64_t>> unencoded_byte_array_data_bytes;
};

/**
 * @brief Thrift-derived struct describing the column index.
 */
struct ColumnIndex {
  std::vector<bool> null_pages;  // Boolean used to determine if a page contains only null values
  std::vector<std::vector<uint8_t>> min_values;  // lower bound for values in each page
  std::vector<std::vector<uint8_t>> max_values;  // upper bound for values in each page
  BoundaryOrder boundary_order =
    BoundaryOrder::UNORDERED;  // Indicates if min and max values are ordered
  thrust::optional<std::vector<int64_t>> null_counts;  // Optional count of null values per page
  // Repetition/definition level histograms for the column chunk
  thrust::optional<std::vector<int64_t>> repetition_level_histogram;
  thrust::optional<std::vector<int64_t>> definition_level_histogram;
};

/**
 * @brief Thrift-derived struct describing page encoding statistics
 */
struct PageEncodingStats {
  PageType page_type;  // The page type (data/dic/...)
  Encoding encoding;   // Encoding of the page
  int32_t count;       // Number of pages of this type with this encoding
};

/**
 * @brief Thrift-derived struct describing column sort order
 */
struct SortingColumn {
  int32_t column_idx;  // The column index (in this row group)
  bool descending;     // If true, indicates this column is sorted in descending order
  bool nulls_first;    // If true, nulls will come before non-null values
};

/**
 * @brief Thrift-derived struct describing a column chunk
 */
struct ColumnChunkMetaData {
  // Type of this column
  Type type = BOOLEAN;
  // Set of all encodings used for this column. The purpose is to validate
  // whether we can decode those pages.
  std::vector<Encoding> encodings;
  // Path in schema
  std::vector<std::string> path_in_schema;
  // Compression codec
  Compression codec = UNCOMPRESSED;
  // Number of values in this column
  int64_t num_values = 0;
  // Total byte size of all uncompressed pages in this column chunk (including the headers)
  int64_t total_uncompressed_size = 0;
  // Total byte size of all compressed pages in this column chunk (including the headers)
  int64_t total_compressed_size = 0;
  // Byte offset from beginning of file to first data page
  int64_t data_page_offset = 0;
  // Byte offset from beginning of file to root index page
  int64_t index_page_offset = 0;
  // Byte offset from the beginning of file to first (only) dictionary page
  int64_t dictionary_page_offset = 0;
  // Optional statistics for this column chunk
  Statistics statistics;
  // Set of all encodings used for pages in this column chunk. This information can be used to
  // determine if all data pages are dictionary encoded for example.
  thrust::optional<std::vector<PageEncodingStats>> encoding_stats;
  // Optional statistics to help estimate total memory when converted to in-memory representations.
  // The histograms contained in these statistics can also be useful in some cases for more
  // fine-grained nullability/list length filter pushdown.
  thrust::optional<SizeStatistics> size_statistics;
};

/**
 * @brief Thrift-derived struct describing a chunk of data for a particular
 * column
 *
 * Each column chunk lives in a particular row group and are guaranteed to be
 * contiguous in the file. Any missing or corrupted chunks can be skipped during
 * reading.
 */
struct ColumnChunk {
  std::string file_path = "";
  int64_t file_offset   = 0;
  ColumnChunkMetaData meta_data;
  int64_t offset_index_offset = 0;  // File offset of ColumnChunk's OffsetIndex
  int32_t offset_index_length = 0;  // Size of ColumnChunk's OffsetIndex, in bytes
  int64_t column_index_offset = 0;  // File offset of ColumnChunk's ColumnIndex
  int32_t column_index_length = 0;  // Size of ColumnChunk's ColumnIndex, in bytes

  // Following fields are derived from other fields
  int schema_idx = -1;  // Index in flattened schema (derived from path_in_schema)
  // The indexes don't really live here, but it's a convenient place to hang them.
  std::optional<OffsetIndex> offset_index;
  std::optional<ColumnIndex> column_index;
};

/**
 * @brief Thrift-derived struct describing a group of row data
 *
 * There may be one or more row groups within a dataset, with each row group
 * consisting of a column chunk for each column.
 */
struct RowGroup {
  // Metadata for each column chunk in this row group.
  std::vector<ColumnChunk> columns;
  // Total byte size of all the uncompressed column data in this row group
  int64_t total_byte_size = 0;
  // Number of rows in this row group
  int64_t num_rows = 0;
  // If set, specifies a sort ordering of the rows in this RowGroup.
  // The sorting columns can be a subset of all the columns.
  thrust::optional<std::vector<SortingColumn>> sorting_columns;
  // Byte offset from beginning of file to first page (data or dictionary) in this row group
  thrust::optional<int64_t> file_offset;
  // Total byte size of all compressed (and potentially encrypted) column data in this row group
  thrust::optional<int64_t> total_compressed_size;
  // Row group ordinal in the file
  thrust::optional<int16_t> ordinal;
};

/**
 * @brief Thrift-derived struct describing a key-value pair, for user metadata
 */
struct KeyValue {
  std::string key;
  std::string value;
};

/**
 * @brief Thrift-derived struct describing file-level metadata
 *
 * The additional information stored in the key_value_metadata can be used
 * during reading to reconstruct the output data to the exact original dataset
 * prior to conversion to Parquet.
 */
struct FileMetaData {
  int32_t version = 0;
  std::vector<SchemaElement> schema;
  int64_t num_rows = 0;
  std::vector<RowGroup> row_groups;
  std::vector<KeyValue> key_value_metadata;
  std::string created_by = "";
  thrust::optional<std::vector<ColumnOrder>> column_orders;
};

/**
 * @brief Thrift-derived struct describing the header for a data page
 */
struct DataPageHeader {
  int32_t num_values                 = 0;  // Number of values, including NULLs, in this data page.
  Encoding encoding                  = Encoding::PLAIN;  // Encoding used for this data page
  Encoding definition_level_encoding = Encoding::PLAIN;  // Encoding used for definition levels
  Encoding repetition_level_encoding = Encoding::PLAIN;  // Encoding used for repetition levels
};

/**
 * @brief Thrift-derived struct describing the header for a V2 data page
 */
struct DataPageHeaderV2 {
  int32_t num_values = 0;  // Number of values, including NULLs, in this data page.
  int32_t num_nulls  = 0;  // Number of NULL values, in this data page.
  int32_t num_rows   = 0;  // Number of rows in this data page. which means
                           // pages change on record boundaries (r = 0)
  Encoding encoding                     = Encoding::PLAIN;  // Encoding used for this data page
  int32_t definition_levels_byte_length = 0;                // length of the definition levels
  int32_t repetition_levels_byte_length = 0;                // length of the repetition levels
  bool is_compressed                    = true;             // whether the values are compressed.
};

/**
 * @brief Thrift-derived struct describing the header for a dictionary page
 */
struct DictionaryPageHeader {
  int32_t num_values = 0;                // Number of values in the dictionary
  Encoding encoding  = Encoding::PLAIN;  // Encoding using this dictionary page
};

/**
 * @brief Thrift-derived struct describing the page header
 *
 * Column data are divided into individual chunks, which are subdivided into
 * pages. Each page has an associated header, describing the page type. There
 * can be multiple page types interleaved in a column chunk, and each page is
 * individually compressed and encoded. Any missing or corrupted pages can be
 * skipped during reading.
 */
struct PageHeader {
  PageType type =
    PageType::DATA_PAGE;  // the type of the page: indicates which of the *_header fields is set
  int32_t uncompressed_page_size = 0;  // Uncompressed page size in bytes (not including the header)
  int32_t compressed_page_size   = 0;  // Compressed page size in bytes (not including the header)
  DataPageHeader data_page_header;
  DictionaryPageHeader dictionary_page_header;
  DataPageHeaderV2 data_page_header_v2;
};

// bit space we are reserving in column_buffer::user_data
constexpr uint32_t PARQUET_COLUMN_BUFFER_SCHEMA_MASK          = (0xff'ffffu);
constexpr uint32_t PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED = (1 << 24);
// if this column has a list parent anywhere above it in the hierarchy
constexpr uint32_t PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT = (1 << 25);

/**
 * @brief Count the number of leading zeros in an unsigned integer
 */
static inline int CountLeadingZeros32(uint32_t value)
{
#if defined(__clang__) || defined(__GNUC__)
  if (value == 0) return 32;
  return static_cast<int>(__builtin_clz(value));
#else
  int bitpos = 0;
  while (value != 0) {
    value >>= 1;
    ++bitpos;
  }
  return 32 - bitpos;
#endif
}

}  // namespace cudf::io::parquet::detail
