/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cudf {
namespace io {
namespace parquet {
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

// thrift generated code simplified.
struct StringType {};
struct MapType {};
struct ListType {};
struct EnumType {};
struct DecimalType {
  int32_t scale     = 0;
  int32_t precision = 0;
};
struct DateType {};

struct MilliSeconds {};
struct MicroSeconds {};
struct NanoSeconds {};
using TimeUnit_isset = struct TimeUnit_isset {
  bool MILLIS{false};
  bool MICROS{false};
  bool NANOS{false};
};

struct TimeUnit {
  TimeUnit_isset isset;
  MilliSeconds MILLIS;
  MicroSeconds MICROS;
  NanoSeconds NANOS;
};

struct TimeType {
  bool isAdjustedToUTC = false;
  TimeUnit unit;
};
struct TimestampType {
  bool isAdjustedToUTC = false;
  TimeUnit unit;
};
struct IntType {
  int8_t bitWidth = 0;
  bool isSigned   = false;
};
struct NullType {};
struct JsonType {};
struct BsonType {};

// thrift generated code simplified.
using LogicalType_isset = struct LogicalType_isset {
  bool STRING{false};
  bool MAP{false};
  bool LIST{false};
  bool ENUM{false};
  bool DECIMAL{false};
  bool DATE{false};
  bool TIME{false};
  bool TIMESTAMP{false};
  bool INTEGER{false};
  bool UNKNOWN{false};
  bool JSON{false};
  bool BSON{false};
};

struct LogicalType {
  LogicalType_isset isset;
  StringType STRING;
  MapType MAP;
  ListType LIST;
  EnumType ENUM;
  DecimalType DECIMAL;
  DateType DATE;
  TimeType TIME;
  TimestampType TIMESTAMP;
  IntType INTEGER;
  NullType UNKNOWN;
  JsonType JSON;
  BsonType BSON;
};

/**
 * @brief Struct for describing an element/field in the Parquet format schema
 *
 * Parquet is a strongly-typed format so the file layout can be interpreted as
 * as a schema tree.
 */
struct SchemaElement {
  Type type                    = UNDEFINED_TYPE;
  ConvertedType converted_type = UNKNOWN;
  LogicalType logical_type;
  int32_t type_length =
    0;  // Byte length of FIXED_LENGTH_BYTE_ARRAY elements, or maximum bit length for other types
  FieldRepetitionType repetition_type = REQUIRED;
  std::string name                    = "";
  int32_t num_children                = 0;
  int32_t decimal_scale               = 0;
  int32_t decimal_precision           = 0;
  std::optional<int32_t> field_id     = std::nullopt;
  bool output_as_byte_array           = false;

  // The following fields are filled in later during schema initialization
  int max_definition_level = 0;
  int max_repetition_level = 0;
  int parent_idx           = 0;
  std::vector<size_t> children_idx;

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
  [[nodiscard]] bool is_one_level_list() const
  {
    return repetition_type == REPEATED and num_children == 0;
  }

  // in parquet terms, a group is a level of nesting in the schema. a group
  // can be a struct or a list
  [[nodiscard]] bool is_struct() const
  {
    return type == UNDEFINED_TYPE &&
           // this assumption might be a little weak.
           ((repetition_type != REPEATED) || (repetition_type == REPEATED && num_children == 2));
  }
};

/**
 * @brief Thrift-derived struct describing column chunk statistics
 */
struct Statistics {
  std::vector<uint8_t> max;        // deprecated max value in signed comparison order
  std::vector<uint8_t> min;        // deprecated min value in signed comparison order
  int64_t null_count     = -1;     // count of null values in the column
  int64_t distinct_count = -1;     // count of distinct values occurring
  std::vector<uint8_t> max_value;  // max value for column determined by ColumnOrder
  std::vector<uint8_t> min_value;  // min value for column determined by ColumnOrder
};

/**
 * @brief Thrift-derived struct describing a column chunk
 */
struct ColumnChunkMetaData {
  Type type = BOOLEAN;
  std::vector<Encoding> encodings;
  std::vector<std::string> path_in_schema;
  Compression codec  = UNCOMPRESSED;
  int64_t num_values = 0;
  int64_t total_uncompressed_size =
    0;  // total byte size of all uncompressed pages in this column chunk (including the headers)
  int64_t total_compressed_size =
    0;  // total byte size of all compressed pages in this column chunk (including the headers)
  int64_t data_page_offset  = 0;  // Byte offset from beginning of file to first data page
  int64_t index_page_offset = 0;  // Byte offset from beginning of file to root index page
  int64_t dictionary_page_offset =
    0;  // Byte offset from the beginning of file to first (only) dictionary page
  std::vector<uint8_t> statistics_blob;  // Encoded chunk-level statistics as binary blob
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
};

/**
 * @brief Thrift-derived struct describing a group of row data
 *
 * There may be one or more row groups within a dataset, with each row group
 * consisting of a column chunk for each column.
 */
struct RowGroup {
  int64_t total_byte_size = 0;
  std::vector<ColumnChunk> columns;
  int64_t num_rows = 0;
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
  std::string created_by         = "";
  uint32_t column_order_listsize = 0;
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
};

/**
 * @brief Thrift-derived struct describing the column index.
 */
struct ColumnIndex {
  std::vector<bool> null_pages;  // Boolean used to determine if a page contains only null values
  std::vector<std::vector<uint8_t>> min_values;  // lower bound for values in each page
  std::vector<std::vector<uint8_t>> max_values;  // upper bound for values in each page
  BoundaryOrder boundary_order =
    BoundaryOrder::UNORDERED;                    // Indicates if min and max values are ordered
  std::vector<int64_t> null_counts;              // Optional count of null values per page
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

}  // namespace parquet
}  // namespace io
}  // namespace cudf
