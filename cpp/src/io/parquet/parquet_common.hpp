/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <cstdint>

#include <string>
#include <vector>

namespace cudf {
namespace io {
namespace parquet {
/**
 * @brief Basic data types in Parquet, determines how data is physically stored
 **/
enum Type {
  UNDEFINED_TYPE       = -1,  // Undefined for non-leaf nodes
  BOOLEAN              = 0,
  INT32                = 1,
  INT64                = 2,
  INT96                = 3,  // Deprecated
  FLOAT                = 4,
  DOUBLE               = 5,
  BYTE_ARRAY           = 6,
  FIXED_LEN_BYTE_ARRAY = 7,
};

/**
 * @brief High-level data types in Parquet, determines how data is logically interpreted
 **/
enum ConvertedType {
  UNKNOWN = -1,  // No type information present
  UTF8    = 0,   // a BYTE_ARRAY actually contains UTF8 encoded chars
  MAP     = 1,   // a map is converted as an optional field containing a repeated key/value pair
  MAP_KEY_VALUE = 2,  // a key/value pair is converted into a group of two fields
  LIST =
    3,  // a list is converted into an optional field containing a repeated field for its values
  ENUM    = 4,      // an enum is converted into a binary field
  DECIMAL = 5,      // A decimal value. 10^(-scale) encoded as 2's complement big endian
                    // (precision=number of digits, scale=location of decimal point)
  DATE        = 6,  // A Date, stored as days since Unix epoch, encoded as the INT32 physical type.
  TIME_MILLIS = 7,  // A time. The total number of milliseconds since midnight.The value is stored
                    // as an INT32 physical type.
  TIME_MICROS = 8,  // A time. The total number of microseconds since midnight.  The value is stored
                    // as an INT64 physical type.
  TIMESTAMP_MILLIS = 9,   // A date/time combination, recorded as milliseconds since the Unix epoch
                          // using physical type of INT64.
  TIMESTAMP_MICROS = 10,  // A date/time combination, microseconds since the Unix epoch as INT64
  UINT_8           = 11,  // An unsigned integer 8-bit value as INT32
  UINT_16          = 12,  // An unsigned integer 16-bit value as INT32
  UINT_32          = 13,  // An unsigned integer 32-bit value as INT32
  UINT_64          = 14,  // An unsigned integer 64-bit value as INT64
  INT_8            = 15,  // A signed integer 8-bit value as INT32
  INT_16           = 16,  // A signed integer 16-bit value as INT32
  INT_32           = 17,  // A signed integer 32-bit value as INT32
  INT_64           = 18,  // A signed integer 8-bit value as INT64
  JSON             = 19,  // A JSON document embedded within a single UTF8 column.
  BSON             = 20,  // A BSON document embedded within a single BINARY column.
  INTERVAL = 21,  // This type annotates a time interval stored as a FIXED_LEN_BYTE_ARRAY of length
                  // 12 for 3 integers {months,days,milliseconds}
};

/**
 * @brief Encoding types for the actual data stream
 **/
enum Encoding {
  PLAIN                   = 0,
  GROUP_VAR_INT           = 1,  // Deprecated, never used
  PLAIN_DICTIONARY        = 2,
  RLE                     = 3,
  BIT_PACKED              = 4,
  DELTA_BINARY_PACKED     = 5,
  DELTA_LENGTH_BYTE_ARRAY = 6,
  DELTA_BYTE_ARRAY        = 7,
  RLE_DICTIONARY          = 8,
};

/**
 * @brief Compression codec used for compressed data pages
 **/
enum Compression {
  UNCOMPRESSED = 0,
  SNAPPY       = 1,
  GZIP         = 2,
  LZO          = 3,
  BROTLI       = 4,  // Added in 2.3.2
  LZ4          = 5,  // Added in 2.3.2
  ZSTD         = 6,  // Added in 2.3.2
};

/**
 * @brief Compression codec used for compressed data pages
 **/
enum FieldRepetitionType {
  NO_REPETITION_TYPE = -1,
  REQUIRED = 0,  // This field is required (can not be null) and each record has exactly 1 value.
  OPTIONAL = 1,  // The field is optional (can be null) and each record has 0 or 1 values.
  REPEATED = 2,  // The field is repeated and can contain 0 or more values
};

/**
 * @brief Types of pages
 **/
enum class PageType : uint8_t {
  DATA_PAGE       = 0,
  INDEX_PAGE      = 1,
  DICTIONARY_PAGE = 2,
  DATA_PAGE_V2    = 3,
};

/**
 * @brief Thrift compact protocol struct field types
 **/
enum {
  ST_FLD_TRUE   = 1,
  ST_FLD_FALSE  = 2,
  ST_FLD_BYTE   = 3,
  ST_FLD_I16    = 4,
  ST_FLD_I32    = 5,
  ST_FLD_I64    = 6,
  ST_FLD_DOUBLE = 7,
  ST_FLD_BINARY = 8,
  ST_FLD_LIST   = 9,
  ST_FLD_SET    = 10,
  ST_FLD_MAP    = 11,
  ST_FLD_STRUCT = 12,
};

}  // namespace parquet
}  // namespace io
}  // namespace cudf
