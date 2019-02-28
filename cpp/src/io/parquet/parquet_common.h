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

#ifndef __IO_PARQUET_COMMON_H__
#define __IO_PARQUET_COMMON_H__

namespace parquet {

// Basic parquet column data types
enum Type {
    BOOLEAN = 0,
    INT32 = 1,
    INT64 = 2,
    INT96 = 3,
    FLOAT = 4,
    DOUBLE = 5,
    BYTE_ARRAY = 6,
    FIXED_LEN_BYTE_ARRAY = 7,
};

// Encodings
enum Encoding {
    PLAIN = 0,
    GROUP_VAR_INT = 1, // Deprecated, never used
    PLAIN_DICTIONARY = 2,
    RLE = 3,
    BIT_PACKED = 4,
    DELTA_BINARY_PACKED = 5,
    DELTA_LENGTH_BYTE_ARRAY = 6,
    DELTA_BYTE_ARRAY = 7,
    RLE_DICTIONARY = 8,
};

// Compression
enum Compression {
    UNCOMPRESSED = 0,
    SNAPPY = 1,
    GZIP = 2,
    LZO = 3,
    BROTLI = 4,    // Added in 2.3.2
    LZ4 = 5,    // Added in 2.3.2
    ZSTD = 6,    // Added in 2.3.2
};

enum FieldRepetitionType {
    REQUIRED = 0,   // This field is required (can not be null) and each record has exactly 1 value.
    OPTIONAL = 1,   // The field is optional (can be null) and each record has 0 or 1 values.
    REPEATED = 2,   // The field is repeated and can contain 0 or more values
};

// Page types
enum PageType {
    DATA_PAGE = 0,
    INDEX_PAGE = 1,
    DICTIONARY_PAGE = 2,
    DATA_PAGE_V2 = 3,
};



}; // namespace parquet


#endif // __IO_PARQUET_COMMON_H__




