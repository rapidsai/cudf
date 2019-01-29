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

// file: orc_types.h
// defines data types 

#ifndef __GDF_ORC_DATA_TYPES_H__
#define __GDF_ORC_DATA_TYPES_H__

#include <stdint.h>
#include <stddef.h>
#include <utility>

// define the byte size of the element for nanoseconds stream of timestamp
#define GDF_ORC_TIMESTAMP_NANO_PRECISION 8

// fixed width data types.
typedef unsigned char orc_uint8;
typedef   signed char orc_sint8;
typedef unsigned char orc_byte;
typedef unsigned char orc_bitmap;

typedef float  orc_float32;
typedef double orc_float64;

typedef uint16_t    orc_uint16;
typedef  int16_t    orc_sint16;
typedef uint32_t    orc_uint32;
typedef  int32_t    orc_sint32;
typedef uint64_t    orc_uint64;
typedef  int64_t    orc_sint64;


typedef struct {
    orc_uint64 hi;
    orc_uint64 low;
} orc_uint128 ;

typedef orc_uint128    orc_sint128;

using gdf_string = std::pair<const char*, size_t> ;

/** ---------------------------------------------------------------------------*
* @brief GDF_ORC error codes
* ---------------------------------------------------------------------------**/
typedef enum {
    GDF_ORC_SUCCESS = 0,                //< reading orc file is succeeded.
    GDF_ORC_UNSUPPORTED_DATA_TYPE,      //< treated as success, but the data types which ORC are not supported are skipped.

    GDF_ORC_FILE_NOT_FOUND,             //< the file is not found.
    GDF_ORC_INVALID_FILE_FORMAT,        //< ORC file format error.
    GDF_ORC_INVALID_FILE_FORMAT_PROTOBUF_FAILURE,   //< google protobuf parse failure, a kind of invalid file format.

    GDF_ORC_OUT_OF_MEMORY,                          //< out of memory failure.
    GDF_ORC_UNSUPPORTED_COMPRESSION_TYPE,           //< the compression kind of ORC file is not supported.

    GDF_ORC_INVALID_API_CALL,           //< cuda API call failure

    GDF_ORC_MAX_ERROR_CODE,             //< not used as a error code, used to handle number of error code
} CudaOrcError_t;

/** ---------------------------------------------------------------------------*
* @brief GDF_ORC element types
* ---------------------------------------------------------------------------**/
enum OrcElementType {
    Uint8,
    Sint8,
    Uint16,
    Sint16,
    Uint32,
    Sint32,
    Uint64,
    Sint64,
    Uint128,
    Sint128,
    Float32,
    Float64,
    None,
};

/** ---------------------------------------------------------------------------*
* @brief GDF_ORC file compression type kinds
* ---------------------------------------------------------------------------**/
enum OrcCompressionKind {
    NONE,
    ZLIB,
    SNAPPY,
    LZO,
    LZ4,
    ZSTD,
};

/** ---------------------------------------------------------------------------*
* @brief GDF_ORC stream data types
* ---------------------------------------------------------------------------**/
enum ORCTypeKind {
    OrcBolean = 0,
    OrcByte,
    OrcShort,
    OrcInt,
    OrcLong,
    OrcFloat,
    OrcDouble,
    OrcString,
    OrcBinary,
    OrcTimestamp,
    OrcList,
    OrcMap,
    OrcStruct,
    OrcUnion,
    OrcDecimal,
    OrcDate,
    OrcVarchar,
    OrcChar,
};

/** ---------------------------------------------------------------------------*
* @brief GDF_ORC stream type kinds
* ---------------------------------------------------------------------------**/
enum ORCStreamKind {
    OrcPresent = 0,
    OrcData,
    OrcLength,
    OrcDictionaryData,
    OrcDictionaryCount,
    OrcSecondary,
    OrcRowIndex,
    OrcBloomFilter,
    OrcBloomFilterUtf8,
};

/** ---------------------------------------------------------------------------*
* @brief GDF_ORC column RLE encoding kinds
* ---------------------------------------------------------------------------**/
enum ORCColumnEncodingKind {
    OrcDirect = 0,
    OrcDictionary,
    OrcDirect_V2,
    OrcDictionary_V2,
};

#endif //  __GDF_ORC_DATA_TYPES_H__
