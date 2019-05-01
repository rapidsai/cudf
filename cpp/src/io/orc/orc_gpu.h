/*
* Copyright (c) 2019, NVIDIA CORPORATION.
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

#ifndef __IO_ORC_GPU_H__
#define __IO_ORC_GPU_H__

#include "io/comp/gpuinflate.h"

namespace orc { namespace gpu {

#define DECIMALS_AS_FLOAT64     1     // 0: store decimals as INT64, 1: store decimals as FLOAT64

#define ORC_TS_CLKRATE          1000  // Rate of the output timestamp clock (1000=milliseconds, 1000000000=nanoseconds)


struct CompressedStreamInfo {
  CompressedStreamInfo() = default;
  explicit constexpr CompressedStreamInfo(uint8_t *compressed_data_,
                                          size_t compressed_size_)
      : compressed_data(compressed_data_),
        compressed_data_size(compressed_size_),
        uncompressed_data(nullptr),
        decctl(nullptr),
        decstatus(nullptr),
        max_compressed_blocks(0),
        num_compressed_blocks(0),
        max_uncompressed_size(0) {}
  uint8_t *compressed_data;         // [in] base ptr to compressed stream data
  uint8_t *uncompressed_data;       // [in] base ptr to uncompressed stream data or NULL if not known yet
  size_t compressed_data_size;      // [in] compressed data size for this stream
  gpu_inflate_input_s *decctl;      // [in] base ptr to decompression structure to be filled
  gpu_inflate_status_s *decstatus;  // [in] results of decompression
  uint32_t max_compressed_blocks;   // [in] number of entries in decctl
  uint32_t num_compressed_blocks;   // [out] total number of compressed blocks in this stream
  uint64_t max_uncompressed_size;   // [out] maximum uncompressed data size
};


enum StreamIndexType {
    CI_DATA = 0,        // Primary data stream
    CI_DATA2,           // Secondary/Length stream
    CI_PRESENT,         // Present stream
    CI_DICTIONARY,      // Dictionary stream
    CI_NUM_STREAMS
};


/**
 * @brief Struct to describe the output of a string datatype
 **/
struct nvstrdesc_s {
    const char *ptr;
    size_t count;
};


/**
* @brief Struct to describe a single entry in the global dictionary
**/
struct DictionaryEntry {
    uint32_t pos;   // Position in data stream
    uint32_t len;   // Length in data stream
};


/**
 * @brief Struct to describe per stripe's column information
 **/
struct ColumnDesc
{
    const uint8_t *streams[CI_NUM_STREAMS];     // ptr to data stream index
    uint32_t strm_id[CI_NUM_STREAMS];           // stream ids
    uint32_t strm_len[CI_NUM_STREAMS];          // stream length
    uint32_t *valid_map_base;                   // base pointer of valid bit map for this column
    void *column_data_base;                     // base pointer of column data
    uint32_t start_row;                         // starting row of the stripe
    uint32_t num_rows;                          // starting row of the stripe
    uint32_t dictionary_start;                  // start position in global dictionary
    uint32_t dict_len;                          // length of local dictionary
    uint32_t null_count;                        // number of null values in this stripe's column
    uint32_t skip_count;                        // number of non-null values to skip
    uint8_t encoding_kind;                      // column encoding kind (orc::ColumnEncodingKind)
    uint8_t type_kind;                          // column data type (orc::TypeKind)
    uint8_t dtype_len;                          // data type length (for types that can be mapped to different sizes)
    uint8_t decimal_scale;                      // number of fractional decimal digits for decimal type
    uint8_t pad[4];
};


/**
 * @brief Launches kernel for parsing the compressed stripe data
 *
 * @param[in] strm_info List of compressed streams
 * @param[in] num_streams Number of compressed streams
 * @param[in] compression_block_size maximum size of compressed blocks (up to 16M)
 * @param[in] log2maxcr log2 of maximum compression ratio (used to infer max uncompressed size from compressed size)
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t ParseCompressedStripeData(CompressedStreamInfo *strm_info, int32_t num_streams, uint32_t compression_block_size, uint32_t log2maxcr = 24, cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Launches kernel for re-assembling decompressed blocks into a single contiguous block
 *
 * @param[in] strm_info List of compressed streams
 * @param[in] num_streams Number of compressed streams
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t PostDecompressionReassemble(CompressedStreamInfo *strm_info, int32_t num_streams, cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Launches kernel for decoding NULLs and building string dictionary index tables
 *
 * @param[in] chunks ColumnDesc device array [stripe][column]
 * @param[in] global_dictionary Global dictionary device array
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] max_rows Maximum number of rows to load
 * @param[in] first_row Crop all rows below first_row
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t DecodeNullsAndStringDictionaries(ColumnDesc *chunks, DictionaryEntry *global_dictionary, uint32_t num_columns, uint32_t num_stripes, size_t max_rows = ~0, size_t first_row = 0, cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Launches kernel for decoding column data
 *
 * @param[in] chunks ColumnDesc device array [stripe][column]
 * @param[in] global_dictionary Global dictionary device array
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] max_rows Maximum number of rows to load
 * @param[in] first_row Crop all rows below first_row
 * @param[in] tz_table Timezone translation table
 * @param[in] tz_len Length of timezone translation table
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t DecodeOrcColumnData(ColumnDesc *chunks, DictionaryEntry *global_dictionary, uint32_t num_columns, uint32_t num_stripes, size_t max_rows = ~0,
                                size_t first_row = 0, int64_t *tz_table = 0, size_t tz_len = 0, cudaStream_t stream = (cudaStream_t)0);


};}; // orc::gpu namespace

#endif // __IO_ORC_GPU_H__
