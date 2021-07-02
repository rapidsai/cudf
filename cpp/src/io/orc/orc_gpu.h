/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include "timezone.cuh"

#include <io/comp/gpuinflate.h>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <io/statistics/statistics.cuh>
#include <io/utilities/column_buffer.hpp>
#include "orc_common.h"

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace io {
namespace orc {
namespace gpu {

using cudf::detail::device_2dspan;

struct CompressedStreamInfo {
  CompressedStreamInfo() = default;
  explicit constexpr CompressedStreamInfo(const uint8_t *compressed_data_, size_t compressed_size_)
    : compressed_data(compressed_data_),
      uncompressed_data(nullptr),
      compressed_data_size(compressed_size_),
      decctl(nullptr),
      decstatus(nullptr),
      copyctl(nullptr),
      num_compressed_blocks(0),
      num_uncompressed_blocks(0),
      max_uncompressed_size(0)
  {
  }
  const uint8_t *compressed_data;  // [in] base ptr to compressed stream data
  uint8_t *uncompressed_data;  // [in] base ptr to uncompressed stream data or NULL if not known yet
  size_t compressed_data_size;      // [in] compressed data size for this stream
  gpu_inflate_input_s *decctl;      // [in] base ptr to decompression structure to be filled
  gpu_inflate_status_s *decstatus;  // [in] results of decompression
  gpu_inflate_input_s
    *copyctl;  // [in] base ptr to copy structure to be filled for uncompressed blocks
  uint32_t num_compressed_blocks;  // [in,out] number of entries in decctl(in), number of compressed
                                   // blocks(out)
  uint32_t num_uncompressed_blocks;  // [in,out] number of entries in copyctl(in), number of
                                     // uncompressed blocks(out)
  uint64_t max_uncompressed_size;    // [out] maximum uncompressed data size
};

enum StreamIndexType {
  CI_DATA = 0,    // Primary data stream
  CI_DATA2,       // Secondary/Length stream
  CI_PRESENT,     // Present stream
  CI_DICTIONARY,  // Dictionary stream
  CI_INDEX,       // Index stream
  CI_NUM_STREAMS
};

/**
 * @brief Struct to describe a single entry in the global dictionary
 */
struct DictionaryEntry {
  uint32_t pos;  // Position in data stream
  uint32_t len;  // Length in data stream
};

/**
 * @brief Mask to indicate conversion from decimals to float64
 */
constexpr int orc_decimal2float64_scale = 0x80;

/**
 * @brief Struct to describe per stripe's column information
 */
struct ColumnDesc {
  const uint8_t *streams[CI_NUM_STREAMS];  // ptr to data stream index
  uint32_t strm_id[CI_NUM_STREAMS];        // stream ids
  uint32_t strm_len[CI_NUM_STREAMS];       // stream length
  uint32_t *valid_map_base;                // base pointer of valid bit map for this column
  void *column_data_base;                  // base pointer of column data
  uint32_t start_row;                      // starting row of the stripe
  uint32_t num_rows;                       // starting row of the stripe
  uint32_t dictionary_start;               // start position in global dictionary
  uint32_t dict_len;                       // length of local dictionary
  uint32_t null_count;                     // number of null values in this stripe's column
  uint32_t skip_count;                     // number of non-null values to skip
  uint32_t rowgroup_id;                    // row group position
  uint8_t encoding_kind;                   // column encoding kind (orc::ColumnEncodingKind)
  uint8_t type_kind;                       // column data type (orc::TypeKind)
  uint8_t dtype_len;      // data type length (for types that can be mapped to different sizes)
  int32_t decimal_scale;  // number of fractional decimal digits for decimal type
  int32_t ts_clock_rate;  // output timestamp clock frequency (0=default, 1000=ms, 1000000000=ns)
};

/**
 * @brief Struct to describe a groups of row belonging to a column stripe
 */
struct RowGroup {
  uint32_t chunk_id;        // Column chunk this entry belongs to
  uint32_t strm_offset[2];  // Index offset for CI_DATA and CI_DATA2 streams
  uint16_t run_pos[2];      // Run position for CI_DATA and CI_DATA2
};

/**
 * @brief Struct to describe an encoder data chunk
 */
struct EncChunk {
  uint32_t start_row;     // start row of this chunk
  uint32_t num_rows;      // number of rows in this chunk
  uint8_t encoding_kind;  // column encoding kind (orc::ColumnEncodingKind)
  uint8_t type_kind;      // column data type (orc::TypeKind)
  uint8_t dtype_len;      // data type length
  int32_t scale;          // scale for decimals or timestamps

  uint32_t *dict_index;  // dictionary index from row index
  device_span<uint32_t> decimal_offsets;
  column_device_view *leaf_column;
};

/**
 * @brief Struct to describe the streams that correspond to a single `EncChunk`.
 */
struct encoder_chunk_streams {
  uint8_t *data_ptrs[CI_NUM_STREAMS];  // encoded output
  int32_t ids[CI_NUM_STREAMS];         // stream id; -1 if stream is not present
  uint32_t lengths[CI_NUM_STREAMS];    // in: max length, out: actual length
};

/**
 * @brief Struct to describe a column stream within a stripe
 */
struct StripeStream {
  size_t bfr_offset;        // Offset of this stream in compressed buffer
  uint32_t stream_size;     // Size of stream in bytes
  uint32_t first_chunk_id;  // First chunk of the stripe
  uint32_t num_chunks;      // Number of chunks in the stripe
  uint32_t column_id;       // column index
  uint32_t first_block;     // First compressed block
  uint8_t stream_type;      // Stream index type
  uint8_t pad[3];
};

/**
 * @brief Struct to describe a dictionary chunk
 */
struct DictionaryChunk {
  uint32_t *dict_data;   // dictionary data (index of non-null rows)
  uint32_t *dict_index;  // row indices of corresponding string (row from dictionary index)
  uint32_t start_row;    // start row of this chunk
  uint32_t num_rows;     // num rows in this chunk
  uint32_t num_strings;  // number of strings in this chunk
  uint32_t
    string_char_count;  // total size of string data (NOTE: assumes less than 4G bytes per chunk)
  uint32_t num_dict_strings;  // number of strings in dictionary
  uint32_t dict_char_count;   // size of dictionary string data for this chunk

  column_device_view *leaf_column;  //!< Pointer to string column
};

/**
 * @brief Struct to describe a dictionary
 */
struct StripeDictionary {
  uint32_t *dict_data;       // row indices of corresponding string (row from dictionary index)
  uint32_t *dict_index;      // dictionary index from row index
  uint32_t column_id;        // real column id
  uint32_t start_chunk;      // first chunk in stripe
  uint32_t num_chunks;       // number of chunks in the stripe
  uint32_t num_strings;      // number of unique strings in the dictionary
  uint32_t dict_char_count;  // total size of dictionary string data

  column_device_view *leaf_column;  //!< Pointer to string column
};

/**
 * @brief Launches kernel for parsing the compressed stripe data
 *
 * @param[in] strm_info List of compressed streams
 * @param[in] num_streams Number of compressed streams
 * @param[in] compression_block_size maximum size of compressed blocks (up to 16M)
 * @param[in] log2maxcr log2 of maximum compression ratio (used to infer max uncompressed size from
 *compressed size)
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 */
void ParseCompressedStripeData(CompressedStreamInfo *strm_info,
                               int32_t num_streams,
                               uint32_t compression_block_size,
                               uint32_t log2maxcr           = 24,
                               rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Launches kernel for re-assembling decompressed blocks into a single contiguous block
 *
 * @param[in] strm_info List of compressed streams
 * @param[in] num_streams Number of compressed streams
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 */
void PostDecompressionReassemble(CompressedStreamInfo *strm_info,
                                 int32_t num_streams,
                                 rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Launches kernel for constructing rowgroup from index streams
 *
 * @param[out] row_groups RowGroup device array [rowgroup][column]
 * @param[in] strm_info List of compressed streams (or NULL if uncompressed)
 * @param[in] chunks ColumnDesc device array [stripe][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] num_rowgroups Number of row groups
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 */
void ParseRowGroupIndex(RowGroup *row_groups,
                        CompressedStreamInfo *strm_info,
                        ColumnDesc *chunks,
                        uint32_t num_columns,
                        uint32_t num_stripes,
                        uint32_t num_rowgroups,
                        uint32_t rowidx_stride,
                        rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Launches kernel for decoding NULLs and building string dictionary index tables
 *
 * @param[in] chunks ColumnDesc device array [stripe][column]
 * @param[in] global_dictionary Global dictionary device array
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] max_rows Maximum number of rows to load
 * @param[in] first_row Crop all rows below first_row
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 */
void DecodeNullsAndStringDictionaries(ColumnDesc *chunks,
                                      DictionaryEntry *global_dictionary,
                                      uint32_t num_columns,
                                      uint32_t num_stripes,
                                      size_t max_rows              = ~0,
                                      size_t first_row             = 0,
                                      rmm::cuda_stream_view stream = rmm::cuda_stream_default);

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
 * @param[in] row_groups Optional row index data
 * @param[in] num_rowgroups Number of row groups in row index data
 * @param[in] rowidx_stride Row index stride
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 */
void DecodeOrcColumnData(ColumnDesc const *chunks,
                         DictionaryEntry *global_dictionary,
                         uint32_t num_columns,
                         uint32_t num_stripes,
                         size_t max_rows              = ~0,
                         size_t first_row             = 0,
                         timezone_table_view tz_table = {},
                         const RowGroup *row_groups   = 0,
                         uint32_t num_rowgroups       = 0,
                         uint32_t rowidx_stride       = 0,
                         rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Launches kernel for encoding column data
 *
 * @param[in] chunks  encoder chunk device array [column][rowgroup]
 * @param[in, out] streams chunk streams device array [column][rowgroup]
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 */
void EncodeOrcColumnData(device_2dspan<EncChunk const> chunks,
                         device_2dspan<encoder_chunk_streams> streams,
                         rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Launches kernel for encoding column dictionaries
 *
 * @param[in] stripes Stripe dictionaries device array [stripe][string_column]
 * @param[in] chunks encoder chunk device array [column][rowgroup]
 * @param[in] num_string_columns Number of string columns
 * @param[in] num_stripes Number of stripes
 * @param[in,out] enc_streams chunk streams device array [column][rowgroup]
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 */
void EncodeStripeDictionaries(StripeDictionary *stripes,
                              device_2dspan<EncChunk const> chunks,
                              uint32_t num_string_columns,
                              uint32_t num_stripes,
                              device_2dspan<encoder_chunk_streams> enc_streams,
                              rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Set leaf column element of EncChunk
 *
 * @param[in] view table device view representing input table
 * @param[in,out] chunks encoder chunk device array [column][rowgroup]
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 */
void set_chunk_columns(const table_device_view &view,
                       device_2dspan<EncChunk> chunks,
                       rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for compacting chunked column data prior to compression
 *
 * @param[in,out] strm_desc StripeStream device array [stripe][stream]
 * @param[in,out] enc_streams chunk streams device array [column][rowgroup]
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 */
void CompactOrcDataStreams(device_2dspan<StripeStream> strm_desc,
                           device_2dspan<encoder_chunk_streams> enc_streams,
                           rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Launches kernel(s) for compressing data streams
 *
 * @param[in] compressed_data Output compressed blocks
 * @param[in] num_compressed_blocks Total number of compressed blocks
 * @param[in] compression Type of compression
 * @param[in] comp_blk_size Compression block size
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 * @param[in,out] strm_desc StripeStream device array [stripe][stream]
 * @param[in,out] enc_streams chunk streams device array [column][rowgroup]
 * @param[out] comp_in Per-block compression input parameters
 * @param[out] comp_out Per-block compression status
 */
void CompressOrcDataStreams(uint8_t *compressed_data,
                            uint32_t num_compressed_blocks,
                            CompressionKind compression,
                            uint32_t comp_blk_size,
                            device_2dspan<StripeStream> strm_desc,
                            device_2dspan<encoder_chunk_streams> enc_streams,
                            gpu_inflate_input_s *comp_in,
                            gpu_inflate_status_s *comp_out,
                            rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Launches kernel for initializing dictionary chunks
 *
 * @param[in] view table device view representing input table
 * @param[in,out] chunks DictionaryChunk device array [rowgroup][column]
 * @param[in] dict_data dictionary data (index of non-null rows)
 * @param[in] dict_index row indices of corresponding string (row from dictionary index)
 * @param[in] row_index_stride Rowgroup size in rows
 * @param[in] str_col_ids List of columns that are strings type
 * @param[in] num_columns Number of columns
 * @param[in] num_rowgroups Number of row groups
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 */
void InitDictionaryIndices(const table_device_view &view,
                           DictionaryChunk *chunks,
                           uint32_t *dict_data,
                           uint32_t *dict_index,
                           size_t row_index_stride,
                           size_type *str_col_ids,
                           uint32_t num_columns,
                           uint32_t num_rowgroups,
                           rmm::cuda_stream_view stream);

/**
 * @brief Launches kernel for building stripe dictionaries
 *
 * @param[in] stripes_dev StripeDictionary device array [stripe][column]
 * @param[in] stripes_host StripeDictionary host array [stripe][column]
 * @param[in] chunks DictionaryChunk device array [rowgroup][column]
 * @param[in] num_stripes Number of stripes
 * @param[in] num_rowgroups Number of row groups
 * @param[in] num_columns Number of columns
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 */
void BuildStripeDictionaries(StripeDictionary *stripes_dev,
                             StripeDictionary *stripes_host,
                             DictionaryChunk const *chunks,
                             uint32_t num_stripes,
                             uint32_t num_rowgroups,
                             uint32_t num_columns,
                             rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Launches kernels to initialize statistics collection
 *
 * @param[out] groups Statistics groups (rowgroup-level)
 * @param[in] cols Column descriptors
 * @param[in] num_columns Number of columns
 * @param[in] num_rowgroups Number of rowgroups
 * @param[in] row_index_stride Rowgroup size in rows
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 */
void orc_init_statistics_groups(statistics_group *groups,
                                const stats_column_desc *cols,
                                uint32_t num_columns,
                                uint32_t num_rowgroups,
                                uint32_t row_index_stride,
                                rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Launches kernels to return statistics buffer offsets and sizes
 *
 * @param[in,out] groups Statistics merge groups
 * @param[in] chunks Statistics chunks
 * @param[in] statistics_count Number of statistics buffers to encode
 * @param[in] stream CUDA stream to use, default `rmm::cuda_stream_default`
 */
void orc_init_statistics_buffersize(statistics_merge_group *groups,
                                    const statistics_chunk *chunks,
                                    uint32_t statistics_count,
                                    rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Launches kernel to encode statistics in ORC protobuf format
 *
 * @param[out] blob_bfr Output buffer for statistics blobs
 * @param[in,out] groups Statistics merge groups
 * @param[in,out] chunks Statistics data
 * @param[in] statistics_count Number of statistics buffers
 */
void orc_encode_statistics(uint8_t *blob_bfr,
                           statistics_merge_group *groups,
                           const statistics_chunk *chunks,
                           uint32_t statistics_count,
                           rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace gpu
}  // namespace orc
}  // namespace io
}  // namespace cudf
