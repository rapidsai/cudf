/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#ifndef __IO_PARQUET_GPU_H__
#define __IO_PARQUET_GPU_H__

#include <cstdint>
#include <io/comp/gpuinflate.h>
#include "parquet_common.h"

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {

/**
 * @brief Enums for the flags in the page header
 **/
enum {
  PAGEINFO_FLAGS_DICTIONARY = 0x01,  // Indicates a dictionary page
};

/**
 * @brief Struct to describe the output of a string datatype
 **/
struct nvstrdesc_s {
  const char *ptr;
  size_t count;
};

/**
 * @brief Struct describing a particular page of column chunk data
 **/
struct PageInfo {
  uint8_t *page_data;                 // Compressed page data before decompression, or uncompressed data after decompression
  int32_t compressed_page_size;       // compressed data size in bytes
  int32_t uncompressed_page_size;     // uncompressed data size in bytes
  int32_t num_values;                 // Number of values in this data page or dictionary
  int32_t chunk_row;                  // starting row of this page relative to the start of the chunk
  int32_t num_rows;                   // number of rows in this page
  int32_t chunk_idx;                  // column chunk this page belongs to
  uint8_t flags;                      // PAGEINFO_FLAGS_XXX
  uint8_t encoding;                   // Encoding for data or dictionary page
  uint8_t definition_level_encoding;  // Encoding used for definition levels (data page)
  uint8_t repetition_level_encoding;  // Encoding used for repetition levels (data page)
  int32_t valid_count;                // Count of valid (non-null) values in this page (negative values indicate data error)
};

/**
 * @brief Struct describing a particular chunk of column data
 **/
struct ColumnChunkDesc {
  ColumnChunkDesc() = default;
  explicit constexpr ColumnChunkDesc(
      size_t compressed_size_, uint8_t *compressed_data_, size_t num_values_,
      uint16_t datatype_, uint16_t datatype_length_, uint32_t start_row_,
      uint32_t num_rows_, int16_t max_definition_level_,
      int16_t max_repetition_level_, uint8_t def_level_bits_,
      uint8_t rep_level_bits_, int8_t codec_, int8_t converted_type_,
      int8_t decimal_scale_, int32_t ts_clock_rate_)
      : compressed_data(compressed_data_),
        compressed_size(compressed_size_),
        num_values(num_values_),
        start_row(start_row_),
        num_rows(num_rows_),
        max_def_level(max_definition_level_),
        max_rep_level(max_repetition_level_),
        def_level_bits(def_level_bits_),
        rep_level_bits(rep_level_bits_),
        data_type(datatype_ | (datatype_length_ << 3)),
        num_data_pages(0),
        num_dict_pages(0),
        max_num_pages(0),
        page_info(nullptr),
        str_dict_index(nullptr),
        valid_map_base(nullptr),
        column_data_base(nullptr),
        codec(codec_),
        converted_type(converted_type_),
        decimal_scale(decimal_scale_),
        ts_clock_rate(ts_clock_rate_) {}

  uint8_t *compressed_data;     // pointer to compressed column chunk data
  size_t compressed_size;       // total compressed data size for this chunk
  size_t num_values;            // total number of values in this column
  size_t start_row;             // starting row of this chunk
  uint32_t num_rows;            // number of rows in this chunk
  int16_t max_def_level;        // max definition level
  int16_t max_rep_level;        // max repetition level
  uint16_t data_type;           // basic column data type, ((type_length << 3) |
                                // parquet::Type)
  uint8_t def_level_bits;       // bits to encode max definition level
  uint8_t rep_level_bits;       // bits to encode max repetition level
  int32_t num_data_pages;       // number of data pages
  int32_t num_dict_pages;       // number of dictionary pages
  int32_t max_num_pages;        // size of page_info array
  PageInfo *page_info;          // output page info for up to num_dict_pages +
                                // num_data_pages (dictionary pages first)
  nvstrdesc_s *str_dict_index;  // index for string dictionary
  uint32_t *valid_map_base;     // base pointer of valid bit map for this column
  void *column_data_base;       // base pointer of column data
  int8_t codec;                 // compressed codec enum
  int8_t converted_type;        // converted type enum
  int8_t decimal_scale;         // decimal scale pow(10, -decimal_scale)
  int32_t ts_clock_rate;        // output timestamp clock frequency (0=default, 1000=ms, 1000000000=ns)
};


/**
 * @brief Struct describing an encoder column
 **/
struct EncColumnDesc
{
  const uint32_t *valid_map_base;   //!< base ptr of column valid map (null if not present)
  const void *column_data_base;     //!< base ptr of column data
  uint32_t num_rows;                //!< number of rows in column
  uint8_t physical_type;            //!< physical data type
  uint8_t converted_type;           //!< logical data type
  uint8_t level_bits;               //!< bits to encode max definition (lower nibble) & repetition (upper nibble) levels
  uint8_t pad;
};

#define MAX_PAGE_FRAGMENT_SIZE  5000    //!< Max number of rows in a page fragment

/**
 * @brief Struct describing an encoder page fragment
 **/
struct PageFragment
{
  uint32_t fragment_data_size;      //!< Size of fragment data in bytes
  uint16_t num_rows;                //!< Number of rows in fragment
  uint16_t non_nulls;               //!< Number of non-null values
};

/**
 * @brief Struct describing an encoder data page
 **/
struct EncPage
{
  uint8_t *page_data;               //!< Ptr to uncompressed page
  uint8_t *compressed_data;         //!< Ptr to compressed page
  uint16_t num_fragments;           //!< Number of fragments in page
  uint8_t page_type;                //!< Page type (0=data, 2=dictionary)
  uint8_t pad;
  uint32_t chunk_id;                //!< Index in chunk array
  uint32_t max_hdr_size;            //!< Maximum size of page header
  uint32_t max_data_size;           //!< Maximum size of coded page data (excluding header)
  uint32_t start_row;               //!< First row of page
  uint32_t num_rows;                //!< Rows in page
};

/**
 * @brief Return worst-case compressed size of compressed data given the uncompressed size
 **/
inline size_t __device__ __host__ GetMaxCompressedBfrSize(size_t uncomp_size, uint32_t num_pages = 1) {
  return uncomp_size + (uncomp_size >> 7) + num_pages * 8;
}

/**
 * @brief Struct describing an encoder column chunk
 **/
struct EncColumnChunk
{
  const EncColumnDesc *col_desc;    //!< Column description
  const PageFragment *fragments;    //!< First fragment in chunk
  uint8_t *uncompressed_bfr;        //!< Uncompressed page data
  uint8_t *compressed_bfr;          //!< Compressed page data
  uint32_t bfr_size;                //!< Uncompressed buffer size
  uint32_t compressed_size;         //!< Compressed buffer size
  uint32_t start_row;               //!< First row of chunk
  uint32_t num_rows;                //!< Number of rows in chunk
  uint32_t first_page;              //!< First page of chunk
  uint32_t num_pages;               //!< Number of pages in chunk
  uint8_t is_compressed;            //!< Nonzero if the chunk uses compression
  uint8_t pad[3];
};

/**
 * @brief Launches kernel for parsing the page headers in the column chunks
 *
 * @param[in] chunks List of column chunks
 * @param[in] num_chunks Number of column chunks
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t DecodePageHeaders(ColumnChunkDesc *chunks, int32_t num_chunks,
                              cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Launches kernel for building the dictionary index for the column
 * chunks
 *
 * @param[in] chunks List of column chunks
 * @param[in] num_chunks Number of column chunks
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t BuildStringDictionaryIndex(ColumnChunkDesc *chunks,
                                       int32_t num_chunks,
                                       cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Launches kernel for reading the column data stored in the pages
 *
 * The page data will be written to the output pointed to in the page's
 * associated column chunk.
 *
 * @param[in] pages List of pages
 * @param[in] num_pages Number of pages
 * @param[in,out] chunks List of column chunks
 * @param[in] num_chunks Number of column chunks
 * @param[in] num_rows Total number of rows to read
 * @param[in] min_row Minimum number of rows to read, default 0
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t DecodePageData(PageInfo *pages, int32_t num_pages,
                           ColumnChunkDesc *chunks, int32_t num_chunks,
                           size_t num_rows, size_t min_row = 0,
                           cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Launches kernel for initializing encoder page fragments
 *
 * @param[out] frag Fragment array [column_id][fragment_id]
 * @param[in] col_desc Column description array [column_id]
 * @param[in] num_fragments Number of fragments per column
 * @param[in] num_columns Number of columns
 * @param[in] fragment_size Number of rows per fragment
 * @param[in] num_rows Number of rows per column
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t InitPageFragments(PageFragment *frag, const EncColumnDesc *col_desc,
                              int32_t num_fragments, int32_t num_columns,
                              uint32_t fragment_size, uint32_t num_rows,
                              cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Launches kernel for initializing encoder data pages
 *
 * @param[in,out] chunks Column chunks [rowgroup][column]
 * @param[out] pages Encode page array (null if just counting pages)
 * @param[in] col_desc Column description array [column_id]
 * @param[in] num_rowgroups Number of fragments per column
 * @param[in] num_columns Number of columns
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t InitEncoderPages(EncColumnChunk *chunks, EncPage *pages, const EncColumnDesc *col_desc,
                             int32_t num_rowgroups, int32_t num_columns,
                             cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Launches kernel for packing column data into parquet pages
 *
 * @param[in,out] pages Device array of EncPages (unordered)
 * @param[in] chunks Column chunks
 * @param[in] num_pages Number of pages
 * @param[in] start_page First page to encode in page array
 * @param[out] comp_in Optionally initializes compressor input params
 * @param[out] comp_out Optionally initializes compressor output params
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t EncodePages(EncPage *pages, const EncColumnChunk *chunks, uint32_t num_pages, uint32_t start_page = 0,
                        gpu_inflate_input_s *comp_in = nullptr, gpu_inflate_status_s *comp_out = nullptr,
                        cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Launches kernel to make the compressed vs uncompressed chunk-level decision
 *
 * @param[in,out] chunks Column chunks
 * @param[in] pages Device array of EncPages
 * @param[in] num_chunks Number of column chunks
 * @param[in] comp_in Compressor input parameters
 * @param[in] comp_out Compressor status
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t DecideCompression(EncColumnChunk *chunks, const EncPage *pages, uint32_t num_chunks,
                              const gpu_inflate_input_s *comp_in, const gpu_inflate_status_s *comp_out,
                              cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Launches kernel to gather pages to a single contiguous block per chunk
 *
 * @param[in,out] chunks Column chunks
 * @param[in] pages Device array of EncPages
 * @param[in] num_chunks Number of column chunks
 * @param[in] comp_out Compressor status
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t GatherPages(EncColumnChunk *chunks, const EncPage *pages, uint32_t num_chunks, cudaStream_t stream = (cudaStream_t)0);


} // namespace gpu
} // namespace parquet
} // namespace io
} // namespace cudf

#endif // __IO_PARQUET_GPU_H__
