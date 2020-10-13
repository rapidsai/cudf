/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <io/parquet/parquet_gpu.hpp>
#include <io/utilities/block_utils.cuh>

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {
// Minimal thrift implementation for parsing page headers

static const __device__ __constant__ uint8_t g_list2struct[16] = {0,
                                                                  1,
                                                                  2,
                                                                  ST_FLD_BYTE,
                                                                  ST_FLD_DOUBLE,
                                                                  5,
                                                                  ST_FLD_I16,
                                                                  7,
                                                                  ST_FLD_I32,
                                                                  9,
                                                                  ST_FLD_I64,
                                                                  ST_FLD_BINARY,
                                                                  ST_FLD_STRUCT,
                                                                  ST_FLD_MAP,
                                                                  ST_FLD_SET,
                                                                  ST_FLD_LIST};

struct byte_stream_s {
  const uint8_t *cur;
  const uint8_t *end;
  const uint8_t *base;
  // Parsed symbols
  PageType page_type;
  PageInfo page;
  ColumnChunkDesc ck;
};

inline __device__ unsigned int getb(byte_stream_s *bs)
{
  return (bs->cur < bs->end) ? *bs->cur++ : 0;
}

inline __device__ void skip_bytes(byte_stream_s *bs, size_t bytecnt)
{
  bytecnt = min(bytecnt, (size_t)(bs->end - bs->cur));
  bs->cur += bytecnt;
}

__device__ uint32_t get_u32(byte_stream_s *bs)
{
  uint32_t v = 0, l = 0, c;
  do {
    c = getb(bs);
    v |= (c & 0x7f) << l;
    l += 7;
  } while (c & 0x80);
  return v;
}

inline __device__ int32_t get_i32(byte_stream_s *bs)
{
  uint32_t u = get_u32(bs);
  return (int32_t)((u >> 1u) ^ -(int32_t)(u & 1));
}

__device__ void skip_struct_field(byte_stream_s *bs, int t)
{
  int struct_depth = 0;
  int rep_cnt      = 0;

  do {
    if (rep_cnt != 0) {
      rep_cnt--;
    } else if (struct_depth != 0) {
      int c;
      do {
        c = getb(bs);
        if (!c) --struct_depth;
      } while (!c && struct_depth);
      if (!struct_depth) break;
      t = c & 0xf;
      if (!(c & 0xf0)) get_i32(bs);
    }
    switch (t) {
      case ST_FLD_TRUE:
      case ST_FLD_FALSE: break;
      case ST_FLD_I16:
      case ST_FLD_I32:
      case ST_FLD_I64: get_u32(bs); break;
      case ST_FLD_BYTE: skip_bytes(bs, 1); break;
      case ST_FLD_DOUBLE: skip_bytes(bs, 8); break;
      case ST_FLD_BINARY: skip_bytes(bs, get_u32(bs)); break;
      case ST_FLD_LIST:
      case ST_FLD_SET: {  // NOTE: skipping a list of lists is not handled
        int c = getb(bs);
        int n = c >> 4;
        if (n == 0xf) n = get_u32(bs);
        t = g_list2struct[c & 0xf];
        if (t == ST_FLD_STRUCT)
          struct_depth += n;
        else
          rep_cnt = n;
      } break;
      case ST_FLD_STRUCT: struct_depth++; break;
    }
  } while (rep_cnt || struct_depth);
}

#define PARQUET_BEGIN_STRUCT(fn)         \
  __device__ bool fn(byte_stream_s *bs)  \
  {                                      \
    int fld = 0;                         \
    for (;;) {                           \
      int c, t, f;                       \
      c = getb(bs);                      \
      if (!c) break;                     \
      f   = c >> 4;                      \
      t   = c & 0xf;                     \
      fld = (f) ? fld + f : get_i32(bs); \
      switch (fld) {
#define PARQUET_FLD_ENUM(id, m, mt)    \
  case id:                             \
    bs->m = (mt)get_i32(bs);           \
    if (t != ST_FLD_I32) return false; \
    break;

#define PARQUET_FLD_INT32(id, m)       \
  case id:                             \
    bs->m = get_i32(bs);               \
    if (t != ST_FLD_I32) return false; \
    break;

#define PARQUET_FLD_STRUCT(id, m)                   \
  case id:                                          \
    if (t != ST_FLD_STRUCT || !m(bs)) return false; \
    break;

#define PARQUET_END_STRUCT()                \
  default: skip_struct_field(bs, t); break; \
    }                                       \
    }                                       \
    return true;                            \
    }

PARQUET_BEGIN_STRUCT(gpuParseDataPageHeader)
PARQUET_FLD_INT32(1, page.num_input_values)
PARQUET_FLD_ENUM(2, page.encoding, Encoding);
PARQUET_FLD_ENUM(3, page.definition_level_encoding, Encoding);
PARQUET_FLD_ENUM(4, page.repetition_level_encoding, Encoding);
PARQUET_END_STRUCT()

PARQUET_BEGIN_STRUCT(gpuParseDictionaryPageHeader)
PARQUET_FLD_INT32(1, page.num_input_values)
PARQUET_FLD_ENUM(2, page.encoding, Encoding);
PARQUET_END_STRUCT()

PARQUET_BEGIN_STRUCT(gpuParseDataPageHeaderV2)
PARQUET_FLD_INT32(1, page.num_input_values)
PARQUET_FLD_INT32(3, page.num_rows)
PARQUET_FLD_ENUM(4, page.encoding, Encoding);
PARQUET_FLD_ENUM(5, page.definition_level_encoding, Encoding);
PARQUET_FLD_ENUM(6, page.repetition_level_encoding, Encoding);
PARQUET_END_STRUCT()

PARQUET_BEGIN_STRUCT(gpuParsePageHeader)
PARQUET_FLD_ENUM(1, page_type, PageType)
PARQUET_FLD_INT32(2, page.uncompressed_page_size)
PARQUET_FLD_INT32(3, page.compressed_page_size)
PARQUET_FLD_STRUCT(5, gpuParseDataPageHeader)
PARQUET_FLD_STRUCT(7, gpuParseDictionaryPageHeader)
PARQUET_FLD_STRUCT(8, gpuParseDataPageHeaderV2)
PARQUET_END_STRUCT()

/**
 * @brief Kernel for outputting page headers from the specified column chunks
 *
 * @param[in] chunks List of column chunks
 * @param[in] num_chunks Number of column chunks
 */
// blockDim {128,1,1}
extern "C" __global__ void __launch_bounds__(128)
  gpuDecodePageHeaders(ColumnChunkDesc *chunks, int32_t num_chunks)
{
  __shared__ byte_stream_s bs_g[4];

  int t                   = threadIdx.x & 0x1f;
  int chunk               = (blockIdx.x << 2) + (threadIdx.x >> 5);
  byte_stream_s *const bs = &bs_g[threadIdx.x >> 5];

  if (chunk < num_chunks) {
    // NOTE: Assumes that sizeof(ColumnChunkDesc) <= 128
    if (t < sizeof(ColumnChunkDesc) / sizeof(uint32_t)) {
      reinterpret_cast<uint32_t *>(&bs->ck)[t] =
        reinterpret_cast<const uint32_t *>(&chunks[chunk])[t];
    }
  }
  __syncthreads();
  if (chunk < num_chunks) {
    size_t num_values, values_found;
    uint32_t data_page_count       = 0;
    uint32_t dictionary_page_count = 0;
    int32_t max_num_pages;
    int32_t num_dict_pages = bs->ck.num_dict_pages;
    PageInfo *page_info;

    if (!t) {
      bs->base = bs->cur      = bs->ck.compressed_data;
      bs->end                 = bs->base + bs->ck.compressed_size;
      bs->page.chunk_idx      = chunk;
      bs->page.src_col_schema = bs->ck.src_col_schema;
      // this computation is only valid for flat schemas. for nested schemas,
      // they will be recomputed in the preprocess step by examining repetition and
      // definition levels
      bs->page.chunk_row = 0;
      bs->page.num_rows  = 0;
    }
    num_values     = bs->ck.num_values;
    page_info      = bs->ck.page_info;
    num_dict_pages = bs->ck.num_dict_pages;
    max_num_pages  = (page_info) ? bs->ck.max_num_pages : 0;
    values_found   = 0;
    SYNCWARP();
    while (values_found < num_values && bs->cur < bs->end) {
      int index_out = -1;

      if (t == 0) {
        // this computation is only valid for flat schemas. for nested schemas,
        // they will be recomputed in the preprocess step by examining repetition and
        // definition levels
        bs->page.chunk_row += bs->page.num_rows;
        bs->page.num_rows = 0;
        if (gpuParsePageHeader(bs) && bs->page.compressed_page_size >= 0) {
          switch (bs->page_type) {
            case PageType::DATA_PAGE:
              // this computation is only valid for flat schemas. for nested schemas,
              // they will be recomputed in the preprocess step by examining repetition and
              // definition levels
              bs->page.num_rows = bs->page.num_input_values;
            case PageType::DATA_PAGE_V2:
              index_out = num_dict_pages + data_page_count;
              data_page_count++;
              bs->page.flags = 0;
              values_found += bs->page.num_input_values;
              break;
            case PageType::DICTIONARY_PAGE:
              index_out = dictionary_page_count;
              dictionary_page_count++;
              bs->page.flags = PAGEINFO_FLAGS_DICTIONARY;
              break;
            default: index_out = -1; break;
          }
          bs->page.page_data = const_cast<uint8_t *>(bs->cur);
          bs->cur += bs->page.compressed_page_size;
        } else {
          bs->cur = bs->end;
        }
      }
      index_out = SHFL0(index_out);
      if (index_out >= 0 && index_out < max_num_pages) {
        // NOTE: Assumes that sizeof(PageInfo) <= 128
        if (t < sizeof(PageInfo) / sizeof(uint32_t)) {
          reinterpret_cast<uint32_t *>(page_info + index_out)[t] =
            reinterpret_cast<const uint32_t *>(&bs->page)[t];
        }
      }
      num_values = SHFL0(num_values);
      SYNCWARP();
    }
    if (t == 0) {
      chunks[chunk].num_data_pages = data_page_count;
      chunks[chunk].num_dict_pages = dictionary_page_count;
    }
  }
}

/**
 * @brief Kernel for building dictionary index for the specified column chunks
 *
 * This function builds an index to point to each dictionary entry
 * (string format is 4-byte little-endian string length followed by character
 * data). The index is a 32-bit integer which contains the offset of each string
 * relative to the beginning of the dictionary page data.
 *
 * @param[in] chunks List of column chunks
 * @param[in] num_chunks Number of column chunks
 */
// blockDim {128,1,1}
extern "C" __global__ void __launch_bounds__(128)
  gpuBuildStringDictionaryIndex(ColumnChunkDesc *chunks, int32_t num_chunks)
{
  __shared__ ColumnChunkDesc chunk_g[4];

  int t                     = threadIdx.x & 0x1f;
  int chunk                 = (blockIdx.x << 2) + (threadIdx.x >> 5);
  ColumnChunkDesc *const ck = &chunk_g[threadIdx.x >> 5];
  if (chunk < num_chunks) {
    // NOTE: Assumes that sizeof(ColumnChunkDesc) <= 128
    if (t < sizeof(ColumnChunkDesc) / sizeof(uint32_t)) {
      reinterpret_cast<uint32_t *>(ck)[t] = reinterpret_cast<const uint32_t *>(&chunks[chunk])[t];
    }
  }
  __syncthreads();
  if (chunk >= num_chunks) { return; }
  if (!t && ck->num_dict_pages > 0 && ck->str_dict_index) {
    // Data type to describe a string
    nvstrdesc_s *dict_index = ck->str_dict_index;
    const uint8_t *dict     = ck->page_info[0].page_data;
    int dict_size           = ck->page_info[0].uncompressed_page_size;
    int num_entries         = ck->page_info[0].num_input_values;
    int pos = 0, cur = 0;
    for (int i = 0; i < num_entries; i++) {
      int len = 0;
      if (cur + 4 <= dict_size) {
        len = dict[cur + 0] | (dict[cur + 1] << 8) | (dict[cur + 2] << 16) | (dict[cur + 3] << 24);
        if (len >= 0 && cur + 4 + len <= dict_size) {
          pos = cur;
          cur = cur + 4 + len;
        } else {
          cur = dict_size;
        }
      }
      // TODO: Could store 8 entries in shared mem, then do a single warp-wide store
      dict_index[i].ptr   = reinterpret_cast<const char *>(dict + pos + 4);
      dict_index[i].count = len;
    }
  }
}

cudaError_t __host__ DecodePageHeaders(ColumnChunkDesc *chunks,
                                       int32_t num_chunks,
                                       cudaStream_t stream)
{
  dim3 dim_block(128, 1);
  dim3 dim_grid((num_chunks + 3) >> 2, 1);  // 1 chunk per warp, 4 warps per block
  gpuDecodePageHeaders<<<dim_grid, dim_block, 0, stream>>>(chunks, num_chunks);
  return cudaSuccess;
}

cudaError_t __host__ BuildStringDictionaryIndex(ColumnChunkDesc *chunks,
                                                int32_t num_chunks,
                                                cudaStream_t stream)
{
  dim3 dim_block(128, 1);
  dim3 dim_grid((num_chunks + 3) >> 2, 1);  // 1 chunk per warp, 4 warps per block
  gpuBuildStringDictionaryIndex<<<dim_grid, dim_block, 0, stream>>>(chunks, num_chunks);
  return cudaSuccess;
}

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf
