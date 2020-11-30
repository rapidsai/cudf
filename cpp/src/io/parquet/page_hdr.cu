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

#include <thrust/tuple.h>
#include <io/parquet/parquet_gpu.hpp>
#include <io/utilities/block_utils.cuh>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {
// Minimal thrift implementation for parsing page headers
// https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md

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

/**
 * @brief Get current byte from the byte stream
 *
 * @param[in] bs Byte stream
 *
 * @return Current byte pointed to by the byte stream
 */
inline __device__ unsigned int getb(byte_stream_s *bs)
{
  return (bs->cur < bs->end) ? *bs->cur++ : 0;
}

inline __device__ void skip_bytes(byte_stream_s *bs, size_t bytecnt)
{
  bytecnt = min(bytecnt, (size_t)(bs->end - bs->cur));
  bs->cur += bytecnt;
}

/**
 * @brief Decode unsigned integer from a byte stream using VarInt encoding
 *
 * Concatenate least significant 7 bits of each byte to form a 32 bit
 * integer. Most significant bit of each byte indicates if more bytes
 * are to be used to form the number.
 *
 * @param[in] bs Byte stream
 *
 * @return Decoded 32 bit integer
 */
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

/**
 * @brief Decode signed integer from a byte stream using zigzag encoding
 *
 * The number n encountered in a byte stream translates to
 * -1^(n%2) * ceil(n/2), with the exception of 0 which remains the same.
 * i.e. 0, 1, 2, 3, 4, 5 etc convert to 0, -1, 1, -2, 2 respectively.
 *
 * @param[in] bs Byte stream
 *
 * @return Decoded 32 bit integer
 */
inline __device__ int32_t get_i32(byte_stream_s *bs)
{
  uint32_t u = get_u32(bs);
  return (int32_t)((u >> 1u) ^ -(int32_t)(u & 1));
}

__device__ void skip_struct_field(byte_stream_s *bs, int field_type)
{
  int struct_depth = 0;
  int rep_cnt      = 0;

  do {
    if (rep_cnt != 0) {
      rep_cnt--;
    } else if (struct_depth != 0) {
      unsigned int c;
      do {
        c = getb(bs);
        if (!c) --struct_depth;
      } while (!c && struct_depth);
      if (!struct_depth) break;
      field_type = c & 0xf;
      if (!(c & 0xf0)) get_i32(bs);
    }
    switch (field_type) {
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
        auto const c = getb(bs);
        int n        = c >> 4;
        if (n == 0xf) n = get_u32(bs);
        field_type = g_list2struct[c & 0xf];
        if (field_type == ST_FLD_STRUCT)
          struct_depth += n;
        else
          rep_cnt = n;
      } break;
      case ST_FLD_STRUCT: struct_depth++; break;
    }
  } while (rep_cnt || struct_depth);
}

/**
 * @brief Functor to set value to 32 bit integer read from byte stream
 *
 * @return True if field type is not int32
 */
struct ParquetFieldInt32 {
  int field;
  int32_t &val;

  __device__ ParquetFieldInt32(int f, int32_t &v) : field(f), val(v) {}

  inline __device__ bool operator()(byte_stream_s *bs, int field_type)
  {
    val = get_i32(bs);
    return (field_type != ST_FLD_I32);
  }
};

/**
 * @brief Functor to set value to enum read from byte stream
 *
 * @return True if field type is not int32
 */
template <typename Enum>
struct ParquetFieldEnum {
  int field;
  Enum &val;

  __device__ ParquetFieldEnum(int f, Enum &v) : field(f), val(v) {}

  inline __device__ bool operator()(byte_stream_s *bs, int field_type)
  {
    val = static_cast<Enum>(get_i32(bs));
    return (field_type != ST_FLD_I32);
  }
};

/**
 * @brief Functor to run operator on byte stream
 *
 * @return True if field type is not struct type or if the calling operator
 * fails
 */
template <typename Operator>
struct ParquetFieldStruct {
  int field;
  Operator op;

  __device__ ParquetFieldStruct(int f) : field(f) {}

  inline __device__ bool operator()(byte_stream_s *bs, int field_type)
  {
    return ((field_type != ST_FLD_STRUCT) || !op(bs));
  }
};

/**
 * @brief Functor to run an operator
 *
 * The purpose of this functor is to replace a switch case. If the field in
 * the argument is equal to the field specified in any element of the tuple
 * of operators then it is run with the byte stream and field type arguments.
 *
 * If the field does not match any of the functors then skip_struct_field is
 * called over the byte stream.
 *
 * @return Return value of the selected operator or false if no operator
 * matched the field value
 */
template <int index>
struct FunctionSwitchImpl {
  template <typename... Operator>
  static inline __device__ bool run(byte_stream_s *bs,
                                    int field_type,
                                    const int &field,
                                    thrust::tuple<Operator...> &ops)
  {
    if (field == thrust::get<index>(ops).field) {
      return thrust::get<index>(ops)(bs, field_type);
    } else {
      return FunctionSwitchImpl<index - 1>::run(bs, field_type, field, ops);
    }
  }
};

template <>
struct FunctionSwitchImpl<0> {
  template <typename... Operator>
  static inline __device__ bool run(byte_stream_s *bs,
                                    int field_type,
                                    const int &field,
                                    thrust::tuple<Operator...> &ops)
  {
    if (field == thrust::get<0>(ops).field) {
      return thrust::get<0>(ops)(bs, field_type);
    } else {
      skip_struct_field(bs, field_type);
      return false;
    }
  }
};

/**
 * @brief Function to parse page header based on the tuple of functors provided
 *
 * Bytes are read from the byte stream and the field delta and field type are
 * matched up against user supplied reading functors. If they match then the
 * corresponding values are written to references pointed to by the functors.
 *
 * @return Returns false if an unexpected field is encountered while reading
 * byte stream. Otherwise true is returned.
 */
template <typename... Operator>
inline __device__ bool parse_header(thrust::tuple<Operator...> &op, byte_stream_s *bs)
{
  constexpr int index = thrust::tuple_size<thrust::tuple<Operator...>>::value - 1;
  int field           = 0;
  while (true) {
    auto const current_byte = getb(bs);
    if (!current_byte) break;
    int const field_delta = current_byte >> 4;
    int const field_type  = current_byte & 0xf;
    field                 = field_delta ? field + field_delta : get_i32(bs);
    bool exit_function    = FunctionSwitchImpl<index>::run(bs, field_type, field, op);
    if (exit_function) { return false; }
  }
  return true;
}

struct gpuParseDataPageHeader {
  __device__ bool operator()(byte_stream_s *bs)
  {
    auto op = thrust::make_tuple(ParquetFieldInt32(1, bs->page.num_input_values),
                                 ParquetFieldEnum<Encoding>(2, bs->page.encoding),
                                 ParquetFieldEnum<Encoding>(3, bs->page.definition_level_encoding),
                                 ParquetFieldEnum<Encoding>(4, bs->page.repetition_level_encoding));
    return parse_header(op, bs);
  }
};

struct gpuParseDictionaryPageHeader {
  __device__ bool operator()(byte_stream_s *bs)
  {
    auto op = thrust::make_tuple(ParquetFieldInt32(1, bs->page.num_input_values),
                                 ParquetFieldEnum<Encoding>(2, bs->page.encoding));
    return parse_header(op, bs);
  }
};

struct gpuParseDataPageHeaderV2 {
  __device__ bool operator()(byte_stream_s *bs)
  {
    auto op = thrust::make_tuple(ParquetFieldInt32(1, bs->page.num_input_values),
                                 ParquetFieldInt32(3, bs->page.num_rows),
                                 ParquetFieldEnum<Encoding>(4, bs->page.encoding),
                                 ParquetFieldEnum<Encoding>(5, bs->page.definition_level_encoding),
                                 ParquetFieldEnum<Encoding>(6, bs->page.repetition_level_encoding));
    return parse_header(op, bs);
  }
};

struct gpuParsePageHeader {
  __device__ bool operator()(byte_stream_s *bs)
  {
    auto op = thrust::make_tuple(ParquetFieldEnum<PageType>(1, bs->page_type),
                                 ParquetFieldInt32(2, bs->page.uncompressed_page_size),
                                 ParquetFieldInt32(3, bs->page.compressed_page_size),
                                 ParquetFieldStruct<gpuParseDataPageHeader>(5),
                                 ParquetFieldStruct<gpuParseDictionaryPageHeader>(7),
                                 ParquetFieldStruct<gpuParseDataPageHeaderV2>(8));
    return parse_header(op, bs);
  }
};

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
  gpuParsePageHeader parse_page_header;
  __shared__ byte_stream_s bs_g[4];

  int lane_id             = threadIdx.x % 32;
  int chunk               = (blockIdx.x * 4) + (threadIdx.x / 32);
  byte_stream_s *const bs = &bs_g[threadIdx.x / 32];

  if (chunk < num_chunks and lane_id == 0) bs->ck = chunks[chunk];
  __syncthreads();

  if (chunk < num_chunks) {
    size_t num_values, values_found;
    uint32_t data_page_count       = 0;
    uint32_t dictionary_page_count = 0;
    int32_t max_num_pages;
    int32_t num_dict_pages = bs->ck.num_dict_pages;
    PageInfo *page_info;

    if (!lane_id) {
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
    __syncwarp();
    while (values_found < num_values && bs->cur < bs->end) {
      int index_out = -1;

      if (lane_id == 0) {
        // this computation is only valid for flat schemas. for nested schemas,
        // they will be recomputed in the preprocess step by examining repetition and
        // definition levels
        bs->page.chunk_row += bs->page.num_rows;
        bs->page.num_rows = 0;
        if (parse_page_header(bs) && bs->page.compressed_page_size >= 0) {
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
      index_out = shuffle(index_out);
      if (index_out >= 0 && index_out < max_num_pages && lane_id == 0)
        page_info[index_out] = bs->page;
      num_values = shuffle(num_values);
      __syncwarp();
    }
    if (lane_id == 0) {
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

  int lane_id               = threadIdx.x % 32;
  int chunk                 = (blockIdx.x * 4) + (threadIdx.x / 32);
  ColumnChunkDesc *const ck = &chunk_g[threadIdx.x / 32];
  if (chunk < num_chunks and lane_id == 0) *ck = chunks[chunk];
  __syncthreads();

  if (chunk >= num_chunks) { return; }
  if (!lane_id && ck->num_dict_pages > 0 && ck->str_dict_index) {
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

void __host__ DecodePageHeaders(ColumnChunkDesc *chunks,
                                int32_t num_chunks,
                                rmm::cuda_stream_view stream)
{
  dim3 dim_block(128, 1);
  dim3 dim_grid((num_chunks + 3) >> 2, 1);  // 1 chunk per warp, 4 warps per block
  gpuDecodePageHeaders<<<dim_grid, dim_block, 0, stream.value()>>>(chunks, num_chunks);
}

void __host__ BuildStringDictionaryIndex(ColumnChunkDesc *chunks,
                                         int32_t num_chunks,
                                         rmm::cuda_stream_view stream)
{
  dim3 dim_block(128, 1);
  dim3 dim_grid((num_chunks + 3) >> 2, 1);  // 1 chunk per warp, 4 warps per block
  gpuBuildStringDictionaryIndex<<<dim_grid, dim_block, 0, stream.value()>>>(chunks, num_chunks);
}

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf
