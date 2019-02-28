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

#include "parquet_gpu.h"

#if (__CUDACC_VER_MAJOR__ >= 9)
#define SHFL0(v)    __shfl_sync(~0, v, 0)
#define SHFL(v, t)  __shfl_sync(~0, v, t)
#define SYNCWARP()  __syncwarp()
#define BALLOT(v)   __ballot_sync(~0, v)
#else
#define SHFL0(v)    __shfl(v, 0)
#define SHFL(v, t)  __shfl(v, t)
#define SYNCWARP()
#define BALLOT(v)   __ballot(v)
#endif

#if (__CUDA_ARCH__ >= 700)
#define NANOSLEEP(d)  __nanosleep(d)
#else
#define NANOSLEEP(d)  clock()
#endif


namespace parquet { namespace gpu {

// NOTE: For maximum SM occupancy with 32 registers/thread, we need sizeof(page_state_s) to be 1024 bytes or below
#define LOG2_INDEX_QUEUE_LEN    7
#define INDEX_QUEUE_LEN         (1 << LOG2_INDEX_QUEUE_LEN)

struct index_queue_s
{
    int32_t finished;   // Set to 1 when there are no more values
    int32_t wr_count;   // WARP0 write count
    int32_t rd_count;   // WARP1 values read
    int32_t buf[INDEX_QUEUE_LEN];   // Buffered values
};


struct page_state_s {
    const uint8_t *lvl_start[2];  // [def,rep]
    const uint8_t *data_start;
    const uint8_t *data_end;
    uint32_t *valid_map;
    const uint8_t *dict_base;       // ptr to dictionary page data
    int32_t dict_size;              // size of dictionary data
    uint8_t *data_out;
    int32_t valid_map_offset;       // offset in valid_map, in bits
    int32_t first_row;
    int32_t num_rows;
    int32_t dtype_len;
    int32_t dict_bits;              // # of bits to store dictionary indices
    uint32_t dict_run;
    int32_t dict_val;
    uint32_t initial_rle_run[2];    // [def,rep]
    int32_t initial_rle_value[2];   // [def,rep]
    int32_t error;
    PageInfo page;
    ColumnChunkDesc col;
    volatile index_queue_s q;   // Index queue
    union {
        int32_t dict_idx[32];       // Dictionary indices for current batch (PLAIN_DICTIONARY, RLE_DICTIONARY)
        nvstrdesc_s str_desc[32];   // String descriptors for current batch (PLAIN encoding for BYTE_ARRAY type)
    } scratch;
};


inline __device__ uint32_t get_vlq32(const uint8_t *&cur, const uint8_t *end)
{
    uint32_t v = *cur++;
    if (v >= 0x80 && cur < end)
    {
        v = (v & 0x7f) | ((*cur++) << 7);
        if (v >= (0x80 << 7) && cur < end)
        {
            v = (v & ((0x7f << 7) | 0x7f)) | ((*cur++) << 14);
            if (v >= (0x80 << 14) && cur < end)
            {
                v = (v & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | ((*cur++) << 21);
                if (v >= (0x80 << 21) && cur < end)
                {
                    v = (v & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) | ((*cur++) << 28);
                }
            }
        }
    }
    return v;
}


// Parse the beginning of the level section (definition or repetition), initializes the initial RLE run & value, and return the section length
__device__ uint32_t InitLevelSection(page_state_s *s, const uint8_t *cur, const uint8_t *end, int encoding, int level_bits, int idx)
{
    int32_t len;
    if (level_bits == 0)
    {
        len = 0;
        s->initial_rle_run[idx] = s->page.num_values * 2; // repeated value
        s->initial_rle_value[idx] = 0;
        s->lvl_start[idx] = cur;
    }
    else if (encoding == RLE)
    {
        if (cur + 4 < end)
        {
            len = 4 + (cur[0]) | (cur[1] << 8) | (cur[2] << 16) | (cur[3] << 24);
            if (len > 0)
            {
                uint32_t run;
                cur += 4;
                run = get_vlq32(cur, end);
                s->initial_rle_run[idx] = run;
                if (!(run & 1))
                {
                    int v = (cur < end) ? cur[0] : 0;
                    if (level_bits > 8)
                    {
                        v |= ((cur < end) ? cur[0] : 0) << 8;
                        cur++;
                    }
                    s->initial_rle_value[idx] = v;
                }
                s->lvl_start[idx] = cur;
                if (cur > end)
                {
                    s->error = 2;
                }
            }
        }
        else
        {
            len = 0;
            s->error = 2;
        }
    }
    else if (encoding == BIT_PACKED)
    {
        len = (s->page.num_values * level_bits + 7) >> 3;
        s->initial_rle_run[idx] = ((s->page.num_values + 7) >> 3) * 2 + 1; // literal run
        s->initial_rle_value[idx] = 0;
        s->lvl_start[idx] = cur;
    }
    else
    {
        s->error = 3;
        len = 0;
    }
    if (cur + (uint32_t)len > end)
    {
        s->error = 4;
    }
    return (uint32_t)len;
}


// WARP0: Decode definition and repetition levels, outputs row indices
__device__ void gpuDecodeLevels(page_state_s *s, int t)
{
    const uint8_t *cur_def = s->lvl_start[0];
    const uint8_t *end = s->data_start;
    uint32_t *valid_map = s->valid_map;
    uint32_t valid_map_offset = s->valid_map_offset;
    uint32_t out_valid = 0, out_valid_mask = (~0) << valid_map_offset;
    int32_t first_row = s->first_row;
    uint32_t def_run = s->initial_rle_run[0];
    int32_t def_val = s->initial_rle_value[0];
    int def_bits = s->col.def_level_bits;
    int def_mask = (1 << def_bits) - 1;
    int max_def_level = s->col.max_def_level;
    int32_t num_values = min(s->page.num_values, first_row + s->num_rows);
    int32_t value_count = 0;    // Row offset of next value
    int32_t coded_count = 0;    // Count of non-null values
    while (value_count < num_values)
    {
        int batch_len, is_valid, valid_mask;
        if (def_run <= 1)
        {
            // Get a new run symbol from the byte stream
            int sym_len = 0;
            if (!t)
            {
                const uint8_t *cur = cur_def;
                if (cur < end)
                {
                    def_run = get_vlq32(cur, end);
                }
                if (!(def_run & 1))
                {
                    if (cur < end)
                        def_val = cur[0];
                    cur++;
                    if (def_bits > 8)
                    {
                        if (cur < end)
                            def_val |= cur[0] << 8;
                        cur++;
                    }
                }
                if (cur > end || def_run <= 1)
                {
                    s->error = 0x10;
                }
                sym_len = (int32_t)(cur - cur_def);
                __threadfence_block();
            }
            sym_len = SHFL0(sym_len);
            def_val = SHFL0(def_val);
            def_run = SHFL0(def_run);
            cur_def += sym_len;
        }
        if (s->error)
        {
            break;
        }
        batch_len = min(num_values - value_count, 32);
        if (def_run & 1)
        {
            // Literal run
            int batch_len8;
            batch_len = min(batch_len, (def_run>>1)*8);
            batch_len8 = (batch_len + 7) >> 3;
            if (t < batch_len)
            {
                int bitpos = t * def_bits;
                const uint8_t *cur = cur_def + (bitpos >> 3);
                bitpos &= 7;
                if (cur < end)
                    def_val = cur[0];
                cur++;
                if (def_bits > 8 - bitpos && cur < end)
                {
                    def_val |= cur[0] << 8;
                    cur++;
                    if (def_bits > 16 - bitpos && cur < end)
                        def_val |= cur[0] << 16;
                }
                def_val = (def_val >> bitpos) & def_mask;
            }
            def_run -= batch_len8 * 2;
            cur_def += batch_len8 * def_bits;
        }
        else
        {
            // Repeated value
            batch_len = min(batch_len, def_run >> 1);
            def_run -= batch_len * 2;
        }
        is_valid = (t < batch_len && def_val >= max_def_level);
        valid_mask = BALLOT(is_valid);
        if (valid_mask)
        {
            if (is_valid)
            {
                int idx = coded_count + __popc(valid_mask & ((1 << t) - 1));
                int ofs = value_count + t - first_row;
                s->q.buf[idx & (INDEX_QUEUE_LEN - 1)] = ofs;
            }
            coded_count += __popc(valid_mask);
            SYNCWARP();
            if (!t)
            {
                // Wait until we can safely output a full batch
                s->q.wr_count = coded_count;
                __threadfence_block();
                while (s->q.rd_count + INDEX_QUEUE_LEN - 32 < coded_count)
                {
                    NANOSLEEP(100);
                    if (s->error)
                        break;
                }
            }
            SYNCWARP();
        }
        value_count += batch_len;
        if (!t && valid_map)
        {
            // If needed, adjust batch length to eliminate rows before the first row
            if (value_count < first_row + batch_len)
            {
                if (value_count > first_row)
                {
                    // At least some values are above the first row
                    unsigned int skip_cnt = batch_len - (value_count - first_row);
                    valid_mask >>= skip_cnt;
                    batch_len -= skip_cnt;
                }
                else // All values are below the first row
                {
                    batch_len = 0;
                    valid_mask = 0;
                }
            }
            out_valid |= valid_mask << valid_map_offset;
            valid_map_offset += batch_len;
            if (valid_map_offset >= 32)
            {
                if (out_valid_mask == ~0) // Safe to output all 32 bits are within the current page
                {
                    *valid_map = out_valid;
                }
                else // Special case for the first valid row, which may not start on a 32-bit boundary (only setting some of the bits)
                {
                    atomicAnd(valid_map, ~out_valid_mask);
                    atomicOr(valid_map, out_valid);
                }
                s->page.valid_count += __popc(out_valid);
                valid_map_offset &= 0x1f;
                out_valid = (valid_map_offset > 0) ? valid_mask >> (unsigned int)(batch_len - valid_map_offset) : 0;
                out_valid_mask = ~0;
                valid_map++;
            }
            __threadfence_block();
        }
    }
    // Store the remaining valid bits
    if (!t && valid_map && valid_map_offset != 0)
    {
        out_valid_mask &= (1 << valid_map_offset) - 1;
        out_valid &= out_valid_mask;
        s->page.valid_count += __popc(out_valid);
        atomicAnd(valid_map, ~out_valid_mask);
        atomicOr(valid_map, out_valid);
    }
}


// Perform RLE decoding of dictionary indices
__device__ int gpuDecodeDictionaryIndices(volatile page_state_s *s, int batch_len, int t)
{
    const uint8_t *end = s->data_end;
    int dict_bits = s->dict_bits;
    int is_literal = 0, dict_idx;

    if (!t)
    {
        uint32_t run = s->dict_run;
        const uint8_t *cur = s->data_start;
        if (run <= 1)
        {
            run = (cur < end) ? get_vlq32(cur, end) : 0;
            if (!(run & 1))
            {
                // Repeated value
                int bytecnt = (dict_bits + 7) >> 3;
                if (cur + bytecnt <= end)
                {
                    int32_t run_val = cur[0];
                    if (bytecnt > 1)
                    {
                        run_val |= cur[1] << 8;
                        if (bytecnt > 2)
                        {
                            run_val |= cur[2] << 16;
                            if (bytecnt > 3)
                            {
                                run_val |= cur[3] << 24;
                            }
                        }
                    }
                    s->dict_val = run_val & ((1 << dict_bits) - 1);
                }
                cur += bytecnt;
            }
        }
        if (run & 1)
        {
            // Literal batch: must output a multiple of 8, except for the last batch
            int batch_len_div8;
            batch_len = max(min(batch_len, (int)(run >> 1) * 8), 1);
            if (batch_len >= 8)
            {
                batch_len &= ~7;
            }
            batch_len_div8 = (batch_len + 7) >> 3;
            run -= batch_len_div8 * 2;
            cur += batch_len_div8 * dict_bits;
        }
        else
        {
            batch_len = max(min(batch_len, (int)(run >> 1)), 1);
            run -= batch_len * 2;
        }
        s->dict_run = run;
        s->data_start = cur;
        is_literal = run & 1;
        __threadfence_block();
    }
    SYNCWARP();
    is_literal = SHFL0(is_literal);
    batch_len = SHFL0(batch_len);
    if (is_literal && t < batch_len)
    {
        int32_t ofs = (t - ((batch_len + 7) & ~7)) * dict_bits;
        const uint8_t *p = s->data_start + (ofs >> 3);
        ofs &= 7;
        if (p < end)
        {
            uint32_t c = 8 - ofs;
            dict_idx = (*p++) >> ofs;
            if (c < dict_bits && p < end)
            {
                dict_idx |= (*p++) << c;
                c += 8;
                if (c < dict_bits && p < end)
                {
                    dict_idx |= (*p++) << c;
                    c += 8;
                    if (c < dict_bits && p < end)
                    {
                        dict_idx |= (*p++) << c;
                    }
                }
            }
            dict_idx &= (1 << dict_bits) - 1;
        }
    }
    else
    {
        dict_idx = s->dict_val;
    }
    s->scratch.dict_idx[t] = dict_idx;
    return batch_len;
}


// Parse the length of strings in the batch and initializes the corresponding string descriptor
__device__ void gpuInitStringDescriptors(volatile page_state_s *s, int batch_len, int t)
{
    // This step is purely serial
    if (!t)
    {
        const uint8_t *cur = s->data_start;
        const uint8_t *end = s->data_end;

        for (int i = 0; i < batch_len; i++)
        {
            if (cur + 4 <= end)
            {
                uint32_t len = (cur[0]) | (cur[1] << 8) | (cur[2] << 16) | (cur[3] << 24);
                if (cur + 4 + len <= end)
                {
                    cur += 4;
                    s->scratch.str_desc[i].ptr = reinterpret_cast<const char *>(cur);
                    s->scratch.str_desc[i].count = len;
                    cur += len;
                }
            }
        }
        s->data_start = cur;
        __threadfence_block();
    }
    SYNCWARP();
}


enum CodingMode {
    PLAIN_FIXED_LENGTH,     // Plain, fixed length symbols
    PLAIN_VARIABLE_LENGTH,  // Plain string 32-bit length followed by data
    DICTIONARY_RLE          // RLE-coded dictionary indices
};


// WARP1: Decode values and store the output at the row position given by WARP0
template<CodingMode mode>
__device__ void gpuDecodeValues(volatile page_state_s *s, int t)
{
    int32_t rd_count = 0, batch_len = 0;
    for (;;)
    {
        const uint8_t *dict;
        unsigned int dict_size;
        int dict_idx, row_idx;
        if (!t)
        {
            // Wait for data from WARP0
            batch_len = min(s->q.wr_count - rd_count, 32);
            if (batch_len < 8) // We want at least 8 values in the batch (simplifies RLE dictionary path)
            {
                for (;;)
                {
                    int finished = s->q.finished;
                    batch_len = min(s->q.wr_count - rd_count, 32);
                    if (batch_len >= 8 || finished)
                        break;
                    NANOSLEEP(100);
                }
            }
        }
        batch_len = SHFL0(batch_len);
        if (batch_len <= 0)
            break;
        if (mode == DICTIONARY_RLE)
        {
            batch_len = gpuDecodeDictionaryIndices(s, batch_len, t); // May lower the value of batch_len
            dict_idx = s->scratch.dict_idx[t];
            dict = s->dict_base;
            dict_size = s->dict_size;
        }
        else if (mode == PLAIN_VARIABLE_LENGTH)
        {
            gpuInitStringDescriptors(s, batch_len, t);
            dict_idx = 0;
            dict = const_cast<const uint8_t *>(reinterpret_cast<volatile uint8_t *>(&s->scratch.str_desc[t]));
            dict_size = sizeof(s->scratch.str_desc[0]);
        }
        else // PLAIN_FIXED_LENGTH
        {
            dict_idx = rd_count + t;
            dict = s->data_start;
            dict_size = s->dict_size;
        }
        if (t < batch_len)
        {
            row_idx = s->q.buf[(rd_count + t) & (INDEX_QUEUE_LEN - 1)];
            if (row_idx >= 0 && row_idx < s->num_rows)
            {
                // Read and store the value
                unsigned int len = s->dtype_len;
                unsigned int dict_pos = dict_idx * len;
                uint8_t *dst8 = s->data_out;
                if (dst8)
                {
                    dst8 += len * row_idx;
                    if (len & 3)
                    {
                        // Generic slow path
                        for (unsigned int i = 0; i < len; i++)
                        {
                            dst8[i] = (dict_pos + i < dict_size) ? dict[dict_pos + i] : 0;
                        }
                    }
                    else
                    {
                        // Copy 4 bytes at a time
                        const uint8_t *src8 = dict;
                        unsigned int ofs = 3 & reinterpret_cast<size_t>(src8);
                        src8 -= ofs;    // align to 32-bit boundary
                        ofs <<= 3;      // bytes -> bits
                        for (unsigned int i = 0; i < len; i += 4)
                        {
                            uint32_t bytebuf = (dict_pos < dict_size) ? *(const uint32_t *)(src8 + dict_pos) : 0;
                            if (ofs)
                            {
                                uint32_t bytebufnext = (dict_pos + 4 < dict_size) ? *(const uint32_t *)(src8 + dict_pos + 4) : 0;
                                bytebuf = __funnelshift_r(bytebuf, bytebufnext, ofs);
                            }
                            dict_pos += 4;
                            *(uint32_t *)(dst8 + i) = bytebuf;
                        }
                    }
                }
            }
        }
        SYNCWARP();
        rd_count += batch_len;
        if (t == 0)
        {
            s->q.rd_count = rd_count;
            __threadfence_block();
        }
    }
}



// blockDim {64,2,1}
extern "C" __global__ void __launch_bounds__(128)
gpuDecodePageData(PageInfo *pages, ColumnChunkDesc *chunks, int32_t num_pages, int32_t num_chunks, size_t min_row, size_t num_rows)
{
    __shared__ __align__(16) page_state_s state_g[2];

    page_state_s * const s = &state_g[threadIdx.y];
    int page_idx = blockIdx.x * 2 + threadIdx.y;
    int t = threadIdx.x;
    
    // Fetch page info
    if (page_idx < num_pages)
    {
        // NOTE: Assumes that sizeof(PageInfo) <= 256
        if (t < sizeof(PageInfo) / sizeof(uint32_t))
        {
            ((uint32_t *)&s->page)[t] = ((const uint32_t *)&pages[page_idx])[t];
        }
    }
    __syncthreads();
    // Fetch column chunk info
    if (page_idx < num_pages)
    {
        int chunk_idx = s->page.chunk_idx;
        if ((uint32_t)chunk_idx < (uint32_t)num_chunks)
        {
            // NOTE: Assumes that sizeof(ColumnChunkDesc) <= 256
            if (t < sizeof(ColumnChunkDesc) / sizeof(uint32_t))
            {
                ((uint32_t *)&s->col)[t] = ((const uint32_t *)&chunks[chunk_idx])[t];
            }
        }
    }
    __syncthreads();
    if (!t)
    {
        s->num_rows = 0;
        s->page.valid_count = 0;
        s->error = 0;
        if (page_idx < num_pages && !(s->page.flags & PAGEINFO_FLAGS_DICTIONARY) && s->page.num_values > 0 && s->page.num_rows > 0)
        {
            uint8_t *cur = s->page.compressed_page_data;
            uint8_t *end = cur + s->page.uncompressed_page_size;
            size_t page_start_row = s->col.start_row + s->page.chunk_row;
            // Validate data type
            switch(s->col.data_type & 7)
            {
            case BOOLEAN:
                s->dtype_len = 1;  // TBD: Output 1 byte per boolean or bitmap ?
                break;
            case INT32:
            case FLOAT:
                s->dtype_len = 4;
                break;
            case INT64:
            case DOUBLE:
                s->dtype_len = 8;
                break;
            case INT96:
                s->dtype_len = 12;
                break;
            case BYTE_ARRAY:
                s->dtype_len = sizeof(nvstrdesc_s);
                break;
            default: // FIXED_LEN_BYTE_ARRAY:
                s->dtype_len = s->col.data_type >> 3;
                s->error |= (s->dtype_len <= 0);
                break;
            }
            // Setup local valid map and compute first & num rows relative to the current page
            s->data_out = reinterpret_cast<uint8_t *>(s->col.column_data_base);
            s->valid_map = s->col.valid_map_base;
            s->valid_map_offset = 0;
            s->first_row = 0;
            if (page_start_row >= min_row)
            {
                if (s->data_out)
                {
                    s->data_out += (page_start_row - min_row) * s->dtype_len;
                }
                if (s->valid_map)
                {
                    s->valid_map += (page_start_row - min_row) >> 5;
                    s->valid_map_offset = (int32_t)((page_start_row - min_row) & 0x1f);
                }
                s->num_rows = s->page.num_rows;
            }
            else // First row starts after the beginning of the page
            {
                s->first_row = (int32_t)min(min_row - page_start_row, (size_t)s->page.num_rows);
                s->num_rows = s->page.num_rows - s->first_row;
            }
            if (page_start_row + s->num_rows > min_row + num_rows)
            {
                s->num_rows = (int32_t)max((int64_t)(min_row + num_rows - page_start_row), INT64_C(0));
            }
            // Find the compressed size of definition levels
            cur += InitLevelSection(s, cur, end, s->page.definition_level_encoding, s->col.def_level_bits, 0);
            // Find the compressed size of repetition levels
            cur += InitLevelSection(s, cur, end, s->page.repetition_level_encoding, s->col.rep_level_bits, 1);
            s->dict_bits = 0;
            s->dict_base = 0;
            s->dict_size = 0;
            switch (s->page.encoding)
            {
            case PLAIN_DICTIONARY:
            case RLE_DICTIONARY:
                // RLE-packed dictionary indices, first byte indicates index length in bits
                if (((s->col.data_type & 7) == BYTE_ARRAY) && (s->col.str_dict_index))
                {
                    // String dictionary: use index
                    s->dict_base = reinterpret_cast<const uint8_t *>(s->col.str_dict_index);
                    s->dict_size = s->col.page_info[0].num_values * sizeof(nvstrdesc_s);
                }
                else
                {
                    s->dict_base = s->col.page_info[0].compressed_page_data; // dictionary is always stored in the first page
                    s->dict_size = s->col.page_info[0].uncompressed_page_size;
                }
                s->dict_run = 0;
                s->dict_val = 0;
                s->dict_bits = (cur < end) ? *cur++ : 0;
                if (s->dict_bits <= 0 || s->dict_bits > 32 || !s->dict_base)
                {
                    s->error = (10 << 8) | s->dict_bits;
                }
                break;
            case PLAIN:
                s->dict_size = static_cast<int32_t>(end - cur);
                break;
            case RLE:
                break;
            default:
                s->error = 1;   // Unsupported encoding
                break;
            }
            if (cur > end)
            {
                s->error = 1;
            }
            s->data_start = cur;
            s->data_end = end;
            s->q.finished = 0;
            s->q.wr_count = 0;
            s->q.rd_count = 0;
        }
        else if (!(s->page.flags & PAGEINFO_FLAGS_DICTIONARY))
        {
            s->error = 1;
        }
        __threadfence_block();
    }
    __syncthreads();
    if (page_idx < num_pages && !(s->page.flags & PAGEINFO_FLAGS_DICTIONARY))
    {
        if (!s->error)
        {
            if (t < 32)
            {
                // WARP0: Decode definition and repetition levels, outputs row indices
                gpuDecodeLevels(s, t);
                if (!t)
                {
                    s->q.finished = 1;
                }
            }
            else
            {
                // WARP1: Decode values
                if (s->dict_base)
                    gpuDecodeValues<DICTIONARY_RLE>(s, t & 0x1f);
                else if ((s->col.data_type & 7) == BYTE_ARRAY)
                    gpuDecodeValues<PLAIN_VARIABLE_LENGTH>(s, t & 0x1f);
                else
                    gpuDecodeValues<PLAIN_FIXED_LENGTH>(s, t & 0x1f);
            }
            __threadfence_block();
        }
        SYNCWARP();
        if (!t)
        {
            // Update the number of rows (after cropping to [min_row, min_row+num_rows-1]), and number of valid values
            pages[page_idx].num_rows = s->num_rows;
            pages[page_idx].valid_count = (s->error) ? -s->error : s->page.valid_count;
        }
    }
}


cudaError_t __host__ DecodePageData(PageInfo *pages, int32_t num_pages, ColumnChunkDesc *chunks, int32_t num_chunks, size_t num_rows, size_t min_row, cudaStream_t stream)
{
    dim3 dim_block(64, 2);
    dim3 dim_grid((num_pages + 1) >> 1, 1); // 2 warps per page, 4 warps per block
    gpuDecodePageData << < dim_grid, dim_block, 0, stream >> >(pages, chunks, num_pages, num_chunks, min_row, num_rows);
    return cudaSuccess;
}



};}; // parquet::gpu namespace
