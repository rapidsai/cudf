/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <io/parquet/decode.cuh>

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {
  
// # of threads we're decoding with
constexpr int decode_block_size = 512;

// the required number of runs in shared memory we will need to provide the 
// rle_stream object
constexpr int rle_run_buffer_size = rle_stream_required_run_buffer_size(decode_block_size);

// the size of the rolling batch buffer
constexpr int rolling_buf_size = LEVEL_DECODE_BUF_SIZE;

namespace {

template<typename state_buf>
__device__ void gpuDecodeValues(
  page_state_s* s, state_buf* const sb, int start, int end, int t)
{
  constexpr int num_warps      = decode_block_size / 32;
  constexpr int max_batch_size = num_warps * 32;

  PageNestingDecodeInfo* nesting_info_base = s->nesting_info;
  uint32_t const skipped_leaf_values       = s->page.skipped_leaf_values;
  bool const has_repetition                = s->col.max_level[level_type::REPETITION] > 0;
  int const dtype                          = s->col.data_type & 7;

  // decode values
  int pos = start;
  while (pos < end) {
    int const batch_size = min(max_batch_size, end - pos);

    int const target_pos = pos + batch_size;
    int const src_pos    = pos + t;

    // the position in the output column/buffer
    int dst_pos = sb->nz_idx[rolling_index<state_buf::nz_buf_size>(src_pos)];

    // for the flat hierarchy case we will be reading from the beginning of the value stream,
    // regardless of the value of first_row. so adjust our destination offset accordingly.
    // example:
    // - user has passed skip_rows = 2, so our first_row to output is 2
    // - the row values we get from nz_idx will be
    //   0, 1, 2, 3, 4 ....
    // - by shifting these values by first_row, the sequence becomes
    //   -1, -2, 0, 1, 2 ...
    // - so we will end up ignoring the first two input rows, and input rows 2..n will
    //   get written to the output starting at position 0.
    //
    if (!has_repetition) { dst_pos -= s->first_row; }

    // target_pos will always be properly bounded by num_rows, but dst_pos may be negative (values
    // before first_row) in the flat hierarchy case.
    if (src_pos < target_pos && dst_pos >= 0) {
      // src_pos represents the logical row position we want to read from. But in the case of
      // nested hierarchies, there is no 1:1 mapping of rows to values.  So our true read position
      // has to take into account the # of values we have to skip in the page to get to the
      // desired logical row.  For flat hierarchies, skipped_leaf_values will always be 0.

      uint32_t val_src_pos = src_pos + skipped_leaf_values;
      // nesting level that is storing actual leaf values
      int leaf_level_index = s->col.max_nesting_depth - 1;

      uint32_t dtype_len = s->dtype_len;
      void* dst =
        nesting_info_base[leaf_level_index].data_out + static_cast<size_t>(dst_pos) * dtype_len;
      if (dtype == BYTE_ARRAY) {
        if (s->col.converted_type == DECIMAL) {
          auto const [ptr, len]        = gpuGetStringData(s, sb, val_src_pos);
          auto const decimal_precision = s->col.decimal_precision;
          if (decimal_precision <= MAX_DECIMAL32_PRECISION) {
            gpuOutputByteArrayAsInt(ptr, len, static_cast<int32_t*>(dst));
          } else if (decimal_precision <= MAX_DECIMAL64_PRECISION) {
            gpuOutputByteArrayAsInt(ptr, len, static_cast<int64_t*>(dst));
          } else {
            gpuOutputByteArrayAsInt(ptr, len, static_cast<__int128_t*>(dst));
          }
        } else {
          gpuOutputString(s, sb, val_src_pos, dst);
        }
      } else if (dtype == BOOLEAN) {
        // for now we only get in here for PLAIN encoding. RLE encoded booleans run through
        // the old path.
        gpuOutputBooleanFast(s->data_start, val_src_pos, static_cast<uint8_t*>(dst));
      } else if (s->col.converted_type == DECIMAL) {
        switch (dtype) {
          case INT32: gpuOutputFast(s, sb, val_src_pos, static_cast<uint32_t*>(dst)); break;
          case INT64: gpuOutputFast(s, sb, val_src_pos, static_cast<uint2*>(dst)); break;
          default:
            if (s->dtype_len_in <= sizeof(int32_t)) {
              gpuOutputFixedLenByteArrayAsInt(s, sb, val_src_pos, static_cast<int32_t*>(dst));
            } else if (s->dtype_len_in <= sizeof(int64_t)) {
              gpuOutputFixedLenByteArrayAsInt(s, sb, val_src_pos, static_cast<int64_t*>(dst));
            } else {
              gpuOutputFixedLenByteArrayAsInt(s, sb, val_src_pos, static_cast<__int128_t*>(dst));
            }
            break;
        }
      } else if (dtype == INT96) {
        gpuOutputInt96Timestamp(s, sb, val_src_pos, static_cast<int64_t*>(dst));
      } else if (dtype_len == 8) {
        if (s->dtype_len_in == 4) {
          // Reading INT32 TIME_MILLIS into 64-bit DURATION_MILLISECONDS
          // TIME_MILLIS is the only duration type stored as int32:
          // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#deprecated-time-convertedtype
          gpuOutputFast(s, sb, val_src_pos, static_cast<uint32_t*>(dst));
        } else if (s->ts_scale) {
          gpuOutputInt64Timestamp(s, sb, val_src_pos, static_cast<int64_t*>(dst));
        } else {
          gpuOutputFast(s, sb, val_src_pos, static_cast<uint2*>(dst));
        }
      } else if (dtype_len == 4) {
        gpuOutputFast(s, sb, val_src_pos, static_cast<uint32_t*>(dst));
      } else {
        gpuOutputGeneric(s, sb, val_src_pos, static_cast<uint8_t*>(dst), dtype_len);
      }
    }

    pos += batch_size;
  }
}

template <bool nullable, typename level_t, typename state_buf>
static __device__ int gpuUpdateValidityOffsetsAndRowIndicesFlat(int32_t target_value_count,
                                                                 page_state_s* s,
                                                                 state_buf* sb,
                                                                 level_t const* const def,
                                                                 int t,
                                                                 int page_idx)
{
  constexpr int num_warps      = decode_block_size / 32;
  constexpr int max_batch_size = num_warps * 32;

  // how many (input) values we've processed in the page so far
  int value_count = s->input_value_count;
  int valid_count = s->nesting_info[0].valid_count;

  int const valid_map_offset = s->nesting_info[0].valid_map_offset / 32;
  int const first_row = s->first_row;
  int const last_row = first_row + s->num_rows;
  int const row_index_lower_bound = s->row_index_lower_bound;  
  int const max_def_level = s->nesting_info[0].max_def_level;
  while (value_count < target_value_count) {
    int const batch_size = min(max_batch_size, target_value_count - value_count);
    
    // definition level. only need to process for nullable columns
    int d;
    if constexpr(nullable){ 
      d = t < batch_size ? static_cast<int>(def[rolling_index<state_buf::nz_buf_size>(value_count + t)]) : -1;
    } 

    int const thread_value_count = t + 1;
    int const block_value_count = batch_size;

    // compute our row index, whether we're in row bounds, and validity
    int const row_index = (thread_value_count + value_count) - 1;
    int const in_row_bounds = (row_index >= row_index_lower_bound) && (row_index < last_row);
    int is_valid;
    if constexpr(nullable){
      is_valid = (d >= max_def_level) && in_row_bounds ? 1 : 0;
    } else {
      is_valid = in_row_bounds;
    }
    
    // thread and block validity count
    int thread_valid_count, block_valid_count;    
    if constexpr(nullable){
      using block_scan   = cub::BlockScan<int, decode_block_size>;
      __shared__ typename block_scan::TempStorage scan_storage;
       block_scan(scan_storage).InclusiveSum(is_valid, thread_valid_count, block_valid_count);
      __syncthreads();

      // write out validity by warp
      uint32_t const warp_validity_mask = ballot(is_valid);
      if(t < batch_size && !(t%32)){
        int const vindex = (value_count + thread_value_count) - 1;
        int const bit_offset = valid_map_offset + vindex;
        int const bit_count = min(32, target_value_count - vindex);
        store_validity(bit_offset,
                       s->nesting_info[0].valid_map,
                       warp_validity_mask,
                       bit_count);
      }
    } 
    // trivial for non-nullable columns
    else {
      thread_valid_count = thread_value_count;
      block_valid_count = block_value_count;
    }
    
    // output offset
    if (is_valid){
      int const dst_pos = (value_count + thread_value_count) - 1;
      int const src_pos = (valid_count + thread_valid_count) - 1;
      sb->nz_idx[rolling_index<state_buf::nz_buf_size>(src_pos)] = dst_pos;
    }

    // update stuff
    value_count += block_value_count;
    valid_count += block_valid_count;    
    __syncthreads();
  }
  
  if (!t) {
    // update valid value count for decoding and total # of values we've processed    
    s->nesting_info[0].valid_count = valid_count;
    s->nesting_info[0].value_count = value_count;
    s->nesting_info[0].null_count = value_count - valid_count;
    s->nz_count          = valid_count;
    s->input_value_count = value_count;
    s->input_row_count   = value_count;
  }

  return valid_count;
}


/**
 * @brief Kernel for co the column data stored in the pages
 *
 * This function will write the page data and the page data's validity to the
 * output specified in the page's column chunk. If necessary, additional
 * conversion will be performed to translate from the Parquet datatype to
 * desired output datatype (ex. 32-bit to 16-bit, string to hash).
 *
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read
 */
template <typename level_t>
__global__ void __launch_bounds__(decode_block_size) gpuDecodePageDataFixed(
  PageInfo* pages, device_span<ColumnChunkDesc const> chunks, size_t min_row, size_t num_rows)
{
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16) page_state_buffers_s<rolling_buf_size,   // size of nz_idx buffer
                                                1,                  // unused in this kernel
                                                1>                  // unused in this kernel
                                                state_buffers;

  page_state_s* const s          = &state_g;
  auto* const sb                 = &state_buffers;
  int page_idx                   = blockIdx.x;
  int t                          = threadIdx.x;
  PageInfo* pp                   = &pages[page_idx];

  if(!(pages[page_idx].kernel_mask & KERNEL_MASK_FIXED_WIDTH_NO_DICT)){ return; }
  if (!setupLocalPageInfo(s, &pages[page_idx], chunks, min_row, num_rows, true)) {
    return;
  }

  // must come after the kernel mask check
  [[maybe_unused]] null_count_back_copier _{s, t};

  // the level stream decoders
  __shared__ rle_run<level_t> def_runs[rle_run_buffer_size];
  rle_stream<level_t, decode_block_size> def_decoder{def_runs};

  bool const has_repetition = false;  
  bool const nullable = s->col.max_level[level_type::DEFINITION] > 0;

  // if we have no work to do (eg, in a skip_rows/num_rows case) in this page.
  //
  // corner case: in the case of lists, we can have pages that contain "0" rows if the current row
  // starts before this page and ends after this page:
  //       P0        P1        P2
  //  |---------|---------|----------|
  //        ^------------------^
  //      row start           row end
  // P1 will contain 0 rows
  //
  if (s->num_rows == 0 && !(has_repetition && (is_bounds_page(s, min_row, num_rows) ||
                                               is_page_contained(s, min_row, num_rows)))) {
    return;
  }

  // initialize the stream decoders (requires values computed in setupLocalPageInfo)
  int const max_batch_size = rolling_buf_size;
  level_t* def             = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  if(nullable){
    def_decoder.init(s->col.level_bits[level_type::DEFINITION],
                     s->abs_lvl_start[level_type::DEFINITION],
                     s->abs_lvl_end[level_type::DEFINITION],
                     max_batch_size,
                     def,
                     s->page.num_input_values);
  }
  __syncthreads();

  // the core loop. decode batches of level stream data using rle_stream objects
  // and pass the results to gpuUpdatePageSizes
  int processed = 0;
  int valid = 0;  
  while (processed < s->page.num_input_values) {
    int next_valid;

    // only need to process definition levels if this is a nullable column
    int this_processed;    
    if(nullable){
      this_processed = def_decoder.decode_next(t);    
      __syncthreads();

      next_valid = gpuUpdateValidityOffsetsAndRowIndicesFlat<true, level_t>(processed + this_processed,
                                                                            s,
                                                                            sb,
                                                                            def,
                                                                            t,
                                                                            page_idx);
    } 
    // if we wanted to split off the skip_rows/num_rows case into a seperate kernel, we could skip this function call entirely
    // since all it will ever generate is a mapping of (i -> i) for nz_idx.  gpuDecodeValues would be the only work that happens.
    else {
      this_processed = min(max_batch_size, s->page.num_input_values - processed);
      next_valid = gpuUpdateValidityOffsetsAndRowIndicesFlat<false, level_t>(processed + this_processed,
                                                                             s,
                                                                             sb,
                                                                             nullptr,
                                                                             t,
                                                                             page_idx);
    }
    __syncthreads();
        
    // decode the values themselves
    gpuDecodeValues(s, sb, valid, next_valid, t);
    __syncthreads();

    processed += this_processed;
    valid = next_valid;
  }
}


} // anonymous namespace

void __host__ DecodePageDataFixed(cudf::detail::hostdevice_vector<PageInfo>& pages,
                                  cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                                  size_t num_rows,
                                  size_t min_row,
                                  int level_type_size,
                                  rmm::cuda_stream_view stream)
{
  dim3 dim_block(decode_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page
    
  if (level_type_size == 1) {
    gpuDecodePageDataFixed<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
        pages.device_ptr(), chunks, min_row, num_rows);
  } else {
    gpuDecodePageDataFixed<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
        pages.device_ptr(), chunks, min_row, num_rows);
  }
}

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf