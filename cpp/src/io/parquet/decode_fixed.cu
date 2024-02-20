/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "page_data.cuh"
#include "page_decode.cuh"
#include "parquet_gpu.hpp"
#include "rle_stream.cuh"

namespace cudf::io::parquet::detail {

namespace {

constexpr int decode_block_size = 128;
constexpr int rolling_buf_size  = decode_block_size * 2;
// the required number of runs in shared memory we will need to provide the
// rle_stream object
constexpr int rle_run_buffer_size = rle_stream_required_run_buffer_size<decode_block_size>();

template <bool nullable, typename level_t, typename state_buf>
static __device__ int gpuUpdateValidityOffsetsAndRowIndicesFlat(
  int32_t target_value_count, page_state_s* s, state_buf* sb, level_t const* const def, int t)
{
  constexpr int num_warps      = decode_block_size / 32;
  constexpr int max_batch_size = num_warps * 32;

  auto& ni = s->nesting_info[0];

  // how many (input) values we've processed in the page so far
  int value_count = s->input_value_count;
  int valid_count = ni.valid_count;

  // cap by last row so that we don't process any rows past what we want to output.
  int const first_row = s->first_row;
  int const last_row  = first_row + s->num_rows;
  target_value_count  = min(target_value_count, last_row);

  int const valid_map_offset      = ni.valid_map_offset;
  int const row_index_lower_bound = s->row_index_lower_bound;

  __syncthreads();

  while (value_count < target_value_count) {
    int const batch_size = min(max_batch_size, target_value_count - value_count);

    // definition level. only need to process for nullable columns
    int d = 0;
    if constexpr (nullable) {
      d = t < batch_size
            ? static_cast<int>(def[rolling_index<state_buf::nz_buf_size>(value_count + t)])
            : -1;
    }

    int const thread_value_count = t + 1;
    int const block_value_count  = batch_size;

    // compute our row index, whether we're in row bounds, and validity
    int const row_index     = (thread_value_count + value_count) - 1;
    int const in_row_bounds = (row_index >= row_index_lower_bound) && (row_index < last_row);
    int is_valid;
    if constexpr (nullable) {
      is_valid = ((d > 0) && in_row_bounds) ? 1 : 0;
    } else {
      is_valid = in_row_bounds;
    }

    // thread and block validity count
    int thread_valid_count, block_valid_count;
    if constexpr (nullable) {
      using block_scan = cub::BlockScan<int, decode_block_size>;
      __shared__ typename block_scan::TempStorage scan_storage;
      block_scan(scan_storage).InclusiveSum(is_valid, thread_valid_count, block_valid_count);
      __syncthreads();

      // validity is processed per-warp
      //
      // nested schemas always read and write to the same bounds (that is, read and write
      // positions are already pre-bounded by first_row/num_rows). flat schemas will start reading
      // at the first value, even if that is before first_row, because we cannot trivially jump to
      // the correct position to start reading. since we are about to write the validity vector
      // here we need to adjust our computed mask to take into account the write row bounds.
      int const in_write_row_bounds = ballot(row_index >= first_row && row_index < last_row);
      int const write_start = __ffs(in_write_row_bounds) - 1;  // first bit in the warp to store
      int warp_null_count   = 0;
      if (write_start >= 0) {
        uint32_t const warp_validity_mask = ballot(is_valid);
        // lane 0 from each warp writes out validity
        if (!(t % 32)) {
          int const vindex = (value_count + thread_value_count) - 1;  // absolute input value index
          int const bit_offset = (valid_map_offset + vindex + write_start) -
                                 first_row;  // absolute bit offset into the output validity map
          int const write_end = 32 - __clz(in_write_row_bounds);  // last bit in the warp to store
          int const bit_count = write_end - write_start;
          warp_null_count     = bit_count - __popc(warp_validity_mask >> write_start);

          store_validity(bit_offset, ni.valid_map, warp_validity_mask >> write_start, bit_count);
        }
      }

      // sum null counts. we have to do it this way instead of just incrementing by (value_count -
      // valid_count) because valid_count also includes rows that potentially start before our row
      // bounds. if we could come up with a way to clean that up, we could remove this and just
      // compute it directly at the end of the kernel.
      size_type block_null_count =
        cudf::detail::single_lane_block_sum_reduce<decode_block_size, 0>(warp_null_count);
      if (!t) { ni.null_count += block_null_count; }
    }
    // trivial for non-nullable columns
    else {
      thread_valid_count = thread_value_count;
      block_valid_count  = block_value_count;
    }

    // output offset
    if (is_valid) {
      int const dst_pos = (value_count + thread_value_count) - 1;
      int const src_pos = (valid_count + thread_valid_count) - 1;
      auto ix           = rolling_index<state_buf::nz_buf_size>(src_pos);
      sb->nz_idx[rolling_index<state_buf::nz_buf_size>(src_pos)] = dst_pos;
    }

    // update stuff
    value_count += block_value_count;
    valid_count += block_valid_count;
  }

  if (!t) {
    // update valid value count for decoding and total # of values we've processed
    ni.valid_count       = valid_count;
    ni.value_count       = value_count;
    s->nz_count          = valid_count;
    s->input_value_count = value_count;
    s->input_row_count   = value_count;
  }

  return valid_count;
}

template <typename state_buf>
__device__ inline void gpuDecodeValues(
  page_state_s* s, state_buf* const sb, int start, int end, int t)
{
  constexpr int num_warps      = decode_block_size / 32;
  constexpr int max_batch_size = num_warps * 32;

  PageNestingDecodeInfo* nesting_info_base = s->nesting_info;
  int const dtype                          = s->col.data_type & 7;

  // decode values
  int pos = start;
  while (pos < end) {
    int const batch_size = min(max_batch_size, end - pos);

    int const target_pos = pos + batch_size;
    int const src_pos    = pos + t;

    // the position in the output column/buffer
    auto nz_idx = sb->nz_idx[rolling_index<state_buf::nz_buf_size>(src_pos)];
    int dst_pos = sb->nz_idx[rolling_index<state_buf::nz_buf_size>(src_pos)] - s->first_row;

    // target_pos will always be properly bounded by num_rows, but dst_pos may be negative (values
    // before first_row) in the flat hierarchy case.
    if (src_pos < target_pos && dst_pos >= 0) {
      // nesting level that is storing actual leaf values
      int leaf_level_index = s->col.max_nesting_depth - 1;

      uint32_t dtype_len = s->dtype_len;
      void* dst =
        nesting_info_base[leaf_level_index].data_out + static_cast<size_t>(dst_pos) * dtype_len;
      if (dtype == BYTE_ARRAY) {
        if (s->col.converted_type == DECIMAL) {
          auto const [ptr, len]        = gpuGetStringData(s, sb, src_pos);
          auto const decimal_precision = s->col.decimal_precision;
          if (decimal_precision <= MAX_DECIMAL32_PRECISION) {
            gpuOutputByteArrayAsInt(ptr, len, static_cast<int32_t*>(dst));
          } else if (decimal_precision <= MAX_DECIMAL64_PRECISION) {
            gpuOutputByteArrayAsInt(ptr, len, static_cast<int64_t*>(dst));
          } else {
            gpuOutputByteArrayAsInt(ptr, len, static_cast<__int128_t*>(dst));
          }
        } else {
          gpuOutputString(s, sb, src_pos, dst);
        }
      } else if (dtype == BOOLEAN) {
        gpuOutputBoolean(sb, src_pos, static_cast<uint8_t*>(dst));
      } else if (s->col.converted_type == DECIMAL) {
        switch (dtype) {
          case INT32: gpuOutputFast(s, sb, src_pos, static_cast<uint32_t*>(dst)); break;
          case INT64: gpuOutputFast(s, sb, src_pos, static_cast<uint2*>(dst)); break;
          default:
            if (s->dtype_len_in <= sizeof(int32_t)) {
              gpuOutputFixedLenByteArrayAsInt(s, sb, src_pos, static_cast<int32_t*>(dst));
            } else if (s->dtype_len_in <= sizeof(int64_t)) {
              gpuOutputFixedLenByteArrayAsInt(s, sb, src_pos, static_cast<int64_t*>(dst));
            } else {
              gpuOutputFixedLenByteArrayAsInt(s, sb, src_pos, static_cast<__int128_t*>(dst));
            }
            break;
        }
      } else if (dtype == INT96) {
        gpuOutputInt96Timestamp(s, sb, src_pos, static_cast<int64_t*>(dst));
      } else if (dtype_len == 8) {
        if (s->dtype_len_in == 4) {
          // Reading INT32 TIME_MILLIS into 64-bit DURATION_MILLISECONDS
          // TIME_MILLIS is the only duration type stored as int32:
          // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#deprecated-time-convertedtype
          gpuOutputFast(s, sb, src_pos, static_cast<uint32_t*>(dst));
        } else if (s->ts_scale) {
          gpuOutputInt64Timestamp(s, sb, src_pos, static_cast<int64_t*>(dst));
        } else {
          gpuOutputFast(s, sb, src_pos, static_cast<uint2*>(dst));
        }
      } else if (dtype_len == 4) {
        gpuOutputFast(s, sb, src_pos, static_cast<uint32_t*>(dst));
      } else {
        gpuOutputGeneric(s, sb, src_pos, static_cast<uint8_t*>(dst), dtype_len);
      }
    }

    pos += batch_size;
  }
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
  __shared__ __align__(16) page_state_buffers_s<rolling_buf_size,  // size of nz_idx buffer
                                                1,                 // unused in this kernel
                                                1>                 // unused in this kernel
    state_buffers;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  const int page_idx    = blockIdx.x;
  const int t           = threadIdx.x;
  PageInfo* pp          = &pages[page_idx];

  if (!(BitAnd(pages[page_idx].kernel_mask, decode_kernel_mask::FIXED_WIDTH_NO_DICT))) { return; }

  // must come after the kernel mask check
  [[maybe_unused]] null_count_back_copier _{s, t};

  if (!setupLocalPageInfo(s,
                          pp,
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{decode_kernel_mask::FIXED_WIDTH_NO_DICT},
                          page_processing_stage::DECODE)) {
    return;
  }

  // the level stream decoders
  __shared__ rle_run<level_t> def_runs[rle_run_buffer_size];
  rle_stream<level_t, decode_block_size, rolling_buf_size> def_decoder{def_runs};

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
  if (s->num_rows == 0) { return; }

  // initialize the stream decoders (requires values computed in setupLocalPageInfo)

  level_t* def = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  if (nullable) {
    def_decoder.init(s->col.level_bits[level_type::DEFINITION],
                     s->abs_lvl_start[level_type::DEFINITION],
                     s->abs_lvl_end[level_type::DEFINITION],
                     def,
                     s->page.num_input_values);
  }
  __syncthreads();

  // the core loop. decode batches of level stream data using rle_stream objects
  // and pass the results to gpuUpdatePageSizes
  int processed = 0;
  int valid     = 0;
  while (processed < s->page.num_input_values) {
    int next_valid;

    // only need to process definition levels if this is a nullable column
    if (nullable) {
      processed += def_decoder.decode_next(t);
      __syncthreads();

      next_valid =
        gpuUpdateValidityOffsetsAndRowIndicesFlat<true, level_t>(processed, s, sb, def, t);
    }
    // if we wanted to split off the skip_rows/num_rows case into a separate kernel, we could skip
    // this function call entirely since all it will ever generate is a mapping of (i -> i) for
    // nz_idx.  gpuDecodeValues would be the only work that happens.
    else {
      processed += min(rolling_buf_size, s->page.num_input_values - processed);
      next_valid =
        gpuUpdateValidityOffsetsAndRowIndicesFlat<false, level_t>(processed, s, sb, nullptr, t);
    }
    __syncthreads();

    // decode the values themselves
    gpuDecodeValues(s, sb, valid, next_valid, t);
    __syncthreads();

    valid = next_valid;
  }
}

template <typename level_t>
__global__ void __launch_bounds__(decode_block_size) gpuDecodePageDataFixedDict(
  PageInfo* pages, device_span<ColumnChunkDesc const> chunks, size_t min_row, size_t num_rows)
{
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16) page_state_buffers_s<rolling_buf_size,  // size of nz_idx buffer
                                                rolling_buf_size,  // dictionary
                                                1>                 // unused in this kernel
    state_buffers;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  const int page_idx    = blockIdx.x;
  const int t           = threadIdx.x;
  PageInfo* pp          = &pages[page_idx];

  if (!(BitAnd(pages[page_idx].kernel_mask, decode_kernel_mask::FIXED_WIDTH_DICT))) { return; }

  // must come after the kernel mask check
  [[maybe_unused]] null_count_back_copier _{s, t};

  if (!setupLocalPageInfo(s,
                          pp,
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{decode_kernel_mask::FIXED_WIDTH_DICT},
                          page_processing_stage::DECODE)) {
    return;
  }

  // the level stream decoders
  // rolling_buf_size = 256
  __shared__ rle_run<level_t> def_runs[rle_run_buffer_size];
  rle_stream<level_t, decode_block_size, rolling_buf_size> def_decoder{def_runs};

  // should the size be 1/2 (128?)
  __shared__ rle_run<uint32_t> dict_runs[rle_run_buffer_size];  // should be array of 6
  rle_stream<uint32_t, decode_block_size, rolling_buf_size> dict_stream{dict_runs};

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
  if (s->num_rows == 0) { return; }

  // initialize the stream decoders (requires values computed in setupLocalPageInfo)
  level_t* def = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  if (nullable) {
    // col.level_bits - 1
    def_decoder.init(s->col.level_bits[level_type::DEFINITION],
                     s->abs_lvl_start[level_type::DEFINITION],
                     s->abs_lvl_end[level_type::DEFINITION],
                     def,  // 2048 sized
                     s->page.num_input_values);
  }

  dict_stream.init(
    s->dict_bits, s->data_start, s->data_end, sb->dict_idx, s->page.num_input_values);
  __syncthreads();

  // the core loop. decode batches of level stream data using rle_stream objects
  // and pass the results to gpuUpdatePageSizes
  int processed = 0;
  int valid     = 0;
  while (processed < s->page.num_input_values) {
    int next_valid;

    // only need to process definition levels if this is a nullable column
    if (nullable) {
      //-1 don't cap it
      processed += def_decoder.decode_next(t);
      __syncthreads();

      // count of valid items in this batch
      next_valid =
        gpuUpdateValidityOffsetsAndRowIndicesFlat<true, level_t>(processed, s, sb, def, t);
    }
    // if we wanted to split off the skip_rows/num_rows case into a separate kernel, we could skip
    // this function call entirely since all it will ever generate is a mapping of (i -> i) for
    // nz_idx.  gpuDecodeValues would be the only work that happens.
    else {
      processed += min(rolling_buf_size, s->page.num_input_values - processed);
      next_valid =
        gpuUpdateValidityOffsetsAndRowIndicesFlat<false, level_t>(processed, s, sb, nullptr, t);
    }
    __syncthreads();

    dict_stream.decode_next(t, (next_valid - valid));
    __syncthreads();

    // decode the values themselves
    gpuDecodeValues(s, sb, valid, next_valid, t);
    __syncthreads();

    valid = next_valid;
  }
}

}  // anonymous namespace

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
    gpuDecodePageDataFixed<uint8_t>
      <<<dim_grid, dim_block, 0, stream.value()>>>(pages.device_ptr(), chunks, min_row, num_rows);
  } else {
    gpuDecodePageDataFixed<uint16_t>
      <<<dim_grid, dim_block, 0, stream.value()>>>(pages.device_ptr(), chunks, min_row, num_rows);
  }
}

void __host__
DecodePageDataFixedDict(cudf::detail::hostdevice_vector<PageInfo>& pages,
                        cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                        size_t num_rows,
                        size_t min_row,
                        int level_type_size,
                        rmm::cuda_stream_view stream)
{
  //  dim3 dim_block(decode_block_size, 1); // decode_block_size = 128 threads per block
  // 1 full warp, and 1 warp of 1 thread
  dim3 dim_block(decode_block_size, 1);  // decode_block_size = 128 threads per block
  dim3 dim_grid(pages.size(), 1);        // 1 thread block per pags => # blocks

  if (level_type_size == 1) {
    gpuDecodePageDataFixedDict<uint8_t>
      <<<dim_grid, dim_block, 0, stream.value()>>>(pages.device_ptr(), chunks, min_row, num_rows);
  } else {
    gpuDecodePageDataFixedDict<uint16_t>
      <<<dim_grid, dim_block, 0, stream.value()>>>(pages.device_ptr(), chunks, min_row, num_rows);
  }
}

}  // namespace cudf::io::parquet::detail
