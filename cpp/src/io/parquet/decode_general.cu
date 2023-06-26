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

constexpr int decode_block_size = 128;
constexpr int rolling_buf_size  = decode_block_size * 2;

namespace {

/**
 * @brief Performs RLE decoding of dictionary indexes, for when dict_size=1
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] target_pos Target write position
 * @param[in] t Thread ID
 *
 * @return The new output position
 */
template <typename state_buf>
__device__ int gpuDecodeRleBooleans(page_state_s* s, state_buf* sb, int target_pos, int t)
{
  uint8_t const* end = s->data_end;
  int pos            = s->dict_pos;

  while (pos < target_pos) {
    int is_literal, batch_len;
    if (!t) {
      uint32_t run       = s->dict_run;
      uint8_t const* cur = s->data_start;
      if (run <= 1) {
        run = (cur < end) ? get_vlq32(cur, end) : 0;
        if (!(run & 1)) {
          // Repeated value
          s->dict_val = (cur < end) ? cur[0] & 1 : 0;
          cur++;
        }
      }
      if (run & 1) {
        // Literal batch: must output a multiple of 8, except for the last batch
        int batch_len_div8;
        batch_len = max(min(32, (int)(run >> 1) * 8), 1);
        if (batch_len >= 8) { batch_len &= ~7; }
        batch_len_div8 = (batch_len + 7) >> 3;
        run -= batch_len_div8 * 2;
        cur += batch_len_div8;
      } else {
        batch_len = max(min(32, (int)(run >> 1)), 1);
        run -= batch_len * 2;
      }
      s->dict_run   = run;
      s->data_start = cur;
      is_literal    = run & 1;
      __threadfence_block();
    }
    __syncwarp();
    is_literal = shuffle(is_literal);
    batch_len  = shuffle(batch_len);
    if (t < batch_len) {
      int dict_idx;
      if (is_literal) {
        int32_t ofs      = t - ((batch_len + 7) & ~7);
        uint8_t const* p = s->data_start + (ofs >> 3);
        dict_idx         = (p < end) ? (p[0] >> (ofs & 7u)) & 1 : 0;
      } else {
        dict_idx = s->dict_val;
      }
      sb->dict_idx[rolling_index<state_buf::dict_buf_size>(pos + t)] = dict_idx;
    }
    pos += batch_len;
  }
  return pos;
}

/**
 * @brief Process a batch of incoming repetition/definition level values and generate
 *        validity, nested column offsets (where appropriate) and decoding indices.
 *
 * @param[in] target_input_value_count The # of repetition/definition levels to process up to
 * @param[in] s Local page information
 * @param[out] sb Page state buffer output
 * @param[in] rep Repetition level buffer
 * @param[in] def Definition level buffer
 * @param[in] t Thread index
 */
template <typename level_t, typename state_buf>
static __device__ void gpuUpdateValidityOffsetsAndRowIndices(int32_t target_input_value_count,
                                                             page_state_s* s,
                                                             state_buf* sb,
                                                             level_t const* const rep,
                                                             level_t const* const def,
                                                             int t)
{
  // max nesting depth of the column
  int const max_depth       = s->col.max_nesting_depth;
  bool const has_repetition = s->col.max_level[level_type::REPETITION] > 0;
  // how many (input) values we've processed in the page so far
  int input_value_count = s->input_value_count;
  // how many rows we've processed in the page so far
  int input_row_count = s->input_row_count;

  PageNestingDecodeInfo* nesting_info_base = s->nesting_info;

  // process until we've reached the target
  while (input_value_count < target_input_value_count) {
    // determine the nesting bounds for this thread (the range of nesting depths we
    // will generate new value indices and validity bits for)
    int start_depth, end_depth, d;
    get_nesting_bounds<state_buf::nz_buf_size, level_t>(
      start_depth, end_depth, d, s, rep, def, input_value_count, target_input_value_count, t);

    // 4 interesting things to track:
    // thread_value_count : # of output values from the view of this thread
    // warp_value_count   : # of output values for the whole warp
    //
    // thread_valid_count : # of valid values from the view of this thread
    // warp_valid_count   : # of valid values for the whole warp
    uint32_t thread_value_count, warp_value_count;
    uint32_t thread_valid_count, warp_valid_count;

    // track (page-relative) row index for the thread so we can compare against input bounds
    // keep track of overall # of rows we've read.
    int const is_new_row               = start_depth == 0 ? 1 : 0;
    uint32_t const warp_row_count_mask = ballot(is_new_row);
    int32_t const thread_row_index =
      input_row_count + ((__popc(warp_row_count_mask & ((1 << t) - 1)) + is_new_row) - 1);
    input_row_count += __popc(warp_row_count_mask);
    // is this thread within read row bounds?
    int const in_row_bounds = thread_row_index >= s->row_index_lower_bound &&
                                  thread_row_index < (s->first_row + s->num_rows)
                                ? 1
                                : 0;

    // compute warp and thread value counts
    uint32_t const warp_count_mask =
      ballot((0 >= start_depth && 0 <= end_depth) && in_row_bounds ? 1 : 0);

    warp_value_count = __popc(warp_count_mask);
    // Note : ((1 << t) - 1) implies "for all threads before me"
    thread_value_count = __popc(warp_count_mask & ((1 << t) - 1));

    // walk from 0 to max_depth
    uint32_t next_thread_value_count, next_warp_value_count;
    for (int s_idx = 0; s_idx < max_depth; s_idx++) {
      PageNestingDecodeInfo* nesting_info = &nesting_info_base[s_idx];

      // if we are within the range of nesting levels we should be adding value indices for
      int const in_nesting_bounds =
        ((s_idx >= start_depth && s_idx <= end_depth) && in_row_bounds) ? 1 : 0;

      // everything up to the max_def_level is a non-null value
      uint32_t const is_valid = d >= nesting_info->max_def_level && in_nesting_bounds ? 1 : 0;

      // compute warp and thread valid counts
      uint32_t const warp_valid_mask =
        // for flat schemas, a simple ballot_sync gives us the correct count and bit positions
        // because every value in the input matches to a value in the output
        !has_repetition
          ? ballot(is_valid)
          :
          // for nested schemas, it's more complicated.  This warp will visit 32 incoming values,
          // however not all of them will necessarily represent a value at this nesting level. so
          // the validity bit for thread t might actually represent output value t-6. the correct
          // position for thread t's bit is cur_value_count. for cuda 11 we could use
          // __reduce_or_sync(), but until then we have to do a warp reduce.
          WarpReduceOr32(is_valid << thread_value_count);

      thread_valid_count = __popc(warp_valid_mask & ((1 << thread_value_count) - 1));
      warp_valid_count   = __popc(warp_valid_mask);

      // if this is the value column emit an index for value decoding
      if (is_valid && s_idx == max_depth - 1) {
        int const src_pos = nesting_info->valid_count + thread_valid_count;
        int const dst_pos = nesting_info->value_count + thread_value_count;
        // nz_idx is a mapping of src buffer indices to destination buffer indices
        sb->nz_idx[rolling_index<state_buf::nz_buf_size>(src_pos)] = dst_pos;
      }

      // compute warp and thread value counts for the -next- nesting level. we need to
      // do this for nested schemas so that we can emit an offset for the -current- nesting
      // level. more concretely : the offset for the current nesting level == current length of the
      // next nesting level
      if (s_idx < max_depth - 1) {
        uint32_t const next_warp_count_mask =
          ballot((s_idx + 1 >= start_depth && s_idx + 1 <= end_depth && in_row_bounds) ? 1 : 0);
        next_warp_value_count   = __popc(next_warp_count_mask);
        next_thread_value_count = __popc(next_warp_count_mask & ((1 << t) - 1));

        // if we're -not- at a leaf column and we're within nesting/row bounds
        // and we have a valid data_out pointer, it implies this is a list column, so
        // emit an offset.
        if (in_nesting_bounds && nesting_info->data_out != nullptr) {
          int const idx             = nesting_info->value_count + thread_value_count;
          cudf::size_type const ofs = nesting_info_base[s_idx + 1].value_count +
                                      next_thread_value_count +
                                      nesting_info_base[s_idx + 1].page_start_value;
          (reinterpret_cast<cudf::size_type*>(nesting_info->data_out))[idx] = ofs;
        }
      }

      // nested schemas always read and write to the same bounds (that is, read and write positions
      // are already pre-bounded by first_row/num_rows). flat schemas will start reading at the
      // first value, even if that is before first_row, because we cannot trivially jump to
      // the correct position to start reading. since we are about to write the validity vector here
      // we need to adjust our computed mask to take into account the write row bounds.
      int const in_write_row_bounds =
        !has_repetition
          ? thread_row_index >= s->first_row && thread_row_index < (s->first_row + s->num_rows)
          : in_row_bounds;
      int const first_thread_in_write_range =
        !has_repetition ? __ffs(ballot(in_write_row_bounds)) - 1 : 0;

      // # of bits to of the validity mask to write out
      int const warp_valid_mask_bit_count =
        first_thread_in_write_range < 0 ? 0 : warp_value_count - first_thread_in_write_range;

      // increment count of valid values, count of total values, and update validity mask
      if (!t) {
        if (nesting_info->valid_map != nullptr && warp_valid_mask_bit_count > 0) {
          uint32_t const warp_output_valid_mask = warp_valid_mask >> first_thread_in_write_range;
          store_validity(nesting_info->valid_map_offset,
                         nesting_info->valid_map,
                         warp_output_valid_mask,
                         warp_valid_mask_bit_count);
          nesting_info->valid_map_offset += warp_valid_mask_bit_count;
          nesting_info->null_count += warp_valid_mask_bit_count - __popc(warp_output_valid_mask);
        }
        nesting_info->valid_count += warp_valid_count;
        nesting_info->value_count += warp_value_count;
      }

      // propagate value counts for the next level
      warp_value_count   = next_warp_value_count;
      thread_value_count = next_thread_value_count;
    }

    input_value_count += min(32, (target_input_value_count - input_value_count));
    __syncwarp();
  }

  // update
  if (!t) {
    // update valid value count for decoding and total # of values we've processed
    s->nz_count          = nesting_info_base[max_depth - 1].valid_count;
    s->input_value_count = input_value_count;
    s->input_row_count   = input_row_count;
  }
}

/**
 * @brief Decode values out of a definition or repetition stream
 *
 * @param[in,out] s Page state input/output
 * @param[in] t target_count Target count of stream values on output
 * @param[in] t Warp0 thread ID (0..31)
 * @param[in] lvl The level type we are decoding - DEFINITION or REPETITION
 */
template <typename level_t>
__device__ void gpuDecodeStream(
  level_t* output, page_state_s* s, int32_t target_count, int t, level_type lvl)
{
  uint8_t const* cur_def    = s->lvl_start[lvl];
  uint8_t const* end        = s->lvl_end;
  uint32_t level_run        = s->initial_rle_run[lvl];
  int32_t level_val         = s->initial_rle_value[lvl];
  int level_bits            = s->col.level_bits[lvl];
  int32_t num_input_values  = s->num_input_values;
  int32_t value_count       = s->lvl_count[lvl];
  int32_t batch_coded_count = 0;

  while (value_count < target_count && value_count < num_input_values) {
    int batch_len;
    if (level_run <= 1) {
      // Get a new run symbol from the byte stream
      int sym_len = 0;
      if (!t) {
        uint8_t const* cur = cur_def;
        if (cur < end) { level_run = get_vlq32(cur, end); }
        if (!(level_run & 1)) {
          if (cur < end) level_val = cur[0];
          cur++;
          if (level_bits > 8) {
            if (cur < end) level_val |= cur[0] << 8;
            cur++;
          }
        }
        if (cur > end || level_run <= 1) { s->error = 0x10; }
        sym_len = (int32_t)(cur - cur_def);
        __threadfence_block();
      }
      sym_len   = shuffle(sym_len);
      level_val = shuffle(level_val);
      level_run = shuffle(level_run);
      cur_def += sym_len;
    }
    if (s->error) { break; }

    batch_len = min(num_input_values - value_count, 32);
    if (level_run & 1) {
      // Literal run
      int batch_len8;
      batch_len  = min(batch_len, (level_run >> 1) * 8);
      batch_len8 = (batch_len + 7) >> 3;
      if (t < batch_len) {
        int bitpos         = t * level_bits;
        uint8_t const* cur = cur_def + (bitpos >> 3);
        bitpos &= 7;
        if (cur < end) level_val = cur[0];
        cur++;
        if (level_bits > 8 - bitpos && cur < end) {
          level_val |= cur[0] << 8;
          cur++;
          if (level_bits > 16 - bitpos && cur < end) level_val |= cur[0] << 16;
        }
        level_val = (level_val >> bitpos) & ((1 << level_bits) - 1);
      }
      level_run -= batch_len8 * 2;
      cur_def += batch_len8 * level_bits;
    } else {
      // Repeated value
      batch_len = min(batch_len, level_run >> 1);
      level_run -= batch_len * 2;
    }
    if (t < batch_len) {
      int idx                                      = value_count + t;
      output[rolling_index<rolling_buf_size>(idx)] = level_val;
    }
    batch_coded_count += batch_len;
    value_count += batch_len;
  }

  // update the stream info
  if (!t) {
    s->lvl_start[lvl]         = cur_def;
    s->initial_rle_run[lvl]   = level_run;
    s->initial_rle_value[lvl] = level_val;
    s->lvl_count[lvl]         = value_count;
  }
}

/**
 * @brief Process repetition and definition levels up to the target count of leaf values.
 *
 * In order to decode actual leaf values from the input stream, we need to generate the
 * list of non-null value positions (page_state_s::nz_idx). We do this by processing
 * the repetition and definition level streams.  This process also generates validity information,
 * and offset column values in the case of nested schemas. Because of the way the streams
 * are encoded, this function may generate slightly more than target_leaf_count.
 *
 * Only runs on 1 warp.
 *
 * @param[in] s The local page state
 * @param[out] sb Page state buffer output
 * @param[in] target_leaf_count Target count of non-null leaf values to generate indices for
 * @param[in] rep Repetition level buffer
 * @param[in] def Definition level buffer
 * @param[in] t Thread index
 */
template <typename level_t, typename state_buf>
__device__ void gpuDecodeLevels(page_state_s* s,
                                state_buf* sb,
                                int32_t target_leaf_count,
                                level_t* const rep,
                                level_t* const def,
                                int t)
{
  bool has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  constexpr int batch_size = 32;
  int cur_leaf_count       = target_leaf_count;
  while (!s->error && s->nz_count < target_leaf_count &&
         s->input_value_count < s->num_input_values) {
    if (has_repetition) { gpuDecodeStream(rep, s, cur_leaf_count, t, level_type::REPETITION); }
    gpuDecodeStream(def, s, cur_leaf_count, t, level_type::DEFINITION);
    __syncwarp();

    // because the rep and def streams are encoded separately, we cannot request an exact
    // # of values to be decoded at once. we can only process the lowest # of decoded rep/def
    // levels we get.
    int actual_leaf_count = has_repetition ? min(s->lvl_count[level_type::REPETITION],
                                                 s->lvl_count[level_type::DEFINITION])
                                           : s->lvl_count[level_type::DEFINITION];

    // process what we got back
    gpuUpdateValidityOffsetsAndRowIndices<level_t>(actual_leaf_count, s, sb, rep, def, t);
    cur_leaf_count = actual_leaf_count + batch_size;
    __syncwarp();
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
__global__ void __launch_bounds__(decode_block_size) gpuDecodePageData(
  PageInfo* pages, device_span<ColumnChunkDesc const> chunks, size_t min_row, size_t num_rows)
{
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16) page_state_buffers_s<rolling_buf_size,  // size of nz_idx buffer
                                                rolling_buf_size,  // size of dict index buffer
                                                rolling_buf_size>  // size of string lengths buffer
    state_buffers;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  int page_idx          = blockIdx.x;
  int t                 = threadIdx.x;
  int out_thread0;

  if (!(pages[page_idx].kernel_mask & KERNEL_MASK_GENERAL)) { return; }
  if (!setupLocalPageInfo(s, &pages[page_idx], chunks, min_row, num_rows, true)) { return; }

  // must come after the kernel mask check
  [[maybe_unused]] null_count_back_copier _{s, t};

  bool const has_repetition = s->col.max_level[level_type::REPETITION] > 0;

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

  if (s->dict_base) {
    out_thread0 = (s->dict_bits > 0) ? 64 : 32;
  } else {
    out_thread0 =
      ((s->col.data_type & 7) == BOOLEAN || (s->col.data_type & 7) == BYTE_ARRAY) ? 64 : 32;
  }

  PageNestingDecodeInfo* nesting_info_base = s->nesting_info;

  __shared__ level_t rep[rolling_buf_size];  // circular buffer of repetition level values
  __shared__ level_t def[rolling_buf_size];  // circular buffer of definition level values

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t skipped_leaf_values = s->page.skipped_leaf_values;
  while (!s->error && (s->input_value_count < s->num_input_values || s->src_pos < s->nz_count)) {
    int target_pos;
    int src_pos = s->src_pos;

    if (t < out_thread0) {
      target_pos = min(src_pos + 2 * (decode_block_size - out_thread0),
                       s->nz_count + (decode_block_size - out_thread0));
    } else {
      target_pos = min(s->nz_count, src_pos + decode_block_size - out_thread0);
      if (out_thread0 > 32) { target_pos = min(target_pos, s->dict_pos); }
    }
    __syncthreads();
    if (t < 32) {
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<level_t>(s, sb, target_pos, rep, def, t);
    } else if (t < out_thread0) {
      // skipped_leaf_values will always be 0 for flat hierarchies.
      uint32_t src_target_pos = target_pos + skipped_leaf_values;

      // WARP1: Decode dictionary indices, booleans or string positions
      if (s->dict_base) {
        src_target_pos = gpuDecodeDictionaryIndices<false>(s, sb, src_target_pos, t & 0x1f).first;
      } else if ((s->col.data_type & 7) == BOOLEAN) {
        src_target_pos = gpuDecodeRleBooleans(s, sb, src_target_pos, t & 0x1f);
      } else if ((s->col.data_type & 7) == BYTE_ARRAY) {
        gpuInitStringDescriptors<false>(s, sb, src_target_pos, t & 0x1f);
      }
      if (t == 32) { *(volatile int32_t*)&s->dict_pos = src_target_pos; }
    } else {
      // WARP1..WARP3: Decode values
      int dtype = s->col.data_type & 7;
      src_pos += t - out_thread0;

      // the position in the output column/buffer
      int dst_pos = sb->nz_idx[rolling_index<rolling_buf_size>(src_pos)];

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
          gpuOutputBoolean(sb, val_src_pos, static_cast<uint8_t*>(dst));
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

      if (t == out_thread0) { *(volatile int32_t*)&s->src_pos = target_pos; }
    }
    __syncthreads();
  }
}

}  // anonymous namespace

void __host__ DecodePageDataGeneral(cudf::detail::hostdevice_vector<PageInfo>& pages,
                                    cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                                    size_t num_rows,
                                    size_t min_row,
                                    int level_type_size,
                                    rmm::cuda_stream_view stream)
{
  dim3 dim_block(decode_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    gpuDecodePageData<uint8_t>
      <<<dim_grid, dim_block, 0, stream.value()>>>(pages.device_ptr(), chunks, min_row, num_rows);
  } else {
    gpuDecodePageData<uint16_t>
      <<<dim_grid, dim_block, 0, stream.value()>>>(pages.device_ptr(), chunks, min_row, num_rows);
  }
}

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf
