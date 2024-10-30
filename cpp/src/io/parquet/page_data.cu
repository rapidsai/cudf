/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/batched_memcpy.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

namespace cudf::io::parquet::detail {

namespace cg = cooperative_groups;

namespace {

constexpr int decode_block_size = 128;
constexpr int rolling_buf_size  = decode_block_size * 2;

/**
 * @brief Kernel for computing the BYTE_STREAM_SPLIT column data stored in the pages
 *
 * This is basically the PLAIN decoder, but with a pared down set of supported data
 * types, and using output functions that piece together the individual streams.
 * Supported physical types include INT32, INT64, FLOAT, DOUBLE and FIXED_LEN_BYTE_ARRAY.
 * The latter is currently only used for large decimals. The Parquet specification also
 * has FLOAT16 and UUID types that are currently not supported. FIXED_LEN_BYTE_ARRAY data
 * that lacks a `LogicalType` annotation will be handled by the string decoder.
 *
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read
 * @param error_code Error code to set if an error is encountered
 */
template <int lvl_buf_size, typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_block_size)
  gpuDecodeSplitPageData(PageInfo* pages,
                         device_span<ColumnChunkDesc const> chunks,
                         size_t min_row,
                         size_t num_rows,
                         kernel_error::pointer error_code)
{
  using cudf::detail::warp_size;
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16)
    page_state_buffers_s<rolling_buf_size, rolling_buf_size, rolling_buf_size>
      state_buffers;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  int page_idx          = blockIdx.x;
  int t                 = threadIdx.x;
  [[maybe_unused]] null_count_back_copier _{s, t};

  if (!setupLocalPageInfo(s,
                          &pages[page_idx],
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{decode_kernel_mask::BYTE_STREAM_SPLIT},
                          page_processing_stage::DECODE)) {
    return;
  }

  bool const has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  auto const data_len    = thrust::distance(s->data_start, s->data_end);
  auto const num_values  = data_len / s->dtype_len_in;
  auto const out_thread0 = warp_size;

  PageNestingDecodeInfo* nesting_info_base = s->nesting_info;

  __shared__ level_t rep[rolling_buf_size];  // circular buffer of repetition level values
  __shared__ level_t def[rolling_buf_size];  // circular buffer of definition level values

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t skipped_leaf_values = s->page.skipped_leaf_values;
  while (s->error == 0 &&
         (s->input_value_count < s->num_input_values || s->src_pos < s->nz_count)) {
    int target_pos;
    int src_pos = s->src_pos;

    if (t < out_thread0) {
      target_pos = min(src_pos + 2 * (decode_block_size - out_thread0),
                       s->nz_count + (decode_block_size - out_thread0));
    } else {
      target_pos = min(s->nz_count, src_pos + decode_block_size - out_thread0);
    }
    // this needs to be here to prevent warp 1 modifying src_pos before all threads have read it
    __syncthreads();

    if (t < warp_size) {
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<lvl_buf_size, level_t>(s, sb, target_pos, rep, def, t);
    } else {
      // WARP1..WARP3: Decode values
      int const dtype = s->col.physical_type;
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
      //   -2, -1, 0, 1, 2 ...
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
        uint8_t const* src = s->data_start + val_src_pos;
        uint8_t* dst =
          nesting_info_base[leaf_level_index].data_out + static_cast<size_t>(dst_pos) * dtype_len;
        auto const is_decimal =
          s->col.logical_type.has_value() and s->col.logical_type->type == LogicalType::DECIMAL;

        // Note: non-decimal FIXED_LEN_BYTE_ARRAY will be handled in the string reader
        if (is_decimal) {
          switch (dtype) {
            case INT32: gpuOutputByteStreamSplit<int32_t>(dst, src, num_values); break;
            case INT64: gpuOutputByteStreamSplit<int64_t>(dst, src, num_values); break;
            case FIXED_LEN_BYTE_ARRAY:
              if (s->dtype_len_in <= sizeof(int32_t)) {
                gpuOutputSplitFixedLenByteArrayAsInt(
                  reinterpret_cast<int32_t*>(dst), src, num_values, s->dtype_len_in);
                break;
              } else if (s->dtype_len_in <= sizeof(int64_t)) {
                gpuOutputSplitFixedLenByteArrayAsInt(
                  reinterpret_cast<int64_t*>(dst), src, num_values, s->dtype_len_in);
                break;
              } else if (s->dtype_len_in <= sizeof(__int128_t)) {
                gpuOutputSplitFixedLenByteArrayAsInt(
                  reinterpret_cast<__int128_t*>(dst), src, num_values, s->dtype_len_in);
                break;
              }
              // unsupported decimal precision
              [[fallthrough]];

            default: s->set_error_code(decode_error::UNSUPPORTED_ENCODING);
          }
        } else if (dtype_len == 8) {
          if (s->dtype_len_in == 4) {
            // Reading INT32 TIME_MILLIS into 64-bit DURATION_MILLISECONDS
            // TIME_MILLIS is the only duration type stored as int32:
            // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#deprecated-time-convertedtype
            gpuOutputByteStreamSplit<int32_t>(dst, src, num_values);
            // zero out most significant bytes
            memset(dst + 4, 0, 4);
          } else if (s->ts_scale) {
            gpuOutputSplitInt64Timestamp(
              reinterpret_cast<int64_t*>(dst), src, num_values, s->ts_scale);
          } else {
            gpuOutputByteStreamSplit<int64_t>(dst, src, num_values);
          }
        } else if (dtype_len == 4) {
          gpuOutputByteStreamSplit<int32_t>(dst, src, num_values);
        } else {
          s->set_error_code(decode_error::UNSUPPORTED_ENCODING);
        }
      }

      if (t == out_thread0) { s->src_pos = target_pos; }
    }
    __syncthreads();
  }
  if (t == 0 and s->error != 0) { set_error(s->error, error_code); }
}

/**
 * @brief Kernel for computing the column data stored in the pages
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
 * @param error_code Error code to set if an error is encountered
 */
template <int lvl_buf_size, typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_block_size)
  gpuDecodePageData(PageInfo* pages,
                    device_span<ColumnChunkDesc const> chunks,
                    size_t min_row,
                    size_t num_rows,
                    kernel_error::pointer error_code)
{
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16)
    page_state_buffers_s<rolling_buf_size, rolling_buf_size, rolling_buf_size>
      state_buffers;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  int page_idx          = blockIdx.x;
  int t                 = threadIdx.x;
  int out_thread0;
  [[maybe_unused]] null_count_back_copier _{s, t};

  if (!setupLocalPageInfo(s,
                          &pages[page_idx],
                          chunks,
                          min_row,
                          num_rows,
                          mask_filter{decode_kernel_mask::GENERAL},
                          page_processing_stage::DECODE)) {
    return;
  }

  bool const has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  if (s->dict_base) {
    out_thread0 = (s->dict_bits > 0) ? 64 : 32;
  } else {
    switch (s->col.physical_type) {
      case BOOLEAN: [[fallthrough]];
      case BYTE_ARRAY: [[fallthrough]];
      case FIXED_LEN_BYTE_ARRAY: out_thread0 = 64; break;
      default: out_thread0 = 32;
    }
  }

  PageNestingDecodeInfo* nesting_info_base = s->nesting_info;

  __shared__ level_t rep[rolling_buf_size];  // circular buffer of repetition level values
  __shared__ level_t def[rolling_buf_size];  // circular buffer of definition level values

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t skipped_leaf_values = s->page.skipped_leaf_values;
  while (s->error == 0 &&
         (s->input_value_count < s->num_input_values || s->src_pos < s->nz_count)) {
    int target_pos;
    int src_pos = s->src_pos;

    if (t < out_thread0) {
      target_pos = min(src_pos + 2 * (decode_block_size - out_thread0),
                       s->nz_count + (decode_block_size - out_thread0));
    } else {
      target_pos = min(s->nz_count, src_pos + decode_block_size - out_thread0);
      if (out_thread0 > 32) { target_pos = min(target_pos, s->dict_pos); }
    }
    // this needs to be here to prevent warp 3 modifying src_pos before all threads have read it
    __syncthreads();
    auto const tile_warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
    if (t < 32) {
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<lvl_buf_size, level_t>(s, sb, target_pos, rep, def, t);
    } else if (t < out_thread0) {
      // skipped_leaf_values will always be 0 for flat hierarchies.
      uint32_t src_target_pos = target_pos + skipped_leaf_values;

      // WARP1: Decode dictionary indices, booleans or string positions
      // NOTE: racecheck complains of a RAW error involving the s->dict_pos assignment below.
      // This is likely a false positive in practice, but could be solved by wrapping the next
      // 9 lines in `if (s->dict_pos < src_target_pos) {}`. If that change is made here, it will
      // be needed in the other DecodeXXX kernels.
      if (s->dict_base) {
        src_target_pos = gpuDecodeDictionaryIndices<false>(s, sb, src_target_pos, t & 0x1f).first;
      } else if (s->col.physical_type == BOOLEAN) {
        src_target_pos = gpuDecodeRleBooleans(s, sb, src_target_pos, t & 0x1f);
      } else if (s->col.physical_type == BYTE_ARRAY or
                 s->col.physical_type == FIXED_LEN_BYTE_ARRAY) {
        gpuInitStringDescriptors<false>(s, sb, src_target_pos, tile_warp);
      }
      if (tile_warp.thread_rank() == 0) { s->dict_pos = src_target_pos; }
    } else {
      // WARP1..WARP3: Decode values
      int const dtype = s->col.physical_type;
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
      //   -2, -1, 0, 1, 2 ...
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
        auto const is_decimal =
          s->col.logical_type.has_value() and s->col.logical_type->type == LogicalType::DECIMAL;
        if (dtype == BYTE_ARRAY) {
          if (is_decimal) {
            auto const [ptr, len]        = gpuGetStringData(s, sb, val_src_pos);
            auto const decimal_precision = s->col.logical_type->precision();
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
        } else if (is_decimal) {
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
        } else if (dtype == FIXED_LEN_BYTE_ARRAY) {
          gpuOutputString(s, sb, val_src_pos, dst);
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

      if (t == out_thread0) { s->src_pos = target_pos; }
    }
    __syncthreads();
  }
  if (t == 0 and s->error != 0) { set_error(s->error, error_code); }
}

struct mask_tform {
  __device__ uint32_t operator()(PageInfo const& p) { return static_cast<uint32_t>(p.kernel_mask); }
};

}  // anonymous namespace

uint32_t GetAggregatedDecodeKernelMask(cudf::detail::hostdevice_span<PageInfo const> pages,
                                       rmm::cuda_stream_view stream)
{
  // determine which kernels to invoke
  auto mask_iter = thrust::make_transform_iterator(pages.device_begin(), mask_tform{});
  return thrust::reduce(
    rmm::exec_policy(stream), mask_iter, mask_iter + pages.size(), 0U, thrust::bit_or<uint32_t>{});
}

/**
 * @copydoc cudf::io::parquet::detail::DecodePageData
 */
void __host__ DecodePageData(cudf::detail::hostdevice_span<PageInfo> pages,
                             cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                             size_t num_rows,
                             size_t min_row,
                             int level_type_size,
                             kernel_error::pointer error_code,
                             rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 dim_block(decode_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    gpuDecodePageData<rolling_buf_size, uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  } else {
    gpuDecodePageData<rolling_buf_size, uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  }
}

/**
 * @copydoc cudf::io::parquet::detail::DecodePageData
 */
void __host__ DecodeSplitPageData(cudf::detail::hostdevice_span<PageInfo> pages,
                                  cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                                  size_t num_rows,
                                  size_t min_row,
                                  int level_type_size,
                                  kernel_error::pointer error_code,
                                  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 dim_block(decode_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    gpuDecodeSplitPageData<rolling_buf_size, uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  } else {
    gpuDecodeSplitPageData<rolling_buf_size, uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, error_code);
  }
}

void WriteFinalOffsets(host_span<size_type const> offsets,
                       host_span<size_type* const> buff_addrs,
                       rmm::cuda_stream_view stream)
{
  // Copy offsets to device and create an iterator
  auto d_src_data = cudf::detail::make_device_uvector_async(
    offsets, stream, cudf::get_current_device_resource_ref());
  // Iterator for the source (scalar) data
  auto src_iter = thrust::make_transform_iterator(
    thrust::make_counting_iterator<std::size_t>(0),
    cuda::proclaim_return_type<cudf::size_type*>(
      [src = d_src_data.begin()] __device__(std::size_t i) { return src + i; }));

  // Copy buffer addresses to device and create an iterator
  auto d_dst_addrs = cudf::detail::make_device_uvector_async(
    buff_addrs, stream, cudf::get_current_device_resource_ref());
  // size_iter is simply a constant iterator of sizeof(size_type) bytes.
  auto size_iter = thrust::make_constant_iterator(sizeof(size_type));

  // Copy offsets to buffers in batched manner.
  cudf::detail::batched_memcpy_async(
    src_iter, d_dst_addrs.begin(), size_iter, offsets.size(), stream);
}

}  // namespace cudf::io::parquet::detail
