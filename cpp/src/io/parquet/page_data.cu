/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include "page_decode.cuh"

#include <io/utilities/column_buffer.hpp>

#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>

#include <rmm/exec_policy.hpp>
#include <thrust/reduce.h>

namespace cudf::io::parquet::detail {

namespace {

constexpr int decode_block_size = 128;
constexpr int rolling_buf_size  = decode_block_size * 2;

/**
 * @brief Output a string descriptor
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dstv Pointer to row output data (string descriptor or 32-bit hash)
 */
template <typename state_buf>
inline __device__ void gpuOutputString(volatile page_state_s* s,
                                       volatile state_buf* sb,
                                       int src_pos,
                                       void* dstv)
{
  auto [ptr, len] = gpuGetStringData(s, sb, src_pos);
  // make sure to only hash `BYTE_ARRAY` when specified with the output type size
  if (s->dtype_len == 4 and (s->col.data_type & 7) == BYTE_ARRAY) {
    // Output hash. This hash value is used if the option to convert strings to
    // categoricals is enabled. The seed value is chosen arbitrarily.
    uint32_t constexpr hash_seed = 33;
    cudf::string_view const sv{ptr, static_cast<size_type>(len)};
    *static_cast<uint32_t*>(dstv) =
      cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>{hash_seed}(sv);
  } else {
    // Output string descriptor
    auto* dst   = static_cast<string_index_pair*>(dstv);
    dst->first  = ptr;
    dst->second = len;
  }
}

/**
 * @brief Output a boolean
 *
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename state_buf>
inline __device__ void gpuOutputBoolean(volatile state_buf* sb, int src_pos, uint8_t* dst)
{
  *dst = sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)];
}

/**
 * @brief Store a 32-bit data element
 *
 * @param[out] dst ptr to output
 * @param[in] src8 raw input bytes
 * @param[in] dict_pos byte position in dictionary
 * @param[in] dict_size size of dictionary
 */
inline __device__ void gpuStoreOutput(uint32_t* dst,
                                      uint8_t const* src8,
                                      uint32_t dict_pos,
                                      uint32_t dict_size)
{
  uint32_t bytebuf;
  unsigned int ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos < dict_size) {
    bytebuf = *reinterpret_cast<uint32_t const*>(src8 + dict_pos);
    if (ofs) {
      uint32_t bytebufnext = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
      bytebuf              = __funnelshift_r(bytebuf, bytebufnext, ofs);
    }
  } else {
    bytebuf = 0;
  }
  *dst = bytebuf;
}

/**
 * @brief Store a 64-bit data element
 *
 * @param[out] dst ptr to output
 * @param[in] src8 raw input bytes
 * @param[in] dict_pos byte position in dictionary
 * @param[in] dict_size size of dictionary
 */
inline __device__ void gpuStoreOutput(uint2* dst,
                                      uint8_t const* src8,
                                      uint32_t dict_pos,
                                      uint32_t dict_size)
{
  uint2 v;
  unsigned int ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos < dict_size) {
    v.x = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 0);
    v.y = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
    if (ofs) {
      uint32_t next = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 8);
      v.x           = __funnelshift_r(v.x, v.y, ofs);
      v.y           = __funnelshift_r(v.y, next, ofs);
    }
  } else {
    v.x = v.y = 0;
  }
  *dst = v;
}

/**
 * @brief Convert an INT96 Spark timestamp to 64-bit timestamp
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[out] dst Pointer to row output data
 */
template <typename state_buf>
inline __device__ void gpuOutputInt96Timestamp(volatile page_state_s* s,
                                               volatile state_buf* sb,
                                               int src_pos,
                                               int64_t* dst)
{
  using cuda::std::chrono::duration_cast;

  uint8_t const* src8;
  uint32_t dict_pos, dict_size = s->dict_size, ofs;

  if (s->dict_base) {
    // Dictionary
    dict_pos =
      (s->dict_bits > 0) ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)] : 0;
    src8 = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    src8     = s->data_start;
  }
  dict_pos *= (uint32_t)s->dtype_len_in;
  ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits

  if (dict_pos + 4 >= dict_size) {
    *dst = 0;
    return;
  }

  uint3 v;
  int64_t nanos, days;
  v.x = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 0);
  v.y = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
  v.z = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 8);
  if (ofs) {
    uint32_t next = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 12);
    v.x           = __funnelshift_r(v.x, v.y, ofs);
    v.y           = __funnelshift_r(v.y, v.z, ofs);
    v.z           = __funnelshift_r(v.z, next, ofs);
  }
  nanos = v.y;
  nanos <<= 32;
  nanos |= v.x;
  // Convert from Julian day at noon to UTC seconds
  days = static_cast<int32_t>(v.z);
  cudf::duration_D d_d{
    days - 2440588};  // TBD: Should be noon instead of midnight, but this matches pyarrow

  *dst = [&]() {
    switch (s->col.ts_clock_rate) {
      case 1:  // seconds
        return duration_cast<duration_s>(d_d).count() +
               duration_cast<duration_s>(duration_ns{nanos}).count();
      case 1'000:  // milliseconds
        return duration_cast<duration_ms>(d_d).count() +
               duration_cast<duration_ms>(duration_ns{nanos}).count();
      case 1'000'000:  // microseconds
        return duration_cast<duration_us>(d_d).count() +
               duration_cast<duration_us>(duration_ns{nanos}).count();
      case 1'000'000'000:  // nanoseconds
      default: return duration_cast<cudf::duration_ns>(d_d).count() + nanos;
    }
  }();
}

/**
 * @brief Output a 64-bit timestamp
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename state_buf>
inline __device__ void gpuOutputInt64Timestamp(volatile page_state_s* s,
                                               volatile state_buf* sb,
                                               int src_pos,
                                               int64_t* dst)
{
  uint8_t const* src8;
  uint32_t dict_pos, dict_size = s->dict_size, ofs;
  int64_t ts;

  if (s->dict_base) {
    // Dictionary
    dict_pos =
      (s->dict_bits > 0) ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)] : 0;
    src8 = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    src8     = s->data_start;
  }
  dict_pos *= (uint32_t)s->dtype_len_in;
  ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos + 4 < dict_size) {
    uint2 v;
    int64_t val;
    int32_t ts_scale;
    v.x = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 0);
    v.y = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
    if (ofs) {
      uint32_t next = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 8);
      v.x           = __funnelshift_r(v.x, v.y, ofs);
      v.y           = __funnelshift_r(v.y, next, ofs);
    }
    val = v.y;
    val <<= 32;
    val |= v.x;
    // Output to desired clock rate
    ts_scale = s->ts_scale;
    if (ts_scale < 0) {
      // round towards negative infinity
      int sign = (val < 0);
      ts       = ((val + sign) / -ts_scale) + sign;
    } else {
      ts = val * ts_scale;
    }
  } else {
    ts = 0;
  }
  *dst = ts;
}

/**
 * @brief Output a byte array as int.
 *
 * @param[in] ptr Pointer to the byte array
 * @param[in] len Byte array length
 * @param[out] dst Pointer to row output data
 */
template <typename T>
__device__ void gpuOutputByteArrayAsInt(char const* ptr, int32_t len, T* dst)
{
  T unscaled = 0;
  for (auto i = 0; i < len; i++) {
    uint8_t v = ptr[i];
    unscaled  = (unscaled << 8) | v;
  }
  // Shift the unscaled value up and back down when it isn't all 8 bytes,
  // which sign extend the value for correctly representing negative numbers.
  unscaled <<= (sizeof(T) - len) * 8;
  unscaled >>= (sizeof(T) - len) * 8;
  *dst = unscaled;
}

/**
 * @brief Output a fixed-length byte array as int.
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename T, typename state_buf>
__device__ void gpuOutputFixedLenByteArrayAsInt(volatile page_state_s* s,
                                                volatile state_buf* sb,
                                                int src_pos,
                                                T* dst)
{
  uint32_t const dtype_len_in = s->dtype_len_in;
  uint8_t const* data         = s->dict_base ? s->dict_base : s->data_start;
  uint32_t const pos =
    (s->dict_base
       ? ((s->dict_bits > 0) ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)] : 0)
       : src_pos) *
    dtype_len_in;
  uint32_t const dict_size = s->dict_size;

  T unscaled = 0;
  for (unsigned int i = 0; i < dtype_len_in; i++) {
    uint32_t v = (pos + i < dict_size) ? data[pos + i] : 0;
    unscaled   = (unscaled << 8) | v;
  }
  // Shift the unscaled value up and back down when it isn't all 8 bytes,
  // which sign extend the value for correctly representing negative numbers.
  if (dtype_len_in < sizeof(T)) {
    unscaled <<= (sizeof(T) - dtype_len_in) * 8;
    unscaled >>= (sizeof(T) - dtype_len_in) * 8;
  }
  *dst = unscaled;
}

/**
 * @brief Output a small fixed-length value
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename T, typename state_buf>
inline __device__ void gpuOutputFast(volatile page_state_s* s,
                                     volatile state_buf* sb,
                                     int src_pos,
                                     T* dst)
{
  uint8_t const* dict;
  uint32_t dict_pos, dict_size = s->dict_size;

  if (s->dict_base) {
    // Dictionary
    dict_pos =
      (s->dict_bits > 0) ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)] : 0;
    dict = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    dict     = s->data_start;
  }
  dict_pos *= (uint32_t)s->dtype_len_in;
  gpuStoreOutput(dst, dict, dict_pos, dict_size);
}

/**
 * @brief Output a N-byte value
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @param[in] dst8 Pointer to row output data
 * @param[in] len Length of element
 */
template <typename state_buf>
static __device__ void gpuOutputGeneric(
  volatile page_state_s* s, volatile state_buf* sb, int src_pos, uint8_t* dst8, int len)
{
  uint8_t const* dict;
  uint32_t dict_pos, dict_size = s->dict_size;

  if (s->dict_base) {
    // Dictionary
    dict_pos =
      (s->dict_bits > 0) ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)] : 0;
    dict = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    dict     = s->data_start;
  }
  dict_pos *= (uint32_t)s->dtype_len_in;
  if (len & 3) {
    // Generic slow path
    for (unsigned int i = 0; i < len; i++) {
      dst8[i] = (dict_pos + i < dict_size) ? dict[dict_pos + i] : 0;
    }
  } else {
    // Copy 4 bytes at a time
    uint8_t const* src8 = dict;
    unsigned int ofs    = 3 & reinterpret_cast<size_t>(src8);
    src8 -= ofs;  // align to 32-bit boundary
    ofs <<= 3;    // bytes -> bits
    for (unsigned int i = 0; i < len; i += 4) {
      uint32_t bytebuf;
      if (dict_pos < dict_size) {
        bytebuf = *reinterpret_cast<uint32_t const*>(src8 + dict_pos);
        if (ofs) {
          uint32_t bytebufnext = *reinterpret_cast<uint32_t const*>(src8 + dict_pos + 4);
          bytebuf              = __funnelshift_r(bytebuf, bytebufnext, ofs);
        }
      } else {
        bytebuf = 0;
      }
      dict_pos += 4;
      *reinterpret_cast<uint32_t*>(dst8 + i) = bytebuf;
    }
  }
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
__global__ void __launch_bounds__(decode_block_size)
  gpuDecodePageData(PageInfo* pages,
                    device_span<ColumnChunkDesc const> chunks,
                    size_t min_row,
                    size_t num_rows,
                    int32_t* error_code)
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
                          true)) {
    return;
  }

  bool const has_repetition = s->col.max_level[level_type::REPETITION] > 0;

  if (s->dict_base) {
    out_thread0 = (s->dict_bits > 0) ? 64 : 32;
  } else {
    switch (s->col.data_type & 7) {
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
    // TODO(ets): see if this sync can be removed
    __syncthreads();
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
      if (s->dict_base) {
        src_target_pos = gpuDecodeDictionaryIndices<false>(s, sb, src_target_pos, t & 0x1f).first;
      } else if ((s->col.data_type & 7) == BOOLEAN) {
        src_target_pos = gpuDecodeRleBooleans(s, sb, src_target_pos, t & 0x1f);
      } else if ((s->col.data_type & 7) == BYTE_ARRAY or
                 (s->col.data_type & 7) == FIXED_LEN_BYTE_ARRAY) {
        gpuInitStringDescriptors<false>(s, sb, src_target_pos, t & 0x1f);
      }
      if (t == 32) { *(volatile int32_t*)&s->dict_pos = src_target_pos; }
    } else {
      // WARP1..WARP3: Decode values
      int const dtype = s->col.data_type & 7;
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

      if (t == out_thread0) { *(volatile int32_t*)&s->src_pos = target_pos; }
    }
    __syncthreads();
  }
  if (t == 0 and s->error != 0) { set_error(s->error, error_code); }
}

struct mask_tform {
  __device__ uint32_t operator()(PageInfo const& p) { return static_cast<uint32_t>(p.kernel_mask); }
};

}  // anonymous namespace

uint32_t GetAggregatedDecodeKernelMask(cudf::detail::hostdevice_vector<PageInfo>& pages,
                                       rmm::cuda_stream_view stream)
{
  // determine which kernels to invoke
  auto mask_iter = thrust::make_transform_iterator(pages.d_begin(), mask_tform{});
  return thrust::reduce(
    rmm::exec_policy(stream), mask_iter, mask_iter + pages.size(), 0U, thrust::bit_or<uint32_t>{});
}

/**
 * @copydoc cudf::io::parquet::detail::DecodePageData
 */
void __host__ DecodePageData(cudf::detail::hostdevice_vector<PageInfo>& pages,
                             cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                             size_t num_rows,
                             size_t min_row,
                             int level_type_size,
                             int32_t* error_code,
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

}  // namespace cudf::io::parquet::detail
