/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cub/cub.cuh>
#include <io/utilities/block_utils.cuh>
#include "orc_common.h"
#include "orc_gpu.h"

// Apache ORC reader does not handle zero-length patch lists for RLEv2 mode2
// Workaround replaces zero-length patch lists by a dummy zero patch
#define ZERO_PLL_WAR 1

namespace cudf {
namespace io {
namespace orc {
namespace gpu {
#define SCRATCH_BFRSZ (512 * 4)

static __device__ __constant__ int64_t kORCTimeToUTC =
  1420070400;  // Seconds from January 1st, 1970 to January 1st, 2015

struct byterle_enc_state_s {
  uint32_t literal_run;
  uint32_t repeat_run;
  volatile uint32_t rpt_map[(512 / 32) + 1];
};

struct intrle_enc_state_s {
  uint32_t literal_run;
  uint32_t delta_run;
  uint32_t literal_mode;
  uint32_t literal_w;
  uint32_t hdr_bytes;
  uint32_t pl_bytes;
  volatile uint32_t delta_map[(512 / 32) + 1];
  volatile union {
    uint32_t u32[(512 / 32) * 2];
    uint64_t u64[(512 / 32) * 2];
  } scratch;
};

struct strdata_enc_state_s {
  uint32_t char_count;
  uint32_t lengths_red[(512 / 32)];
  const char *str_data[512];
};

struct orcenc_state_s {
  uint32_t cur_row;       // Current row in group
  uint32_t present_rows;  // # of rows in present buffer
  uint32_t present_out;   // # of rows in present buffer that have been flushed
  uint32_t nrows;         // # of rows in current batch
  uint32_t numvals;       // # of non-zero values in current batch (<=nrows)
  uint32_t numlengths;    // # of non-zero values in DATA2 batch
  uint32_t nnz;           // Running count of non-null values
  EncChunk chunk;
  uint32_t strm_pos[CI_NUM_STREAMS];
  uint8_t valid_buf[512];  // valid map bits
  union {
    byterle_enc_state_s byterle;
    intrle_enc_state_s intrle;
    strdata_enc_state_s strenc;
    StripeDictionary dict_stripe;
  } u;
  union {
    uint8_t u8[SCRATCH_BFRSZ];  // general scratch buffer
    uint32_t u32[SCRATCH_BFRSZ / 4];
  } buf;
  union {
    uint8_t u8[2048];
    uint32_t u32[1024];
    int32_t i32[1024];
    uint64_t u64[1024];
    int64_t i64[1024];
  } vals;
  union {
    uint8_t u8[2048];
    uint32_t u32[1024];
  } lengths;
};

static inline __device__ uint32_t zigzag32(int32_t v)
{
  int32_t s = (v >> 31);
  return ((v ^ s) * 2) - s;
}
static inline __device__ uint64_t zigzag64(int64_t v)
{
  int64_t s = (v < 0) ? 1 : 0;
  return ((v ^ -s) * 2) + s;
}
static inline __device__ uint32_t CountLeadingBytes32(uint32_t v) { return __clz(v) >> 3; }
static inline __device__ uint32_t CountLeadingBytes64(uint64_t v) { return __clzll(v) >> 3; }

/**
 * @brief Raw data output
 *
 * @param[in] cid stream type (strm_pos[cid] will be updated and output stored at
 *streams[cid]+strm_pos[cid])
 * @param[in] inmask input buffer position mask for circular buffers
 * @param[in] s encoder state
 * @param[in] inbuf base input buffer
 * @param[in] inpos position in input buffer
 * @param[in] count number of bytes to encode
 * @param[in] t thread id
 *
 **/
template <StreamIndexType cid, uint32_t inmask>
static __device__ void StoreBytes(
  orcenc_state_s *s, const uint8_t *inbuf, uint32_t inpos, uint32_t count, int t)
{
  uint8_t *dst = s->chunk.streams[cid] + s->strm_pos[cid];
  while (count > 0) {
    uint32_t n = min(count, 512);
    if (t < n) { dst[t] = inbuf[(inpos + t) & inmask]; }
    dst += n;
    inpos += n;
    count -= n;
  }
  __syncthreads();
  if (!t) { s->strm_pos[cid] = static_cast<uint32_t>(dst - s->chunk.streams[cid]); }
}

/**
 * @brief ByteRLE encoder
 *
 * @param[in] cid stream type (strm_pos[cid] will be updated and output stored at
 *streams[cid]+strm_pos[cid])
 * @param[in] s encoder state
 * @param[in] inbuf base input buffer
 * @param[in] inpos position in input buffer
 * @param[in] inmask input buffer position mask for circular buffers
 * @param[in] numvals max number of values to encode
 * @param[in] flush encode all remaining values if nonzero
 * @param[in] t thread id
 *
 * @return number of input values encoded
 *
 **/
template <StreamIndexType cid, uint32_t inmask>
static __device__ uint32_t ByteRLE(
  orcenc_state_s *s, const uint8_t *inbuf, uint32_t inpos, uint32_t numvals, uint32_t flush, int t)
{
  uint8_t *dst     = s->chunk.streams[cid] + s->strm_pos[cid];
  uint32_t out_cnt = 0;

  while (numvals > 0) {
    uint8_t v0       = (t < numvals) ? inbuf[(inpos + t) & inmask] : 0;
    uint8_t v1       = (t + 1 < numvals) ? inbuf[(inpos + t + 1) & inmask] : 0;
    uint32_t rpt_map = BALLOT(t + 1 < numvals && v0 == v1), literal_run, repeat_run,
             maxvals = min(numvals, 512);
    if (!(t & 0x1f)) s->u.byterle.rpt_map[t >> 5] = rpt_map;
    __syncthreads();
    if (t == 0) {
      // Find the start of an identical 3-byte sequence
      // TBD: The two loops below could be eliminated using more ballot+ffs using warp0
      literal_run = 0;
      repeat_run  = 0;
      while (literal_run < maxvals) {
        uint32_t next = s->u.byterle.rpt_map[(literal_run >> 5) + 1];
        uint32_t mask = rpt_map & __funnelshift_r(rpt_map, next, 1);
        if (mask) {
          uint32_t literal_run_ofs = __ffs(mask) - 1;
          literal_run += literal_run_ofs;
          repeat_run = __ffs(~((rpt_map >> literal_run_ofs) >> 1));
          if (repeat_run + literal_run_ofs == 32) {
            while (next == ~0) {
              uint32_t next_idx = ((literal_run + repeat_run) >> 5) + 1;
              next              = (next_idx < 512 / 32) ? s->u.byterle.rpt_map[next_idx] : 0;
              repeat_run += 32;
            }
            repeat_run += __ffs(~next) - 1;
          }
          repeat_run = min(repeat_run + 1, maxvals - min(literal_run, maxvals));
          if (repeat_run < 3) {
            literal_run += (flush && literal_run + repeat_run >= numvals) ? repeat_run : 0;
            repeat_run = 0;
          }
          break;
        }
        rpt_map = next;
        literal_run += 32;
      }
      if (repeat_run >= 130) {
        // Limit large runs to multiples of 130
        repeat_run = (repeat_run >= 3 * 130) ? 3 * 130 : (repeat_run >= 2 * 130) ? 2 * 130 : 130;
      } else if (literal_run && literal_run + repeat_run == maxvals) {
        repeat_run = 0;  // Try again at next iteration
      }
      s->u.byterle.repeat_run  = repeat_run;
      s->u.byterle.literal_run = min(literal_run, maxvals);
    }
    __syncthreads();
    literal_run = s->u.byterle.literal_run;
    if (!flush && literal_run == numvals) {
      literal_run &= ~0x7f;
      if (!literal_run) break;
    }
    if (literal_run > 0) {
      uint32_t num_runs = (literal_run + 0x7f) >> 7;
      if (t < literal_run) {
        uint32_t run_id = t >> 7;
        uint32_t run    = min(literal_run - run_id * 128, 128);
        if (!(t & 0x7f)) dst[run_id + t] = 0x100 - run;
        dst[run_id + t + 1] = (cid == CI_PRESENT) ? __brev(v0) >> 24 : v0;
      }
      dst += num_runs + literal_run;
      out_cnt += literal_run;
      numvals -= literal_run;
      inpos += literal_run;
    }
    repeat_run = s->u.byterle.repeat_run;
    if (repeat_run > 0) {
      while (repeat_run >= 130) {
        if (t == literal_run)  // repeat_run follows literal_run
        {
          dst[0] = 0x7f;
          dst[1] = (cid == CI_PRESENT) ? __brev(v0) >> 24 : v0;
        }
        dst += 2;
        out_cnt += 130;
        numvals -= 130;
        inpos += 130;
        repeat_run -= 130;
      }
      if (!flush && repeat_run == numvals) {
        // Wait for more data in case we can continue the run later
        break;
      }
      if (repeat_run >= 3) {
        if (t == literal_run)  // repeat_run follows literal_run
        {
          dst[0] = repeat_run - 3;
          dst[1] = (cid == CI_PRESENT) ? __brev(v0) >> 24 : v0;
        }
        dst += 2;
        out_cnt += repeat_run;
        numvals -= repeat_run;
        inpos += repeat_run;
      }
    }
  }
  if (!t) { s->strm_pos[cid] = static_cast<uint32_t>(dst - s->chunk.streams[cid]); }
  __syncthreads();
  return out_cnt;
}

/**
 * @brief Maps the symbol size in bytes to RLEv2 5-bit length code
 **/
static const __device__ __constant__ uint8_t kByteLengthToRLEv2_W[9] = {
  0, 7, 15, 23, 27, 28, 29, 30, 31};

/**
 * @brief Encode a varint value, return the number of bytes written
 **/
static inline __device__ uint32_t StoreVarint(uint8_t *dst, uint64_t v)
{
  uint32_t bytecnt = 0;
  for (;;) {
    uint32_t c = (uint32_t)(v & 0x7f);
    v >>= 7u;
    if (v == 0) {
      dst[bytecnt++] = c;
      break;
    } else {
      dst[bytecnt++] = c + 0x80;
    }
  }
  return bytecnt;
}

static inline __device__ void intrle_minmax(int64_t &vmin, int64_t &vmax)
{
  vmin = INT64_MIN;
  vmax = INT64_MAX;
}
// static inline __device__ void intrle_minmax(uint64_t &vmin, uint64_t &vmax) { vmin = UINT64_C(0);
// vmax = UINT64_MAX; }
static inline __device__ void intrle_minmax(int32_t &vmin, int32_t &vmax)
{
  vmin = INT32_MIN;
  vmax = INT32_MAX;
}
static inline __device__ void intrle_minmax(uint32_t &vmin, uint32_t &vmax)
{
  vmin = UINT32_C(0);
  vmax = UINT32_MAX;
}

template <class T>
static inline __device__ void StoreBytesBigEndian(uint8_t *dst, T v, uint32_t w)
{
  for (uint32_t i = 0, b = w * 8; i < w; ++i) {
    b -= 8;
    dst[i] = static_cast<uint8_t>(v >> b);
  }
}

// Combine and store bits for symbol widths less than 8
static inline __device__ void StoreBitsBigEndian(
  uint8_t *dst, uint32_t v, uint32_t w, int num_vals, int t)
{
  if (t <= (num_vals | 0x1f)) {
    uint32_t mask;
    if (w <= 1) {
      v    = (v << 1) | (SHFL_XOR(v, 1) & 0x1);
      v    = (v << 2) | (SHFL_XOR(v, 2) & 0x3);
      v    = (v << 4) | (SHFL_XOR(v, 4) & 0xf);
      mask = 0x7;
    } else if (w <= 2) {
      v    = (v << 2) | (SHFL_XOR(v, 1) & 0x3);
      v    = (v << 4) | (SHFL_XOR(v, 2) & 0xf);
      mask = 0x3;
    } else  // if (w <= 4)
    {
      v    = (v << 4) | (SHFL_XOR(v, 1) & 0xf);
      mask = 0x1;
    }
    if (t < num_vals && !(t & mask)) { dst[(t * w) >> 3] = static_cast<uint8_t>(v); }
  }
}

/**
 * @brief Integer RLEv2 encoder
 *
 * @param[in] cid stream type (strm_pos[cid] will be updated and output stored at
 *streams[cid]+strm_pos[cid])
 * @param[in] s encoder state
 * @param[in] inbuf base input buffer
 * @param[in] inpos position in input buffer
 * @param[in] inmask input buffer position mask for circular buffers
 * @param[in] numvals max number of values to encode
 * @param[in] flush encode all remaining values if nonzero
 * @param[in] t thread id
 *
 * @return number of input values encoded
 *
 **/
template <StreamIndexType cid,
          class T,
          bool is_signed,
          uint32_t inmask,
          typename FullStorage,
          typename HalfStorage>
static __device__ uint32_t IntegerRLE(orcenc_state_s *s,
                                      const T *inbuf,
                                      uint32_t inpos,
                                      uint32_t numvals,
                                      uint32_t flush,
                                      int t,
                                      FullStorage &temp_storage_full,
                                      HalfStorage &temp_storage_half)
{
  using warp_reduce      = cub::WarpReduce<T>;
  using half_warp_reduce = cub::WarpReduce<T, 16>;
  uint8_t *dst           = s->chunk.streams[cid] + s->strm_pos[cid];
  uint32_t out_cnt       = 0;

  while (numvals > 0) {
    T v0               = (t < numvals) ? inbuf[(inpos + t) & inmask] : 0;
    T v1               = (t + 1 < numvals) ? inbuf[(inpos + t + 1) & inmask] : 0;
    T v2               = (t + 2 < numvals) ? inbuf[(inpos + t + 2) & inmask] : 0;
    uint32_t delta_map = BALLOT(t + 2 < numvals && v1 - v0 == v2 - v1), maxvals = min(numvals, 512),
             literal_run, delta_run;
    if (!(t & 0x1f)) s->u.intrle.delta_map[t >> 5] = delta_map;
    __syncthreads();
    if (!t) {
      // Find the start of the next delta run (2 consecutive values with the same delta)
      literal_run = delta_run = 0;
      while (literal_run < maxvals) {
        if (delta_map != 0) {
          uint32_t literal_run_ofs = __ffs(delta_map) - 1;
          literal_run += literal_run_ofs;
          delta_run = __ffs(~((delta_map >> literal_run_ofs) >> 1));
          if (literal_run_ofs + delta_run == 32) {
            for (;;) {
              uint32_t delta_idx = (literal_run + delta_run) >> 5;
              delta_map          = (delta_idx < 512 / 32) ? s->u.intrle.delta_map[delta_idx] : 0;
              if (delta_map != ~0) break;
              delta_run += 32;
            }
            delta_run += __ffs(~delta_map) - 1;
          }
          delta_run += 2;
          break;
        }
        literal_run += 32;
        delta_map = s->u.intrle.delta_map[(literal_run >> 5)];
      }
      literal_run             = min(literal_run, maxvals);
      s->u.intrle.literal_run = literal_run;
      s->u.intrle.delta_run   = min(delta_run, maxvals - literal_run);
    }
    __syncthreads();
    literal_run = s->u.intrle.literal_run;
    // Find minimum and maximum values
    if (literal_run > 0) {
      // Find min & max
      T vmin, vmax;
      uint32_t literal_mode, literal_w;
      if (t < literal_run) {
        vmin = vmax = v0;
      } else {
        intrle_minmax(vmax, vmin);
      }
      vmin = warp_reduce(temp_storage_full[t / 32]).Reduce(vmin, cub::Min());
      __syncwarp();
      vmax = warp_reduce(temp_storage_full[t / 32]).Reduce(vmax, cub::Max());
      __syncwarp();
      if (!(t & 0x1f)) {
        s->u.intrle.scratch.u64[(t >> 5) * 2 + 0] = vmin;
        s->u.intrle.scratch.u64[(t >> 5) * 2 + 1] = vmax;
      }
      __syncthreads();
      if (t < 32) {
        vmin = (T)s->u.intrle.scratch.u64[(t & 0xf) * 2 + 0];
        vmax = (T)s->u.intrle.scratch.u64[(t & 0xf) * 2 + 1];
        vmin = half_warp_reduce(temp_storage_half[t / 32]).Reduce(vmin, cub::Min());
        __syncwarp();
        vmax = half_warp_reduce(temp_storage_half[t / 32]).Reduce(vmax, cub::Max());
        __syncwarp();
        if (t == 0) {
          uint32_t mode1_w, mode2_w;
          T vrange_mode1, vrange_mode2;
          s->u.intrle.scratch.u64[0] = (uint64_t)vmin;
          if (sizeof(T) > 4) {
            vrange_mode1 = (is_signed) ? max(zigzag64(vmin), zigzag64(vmax)) : vmax;
            vrange_mode2 = vmax - vmin;
            mode1_w      = 8 - min(CountLeadingBytes64(vrange_mode1), 7);
            mode2_w      = 8 - min(CountLeadingBytes64(vrange_mode2), 7);
          } else {
            vrange_mode1 = (is_signed) ? max(zigzag32(vmin), zigzag32(vmax)) : vmax;
            vrange_mode2 = vmax - vmin;
            mode1_w      = 4 - min(CountLeadingBytes32(vrange_mode1), 3);
            mode2_w      = 4 - min(CountLeadingBytes32(vrange_mode2), 3);
          }
          // Decide between mode1 & mode2 (also mode3 for length=2 repeat)
          if (vrange_mode2 == 0 && mode1_w > 1) {
            // Should only occur if literal_run==2 (otherwise would have resulted in repeat_run >=
            // 3)
            uint32_t bytecnt = 2;
            dst[0]           = 0xC0 + ((literal_run - 1) >> 8);
            dst[1]           = (literal_run - 1) & 0xff;
            bytecnt += StoreVarint(dst + 2, vrange_mode1);
            dst[bytecnt++]           = 0;  // Zero delta
            s->u.intrle.literal_mode = 3;
            s->u.intrle.literal_w    = bytecnt;
          } else {
            uint32_t range, w;
            if (mode1_w > mode2_w && (literal_run - 1) * (mode1_w - mode2_w) > 4) {
              s->u.intrle.literal_mode = 2;
              w                        = mode2_w;
              range                    = (uint32_t)vrange_mode2;
            } else {
              s->u.intrle.literal_mode = 1;
              w                        = mode1_w;
              range                    = (uint32_t)vrange_mode1;
            }
            if (w == 1)
              w = (range >= 16) ? w << 3 : (range >= 4) ? 4 : (range >= 2) ? 2 : 1;
            else
              w <<= 3;  // bytes -> bits
            s->u.intrle.literal_w = w;
          }
        }
      }
      __syncthreads();
      vmin         = (T)s->u.intrle.scratch.u64[0];
      literal_mode = s->u.intrle.literal_mode;
      literal_w    = s->u.intrle.literal_w;
      if (literal_mode == 1) {
        // Direct mode
        if (!t) {
          dst[0] = 0x40 +
                   ((literal_w < 8) ? literal_w - 1 : kByteLengthToRLEv2_W[literal_w >> 3]) * 2 +
                   ((literal_run - 1) >> 8);
          dst[1] = (literal_run - 1) & 0xff;
        }
        dst += 2;
        if (t < literal_run && is_signed) {
          if (sizeof(T) > 4)
            v0 = zigzag64(v0);
          else
            v0 = zigzag32(v0);
        }
        if (literal_w < 8)
          StoreBitsBigEndian(dst, (uint32_t)v0, literal_w, literal_run, t);
        else if (t < literal_run)
          StoreBytesBigEndian(dst + t * (literal_w >> 3), v0, (literal_w >> 3));
      } else if (literal_mode == 2) {
        // Patched base mode
        if (!t) {
          uint32_t bw, pw = 1, pll, pgw = 1, bv_scale = (is_signed) ? 0 : 1;
          vmax = (is_signed) ? ((vmin < 0) ? -vmin : vmin) * 2 : vmin;
          bw   = (sizeof(T) > 4) ? (8 - min(CountLeadingBytes64(vmax << bv_scale), 7))
                               : (4 - min(CountLeadingBytes32(vmax << bv_scale), 3));
#if ZERO_PLL_WAR
          // Insert a dummy zero patch
          pll                                                    = 1;
          dst[4 + bw + ((literal_run * literal_w + 7) >> 3) + 0] = 0;
          dst[4 + bw + ((literal_run * literal_w + 7) >> 3) + 1] = 0;
#else
          pll = 0;
#endif
          dst[0] = 0x80 +
                   ((literal_w < 8) ? literal_w - 1 : kByteLengthToRLEv2_W[literal_w >> 3]) * 2 +
                   ((literal_run - 1) >> 8);
          dst[1] = (literal_run - 1) & 0xff;
          dst[2] = ((bw - 1) << 5) | kByteLengthToRLEv2_W[pw];
          dst[3] = ((pgw - 1) << 5) | pll;
          if (is_signed) {
            vmax >>= 1;
            vmax |= vmin & ((T)1 << (bw * 8 - 1));
          }
          StoreBytesBigEndian(dst + 4, vmax, bw);
          s->u.intrle.hdr_bytes = 4 + bw;
          s->u.intrle.pl_bytes  = (pll * (pw * 8 + pgw) + 7) >> 3;
        }
        __syncthreads();
        dst += s->u.intrle.hdr_bytes;
        v0 -= (t < literal_run) ? vmin : 0;
        if (literal_w < 8)
          StoreBitsBigEndian(dst, (uint32_t)v0, literal_w, literal_run, t);
        else if (t < literal_run)
          StoreBytesBigEndian(dst + t * (literal_w >> 3), v0, (literal_w >> 3));
        dst += s->u.intrle.pl_bytes;
      } else {
        // Delta mode
        dst += literal_w;
        literal_w = 0;
      }
      dst += (literal_run * literal_w + 7) >> 3;
      numvals -= literal_run;
      inpos += literal_run;
      out_cnt += literal_run;
      __syncthreads();
    }
    delta_run = s->u.intrle.delta_run;
    if (delta_run > 0) {
      if (t == literal_run) {
        int64_t delta       = (int64_t)v1 - (int64_t)v0;
        uint64_t delta_base = (is_signed) ? (sizeof(T) > 4) ? zigzag64(v0) : zigzag32(v0) : v0;
        if (delta == 0 && delta_run >= 3 && delta_run <= 10) {
          // Short repeat
          uint32_t delta_bw = 8 - min(CountLeadingBytes64(delta_base), 7);
          dst[0]            = ((delta_bw - 1) << 3) + (delta_run - 3);
          for (uint32_t i = 0, b = delta_bw * 8; i < delta_bw; i++) {
            b -= 8;
            dst[1 + i] = static_cast<uint8_t>(delta_base >> b);
          }
          s->u.intrle.hdr_bytes = 1 + delta_bw;
        } else {
          // Delta
          uint64_t delta_u = zigzag64(delta);
          uint32_t bytecnt = 2;
          dst[0]           = 0xC0 + ((delta_run - 1) >> 8);
          dst[1]           = (delta_run - 1) & 0xff;
          bytecnt += StoreVarint(dst + bytecnt, delta_base);
          bytecnt += StoreVarint(dst + bytecnt, delta_u);
          s->u.intrle.hdr_bytes = bytecnt;
        }
      }
      __syncthreads();
      dst += s->u.intrle.hdr_bytes;
      numvals -= delta_run;
      inpos += delta_run;
      out_cnt += delta_run;
    }
  }
  if (!t) { s->strm_pos[cid] = static_cast<uint32_t>(dst - s->chunk.streams[cid]); }
  __syncthreads();
  return out_cnt;
}

/**
 * @brief Store a group of strings as a single concatenated string
 *
 * @param[in] dst destination buffer
 * @param[in] strenc string encoder state
 * @param[in] len(t) string length (per thread)
 * @param[in] t thread id
 *
 **/
static __device__ void StoreStringData(uint8_t *dst,
                                       strdata_enc_state_s *strenc,
                                       uint32_t len,
                                       int t)
{
  // Start with summing up all the lengths
  uint32_t pos = len;
  uint32_t wt  = t & 0x1f;
  for (uint32_t n = 1; n < 32; n <<= 1) {
    uint32_t tmp = SHFL(pos, (wt & ~n) | (n - 1));
    pos += (wt & n) ? tmp : 0;
  }
  if (wt == 0x1f) { strenc->lengths_red[t >> 5] = pos; }
  dst += pos - len;
  __syncthreads();
  if (t < 32) {
    uint32_t wlen = (wt < 16) ? strenc->lengths_red[wt] : 0;
    uint32_t wpos = wlen;
    for (uint32_t n = 1; n < 16; n <<= 1) {
      uint32_t tmp = SHFL(wpos, (wt & ~n) | (n - 1));
      wpos += (wt & n) ? tmp : 0;
    }
    if (wt < 16) { strenc->lengths_red[wt] = wpos - wlen; }
    if (wt == 0xf) {
      strenc->char_count = wpos;  // Update stream position
    }
  }
  __syncthreads();
  // TBD: Might be more efficient to loop over 4 strings and copy 8 consecutive character at a time
  // rather than have each thread to a memcpy
  if (len > 0) { memcpy(dst + strenc->lengths_red[t >> 5], strenc->str_data[t], len); }
}

/**
 * @brief In-place conversion from lengths to positions
 *
 * @param[in] vals input values
 * @param[in] numvals number of values
 * @param[in] t thread id
 *
 **/
template <class T>
inline __device__ void lengths_to_positions(volatile T *vals, uint32_t numvals, unsigned int t)
{
  for (uint32_t n = 1; n < numvals; n <<= 1) {
    __syncthreads();
    if ((t & n) && (t < numvals)) vals[t] += vals[(t & ~n) | (n - 1)];
  }
}

/**
 * @brief Timestamp scale table (powers of 10)
 **/
static const __device__ __constant__ int32_t kTimeScale[10] = {
  1000000000, 100000000, 10000000, 1000000, 100000, 10000, 1000, 100, 10, 1};

/**
 * @brief Encode column data
 *
 * @param[in] chunks EncChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_rowgroups Number of row groups
 *
 **/
// blockDim {512,1,1}
template <int block_size>
__global__ void __launch_bounds__(block_size)
  gpuEncodeOrcColumnData(EncChunk *chunks, uint32_t num_columns, uint32_t num_rowgroups)
{
  __shared__ __align__(16) orcenc_state_s state_g;
  __shared__ union {
    typename cub::WarpReduce<int32_t>::TempStorage full_i32[block_size / 32];
    typename cub::WarpReduce<int64_t>::TempStorage full_i64[block_size / 32];
    typename cub::WarpReduce<uint32_t>::TempStorage full_u32[block_size / 32];
    typename cub::WarpReduce<int32_t, 16>::TempStorage half_i32[block_size / 32];
    typename cub::WarpReduce<int64_t, 16>::TempStorage half_i64[block_size / 32];
    typename cub::WarpReduce<uint32_t, 16>::TempStorage half_u32[block_size / 32];
  } temp_storage;

  orcenc_state_s *const s = &state_g;
  uint32_t col_id         = blockIdx.x;
  uint32_t group_id       = blockIdx.y;
  int t                   = threadIdx.x;

  if (t < sizeof(EncChunk) / sizeof(uint32_t)) {
    ((volatile uint32_t *)&s->chunk)[t] =
      ((const uint32_t *)&chunks[group_id * num_columns + col_id])[t];
  }
  if (t < CI_NUM_STREAMS) { s->strm_pos[t] = 0; }
  __syncthreads();
  if (!t) {
    s->cur_row      = 0;
    s->present_rows = 0;
    s->present_out  = 0;
    s->numvals      = 0;
    s->numlengths   = 0;
    s->nnz          = 0;
    // Dictionary data is encoded in a separate kernel
    if (s->chunk.encoding_kind == DICTIONARY_V2) {
      s->strm_pos[CI_DATA2]      = s->chunk.strm_len[CI_DATA2];
      s->strm_pos[CI_DICTIONARY] = s->chunk.strm_len[CI_DICTIONARY];
    }
  }
  __syncthreads();
  while (s->cur_row < s->chunk.num_rows || s->numvals + s->numlengths != 0) {
    // Encode valid map
    if (s->present_rows < s->chunk.num_rows) {
      uint32_t present_rows = s->present_rows;
      uint32_t nrows        = min(s->chunk.num_rows - present_rows,
                           512 * 8 - (present_rows - (min(s->cur_row, s->present_out) & ~7)));
      uint32_t nrows_out;
      if (t * 8 < nrows) {
        uint32_t row  = s->chunk.start_row + present_rows + t * 8;
        uint8_t valid = 0;
        if (row < s->chunk.valid_rows) {
          const uint8_t *valid_map_base =
            reinterpret_cast<const uint8_t *>(s->chunk.valid_map_base);
          valid = (valid_map_base) ? valid_map_base[row >> 3] : 0xff;
          if (row + 7 > s->chunk.valid_rows) {
            valid = valid & ((1 << (s->chunk.valid_rows & 7)) - 1);
          }
        }
        s->valid_buf[(row >> 3) & 0x1ff] = valid;
      }
      __syncthreads();
      present_rows += nrows;
      if (!t) { s->present_rows = present_rows; }
      // RLE encode the present stream
      nrows_out =
        present_rows -
        s->present_out;  // Should always be a multiple of 8 except at the end of the last row group
      if (nrows_out > ((present_rows < s->chunk.num_rows) ? 130 * 8 : 0)) {
        uint32_t present_out = s->present_out;
        if (s->chunk.strm_id[CI_PRESENT] >= 0) {
          uint32_t flush = (present_rows < s->chunk.num_rows) ? 0 : 7;
          nrows_out      = (nrows_out + flush) >> 3;
          nrows_out =
            ByteRLE<CI_PRESENT, 0x1ff>(
              s, s->valid_buf, (s->chunk.start_row + present_out) >> 3, nrows_out, flush, t) *
            8;
        }
        __syncthreads();
        if (!t) { s->present_out = min(present_out + nrows_out, present_rows); }
      }
      __syncthreads();
    }
    // Fetch non-null values
    if (!s->chunk.streams[CI_DATA]) {
      // Pass-through
      __syncthreads();
      if (!t) {
        s->cur_row           = s->present_rows;
        s->strm_pos[CI_DATA] = s->cur_row * s->chunk.dtype_len;
      }
      __syncthreads();
    } else if (s->cur_row < s->present_rows) {
      uint32_t maxnumvals = (s->chunk.type_kind == BOOLEAN) ? 2048 : 1024;
      uint32_t nrows =
        min(min(s->present_rows - s->cur_row, maxnumvals - max(s->numvals, s->numlengths)), 512);
      uint32_t row   = s->chunk.start_row + s->cur_row + t;
      uint32_t valid = (t < nrows) ? (s->valid_buf[(row >> 3) & 0x1ff] >> (row & 7)) & 1 : 0;
      s->buf.u32[t]  = valid;

      // TODO: Could use a faster reduction relying on _popc() for the initial phase
      lengths_to_positions(s->buf.u32, 512, t);
      __syncthreads();
      if (valid) {
        int nz_idx       = (s->nnz + s->buf.u32[t] - 1) & (maxnumvals - 1);
        void const *base = s->chunk.column_data_base;
        switch (s->chunk.type_kind) {
          case INT:
          case DATE:
          case FLOAT: s->vals.u32[nz_idx] = static_cast<const uint32_t *>(base)[row]; break;
          case DOUBLE:
          case LONG: s->vals.u64[nz_idx] = static_cast<const uint64_t *>(base)[row]; break;
          case SHORT: s->vals.u32[nz_idx] = static_cast<const uint16_t *>(base)[row]; break;
          case BOOLEAN:
          case BYTE: s->vals.u8[nz_idx] = static_cast<const uint8_t *>(base)[row]; break;
          case TIMESTAMP: {
            int64_t ts       = static_cast<const int64_t *>(base)[row];
            int32_t ts_scale = kTimeScale[min(s->chunk.scale, 9)];
            int64_t seconds  = ts / ts_scale;
            int32_t nanos    = (ts - seconds * ts_scale);
            // There is a bug in the ORC spec such that for negative timestamps, it is understood
            // between the writer and reader that nanos will be adjusted to their positive component
            // but the negative seconds will be left alone. This means that -2.6 is encoded as
            // seconds = -2 and nanos = 1+(-0.6) = 0.4
            // This leads to an error in decoding time where -1 < time (s) < 0
            // Details: https://github.com/rapidsai/cudf/pull/5529#issuecomment-648768925
            if (nanos < 0) { nanos += ts_scale; }
            s->vals.i64[nz_idx] = seconds - kORCTimeToUTC;
            if (nanos != 0) {
              // Trailing zeroes are encoded in the lower 3-bits
              uint32_t zeroes = 0;
              nanos *= kTimeScale[9 - min(s->chunk.scale, 9)];
              if (!(nanos % 100)) {
                nanos /= 100;
                zeroes = 1;
                while (zeroes < 7 && !(nanos % 10)) {
                  nanos /= 10;
                  zeroes++;
                }
              }
              nanos = (nanos << 3) + zeroes;
            }
            s->lengths.u32[nz_idx] = nanos;
            break;
          }
          case STRING:
            if (s->chunk.encoding_kind == DICTIONARY_V2) {
              uint32_t dict_idx = static_cast<const uint32_t *>(base)[row];
              if (dict_idx > 0x7fffffffu)
                dict_idx = static_cast<const uint32_t *>(base)[dict_idx & 0x7fffffffu];
              s->vals.u32[nz_idx] = dict_idx;
            } else {
              const nvstrdesc_s *str_desc = static_cast<const nvstrdesc_s *>(base) + row;
              const char *ptr             = str_desc->ptr;
              uint32_t count              = static_cast<uint32_t>(str_desc->count);
              s->u.strenc.str_data[s->buf.u32[t] - 1] = ptr;
              s->lengths.u32[nz_idx]                  = count;
            }
            break;
          default: break;
        }
      }
      __syncthreads();
      if (s->chunk.type_kind == STRING && s->chunk.encoding_kind != DICTIONARY_V2) {
        // Store string data
        uint32_t nz     = s->buf.u32[511];
        uint32_t nz_idx = (s->nnz + t) & 0x3ff;
        uint32_t len    = (t < nz && s->u.strenc.str_data[t]) ? s->lengths.u32[nz_idx] : 0;
        StoreStringData(s->chunk.streams[CI_DATA] + s->strm_pos[CI_DATA], &s->u.strenc, len, t);
        if (!t) { s->strm_pos[CI_DATA] += s->u.strenc.char_count; }
        __syncthreads();
      } else if (s->chunk.type_kind == BOOLEAN) {
        // bool8 -> 8x bool1
        uint32_t nz = s->buf.u32[511];
        uint8_t n   = ((s->nnz + nz) - (s->nnz & ~7) + 7) >> 3;
        if (t < n) {
          uint32_t idx8                              = (s->nnz & ~7) + (t << 3);
          s->lengths.u8[((s->nnz >> 3) + t) & 0x1ff] = ((s->vals.u8[(idx8 + 0) & 0x7ff] & 1) << 7) |
                                                       ((s->vals.u8[(idx8 + 1) & 0x7ff] & 1) << 6) |
                                                       ((s->vals.u8[(idx8 + 2) & 0x7ff] & 1) << 5) |
                                                       ((s->vals.u8[(idx8 + 3) & 0x7ff] & 1) << 4) |
                                                       ((s->vals.u8[(idx8 + 4) & 0x7ff] & 1) << 3) |
                                                       ((s->vals.u8[(idx8 + 5) & 0x7ff] & 1) << 2) |
                                                       ((s->vals.u8[(idx8 + 6) & 0x7ff] & 1) << 1) |
                                                       ((s->vals.u8[(idx8 + 7) & 0x7ff] & 1) << 0);
        }
        __syncthreads();
      }
      if (!t) {
        uint32_t nz = s->buf.u32[511];
        s->nnz += nz;
        s->numvals += nz;
        s->numlengths += (s->chunk.type_kind == TIMESTAMP ||
                          (s->chunk.type_kind == STRING && s->chunk.encoding_kind != DICTIONARY_V2))
                           ? nz
                           : 0;
        s->cur_row += nrows;
      }
      __syncthreads();
      // Encode values
      if (s->numvals > 0) {
        uint32_t flush = (s->cur_row == s->chunk.num_rows) ? 7 : 0, n;
        switch (s->chunk.type_kind) {
          case SHORT:
          case INT:
          case DATE:
            n = IntegerRLE<CI_DATA, int32_t, true, 0x3ff>(s,
                                                          s->vals.i32,
                                                          s->nnz - s->numvals,
                                                          s->numvals,
                                                          flush,
                                                          t,
                                                          temp_storage.full_i32,
                                                          temp_storage.half_i32);
            break;
          case LONG:
          case TIMESTAMP:
            n = IntegerRLE<CI_DATA, int64_t, true, 0x3ff>(s,
                                                          s->vals.i64,
                                                          s->nnz - s->numvals,
                                                          s->numvals,
                                                          flush,
                                                          t,
                                                          temp_storage.full_i64,
                                                          temp_storage.half_i64);
            break;
          case BYTE:
            n = ByteRLE<CI_DATA, 0x3ff>(s, s->vals.u8, s->nnz - s->numvals, s->numvals, flush, t);
            break;
          case BOOLEAN:
            n = ByteRLE<CI_DATA, 0x1ff>(s,
                                        s->lengths.u8,
                                        (s->nnz - s->numvals + flush) >> 3,
                                        (s->numvals + flush) >> 3,
                                        flush,
                                        t) *
                8;
            break;
          case FLOAT:
            StoreBytes<CI_DATA, 0xfff>(s, s->vals.u8, (s->nnz - s->numvals) * 4, s->numvals * 4, t);
            n = s->numvals;
            break;
          case DOUBLE:
            StoreBytes<CI_DATA, 0x1fff>(
              s, s->vals.u8, (s->nnz - s->numvals) * 8, s->numvals * 8, t);
            n = s->numvals;
            break;
          case STRING:
            if (s->chunk.encoding_kind == DICTIONARY_V2) {
              n = IntegerRLE<CI_DATA, uint32_t, false, 0x3ff>(s,
                                                              s->vals.u32,
                                                              s->nnz - s->numvals,
                                                              s->numvals,
                                                              flush,
                                                              t,
                                                              temp_storage.full_u32,
                                                              temp_storage.half_u32);
            } else {
              n = s->numvals;
            }
            break;
          default: n = s->numvals; break;
        }
        __syncthreads();
        if (!t) { s->numvals -= min(n, s->numvals); }
      }
      // Encode secondary stream values
      if (s->numlengths > 0) {
        uint32_t flush = (s->cur_row == s->chunk.num_rows) ? 1 : 0, n;
        switch (s->chunk.type_kind) {
          case TIMESTAMP:
          case STRING:
            n = IntegerRLE<CI_DATA2, uint32_t, false, 0x3ff>(s,
                                                             s->lengths.u32,
                                                             s->nnz - s->numlengths,
                                                             s->numlengths,
                                                             flush,
                                                             t,
                                                             temp_storage.full_u32,
                                                             temp_storage.half_u32);
            break;
          default: n = s->numlengths; break;
        }
        __syncthreads();
        if (!t) { s->numlengths -= min(n, s->numlengths); }
      }
    }
    __syncthreads();
  }
  __syncthreads();
  if (t <= CI_PRESENT && s->chunk.strm_id[t] >= 0) {
    // Update actual compressed length
    chunks[group_id * num_columns + col_id].strm_len[t] = s->strm_pos[t];
    if (!s->chunk.streams[t]) {
      chunks[group_id * num_columns + col_id].streams[t] =
        static_cast<uint8_t *>(const_cast<void *>(s->chunk.column_data_base)) +
        s->chunk.start_row * s->chunk.dtype_len;
    }
  }
}

/**
 * @brief Encode column dictionaries
 *
 * @param[in] stripes Stripe dictionaries device array [stripe][string_column]
 * @param[in] chunks EncChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 *
 **/
// blockDim {512,1,1}
template <int block_size>
__global__ void __launch_bounds__(block_size)
  gpuEncodeStringDictionaries(StripeDictionary *stripes, EncChunk *chunks, uint32_t num_columns)
{
  __shared__ __align__(16) orcenc_state_s state_g;
  __shared__ union {
    typename cub::WarpReduce<uint32_t>::TempStorage full_u32[block_size / 32];
    typename cub::WarpReduce<uint32_t, 16>::TempStorage half_u32[block_size / 32];
  } temp_storage;

  orcenc_state_s *const s = &state_g;
  uint32_t stripe_id      = blockIdx.x;
  uint32_t cid            = (blockIdx.y) ? CI_DICTIONARY : CI_DATA2;
  uint32_t chunk_id;
  int t = threadIdx.x;
  const nvstrdesc_s *str_desc;
  const uint32_t *dict_data;

  if (t < sizeof(StripeDictionary) / sizeof(uint32_t)) {
    ((volatile uint32_t *)&s->u.dict_stripe)[t] = ((const uint32_t *)&stripes[stripe_id])[t];
  }
  __syncthreads();
  chunk_id = s->u.dict_stripe.start_chunk * num_columns + s->u.dict_stripe.column_id;
  if (t < sizeof(EncChunk) / sizeof(uint32_t)) {
    ((volatile uint32_t *)&s->chunk)[t] = ((const uint32_t *)&chunks[chunk_id])[t];
  }
  if (t == 0) {
    s->strm_pos[cid] = 0;
    s->numlengths    = 0;
    s->nrows         = s->u.dict_stripe.num_strings;
    s->cur_row       = 0;
  }
  str_desc  = static_cast<const nvstrdesc_s *>(s->u.dict_stripe.column_data_base);
  dict_data = s->u.dict_stripe.dict_data;
  __syncthreads();
  if (s->chunk.encoding_kind != DICTIONARY_V2) {
    return;  // This column isn't using dictionary encoding -> bail out
  }

  while (s->cur_row < s->nrows || s->numlengths != 0) {
    uint32_t numvals    = min(s->nrows - s->cur_row, min(1024 - s->numlengths, 512));
    uint32_t string_idx = (t < numvals) ? dict_data[s->cur_row + t] : 0;
    if (cid == CI_DICTIONARY) {
      // Encoding string contents
      const char *ptr = (t < numvals) ? str_desc[string_idx].ptr : 0;
      uint32_t count  = (t < numvals) ? static_cast<uint32_t>(str_desc[string_idx].count) : 0;
      s->u.strenc.str_data[t] = ptr;
      StoreStringData(s->chunk.streams[CI_DICTIONARY] + s->strm_pos[CI_DICTIONARY],
                      &s->u.strenc,
                      (ptr) ? count : 0,
                      t);
      if (!t) { s->strm_pos[CI_DICTIONARY] += s->u.strenc.char_count; }
    } else {
      // Encoding string lengths
      uint32_t count  = (t < numvals) ? static_cast<uint32_t>(str_desc[string_idx].count) : 0;
      uint32_t nz_idx = (s->cur_row + t) & 0x3ff;
      if (t < numvals) s->lengths.u32[nz_idx] = count;
      __syncthreads();
      if (s->numlengths + numvals > 0) {
        uint32_t flush = (s->cur_row + numvals == s->nrows) ? 1 : 0;
        uint32_t n     = IntegerRLE<CI_DATA2, uint32_t, false, 0x3ff>(s,
                                                                  s->lengths.u32,
                                                                  s->cur_row,
                                                                  s->numlengths + numvals,
                                                                  flush,
                                                                  t,
                                                                  temp_storage.full_u32,
                                                                  temp_storage.half_u32);
        __syncthreads();
        if (!t) {
          s->numlengths += numvals;
          s->numlengths -= min(n, s->numlengths);
        }
      }
    }
    if (t == 0) { s->cur_row += numvals; }
    __syncthreads();
  }
  if (t == 0) { chunks[chunk_id].strm_len[cid] = s->strm_pos[cid]; }
}

/**
 * @brief Merge chunked column data into a single contiguous stream
 *
 * @param[in] strm_desc StripeStream device array [stripe][stream]
 * @param[in] chunks EncChunk device array [rowgroup][column]
 * @param[in] num_stripe_streams Total number of streams
 * @param[in] num_columns Number of columns
 *
 **/
// blockDim {1024,1,1}
__global__ void __launch_bounds__(1024)
  gpuCompactOrcDataStreams(StripeStream *strm_desc, EncChunk *chunks, uint32_t num_columns)
{
  __shared__ __align__(16) StripeStream ss;
  __shared__ __align__(16) EncChunk ck0;
  __shared__ uint8_t *volatile ck_curptr_g;
  __shared__ uint32_t volatile ck_curlen_g;

  uint32_t strm_id = blockIdx.x;
  uint32_t ck0_id, cid;
  uint32_t t = threadIdx.x;
  uint8_t *dst_ptr;

  if (t < sizeof(StripeStream) / sizeof(uint32_t)) {
    ((volatile uint32_t *)&ss)[t] = ((const uint32_t *)&strm_desc[strm_id])[t];
  }
  __syncthreads();
  ck0_id = ss.first_chunk_id;
  if (t < sizeof(EncChunk) / sizeof(uint32_t)) {
    ((volatile uint32_t *)&ck0)[t] = ((const uint32_t *)&chunks[ck0_id])[t];
  }
  __syncthreads();
  cid     = ss.stream_type;
  dst_ptr = ck0.streams[cid] + ck0.strm_len[cid];
  for (uint32_t g = 1; g < ss.num_chunks; g++) {
    uint8_t *src_ptr;
    uint32_t len;
    if (t == 0) {
      src_ptr = chunks[ck0_id + g * num_columns].streams[cid];
      len     = chunks[ck0_id + g * num_columns].strm_len[cid];
      if (src_ptr != dst_ptr) { chunks[ck0_id + g * num_columns].streams[cid] = dst_ptr; }
      ck_curptr_g = src_ptr;
      ck_curlen_g = len;
    }
    __syncthreads();
    src_ptr = ck_curptr_g;
    len     = ck_curlen_g;
    if (len > 0 && src_ptr != dst_ptr) {
      for (uint32_t i = 0; i < len; i += 1024) {
        uint8_t v = (i + t < len) ? src_ptr[i + t] : 0;
        __syncthreads();
        if (i + t < len) { dst_ptr[i + t] = v; }
      }
    }
    dst_ptr += len;
    __syncthreads();
  }
  if (!t) { strm_desc[strm_id].stream_size = dst_ptr - ck0.streams[cid]; }
}

/**
 * @brief Initializes compression input/output structures
 *
 * @param[in] strm_desc StripeStream device array [stripe][stream]
 * @param[in] chunks EncChunk device array [rowgroup][column]
 * @param[out] comp_in Per-block compression input parameters
 * @param[out] comp_out Per-block compression status
 * @param[in] compressed_bfr Compression output buffer
 * @param[in] comp_blk_size Compression block size
 *
 **/
// blockDim {256,1,1}
__global__ void __launch_bounds__(256) gpuInitCompressionBlocks(StripeStream *strm_desc,
                                                                EncChunk *chunks,
                                                                gpu_inflate_input_s *comp_in,
                                                                gpu_inflate_status_s *comp_out,
                                                                uint8_t *compressed_bfr,
                                                                uint32_t comp_blk_size)
{
  __shared__ __align__(16) StripeStream ss;
  __shared__ uint8_t *volatile uncomp_base_g;

  uint32_t strm_id = blockIdx.x;
  uint32_t t       = threadIdx.x;
  uint32_t num_blocks;
  uint8_t *src, *dst;

  if (t < sizeof(StripeStream) / sizeof(uint32_t)) {
    ((volatile uint32_t *)&ss)[t] = ((const uint32_t *)&strm_desc[strm_id])[t];
  }
  __syncthreads();
  if (t == 0) { uncomp_base_g = chunks[ss.first_chunk_id].streams[ss.stream_type]; }
  __syncthreads();
  src        = uncomp_base_g;
  dst        = compressed_bfr + ss.bfr_offset;
  num_blocks = (ss.stream_size > 0) ? (ss.stream_size - 1) / comp_blk_size + 1 : 1;
  for (uint32_t b = t; b < num_blocks; b += 256) {
    gpu_inflate_input_s *blk_in   = &comp_in[ss.first_block + b];
    gpu_inflate_status_s *blk_out = &comp_out[ss.first_block + b];
    uint32_t blk_size = min(comp_blk_size, ss.stream_size - min(b * comp_blk_size, ss.stream_size));
    blk_in->srcDevice = src + b * comp_blk_size;
    blk_in->srcSize   = blk_size;
    blk_in->dstDevice = dst + b * (3 + comp_blk_size) + 3;  // reserve 3 bytes for block header
    blk_in->dstSize   = blk_size;
    blk_out->bytes_written = blk_size;
    blk_out->status        = 1;
    blk_out->reserved      = 0;
  }
}

/**
 * @brief Compacts compressed blocks in a single contiguous stream, and update 3-byte block length
 *fields
 *
 * @param[in,out] strm_desc StripeStream device array [stripe][stream]
 * @param[in] chunks EncChunk device array [rowgroup][column]
 * @param[in] comp_in Per-block compression input parameters
 * @param[in] comp_out Per-block compression status
 * @param[in] compressed_bfr Compression output buffer
 * @param[in] comp_blk_size Compression block size
 *
 **/
// blockDim {1024,1,1}
__global__ void __launch_bounds__(1024) gpuCompactCompressedBlocks(StripeStream *strm_desc,
                                                                   gpu_inflate_input_s *comp_in,
                                                                   gpu_inflate_status_s *comp_out,
                                                                   uint8_t *compressed_bfr,
                                                                   uint32_t comp_blk_size)
{
  __shared__ __align__(16) StripeStream ss;
  __shared__ const uint8_t *volatile comp_src_g;
  __shared__ uint32_t volatile comp_len_g;

  uint32_t strm_id = blockIdx.x;
  uint32_t t       = threadIdx.x;
  uint32_t num_blocks, b, blk_size;
  const uint8_t *src;
  uint8_t *dst;

  if (t < sizeof(StripeStream) / sizeof(uint32_t)) {
    ((volatile uint32_t *)&ss)[t] = ((const uint32_t *)&strm_desc[strm_id])[t];
  }
  __syncthreads();
  num_blocks = (ss.stream_size > 0) ? (ss.stream_size - 1) / comp_blk_size + 1 : 0;
  dst        = compressed_bfr + ss.bfr_offset;
  b          = 0;
  do {
    if (t == 0) {
      gpu_inflate_input_s *blk_in   = &comp_in[ss.first_block + b];
      gpu_inflate_status_s *blk_out = &comp_out[ss.first_block + b];
      uint32_t src_len =
        min(comp_blk_size, ss.stream_size - min(b * comp_blk_size, ss.stream_size));
      uint32_t dst_len = (blk_out->status == 0) ? blk_out->bytes_written : src_len;
      uint32_t blk_size24;
      if (dst_len >= src_len) {
        // Copy from uncompressed source
        src                    = static_cast<const uint8_t *>(blk_in->srcDevice);
        blk_out->bytes_written = src_len;
        dst_len                = src_len;
        blk_size24             = dst_len * 2 + 1;
      } else {
        // Compressed block
        src        = static_cast<const uint8_t *>(blk_in->dstDevice);
        blk_size24 = dst_len * 2 + 0;
      }
      dst[0]     = static_cast<uint8_t>(blk_size24 >> 0);
      dst[1]     = static_cast<uint8_t>(blk_size24 >> 8);
      dst[2]     = static_cast<uint8_t>(blk_size24 >> 16);
      comp_src_g = src;
      comp_len_g = dst_len;
    }
    __syncthreads();
    src      = comp_src_g;
    blk_size = comp_len_g;
    dst += 3;  // skip over length written by thread0
    if (src != dst) {
      for (uint32_t i = 0; i < blk_size; i += 1024) {
        uint8_t v = (i + t < blk_size) ? src[i + t] : 0;
        __syncthreads();
        if (i + t < blk_size) { dst[i + t] = v; }
      }
    }
    dst += blk_size;
    __syncthreads();
  } while (++b < num_blocks);
  // Update stripe stream with the compressed size
  if (t == 0) {
    strm_desc[strm_id].stream_size = static_cast<uint32_t>(dst - (compressed_bfr + ss.bfr_offset));
  }
}

/**
 * @brief Launches kernel for encoding column data
 *
 * @param[in] chunks EncChunk device array [rowgroup][column]
 * @param[in] num_columns Number of columns
 * @param[in] num_rowgroups Number of row groups
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t EncodeOrcColumnData(EncChunk *chunks,
                                uint32_t num_columns,
                                uint32_t num_rowgroups,
                                cudaStream_t stream)
{
  dim3 dim_block(512, 1);  // 512 threads per chunk
  dim3 dim_grid(num_columns, num_rowgroups);
  gpuEncodeOrcColumnData<512>
    <<<dim_grid, dim_block, 0, stream>>>(chunks, num_columns, num_rowgroups);
  return cudaSuccess;
}

/**
 * @brief Launches kernel for encoding column dictionaries
 *
 * @param[in] stripes Stripe dictionaries device array [stripe][string_column]
 * @param[in] chunks EncChunk device array [rowgroup][column]
 * @param[in] num_string_columns Number of string columns
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t EncodeStripeDictionaries(StripeDictionary *stripes,
                                     EncChunk *chunks,
                                     uint32_t num_string_columns,
                                     uint32_t num_columns,
                                     uint32_t num_stripes,
                                     cudaStream_t stream)
{
  dim3 dim_block(512, 1);  // 512 threads per dictionary
  dim3 dim_grid(num_string_columns * num_stripes, 2);
  gpuEncodeStringDictionaries<512>
    <<<dim_grid, dim_block, 0, stream>>>(stripes, chunks, num_columns);
  return cudaSuccess;
}

/**
 * @brief Launches kernel for compacting chunked column data prior to compression
 *
 * @param[in] strm_desc StripeStream device array [stripe][stream]
 * @param[in] chunks EncChunk device array [rowgroup][column]
 * @param[in] num_stripe_streams Total number of streams
 * @param[in] num_columns Number of columns
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t CompactOrcDataStreams(StripeStream *strm_desc,
                                  EncChunk *chunks,
                                  uint32_t num_stripe_streams,
                                  uint32_t num_columns,
                                  cudaStream_t stream)
{
  dim3 dim_block(1024, 1);
  dim3 dim_grid(num_stripe_streams, 1);
  gpuCompactOrcDataStreams<<<dim_grid, dim_block, 0, stream>>>(strm_desc, chunks, num_columns);
  return cudaSuccess;
}

/**
 * @brief Launches kernel(s) for compressing data streams
 *
 * @param[in] compressed_data Output compressed blocks
 * @param[in] strm_desc StripeStream device array [stripe][stream]
 * @param[in] chunks EncChunk device array [rowgroup][column]
 * @param[out] comp_in Per-block compression input parameters
 * @param[out] comp_out Per-block compression status
 * @param[in] num_stripe_streams Total number of streams
 * @param[in] num_compressed_blocks Total number of compressed blocks
 * @param[in] compression Type of compression
 * @param[in] comp_blk_size Compression block size
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t CompressOrcDataStreams(uint8_t *compressed_data,
                                   StripeStream *strm_desc,
                                   EncChunk *chunks,
                                   gpu_inflate_input_s *comp_in,
                                   gpu_inflate_status_s *comp_out,
                                   uint32_t num_stripe_streams,
                                   uint32_t num_compressed_blocks,
                                   CompressionKind compression,
                                   uint32_t comp_blk_size,
                                   cudaStream_t stream)
{
  dim3 dim_block_init(256, 1);
  dim3 dim_grid(num_stripe_streams, 1);
  gpuInitCompressionBlocks<<<dim_grid, dim_block_init, 0, stream>>>(
    strm_desc, chunks, comp_in, comp_out, compressed_data, comp_blk_size);
  if (compression == SNAPPY) { gpu_snap(comp_in, comp_out, num_compressed_blocks, stream); }
  dim3 dim_block_compact(1024, 1);
  gpuCompactCompressedBlocks<<<dim_grid, dim_block_compact, 0, stream>>>(
    strm_desc, comp_in, comp_out, compressed_data, comp_blk_size);
  return cudaSuccess;
}

}  // namespace gpu
}  // namespace orc
}  // namespace io
}  // namespace cudf
