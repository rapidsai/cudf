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

#include "orc_common.h"
#include "orc_gpu.h"

#if (__CUDACC_VER_MAJOR__ >= 9)
#define SHFL0(v)        __shfl_sync(~0, v, 0)
#define SHFL(v, t)      __shfl_sync(~0, v, t)
#define SHFL_XOR(v, m)  __shfl_xor_sync(~0, v, m)
#define SYNCWARP()      __syncwarp()
#define BALLOT(v)       __ballot_sync(~0, v)
#else
#define SHFL0(v)        __shfl(v, 0)
#define SHFL(v, t)      __shfl(v, t)
#define SHFL_XOR(v, m)  __shfl_xor(v, m)
#define SYNCWARP()
#define BALLOT(v)       __ballot(v)
#endif

#define LOG2_BYTESTREAM_BFRSZ   13  // Must be able to handle 512x 8-byte values

#define BYTESTREAM_BFRSZ        (1 << LOG2_BYTESTREAM_BFRSZ)
#define BYTESTREAM_BFRMASK32    ((BYTESTREAM_BFRSZ-1) >> 2)
#define LOG2_NWARPS             5   // Log2 of number of warps per threadblock
#define LOG2_NTHREADS           (LOG2_NWARPS+5)
#define NWARPS                  (1 << LOG2_NWARPS)
#define NTHREADS                (1 << LOG2_NTHREADS)
#define ROWDEC_BFRSZ            (NTHREADS + 128)    // Add some margin to look ahead to future rows in case there are many zeroes

#define IS_RLEv1(encoding_mode)         ((encoding_mode) < DIRECT_V2)
#define IS_RLEv2(encoding_mode)         ((encoding_mode) >= DIRECT_V2)
#define IS_DICTIONARY(encoding_mode)    ((encoding_mode) & 1)

namespace orc { namespace gpu {

static __device__ __constant__ int64_t kORCTimeToUTC = 1420099200; // Seconds from January 1st, 1970 to January 1st, 2015

struct orc_bytestream_s
{
    const uint8_t *base;
    uint32_t pos;
    uint32_t len;
    uint32_t fill_pos;
    uint32_t fill_count;
    union {
        uint8_t u8[BYTESTREAM_BFRSZ];
        uint32_t u32[BYTESTREAM_BFRSZ >> 2];
        uint2 u64[BYTESTREAM_BFRSZ >> 3];
    } buf;
};

struct orc_rlev1_state_s
{
    uint32_t num_runs;
    uint32_t num_vals;
    int32_t run_data[NWARPS*12];    // (delta << 24) | (count << 16) | (first_val)
};

struct orc_rlev2_state_s
{
    uint32_t num_runs;
    uint32_t num_vals;
    union {
        uint32_t u32[NWARPS];
        uint64_t u64[NWARPS];
    } baseval;
    uint16_t m2_pw_byte3[NWARPS];
    union {
        int32_t i32[NWARPS];
        int64_t i64[NWARPS];
    } delta;
    uint16_t runs_loc[NTHREADS];
};

struct orc_byterle_state_s
{
    uint32_t num_runs;
    uint32_t num_vals;
    uint32_t runs_loc[NWARPS];
    uint32_t runs_pos[NWARPS];
};

struct orc_rowdec_state_s
{
    uint32_t nz_count;
    uint32_t last_row[NWARPS];
    uint32_t row[ROWDEC_BFRSZ];     // 0=skip, >0: row position relative to cur_row
};

struct orc_strdict_state_s
{
    uint2 *local_dict;
    uint32_t dict_pos;
    uint32_t dict_len;
};

struct orc_nulldec_state_s
{
    uint32_t row;
    uint32_t null_count[NWARPS];
};

struct orc_datadec_state_s
{
    uint32_t cur_row;                   // starting row of current batch
    uint32_t end_row;                   // ending row of this chunk (start_row + num_rows)
    uint32_t max_vals;                  // max # of non-zero values to decode in this batch
    uint32_t nrows;                     // # of rows in current batch (up to NTHREADS)
    uint32_t buffered_count;            // number of buffered values in the secondary data stream
    uint32_t tz_num_entries;            // number of entries in timezone table
    uint32_t tz_dst_cycle;              // number of entries in timezone daylight savings cycle
    int64_t first_tz_transition;        // first transition in timezone table
    int64_t last_tz_transition;         // last transition in timezone table
};


struct orcdec_state_s
{
    ColumnDesc chunk;
    orc_bytestream_s bs;
    orc_bytestream_s bs2;
    union {
        orc_strdict_state_s dict;
        orc_nulldec_state_s nulls;
        orc_datadec_state_s data;
    } top;
    union {
        orc_rlev1_state_s rlev1;
        orc_rlev2_state_s rlev2;
        orc_byterle_state_s rle8;
        orc_rowdec_state_s rowdec;
    } u;
    union {
        uint8_t u8[NTHREADS * 8];
        uint32_t u32[NTHREADS * 2];
        int32_t i32[NTHREADS * 2];
        uint64_t u64[NTHREADS];
        int64_t i64[NTHREADS];
    } vals;
};


/**
 * @brief Initializes byte stream, modifying length and start position to keep the read pointer 8-byte aligned
 *        Assumes that the address range [start_address & ~7, (start_address + len - 1) | 7] is valid
 *
 * @param[in] bs Byte stream input
 * @param[in] base Pointer to raw byte stream data
 * @param[in] len Stream length in bytes
 *
 **/
static __device__ void bytestream_init(volatile orc_bytestream_s *bs, const uint8_t *base, uint32_t len)
{
    uint32_t pos = static_cast<uint32_t>(7 & reinterpret_cast<size_t>(base));
    bs->base = base - pos;
    bs->pos = (len > 0) ? pos : 0;
    bs->len = (len + pos + 7) & ~7;
    bs->fill_pos = 0;
    bs->fill_count = min(bs->len, BYTESTREAM_BFRSZ) >> 3;
}

/**
 * @brief Increment the read position, returns number of 64-bit slots to fill
 *
 * @param[in] bs Byte stream input
 * @param[in] bytes_consumed Number of bytes that were consumed
 *
 **/
static __device__ void bytestream_flush_bytes(volatile orc_bytestream_s *bs, uint32_t bytes_consumed)
{
    uint32_t pos = bs->pos;
    uint32_t len = bs->len;
    uint32_t pos_new = min(pos + bytes_consumed, len);
    bs->pos = pos_new;
    pos = min(pos + BYTESTREAM_BFRSZ, len);
    pos_new = min(pos_new + BYTESTREAM_BFRSZ, len);
    bs->fill_pos = pos;
    bs->fill_count = (pos_new >> 3) - (pos >> 3);
}

/**
 * @brief Refill the byte stream buffer
 *
 * @param[in] bs Byte stream input
 * @param[in] t thread id
 *
 **/
static __device__ void bytestream_fill(orc_bytestream_s *bs, int t)
{
    int count = bs->fill_count;
    if (t < count)
    {
        int pos8 = (bs->fill_pos >> 3) + t;
        bs->buf.u64[pos8 & ((BYTESTREAM_BFRSZ >> 3) - 1)] = (reinterpret_cast<const uint2 *>(bs->base))[pos8];
    }
}

/**
 * @brief Read a byte from the byte stream (byte aligned)
 *
 * @param[in] bs Byte stream input
 * @param[in] pos Position in byte stream
 * @return byte
 *
 **/
inline __device__ uint8_t bytestream_readbyte(volatile orc_bytestream_s *bs, int pos)
{
    return bs->buf.u8[pos & (BYTESTREAM_BFRSZ - 1)];
}

/**
 * @brief Read 32 bits from a byte stream (little endian, byte aligned)
 *
 * @param[in] bs Byte stream input
 * @param[in] pos Position in byte stream
 * @result bits
 *
 **/
inline __device__ uint32_t bytestream_readu32(volatile orc_bytestream_s *bs, int pos)
{
    uint32_t a = bs->buf.u32[(pos & (BYTESTREAM_BFRSZ - 1)) >> 2];
    uint32_t b = bs->buf.u32[((pos + 4) & (BYTESTREAM_BFRSZ - 1)) >> 2];
    return __funnelshift_r(a, b, (pos & 3) * 8);
}

/**
 * @brief Read 64 bits from a byte stream (little endian, byte aligned)
 *
 * @param[in] bs Byte stream input
 * @param[in] pos Position in byte stream
 * @param[in] numbits number of bits
 * @return bits
 *
 **/
inline __device__ uint64_t bytestream_readu64(volatile orc_bytestream_s *bs, int pos)
{
    uint32_t a = bs->buf.u32[(pos & (BYTESTREAM_BFRSZ - 1)) >> 2];
    uint32_t b = bs->buf.u32[((pos + 4) & (BYTESTREAM_BFRSZ - 1)) >> 2];
    uint32_t c = bs->buf.u32[((pos + 8) & (BYTESTREAM_BFRSZ - 1)) >> 2];
    uint32_t lo32 = __funnelshift_r(a, b, (pos & 3) * 8);
    uint32_t hi32 = __funnelshift_r(b, c, (pos & 3) * 8);
    uint64_t v = hi32;
    v <<= 32;
    v |= lo32;
    return v;
}

/**
 * @brief Read up to 32-bits from a byte stream (big endian)
 *
 * @param[in] bs Byte stream input
 * @param[in] bitpos Position in byte stream
 * @param[in] numbits number of bits
 * @return decoded value
 *
 **/
inline __device__ uint32_t bytestream_readbits(volatile orc_bytestream_s *bs, int bitpos, uint32_t numbits)
{
    int idx = bitpos >> 5;
    uint32_t a = __byte_perm(bs->buf.u32[(idx + 0) & BYTESTREAM_BFRMASK32], 0, 0x0123);
    uint32_t b = __byte_perm(bs->buf.u32[(idx + 1) & BYTESTREAM_BFRMASK32], 0, 0x0123);
    return __funnelshift_l(b, a, bitpos & 0x1f) >> (32 - numbits);
}

/**
 * @brief Read up to 64-bits from a byte stream (big endian)
 *
 * @param[in] bs Byte stream input
 * @param[in] bitpos Position in byte stream
 * @param[in] numbits number of bits
 * @return decoded value
 *
 **/
inline __device__ uint64_t bytestream_readbits64(volatile orc_bytestream_s *bs, int bitpos, uint32_t numbits)
{
    int idx = bitpos >> 5;
    uint32_t a = __byte_perm(bs->buf.u32[(idx + 0) & BYTESTREAM_BFRMASK32], 0, 0x0123);
    uint32_t b = __byte_perm(bs->buf.u32[(idx + 1) & BYTESTREAM_BFRMASK32], 0, 0x0123);
    uint32_t c = __byte_perm(bs->buf.u32[(idx + 2) & BYTESTREAM_BFRMASK32], 0, 0x0123);
    uint32_t hi32 = __funnelshift_l(b, a, bitpos & 0x1f);
    uint32_t lo32 = __funnelshift_l(c, b, bitpos & 0x1f);
    uint64_t v = hi32;
    v <<= 32;
    v |= lo32;
    v >>= (64 - numbits);
    return v;
}

/**
 * @brief Decode a big-endian unsigned 32-bit value
 *
 * @param[in] bs Byte stream input
 * @param[in] bitpos Position in byte stream
 * @param[in] numbits number of bits
 * @param[out] result decoded value
 *
 **/
inline __device__ void bytestream_readbe(volatile orc_bytestream_s *bs, int bitpos, uint32_t numbits, uint32_t &result)
{
    result = bytestream_readbits(bs, bitpos, numbits);
}

/**
 * @brief Decode a big-endian signed 32-bit value
 *
 * @param[in] bs Byte stream input
 * @param[in] bitpos Position in byte stream
 * @param[in] numbits number of bits
 * @param[out] result decoded value
 *
 **/
inline __device__ void bytestream_readbe(volatile orc_bytestream_s *bs, int bitpos, uint32_t numbits, int32_t &result)
{
    uint32_t u = bytestream_readbits(bs, bitpos, numbits);
    result = (int32_t)((u >> 1u) ^ -(int32_t)(u & 1));
}

/**
 * @brief Decode a big-endian unsigned 64-bit value
 *
 * @param[in] bs Byte stream input
 * @param[in] bitpos Position in byte stream
 * @param[in] numbits number of bits
 * @param[out] result decoded value
 *
 **/
inline __device__ void bytestream_readbe(volatile orc_bytestream_s *bs, int bitpos, uint32_t numbits, uint64_t &result)
{
    result = bytestream_readbits64(bs, bitpos, numbits);
}

/**
 * @brief Decode a big-endian signed 64-bit value
 *
 * @param[in] bs Byte stream input
 * @param[in] bitpos Position in byte stream
 * @param[in] numbits number of bits
 * @param[out] result decoded value
 *
 **/
inline __device__ void bytestream_readbe(volatile orc_bytestream_s *bs, int bitpos, uint32_t numbits, int64_t &result)
{
    uint64_t u = bytestream_readbits64(bs, bitpos, numbits);
    result = (int64_t)((u >> 1u) ^ -(int64_t)(u & 1));
}


/**
 * @brief Return the length of a base-128 varint
 *
 * @param[in] bs Byte stream input
 * @param[in] pos Position in circular byte stream buffer
 * @return length of varint in bytes
 **/
template <class T>
inline __device__ uint32_t varint_length(volatile orc_bytestream_s *bs, int pos)
{
    if (bytestream_readbyte(bs, pos) > 0x7f)
    {
        uint32_t next32 = bytestream_readu32(bs, pos + 1);
        uint32_t zbit = __ffs((~next32) & 0x80808080);
        if (sizeof(T) <= 4 || zbit)
        {
            return 1 + (zbit >> 3); // up to 5x7 bits
        }
        else
        {
            next32 = bytestream_readu32(bs, pos + 5);
            zbit = __ffs((~next32) & 0x80808080);
            if (zbit)
            {
                return 5 + (zbit >> 3); // up to 9x7 bits
            }
            else if ((sizeof(T) <= 8) || (bytestream_readbyte(bs, pos + 9) <= 0x7f))
            {
                return 10;  // up to 70 bits
            }
            else
            {
                uint64_t next64 = bytestream_readu64(bs, pos + 10);
                zbit = __ffsll((~next64) & 0x8080808080808080ull);
                if (zbit)
                {
                    return 10 + (zbit >> 3); // Up to 18x7 bits (126)
                }
                else
                {
                    return 19; // Up to 19x7 bits (133)
                }
            }
        }
    }
    else
    {
        return 1;
    }
}

/**
 * @brief Decodes a base-128 varint
 *
 * @param[in] bs Byte stream input
 * @param[in] pos Position in circular byte stream buffer
 * @param[in] result Unpacked value
 * @return new position in byte stream buffer
 **/
template <class T>
inline __device__ int decode_base128_varint(volatile orc_bytestream_s *bs, int pos, T &result)
{
    uint32_t v = bytestream_readbyte(bs, pos++);
    if (v > 0x7f)
    {
        uint32_t b = bytestream_readbyte(bs, pos++);
        v = (v & 0x7f) | (b << 7);
        if (b > 0x7f)
        {
            b = bytestream_readbyte(bs, pos++);
            v = (v & 0x3fff) | (b << 14);
            if (b > 0x7f)
            {
                b = bytestream_readbyte(bs, pos++);
                v = (v & 0x1fffff) | (b << 21);
                if (b > 0x7f)
                {
                    b = bytestream_readbyte(bs, pos++);
                    v = (v & 0x0fffffff) | (b << 28);
                    if (sizeof(T) > 4)
                    {
                        uint32_t lo = v;
                        uint64_t hi;
                        v = b >> 4;
                        if (b > 0x7f)
                        {
                            b = bytestream_readbyte(bs, pos++);
                            v = (v & 7) | (b << 3);
                            if (b > 0x7f)
                            {
                                b = bytestream_readbyte(bs, pos++);
                                v = (v & 0x3ff) | (b << 10);
                                if (b > 0x7f)
                                {
                                    b = bytestream_readbyte(bs, pos++);
                                    v = (v & 0x1ffff) | (b << 17);
                                    if (b > 0x7f)
                                    {
                                        b = bytestream_readbyte(bs, pos++);
                                        v = (v & 0xffffff) | (b << 24);
                                        if (b > 0x7f)
                                        {
                                            pos++; // last bit is redundant (extra byte implies bit63 is 1)
                                        }
                                    }
                                }
                            }
                        }
                        hi = v;
                        hi <<= 32;
                        result = hi | lo;
                        return pos;
                    }
                }
            }
        }
    }
    result = v;
    return pos;
}


/**
 * @brief Decodes an unsigned 32-bit varint
 **/
inline __device__ int decode_varint(volatile orc_bytestream_s *bs, int pos, uint32_t &result)
{
    uint32_t u;
    pos = decode_base128_varint<uint32_t>(bs, pos, u);
    result = u;
    return pos;
}

/**
 * @brief Decodes an unsigned 64-bit varint
 **/
inline __device__ int decode_varint(volatile orc_bytestream_s *bs, int pos, uint64_t &result)
{
    uint64_t u;
    pos = decode_base128_varint<uint64_t>(bs, pos, u);
    result = u;
    return pos;
}

/**
 * @brief Signed version of 32-bit decode_varint
 **/
inline __device__ int decode_varint(volatile orc_bytestream_s *bs, int pos, int32_t &result)
{
    uint32_t u;
    pos = decode_base128_varint<uint32_t>(bs, pos, u);
    result = (int32_t)((u >> 1u) ^ -(int32_t)(u & 1));
    return pos;
}

/**
 * @brief Signed version of 64-bit decode_varint
 **/
inline __device__ int decode_varint(volatile orc_bytestream_s *bs, int pos, int64_t &result)
{
    uint64_t u;
    pos = decode_base128_varint<uint64_t>(bs, pos, u);
    result = (int64_t)((u >> 1u) ^ -(int64_t)(u & 1));
    return pos;
}


/**
 * @brief In-place conversion from lengths to positions
 *
 * @param[in] vals input values
 * @param[in] numvals number of values
 * @param[in] t thread id
 *
 * @return number of values decoded
 **/
template<class T>
inline __device__ void lengths_to_positions(volatile T *vals, uint32_t numvals, unsigned int t)
{
    for (uint32_t n = 1; n<numvals; n <<= 1)
    {
        __syncthreads();
        if ((t & n) && (t < numvals))
            vals[t] += vals[(t & ~n) | (n - 1)];
    }
}


/**
 * @brief ORC Integer RLEv1 decoding
 *
 * @param[in] bs input byte stream
 * @param[in] rle RLE state
 * @param[in] vals buffer for output values (uint32_t, int32_t, uint64_t or int64_t)
 * @param[in] maxvals maximum number of values to decode
 * @param[in] t thread id
 *
 * @return number of values decoded
 **/
template <class T>
static __device__ uint32_t Integer_RLEv1(orc_bytestream_s *bs, volatile orc_rlev1_state_s *rle, volatile T *vals, uint32_t maxvals, int t)
{
    uint32_t numvals, numruns;
    if (t == 0)
    {
        uint32_t maxpos = min(bs->len, bs->pos + (BYTESTREAM_BFRSZ - 8u));
        uint32_t lastpos = bs->pos;
        numvals = numruns = 0;
        // Find the length and start location of each run
        while (numvals < maxvals &&  numruns < NWARPS*12)
        {
            uint32_t pos = lastpos;
            uint32_t n = bytestream_readbyte(bs, pos++);
            if (n <= 0x7f)
            {
                // Run
                int32_t delta;
                n = n + 3;
                if (numvals + n > maxvals)
                    break;
                delta = bytestream_readbyte(bs, pos++);
                vals[numvals] = pos & 0xffff;
                pos += varint_length<T>(bs, pos);
                if (pos > maxpos)
                    break;
                rle->run_data[numruns++] = (delta << 24) | (n << 16) | numvals;
                numvals += n;
            }
            else
            {
                // Literals
                uint32_t i;
                n = 0x100 - n;
                if (numvals + n > maxvals)
                    break;
                i = 0;
                do
                {
                    vals[numvals + i] = pos & 0xffff;
                    pos += varint_length<T>(bs, pos);
                } while (++i < n);
                if (pos > maxpos)
                    break;
                numvals += n;
            }
            lastpos = pos;
        }
        rle->num_runs = numruns;
        rle->num_vals = numvals;
        bytestream_flush_bytes(bs, lastpos - bs->pos);
    }
    __syncthreads();
    // Expand the runs
    numruns = rle->num_runs;
    if (numruns > 0)
    {
        int r = t >> 5;
        int tr = t & 0x1f;
        for (uint32_t run = r; run < numruns; run += NWARPS)
        {
            int32_t run_data = rle->run_data[run];
            int n = (run_data >> 16) & 0xff;
            int delta = run_data >> 24;
            uint32_t base = run_data & 0x3ff;
            uint32_t pos = vals[base] & 0xffff;
            for (int i = 1+tr; i < n; i += 32)
            {
                vals[base + i] = ((delta * i) << 16) | pos;
            }
        }
        __syncthreads();
    }
    numvals = rle->num_vals;
    // Decode individual 32-bit varints
    if (t < numvals)
    {
        int32_t pos = vals[t];
        int32_t delta = pos >> 16;
        T v;
        decode_varint(bs, pos, v);
        vals[t] = v + delta;
    }
    __syncthreads();
    return numvals;
}


/**
 * @brief Maps the RLEv2 5-bit length code to 6-bit length
 *
 **/
static const __device__ __constant__ uint8_t kRLEv2_W[32] =
{
    1,2,3,4,        5,6,7,8,        9,10,11,12,     13,14,15,16,
    17,18,19,20,    21,22,23,24,    26,28,30,32,    40,48,56,64
};

/**
 * @brief ORC Integer RLEv2 decoding
 *
 * @param[in] bs input byte stream
 * @param[in] rle RLE state
 * @param[in] vals buffer for output values (uint32_t, int32_t, uint64_t or int64_t)
 * @param[in] maxvals maximum number of values to decode
 * @param[in] t thread id
 *
 * @return number of values decoded
 **/
template <class T>
static __device__ uint32_t Integer_RLEv2(orc_bytestream_s *bs, volatile orc_rlev2_state_s *rle, volatile T *vals, uint32_t maxvals, int t)
{
    uint32_t numvals, numruns;
    int r, tr;

    if (t == 0)
    {
        uint32_t maxpos = min(bs->len, bs->pos + (BYTESTREAM_BFRSZ - 8u));
        uint32_t lastpos = bs->pos;
        numvals = numruns = 0;
        // Find the length and start location of each run
        while (numvals < maxvals)
        {
            uint32_t pos = lastpos;
            uint32_t byte0 = bytestream_readbyte(bs, pos++);
            uint32_t n, l;
            int mode = byte0 >> 6;
            rle->runs_loc[numruns] = numvals;
            vals[numvals] = lastpos;           
            if (mode == 0)
            {
                // 00lllnnn: short repeat encoding
                l = 1 + ((byte0 >> 3) & 7); // 1 to 8 bytes
                n = 3 + (byte0 & 7); // 3 to 10 values
            }
            else
            {
                l = kRLEv2_W[(byte0 >> 1) & 0x1f];
                n = 1 + ((byte0 & 1) << 8) + bytestream_readbyte(bs, pos++);
                if (mode == 1)
                {
                    // 01wwwwwn.nnnnnnnn: direct encoding
                    l = (l * n + 7) >> 3;
                }
                else if (mode == 2)
                {
                    // 10wwwwwn.nnnnnnnn.xxxxxxxx.yyyyyyyy: patched base encoding
                    uint32_t byte2 = bytestream_readbyte(bs, pos++);
                    uint32_t byte3 = bytestream_readbyte(bs, pos++);
                    uint32_t bw = 1 + (byte2 >> 5); // base value width, 1 to 8 bytes
                    uint32_t pw = kRLEv2_W[byte2 & 0x1f]; // patch width, 1 to 64 bits
                    uint32_t pgw = 1 + (byte3 >> 5); // patch gap width, 1 to 8 bits
                    uint32_t pll = byte3 & 0x1f;    // patch list length
                    l = (l * n + 7) >> 3;
                    l += bw;
                    l += (pll * (pgw + pw) + 7) >> 3;
                }
                else
                {
                    // 11wwwwwn.nnnnnnnn.<base>.<delta>: delta encoding
                    uint32_t deltapos = varint_length<T>(bs, pos);
                    deltapos += varint_length<T>(bs, pos + deltapos);
                    l = (l > 1) ? (l * n + 7) >> 3 : 0;
                    l += deltapos;
                }
            }
            if (numvals + n > maxvals)
                break;
            pos += l;
            if (pos > maxpos)
                break;
            lastpos = pos;
            numvals += n;
            numruns++;
        }
        rle->num_vals = numvals;
        rle->num_runs = numruns;
        bytestream_flush_bytes(bs, lastpos - bs->pos);
    }
    __syncthreads();
    // Process the runs, 1 warp per run
    numruns = rle->num_runs;
    r = t >> 5;
    tr = t & 0x1f;
    for (uint32_t run = r; run < numruns; run += NWARPS)
    {
        uint32_t base, pos, w, n;
        int mode;
        if (tr == 0)
        {
            uint32_t byte0;
            base = rle->runs_loc[run];
            pos = vals[base];
            byte0 = bytestream_readbyte(bs, pos++);
            mode = byte0 >> 6;
            if (mode == 0)
            {
                T baseval;
                // 00lllnnn: short repeat encoding
                w = 8 + (byte0 & 0x38); // 8 to 64 bits
                n = 3 + (byte0 & 7); // 3 to 10 values
                bytestream_readbe(bs, pos*8, w, baseval);
                if (sizeof(T) <= 4)
                {
                    rle->baseval.u32[r] = baseval;
                }
                else
                {
                    rle->baseval.u64[r] = baseval;
                }
            }
            else
            {
                w = kRLEv2_W[(byte0 >> 1) & 0x1f];
                n = 1 + ((byte0 & 1) << 8) + bytestream_readbyte(bs, pos++);
                if (mode > 1)
                {
                    if (mode == 2)
                    {
                        // Patched base
                        uint32_t byte2 = bytestream_readbyte(bs, pos++);
                        uint32_t byte3 = bytestream_readbyte(bs, pos++);
                        uint32_t bw = 1 + (byte2 >> 5); // base value width, 1 to 8 bytes
                        uint32_t pw = kRLEv2_W[byte2 & 0x1f]; // patch width, 1 to 64 bits
                        if (sizeof(T) <= 4)
                        {
                            uint32_t baseval, mask;
                            bytestream_readbe(bs, pos * 8, bw * 8, baseval);
                            mask = (1 << (bw*8-1)) - 1;
                            rle->baseval.u32[r] = (baseval > mask) ? (-(int32_t)(baseval & mask)) : baseval;
                        }
                        else
                        {
                            uint64_t baseval, mask;
                            bytestream_readbe(bs, pos * 8, bw * 8, baseval);
                            mask = 2;
                            mask <<= (bw*8) - 1;
                            mask -= 1;
                            rle->baseval.u64[r] = (baseval > mask) ? (-(int64_t)(baseval & mask)) : baseval;
                        }
                        rle->m2_pw_byte3[r] = (pw << 8) | byte3;
                        pos += bw;
                    }
                    else
                    {
                        T baseval;
                        // Delta
                        pos = decode_varint(bs, pos, baseval);
                        if (sizeof(T) <= 4)
                        {
                            int32_t delta;
                            pos = decode_varint(bs, pos, delta);
                            rle->baseval.u32[r] = baseval;
                            rle->delta.i32[r] = delta;
                        }
                        else
                        {
                            int64_t delta;
                            pos = decode_varint(bs, pos, delta);
                            rle->baseval.u64[r] = baseval;
                            rle->delta.i64[r] = delta;
                        }
                    }
                }
            }
        }
        base = SHFL0(base);
        mode = SHFL0(mode);
        pos = SHFL0(pos);
        n = SHFL0(n);
        w = SHFL0(w);
        for (uint32_t i = tr; i < n; i += 32)
        {
            if (sizeof(T) <= 4)
            {
                if (mode == 0)
                {
                    vals[base + i] = rle->baseval.u32[r];
                }
                else if (mode == 1)
                {
                    T v;
                    bytestream_readbe(bs, pos * 8 + i*w, w, v);
                    vals[base + i] = v;
                }
                else if (mode == 2)
                {
                    uint32_t ofs = bytestream_readbits(bs, pos * 8 + i*w, w);
                    vals[base + i] = rle->baseval.u32[r] + ofs;
                }
                else
                {
                    int32_t delta = rle->delta.i32[r];
                    uint32_t ofs = (i == 0) ? 0 : (w > 1 && i > 1) ? bytestream_readbits(bs, pos * 8 + (i - 2)*w, w) : abs(delta);
                    vals[base + i] = (delta < 0) ? -ofs : ofs;
                }
            }
            else
            {
                if (mode == 0)
                {
                    vals[base + i] = rle->baseval.u64[r];
                }
                else if (mode == 1)
                {
                    T v;
                    bytestream_readbe(bs, pos * 8 + i*w, w, v);
                    vals[base + i] = v;
                }
                else if (mode == 2)
                {
                    uint32_t ofs = bytestream_readbits64(bs, pos * 8 + i*w, w);
                    vals[base + i] = rle->baseval.u64[r] + ofs;
                }
                else
                {
                    int64_t delta = rle->delta.i64[r];
                    uint64_t ofs = (i == 0) ? 0 : (w > 1 && i > 1) ? bytestream_readbits64(bs, pos * 8 + (i - 2)*w, w) : llabs(delta);
                    vals[base + i] = (delta < 0) ? -ofs : ofs;
                }
            }
        }
        SYNCWARP();
        // Patch values
        if (mode == 2)
        {
            uint32_t pw_byte3 = rle->m2_pw_byte3[r];
            uint32_t pw = pw_byte3 >> 8;
            uint32_t pgw = 1 + ((pw_byte3 >> 5) & 7); // patch gap width, 1 to 8 bits
            uint32_t pll = pw_byte3 & 0x1f;    // patch list length
            uint32_t patch_pos = (tr < pll) ? bytestream_readbits(bs, pos * 8 + n*w, pgw+pw) + 1 : 0; // FIXME: pgw+pw > 32
            uint32_t patch = patch_pos & ((1 << pw) - 1);
            patch_pos >>= pw;
            for (uint32_t k = 1; k < pll; k <<= 1)
            {
                uint32_t tmp = SHFL(patch_pos, (tr & ~k) | (k-1));
                patch_pos += (tr & k) ? tmp : 0;
            }
            if (tr < pll && patch_pos < n)
            {
                vals[base + patch_pos] += patch << w;
            }
        }
        SYNCWARP();
        if (mode == 3)
        {
            T baseval;
            for (uint32_t i = 1; i < n; i <<= 1)
            {
                SYNCWARP();
                for (uint32_t j = tr; j < n; j += 32)
                {
                    if (j & i)
                        vals[base + j] += vals[base + ((j & ~i) | (i - 1))];
                }
            }
            if (sizeof(T) <= 4)
                baseval = rle->baseval.u32[r];
            else
                baseval = rle->baseval.u64[r];
            for (uint32_t j = tr; j < n; j += 32)
            {
                vals[base + j] += baseval;
            }
        }
    }
    __syncthreads();
    return rle->num_vals;
}


/**
 * @brief Reads 32 booleans as a packed 32-bit value
 *
 * @param[in] vals 32-bit array of values (little-endian)
 * @param[in] bitpos bit position
 *
 * @return 32-bit value
 **/
inline __device__ uint32_t rle8_read_bool32(volatile uint32_t *vals, uint32_t bitpos)
{
    uint32_t a = vals[(bitpos >> 5) + 0];
    uint32_t b = vals[(bitpos >> 5) + 1];
    a = __byte_perm(a, 0, 0x0123);
    b = __byte_perm(b, 0, 0x0123);
    return __brev(__funnelshift_l(b, a, bitpos));
}


/**
 * @brief ORC Byte RLE decoding
 *
 * @param[in] bs Input byte stream
 * @param[in] rle RLE state
 * @param[in] vals output buffer for decoded 8-bit values
 * @param[in] maxvals Maximum number of values to decode
 * @param[in] t thread id
 *
 * @return number of values decoded
 **/
static __device__ uint32_t Byte_RLE(orc_bytestream_s *bs, volatile orc_byterle_state_s *rle, volatile uint8_t *vals, uint32_t maxvals, int t)
{
    uint32_t numvals, numruns;
    int r, tr;
    if (t == 0)
    {
        uint32_t maxpos = min(bs->len, bs->pos + (BYTESTREAM_BFRSZ - 8u));
        uint32_t lastpos = bs->pos;
        numvals = numruns = 0;
        // Find the length and start location of each run
        while (numvals < maxvals && numruns < NWARPS)
        {
            uint32_t pos = lastpos, n;
            rle->runs_pos[numruns] = pos;
            rle->runs_loc[numruns] = numvals;
            n = bytestream_readbyte(bs, pos++);
            if (n <= 0x7f)
            {
                // Run
                n = n + 3;
                pos++;
            }
            else
            {
                // Literals
                n = 0x100 - n;
                pos += n;
            }
            if (pos > maxpos || numvals + n > maxvals)
                break;
            numruns++;
            numvals += n;
            lastpos = pos;
        }
        rle->num_runs = numruns;
        rle->num_vals = numvals;
        bytestream_flush_bytes(bs, lastpos - bs->pos);
    }
    __syncthreads();
    numruns = rle->num_runs;
    r = t >> 5;
    tr = t & 0x1f;
    for (int run = r; run < numruns; run += NWARPS)
    {
        uint32_t pos = rle->runs_pos[run];
        uint32_t loc = rle->runs_loc[run];
        uint32_t n = bytestream_readbyte(bs, pos++);
        uint32_t literal_mask;
        if (n <= 0x7f)
        {
            literal_mask = 0;
            n += 3;
        }
        else
        {
            literal_mask = ~0;
            n = 0x100 - n;
        }
        for (uint32_t i = tr; i < n; i += 32)
        {
            vals[loc + i] = bytestream_readbyte(bs, pos + (i & literal_mask));
        }
    }
    __syncthreads();
    return rle->num_vals;
}

/**
 * @brief Powers of 10
 *
 **/
#if DECIMALS_AS_FLOAT64
static const __device__ __constant__ double kPow10[40] =
{
    1.0,    1.e1,   1.e2,   1.e3,   1.e4,   1.e5,   1.e6,   1.e7,
    1.e8,   1.e9,   1.e10,  1.e11,  1.e12,  1.e13,  1.e14,  1.e15,
    1.e16,  1.e17,  1.e18,  1.e19,  1.e20,  1.e21,  1.e22,  1.e23,
    1.e24,  1.e25,  1.e26,  1.e27,  1.e28,  1.e29,  1.e30,  1.e31,
    1.e32,  1.e33,  1.e34,  1.e35,  1.e36,  1.e37,  1.e38,  1.e39,
};
#else // DECIMALS_AS_FLOAT64
static const __device__ __constant__ int64_t kPow10[19] =
{
    1,
    10,
    100,
    1000,
    10000,
    100000,
    1000000,
    10000000,
    100000000,
    1000000000,
    10000000000ll,
    100000000000ll,
    1000000000000ll,
    10000000000000ll,
    100000000000000ll,
    1000000000000000ll,
    10000000000000000ll,
    100000000000000000ll,
    1000000000000000000ll
};
#endif // DECIMALS_AS_FLOAT64

/**
 * @brief ORC Decimal decoding (unbounded base-128 varints)
 *
 * @param[in] bs Input byte stream
 * @param[in,out] vals on input: scale from secondary stream, on output: value
 * @param[in] numvals Number of values to decode
 * @param[in] t thread id
 *
 * @return number of values decoded
 *
 **/
static __device__ int Decode_Decimals(orc_bytestream_s *bs, volatile orc_byterle_state_s *scratch, volatile int64_t *vals, int numvals, int col_scale, int t)
{
#if DECIMALS_AS_FLOAT64
    int scale = (t < numvals) ? (int)vals[t] : 0;
#else
    int scale = (t < numvals) ? col_scale - (int)vals[t] : 0;
#endif
    if (t == 0)
    {
        uint32_t maxpos = min(bs->len, bs->pos + (BYTESTREAM_BFRSZ - 8u));
        uint32_t lastpos = bs->pos;
        uint32_t n;
        for (n = 0; n < numvals; n++)
        {
            uint32_t pos = lastpos;
            *(volatile int32_t *)&vals[n] = lastpos;
            pos += varint_length<uint4>(bs, pos);
            if (pos > maxpos)
                break;
            lastpos = pos;
        }
        scratch->num_vals = n;
        bytestream_flush_bytes(bs, lastpos - bs->pos);
    }
    __syncthreads();
    numvals = scratch->num_vals;
    if (t < numvals)
    {
        int pos = *(volatile int32_t *)&vals[t];
        int64_t v;
        decode_varint(bs, pos, v);
#if DECIMALS_AS_FLOAT64
        if (scale >= 0)
            reinterpret_cast<volatile double *>(vals)[t] = __ll2double_rn(v) / kPow10[min(scale, 39)];
        else
            reinterpret_cast<volatile double *>(vals)[t] = __ll2double_rn(v) * kPow10[min(-scale, 39)];
#else
        if (scale > 0)
        {
            v *= kPow10[min(scale, 18)];
        }
        else if (scale < 0)
        {
            v /= kPow10[min(-scale, 18)];
        }
        vals[t] = v;
#endif
    }
    return numvals;
}


/**
 * @brief Decoding NULLs and builds string dictionary index tables
 *
 * @param[in] chunks ColumnDesc device array [stripe][column]
 * @param[in] global_dictionary Global dictionary device array
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] max_num_rows Maximum number of rows to load
 * @param[in] first_row Crop all rows below first_row
 *
 **/
// blockDim {NTHREADS,1,1}
extern "C" __global__ void __launch_bounds__(NTHREADS)
gpuDecodeNullsAndStringDictionaries(ColumnDesc *chunks, DictionaryEntry *global_dictionary, uint32_t num_columns, uint32_t num_stripes, size_t max_num_rows, size_t first_row)
{
    __shared__ __align__(16) orcdec_state_s state_g;
    
    orcdec_state_s * const s = &state_g;
    bool is_nulldec = (blockIdx.y >= num_stripes);
    uint32_t column = blockIdx.x;
    uint32_t stripe = (is_nulldec) ? blockIdx.y - num_stripes : blockIdx.y;
    uint32_t chunk_id = stripe * num_columns + column;
    int t = threadIdx.x;
    
    if (t < sizeof(ColumnDesc) / sizeof(uint32_t))
    {
        ((volatile uint32_t *)&s->chunk)[t] = ((const uint32_t *)&chunks[chunk_id])[t];
    }
    __syncthreads();
    if (is_nulldec)
    {
        uint32_t null_count = 0;
        // Decode NULLs
        if (t == 0)
        {
            s->top.nulls.row = 0;
            bytestream_init(&s->bs, s->chunk.streams[CI_PRESENT], s->chunk.strm_len[CI_PRESENT]);
        }
        __syncthreads();
        if (s->chunk.strm_len[CI_PRESENT] == 0)
        {
            // No present stream: all rows are valid
            s->vals.u32[t] = ~0;
        }
        while (s->top.nulls.row < s->chunk.num_rows)
        {
            uint32_t nrows_max = min(s->chunk.num_rows - s->top.nulls.row, NTHREADS*32);
            uint32_t nrows;
            size_t row_in;

            bytestream_fill(&s->bs, t);
            __syncthreads();
            if (s->chunk.strm_len[CI_PRESENT] > 0)
            {
                uint32_t nbytes = Byte_RLE(&s->bs, &s->u.rle8, s->vals.u8, (nrows_max + 7) >> 3, t);
                nrows = min(nrows_max, nbytes * 8u);
                if (!nrows)
                {
                    // Error: mark all remaining rows as null
                    nrows = nrows_max;
                    if (t * 32 < nrows)
                    {
                        s->vals.u32[t] = 0;
                    }
                }
            }
            else
            {
                nrows = nrows_max;
            }
            __syncthreads();
            row_in = s->chunk.start_row + s->top.nulls.row;
            if (row_in + nrows > first_row && row_in < first_row + max_num_rows && s->chunk.valid_map_base != NULL)
            {
                int64_t dst_row = row_in - first_row;
                int64_t dst_pos = max(dst_row, (int64_t)0);
                uint32_t startbit = -static_cast<int32_t>(min(dst_row, (int64_t)0));
                uint32_t nbits = nrows - min(startbit, nrows);
                uint32_t *valid = s->chunk.valid_map_base + (dst_pos >> 5);
                uint32_t bitpos = static_cast<uint32_t>(dst_pos) & 0x1f;
                if ((size_t)(dst_pos + nbits) > max_num_rows)
                {
                    nbits = static_cast<uint32_t>(max_num_rows - min((size_t)dst_pos, max_num_rows));
                }
                // Store bits up to the next 32-bit aligned boundary
                if (bitpos != 0)
                {
                    uint32_t n = min(32u - bitpos, nbits);
                    if (t == 0)
                    {
                        uint32_t mask = ((1 << n) - 1) << bitpos;
                        uint32_t bits = (rle8_read_bool32(s->vals.u32, startbit) << bitpos) & mask;
                        atomicAnd(valid, ~mask);
                        atomicOr(valid, bits);
                        null_count += __popc((~bits) & mask);
                    }
                    nbits -= n;
                    startbit += n;
                    valid++;
                }
                // Store bits aligned
                if (t * 32 + 32 <= nbits)
                {
                    uint32_t bits = rle8_read_bool32(s->vals.u32, startbit + t * 32);
                    valid[t] = bits;
                    null_count += __popc(~bits);
                }
                else if (t * 32 < nbits)
                {
                    uint32_t n = nbits - t*32;
                    uint32_t mask = (1 << n) - 1;
                    uint32_t bits = rle8_read_bool32(s->vals.u32, startbit + t * 32) & mask;
                    atomicAnd(valid + t, ~mask);
                    atomicOr(valid + t, bits);
                    null_count += __popc((~bits) & mask);
                }
                __syncthreads();
            }
            // We may have some valid values that are not decoded below first_row -> count these in skip_count,
            // so that subsequent kernel can infer the correct row position
            if (row_in < first_row && t < 32)
            {
                uint32_t skippedrows = min(static_cast<uint32_t>(first_row - row_in), nrows);
                uint32_t skip_count = 0;
                for (uint32_t i = 0; i < skippedrows; i += 32)
                {
                    uint32_t bits = s->vals.u32[i >> 5];
                    if (i + 32 > skippedrows)
                    {
                        bits &= (1 << (skippedrows - i)) - 1;
                    }
                    skip_count += __popc(bits);
                }
                skip_count += SHFL_XOR(skip_count, 1);
                skip_count += SHFL_XOR(skip_count, 2);
                skip_count += SHFL_XOR(skip_count, 4);
                skip_count += SHFL_XOR(skip_count, 8);
                skip_count += SHFL_XOR(skip_count, 16);
                if (t == 0)
                {
                    s->chunk.skip_count += skip_count;
                }
            }
            __syncthreads();
            if (t == 0)
            {
                s->top.nulls.row += nrows;
            }
            __syncthreads();
        }
        __syncthreads();
        // Sum up the valid counts and infer null_count
        null_count += SHFL_XOR(null_count, 1);
        null_count += SHFL_XOR(null_count, 2);
        null_count += SHFL_XOR(null_count, 4);
        null_count += SHFL_XOR(null_count, 8);
        null_count += SHFL_XOR(null_count, 16);
        if (!(t & 0x1f))
        {
            s->top.nulls.null_count[t >> 5] = null_count;
        }
        __syncthreads();
        if (t < 32)
        {
            null_count = (t < NWARPS) ? s->top.nulls.null_count[t] : 0;
            null_count += SHFL_XOR(null_count, 1);
            null_count += SHFL_XOR(null_count, 2);
            null_count += SHFL_XOR(null_count, 4);
            null_count += SHFL_XOR(null_count, 8);
            null_count += SHFL_XOR(null_count, 16);
            if (t == 0)
            {
                chunks[chunk_id].null_count = null_count;
            }
        }
    }
    else
    {
        // Decode string dictionary
        int encoding_kind = s->chunk.encoding_kind;
        if ((encoding_kind == DICTIONARY || encoding_kind == DICTIONARY_V2) && (s->chunk.dict_len > 0))
        {
            if (t == 0)
            {
                s->top.dict.dict_len = s->chunk.dict_len;
                s->top.dict.local_dict = (uint2 *)(global_dictionary + s->chunk.dictionary_start);  // Local dictionary
                s->top.dict.dict_pos = 0;
                // CI_DATA2 contains the LENGTH stream coding the length of individual dictionary entries
                bytestream_init(&s->bs, s->chunk.streams[CI_DATA2], s->chunk.strm_len[CI_DATA2]);
            }
            __syncthreads();
            while (s->top.dict.dict_len > 0)
            {
                uint32_t numvals = min(s->top.dict.dict_len, NTHREADS), len;
                volatile uint32_t *vals = s->vals.u32;
                bytestream_fill(&s->bs, t);
                __syncthreads();
                if (IS_RLEv1(s->chunk.encoding_kind))
                {
                    numvals = Integer_RLEv1(&s->bs, &s->u.rlev1, vals, numvals, t);
                }
                else // RLEv2
                {
                    numvals = Integer_RLEv2(&s->bs, &s->u.rlev2, vals, numvals, t);
                }
                __syncthreads();
                len = (t < numvals) ? vals[t] : 0;
                lengths_to_positions(vals, numvals, t);
                __syncthreads();
                if (numvals == 0)
                {
                    // This is an error (ran out of data)
                    numvals = min(s->top.dict.dict_len, NTHREADS);
                    vals[t] = 0;
                }
                if (t < numvals)
                {
                    uint2 dict_entry;
                    dict_entry.x = s->top.dict.dict_pos + vals[t] - len;
                    dict_entry.y = len;
                    s->top.dict.local_dict[t] = dict_entry;
                }
                __syncthreads();
                if (t == 0)
                {
                    s->top.dict.dict_pos += vals[numvals - 1];
                    s->top.dict.dict_len -= numvals;
                    s->top.dict.local_dict += numvals;
                }
                __syncthreads();
            }
        }
    }
}


/**
 * @brief Decode row positions from valid bits
 *
 * @param[in,out] s Column chunk decoder state
 * @param[in] first_row crop all rows below first rows
 * @param[in] t thread id
 *
 **/
static __device__ void DecodeRowPositions(orcdec_state_s *s, size_t first_row, int t)
{
    if (t == 0)
    {
        s->u.rowdec.nz_count = min(s->chunk.skip_count, NTHREADS);
        s->chunk.skip_count -= s->u.rowdec.nz_count;
    }
    __syncthreads();
    if (t < s->u.rowdec.nz_count)
    {
        s->u.rowdec.row[t] = 0; // Skipped values (below first_row)
    }
    while (s->u.rowdec.nz_count < s->top.data.max_vals && s->top.data.cur_row + s->top.data.nrows < s->top.data.end_row)
    {
        uint32_t nrows = min(s->top.data.end_row - s->top.data.cur_row, min((ROWDEC_BFRSZ - s->u.rowdec.nz_count)*2, NTHREADS));
        if (s->chunk.strm_len[CI_PRESENT] > 0)
        {
            // We have a present stream
            uint32_t rmax = s->top.data.end_row - min((uint32_t)first_row, s->top.data.end_row);
            uint32_t r = (uint32_t)(s->top.data.cur_row + s->top.data.nrows + t - first_row);
            uint32_t valid = (t < nrows && r < rmax) ? (((const uint8_t *)s->chunk.valid_map_base)[r >> 3] >> (r & 7)) & 1 : 0;
            volatile uint32_t *rows = &s->u.rowdec.row[s->u.rowdec.nz_count];
            volatile uint16_t *row_ofs_plus1 = (volatile uint16_t *)rows;
            uint32_t nz_pos, row_plus1, nz_count = s->u.rowdec.nz_count, last_row;
            if (t < nrows)
            {
                row_ofs_plus1[t] = valid;
            }
            lengths_to_positions<uint16_t>(row_ofs_plus1, nrows, t);
            if (t < nrows)
            {
                nz_count += row_ofs_plus1[t];
                row_plus1 = s->top.data.nrows + t + 1;
            }
            else
            {
                row_plus1 = 0;
            }
            if (t == nrows - 1)
            {
                s->u.rowdec.nz_count = min(nz_count, s->top.data.max_vals);
            }
            __syncthreads();
            // TBD: Brute-forcing this, there might be a more efficient way to find the thread with the last row
            last_row = (nz_count == s->u.rowdec.nz_count) ? row_plus1 : 0;
            last_row = max(last_row, SHFL_XOR(last_row, 1));
            last_row = max(last_row, SHFL_XOR(last_row, 2));
            last_row = max(last_row, SHFL_XOR(last_row, 4));
            last_row = max(last_row, SHFL_XOR(last_row, 8));
            last_row = max(last_row, SHFL_XOR(last_row, 16));
            if (!(t & 0x1f))
            {
                *(volatile uint32_t *)&s->u.rowdec.last_row[t >> 5] = last_row;
            }
            nz_pos = (valid) ? row_ofs_plus1[t] : 0;
            __syncthreads();
            if (t < 32)
            {
                last_row = (t < NWARPS) ? *(volatile uint32_t *)&s->u.rowdec.last_row[t] : 0;
                last_row = max(last_row, SHFL_XOR(last_row, 1));
                last_row = max(last_row, SHFL_XOR(last_row, 2));
                last_row = max(last_row, SHFL_XOR(last_row, 4));
                last_row = max(last_row, SHFL_XOR(last_row, 8));
                last_row = max(last_row, SHFL_XOR(last_row, 16));
                if (t == 0)
                {
                    s->top.data.nrows = last_row;
                }
            }
            if (valid && nz_pos - 1 < s->u.rowdec.nz_count)
            {
                rows[nz_pos - 1] = row_plus1;
            }
            __syncthreads();
        }
        else
        {
            // All values are valid
            nrows = min(nrows, s->top.data.max_vals - s->u.rowdec.nz_count);
            if (t < nrows)
            {
                s->u.rowdec.row[s->u.rowdec.nz_count + t] = s->top.data.nrows + t + 1;
            }
            __syncthreads();
            if (t == 0)
            {
                s->top.data.nrows += nrows;
                s->u.rowdec.nz_count += nrows;
            }
            __syncthreads();
        }
    }
}


/**
 * @brief Convert seconds from writer timezone to UTC
 *
 * @param[in] s Orc data decoder state
 * @param[in] table Timezone translation table
 * @param[in] ts Local time in seconds
 *
 * @return UTC time in seconds
 *
 **/
static __device__ int64_t ConvertToUTC(const orc_datadec_state_s *s, const int64_t *table, int64_t ts)
{
    uint32_t num_entries = s->tz_num_entries;
    uint32_t dst_cycle = s->tz_dst_cycle;
    int64_t first_transition = s->first_tz_transition;
    int64_t last_transition = s->last_tz_transition;
    int64_t tsbase;
    uint32_t first, last;

    if (ts <= first_transition)
    {
        return ts + table[0 * 2 + 1];
    }
    else if (ts <= last_transition)
    {
        first = 0;
        last = num_entries - 1;
        tsbase = ts;
    }
    else if (!dst_cycle)
    {
        return ts + table[(num_entries - 1) * 2 + 1];
    }
    else
    {
        // Apply 400-year cycle rule
        const int64_t k400Years = (365 * 400 + (100 - 3)) * 24 * 60 * 60ll;
        tsbase = ts;
        ts %= k400Years;
        if (ts < 0)
        {
            ts += k400Years;
        }
        first = num_entries;
        last = num_entries + dst_cycle - 1;
        if (ts < table[num_entries * 2])
        {
            return tsbase + table[last * 2 + 1];
        }
    }
    // Binary search the table from first to last for ts
    do
    {
        uint32_t mid = first + ((last - first + 1) >> 1);
        int64_t tmid = table[mid * 2];
        if (tmid <= ts)
        {
            first = mid;
        }
        else
        {
            if (mid == last)
            {
                break;
            }
            last = mid;
        }
    } while (first < last);
    return tsbase + table[first * 2 + 1];
}


/**
 * @brief Trailing zeroes for decoding timestamp nanoseconds
 *
 **/
static const __device__ __constant__ uint32_t kTimestampNanoScale[8] =
{
    1, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000
};

/**
 * @brief Decodes column data
 *
 * @param[in] chunks ColumnDesc device array
 * @param[in] global_dictionary Global dictionary device array
 * @param[in] max_num_rows Maximum number of rows to load
 * @param[in] first_row Crop all rows below first_row
 * @param[in] num_chunks Number of column chunks (num_columns * num_stripes)
 * @param[in] stream CUDA stream to use, default 0
 *
 **/
// blockDim {NTHREADS,1,1}
extern "C" __global__ void __launch_bounds__(NTHREADS)
gpuDecodeOrcColumnData(ColumnDesc *chunks, DictionaryEntry *global_dictionary, int64_t *tz_table, size_t max_num_rows, size_t first_row, uint32_t num_chunks, uint32_t tz_len)
{
    __shared__ __align__(16) orcdec_state_s state_g;

    orcdec_state_s * const s = &state_g;
    uint32_t chunk_id = blockIdx.x;
    int t = threadIdx.x;

    if (t < sizeof(ColumnDesc) / sizeof(uint32_t))
    {
        ((volatile uint32_t *)&s->chunk)[t] = ((const uint32_t *)&chunks[chunk_id])[t];
    }
    __syncthreads();
    if (t == 0)
    {
        s->top.data.cur_row = s->chunk.start_row;
        s->top.data.end_row = s->chunk.start_row + s->chunk.num_rows;
        s->top.data.buffered_count = 0;
        if (s->top.data.end_row > first_row + max_num_rows)
        {
            s->top.data.end_row = static_cast<uint32_t>(first_row + max_num_rows);
        }
        if (!IS_DICTIONARY(s->chunk.encoding_kind))
        {
            s->chunk.dictionary_start = 0;
        }
        if (tz_len > 0)
        {
            if (tz_len > 800) // 2 entries/year for 400 years
            {
                s->top.data.tz_num_entries = tz_len - 800;
                s->top.data.tz_dst_cycle = 800;
            }
            else
            {
                s->top.data.tz_num_entries = tz_len;
                s->top.data.tz_dst_cycle = 0;
            }
            if (tz_len > 0)
            {
                s->top.data.first_tz_transition = tz_table[0];
                s->top.data.last_tz_transition = tz_table[(s->top.data.tz_num_entries - 1) * 2];
            }
        }
        bytestream_init(&s->bs, s->chunk.streams[CI_DATA], s->chunk.strm_len[CI_DATA]);
        bytestream_init(&s->bs2, s->chunk.streams[CI_DATA2], s->chunk.strm_len[CI_DATA2]);
    }
    __syncthreads();
    while (s->top.data.cur_row < s->top.data.end_row)
    {
        bytestream_fill(&s->bs, t);
        bytestream_fill(&s->bs2, t);
        __syncthreads();
        if (t == 0)
        {
            s->bs.fill_count = 0;
            s->bs2.fill_count = 0;
            s->top.data.nrows = 0;
            s->top.data.max_vals = min(s->chunk.start_row + s->chunk.num_rows - s->top.data.cur_row, NTHREADS);
        }
        __syncthreads();
        // Decode data streams
        {
            uint32_t numvals = s->top.data.max_vals, secondary_val;
            if (s->chunk.type_kind == STRING || s->chunk.type_kind == BINARY || s->chunk.type_kind == VARCHAR || s->chunk.type_kind == CHAR
             || s->chunk.type_kind == TIMESTAMP)
            {
                // For these data types, we have a secondary unsigned 32-bit data stream
                orc_bytestream_s *bs = (IS_DICTIONARY(s->chunk.encoding_kind)) ? &s->bs : &s->bs2;
                uint32_t ofs = 0;
                if (s->chunk.type_kind == TIMESTAMP)
                {
                    // Restore buffered secondary stream values, if any
                    ofs = s->top.data.buffered_count;
                    if (ofs > 0)
                    {
                        __syncthreads();
                        if (t == 0)
                        {
                            s->top.data.buffered_count = 0;
                        }
                    }
                }
                if (numvals > ofs)
                {
                    if (IS_RLEv1(s->chunk.encoding_kind))
                    {
                        numvals = ofs + Integer_RLEv1(bs, &s->u.rlev1, &s->vals.u32[ofs], numvals - ofs, t);
                    }
                    else
                    {
                        numvals = ofs + Integer_RLEv2(bs, &s->u.rlev2, &s->vals.u32[ofs], numvals - ofs, t);
                    }
                    __syncthreads();
                    if (numvals <= ofs && t >= ofs && t < s->top.data.max_vals)
                    {
                        s->vals.u32[t] = 0;
                    }
                }
                __syncthreads();
                // For strings with direct encoding, we need to convert the lengths into an offset
                if (!IS_DICTIONARY(s->chunk.encoding_kind))
                {
                    secondary_val = (t < numvals) ? s->vals.u32[t] : 0;
                    if (s->chunk.type_kind != TIMESTAMP)
                    {
                        lengths_to_positions(s->vals.u32, numvals, t);
                        __syncthreads();
                    }
                }
            }
            __syncthreads();
            // Adjust the maximum number of values
            if (t == 0 && numvals > 0 && numvals < s->top.data.max_vals)
            {
                s->top.data.max_vals = numvals;
            }
            __syncthreads();
            // Decode the primary data stream
            if (s->chunk.type_kind == INT || s->chunk.type_kind == DATE || s->chunk.type_kind == SHORT)
            {
                // Signed int32 primary data stream
                if (IS_RLEv1(s->chunk.encoding_kind))
                {
                    numvals = Integer_RLEv1(&s->bs, &s->u.rlev1, s->vals.i32, numvals, t);
                }
                else
                {
                    numvals = Integer_RLEv2(&s->bs, &s->u.rlev2, s->vals.i32, numvals, t);
                }
                __syncthreads();
            }
            else if (s->chunk.type_kind == BYTE)
            {
                numvals = Byte_RLE(&s->bs, &s->u.rle8, s->vals.u8, numvals, t);
                __syncthreads();
            }
            else if (s->chunk.type_kind == BOOLEAN)
            {
                numvals = Byte_RLE(&s->bs, &s->u.rle8, s->vals.u8, (numvals + 7) >> 3, t);
                numvals = min(numvals << 3u, s->top.data.max_vals);
                __syncthreads();
            }
            else if (s->chunk.type_kind == LONG || s->chunk.type_kind == TIMESTAMP || s->chunk.type_kind == DECIMAL)
            {
                orc_bytestream_s *bs = (s->chunk.type_kind == DECIMAL) ? &s->bs2 : &s->bs;
                if (IS_RLEv1(s->chunk.encoding_kind))
                {
                    numvals = Integer_RLEv1<int64_t>(bs, &s->u.rlev1, s->vals.i64, numvals, t);
                }
                else
                {
                    numvals = Integer_RLEv2<int64_t>(bs, &s->u.rlev2, s->vals.i64, numvals, t);
                }
                if (s->chunk.type_kind == DECIMAL)
                {
                    __syncthreads();
                    numvals = Decode_Decimals(&s->bs, &s->u.rle8, s->vals.i64, numvals, s->chunk.decimal_scale, t);
                }
                __syncthreads();
            }
            else if (s->chunk.type_kind == FLOAT)
            {
                numvals = min(numvals, (BYTESTREAM_BFRSZ - 8u) >> 2);
                if (t < numvals)
                {
                    s->vals.u32[t] = bytestream_readu32(&s->bs, s->bs.pos + t * 4);
                }
                __syncthreads();
                if (t == 0)
                {
                    bytestream_flush_bytes(&s->bs, numvals * 4);
                }
                __syncthreads();
            }
            else if (s->chunk.type_kind == DOUBLE)
            {
                numvals = min(numvals, (BYTESTREAM_BFRSZ - 8u) >> 3);
                if (t < numvals)
                {
                    s->vals.u64[t] = bytestream_readu64(&s->bs, s->bs.pos + t * 8);
                }
                __syncthreads();
                if (t == 0)
                {
                    bytestream_flush_bytes(&s->bs, numvals * 8);
                }
                __syncthreads();
            }
            __syncthreads();
            if (t == 0 && numvals > 0 && numvals < s->top.data.max_vals)
            {
                if (s->chunk.type_kind == TIMESTAMP)
                {
                    s->top.data.buffered_count = s->top.data.max_vals - numvals;
                }
                s->top.data.max_vals = numvals;
            }
            __syncthreads();
            // Use the valid bits to compute non-null row positions until we get a full batch of values to decode
            DecodeRowPositions(s, first_row, t);
            // Store decoded values to output
            if (t < s->top.data.max_vals && s->u.rowdec.row[t] != 0)
            {
                size_t row = s->top.data.cur_row + s->u.rowdec.row[t] - 1 - first_row;
                if (row < max_num_rows)
                {
                    void *data_out = s->chunk.column_data_base;
                    switch (s->chunk.type_kind)
                    {
                    case FLOAT:
                    case INT:
                        reinterpret_cast<uint32_t *>(data_out)[row] = s->vals.u32[t];
                        break;
                    case DOUBLE:
                    case LONG:
                    case DECIMAL:
                        reinterpret_cast<uint64_t *>(data_out)[row] = s->vals.u64[t];
                        break;
                    case SHORT:
                        reinterpret_cast<uint16_t *>(data_out)[row] = static_cast<uint16_t>(s->vals.u32[t]);
                        break;
                    case BYTE:
                        reinterpret_cast<uint8_t *>(data_out)[row] = s->vals.u8[t];
                        break;
                    case BOOLEAN:
                        reinterpret_cast<uint8_t *>(data_out)[row] = (s->vals.u8[t >> 3] >> ((~t) & 7)) & 1;
                        break;
                    case DATE:
                        if (s->chunk.dtype_len == 8)
                        {
                            // Convert from days to milliseconds by multiplying by 24*3600*1000
                            reinterpret_cast<int64_t *>(data_out)[row] = 86400000ll * (int64_t)s->vals.i32[t];
                        }
                        else
                        {
                            reinterpret_cast<uint32_t *>(data_out)[row] = s->vals.u32[t];
                        }
                        break;
                    case STRING:
                    case BINARY:
                    case VARCHAR:
                    case CHAR:
                    {
                        nvstrdesc_s *strdesc = &reinterpret_cast<nvstrdesc_s *>(data_out)[row];
                        const uint8_t *ptr;
                        uint32_t count;
                        if (IS_DICTIONARY(s->chunk.encoding_kind))
                        {
                            uint32_t dict_idx = s->vals.u32[t];
                            ptr = s->chunk.streams[CI_DICTIONARY];
                            if (dict_idx < s->chunk.dict_len)
                            {
                                ptr += global_dictionary[s->chunk.dictionary_start + dict_idx].pos;
                                count = global_dictionary[s->chunk.dictionary_start + dict_idx].len;
                            }
                            else
                            {
                                count = 0;
                                //ptr = (uint8_t *)0xdeadbeef;
                            }
                        }
                        else
                        {
                            uint32_t dict_idx = s->chunk.dictionary_start + s->vals.u32[t] - secondary_val;
                            count = secondary_val;
                            ptr = s->chunk.streams[CI_DATA] + dict_idx;
                            if (dict_idx + count > s->chunk.strm_len[CI_DATA])
                            {
                                count = 0;
                                //ptr = (uint8_t *)0xdeadbeef;
                            }
                        }
                        strdesc->ptr = reinterpret_cast<const char *>(ptr);
                        strdesc->count = count;
                        break;
                    }
                    case TIMESTAMP:
                    {
                        int64_t seconds = s->vals.i64[t] + kORCTimeToUTC;
                        uint32_t nanos = secondary_val;
                        nanos = (nanos >> 3) * kTimestampNanoScale[nanos & 7];
                        if (tz_len > 0)
                        {
                            seconds = ConvertToUTC(&s->top.data, tz_table, seconds);
                        }
                        if (seconds < 0 && nanos != 0)
                        {
                            seconds -= 1;
                        }
                        reinterpret_cast<int64_t *>(data_out)[row] = seconds * ORC_TS_CLKRATE + (nanos + (499999999 / ORC_TS_CLKRATE)) / (1000000000 / ORC_TS_CLKRATE); // Output to desired clock rate
                        break;
                    }
                    }
                }
            }
            __syncthreads();
            // Buffer secondary stream values
            if (s->chunk.type_kind == TIMESTAMP && t >= s->top.data.max_vals && t < s->top.data.max_vals + s->top.data.buffered_count)
            {
                s->vals.u32[t - s->top.data.max_vals] = secondary_val;
            }
        }
        __syncthreads();
        if (t == 0)
        {
            s->top.data.cur_row += s->top.data.nrows;
            if ((s->chunk.type_kind == STRING || s->chunk.type_kind == BINARY || s->chunk.type_kind == VARCHAR || s->chunk.type_kind == CHAR)
             && !IS_DICTIONARY(s->chunk.encoding_kind) && s->top.data.max_vals > 0)
            {
                s->chunk.dictionary_start += s->vals.u32[s->top.data.max_vals - 1];
            }
        }
        __syncthreads();
        if (!s->top.data.nrows)
        {
            // This is a bug (could happen with bitstream errors with a bad run that would produce more values than the number of remaining rows)
            break;
        }
    }
}


/**
 * @brief Launches kernel for decoding NULLs and building string dictionary index tables
 *
 * @param[in] chunks ColumnDesc device array [stripe][column]
 * @param[in] global_dictionary Global dictionary device array
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] max_rows Maximum number of rows to load
 * @param[in] first_row Crop all rows below first_row
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t __host__ DecodeNullsAndStringDictionaries(ColumnDesc *chunks, DictionaryEntry *global_dictionary, uint32_t num_columns, uint32_t num_stripes, size_t max_num_rows, size_t first_row, cudaStream_t stream)
{
    dim3 dim_block(NTHREADS, 1);
    dim3 dim_grid(num_columns, num_stripes * 2); // 1024 threads per chunk
    gpuDecodeNullsAndStringDictionaries <<< dim_grid, dim_block, 0, stream >>>(chunks, global_dictionary, num_columns, num_stripes, max_num_rows, first_row);
    return cudaSuccess;
}

/**
 * @brief Launches kernel for decoding column data
 *
 * @param[in] chunks ColumnDesc device array [stripe][column]
 * @param[in] global_dictionary Global dictionary device array
 * @param[in] num_columns Number of columns
 * @param[in] num_stripes Number of stripes
 * @param[in] max_rows Maximum number of rows to load
 * @param[in] first_row Crop all rows below first_row
 * @param[in] tz_table Timezone translation table
 * @param[in] tz_len Length of timezone translation table
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t __host__ DecodeOrcColumnData(ColumnDesc *chunks, DictionaryEntry *global_dictionary, uint32_t num_columns, uint32_t num_stripes, size_t max_num_rows, size_t first_row,
    int64_t *tz_table, size_t tz_len, cudaStream_t stream)
{
    uint32_t num_chunks = num_columns * num_stripes;
    dim3 dim_block(NTHREADS, 1);
    dim3 dim_grid(num_chunks, 1); // 1024 threads per chunk
    gpuDecodeOrcColumnData <<< dim_grid, dim_block, 0, stream >>>(chunks, global_dictionary, tz_table, max_num_rows, first_row, num_chunks, (uint32_t)(tz_len >> 1));
    return cudaSuccess;
}



};}; // orc::gpu namespace
