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

/* gpuinflate.cu

  Derived from zlib's contrib/puff.c, original copyright notice below

*/

/*
Copyright (C) 2002-2013 Mark Adler, all rights reserved
version 2.3, 21 Jan 2013

This software is provided 'as-is', without any express or implied
warranty.  In no event will the author be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

Mark Adler    madler@alumni.caltech.edu
*/

#include "gpuinflate.h"
#include <io/utilities/block_utils.cuh>

namespace cudf {
namespace io {

#define NUMTHREADS 128  // Threads per block

#define MAXBITS 15              // maximum bits in a code
#define MAXLCODES 286           // maximum number of literal/length codes
#define MAXDCODES 30            // maximum number of distance codes
#define MAXCODES (MAXLCODES+MAXDCODES)  // maximum codes lengths to read
#define FIXLCODES 288           // number of fixed literal/length codes

#define LOG2LENLUT  10
#define LOG2DISTLUT 8

struct scratch_arr
{
    int16_t lengths[MAXCODES];  // descriptor code lengths
    int16_t offs[MAXBITS + 1];  // offset in symbol table for each length (scratch)
};

struct lut_arr
{
    int32_t lenlut[1 << LOG2LENLUT];    // LUT for length decoding
    int32_t distlut[1 << LOG2DISTLUT];  // LUT for fast distance decoding
};


// 4 batches of 32 symbols
#define LOG2_BATCH_COUNT   2 // 1..5
#define LOG2_BATCH_SIZE    5
#define BATCH_COUNT        (1 << LOG2_BATCH_COUNT)
#define BATCH_SIZE         (1 << LOG2_BATCH_SIZE)

struct xwarp_s
{
    int32_t batch_len[BATCH_COUNT]; // Length of each batch - <0:end, 0:not ready, >0:symbol count
    union {
        uint32_t symqueue[BATCH_COUNT * BATCH_SIZE];
        uint8_t symqueue8[BATCH_COUNT * BATCH_SIZE * 4];
    } u;
};

#define ENABLE_PREFETCH     1

#if ENABLE_PREFETCH
#define LOG2_PREFETCH_SIZE  9 // Must be at least LOG2_BATCH_SIZE+3
#define PREFETCH_SIZE       (1 << LOG2_PREFETCH_SIZE)

#define PREFETCH_ADDR32(q, p)    (uint32_t *)(&q.pref_data[(PREFETCH_SIZE - 4) & (size_t)(p)])
struct prefetch_queue_s
{
    const uint8_t *cur_p; // Prefetch location
    int run; // prefetcher will exit when run=0
    uint8_t pref_data[PREFETCH_SIZE];
};

#endif // ENABLE_PREFETCH


struct inflate_state_s {
    // output state
    uint8_t *out;           // output buffer
    uint8_t *outbase;       // start of output buffer
    uint8_t *outend;        // end of output buffer
    // Input state
    uint8_t *cur;           // input buffer
    uint8_t *end;           // end of input buffer

    uint2    bitbuf;        // bit buffer (64-bit)
    uint32_t bitpos;        // position in bit buffer

    int32_t err;
    int     btype;          // current block type
    int     blast;          // last block
    uint32_t stored_blk_len;// length of stored (uncompressed) block

    uint16_t first_slow_len;    // first code not in fast LUT
    uint16_t index_slow_len;
    uint16_t first_slow_dist;
    uint16_t index_slow_dist;

    volatile xwarp_s x;
#if ENABLE_PREFETCH
    volatile prefetch_queue_s pref;
#endif

    int16_t lencnt[MAXBITS+1];
    int16_t lensym[FIXLCODES]; // Assumes FIXLCODES >= MAXLCODES
    int16_t distcnt[MAXBITS+1];
    int16_t distsym[MAXDCODES];

    union {
        scratch_arr scratch;
        lut_arr lut;
    } u;
};

inline __device__ unsigned int bfe(unsigned int source, unsigned int bit_start, unsigned int num_bits)
{
    unsigned int bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "r"(bit_start), "r"(num_bits));
    return bits;
};


inline __device__ uint32_t showbits(inflate_state_s *s, uint32_t n)
{
    uint32_t next32 = __funnelshift_rc(s->bitbuf.x, s->bitbuf.y, s->bitpos);
    return (next32 & ((1 << n) - 1));
}

inline __device__ uint32_t nextbits32(inflate_state_s *s)
{
    return __funnelshift_rc(s->bitbuf.x, s->bitbuf.y, s->bitpos);
}

inline __device__ void skipbits(inflate_state_s *s, uint32_t n)
{
    uint32_t bitpos = s->bitpos + n;
    if (bitpos >= 32)
    {
        uint8_t *cur = s->cur + 8;
        s->bitbuf.x = s->bitbuf.y;
        s->bitbuf.y = (cur < s->end) ? *(uint32_t *)cur : 0;
        s->cur = cur - 4;
        bitpos &= 0x1f;
    }
    s->bitpos = bitpos;
}


// TODO: If we require 4-byte alignment of input bitstream & length (padded), reading bits would become quite a bit faster
__device__ uint32_t getbits(inflate_state_s *s, uint32_t n)
{
    uint32_t v = showbits(s, n);
    skipbits(s, n);
    return v;
}


/*
* Decode a code from the stream s using huffman table {symbols,counts}.
* Return the symbol or a negative value if there is an error.
* If all of the lengths are zero, i.e. an empty code, or if the code is 
* incomplete and an invalid code is received, then -10 is returned after
* reading MAXBITS bits.
*
* Format notes:
*
* - The codes as stored in the compressed data are bit-reversed relative to
*   a simple integer ordering of codes of the same lengths.  Hence below the
*   bits are pulled from the compressed data one at a time and used to
*   build the code value reversed from what is in the stream in order to
*   permit simple integer comparisons for decoding.  A table-based decoding
*   scheme (as used in zlib) does not need to do this reversal.
*
* - The first code for the shortest length is all zeros.  Subsequent codes of
*   the same length are simply integer increments of the previous code.  When
*   moving up a length, a zero bit is appended to the code.  For a complete
*   code, the last code of the longest length will be all ones.
*
* - Incomplete codes are handled by this decoder, since they are permitted
*   in the deflate format.  See the format notes for fixed() and dynamic().
*/
__device__ int decode(inflate_state_s *s, const int16_t *counts, const int16_t *symbols)
{
    unsigned int len;            // current number of bits in code
    unsigned int code;           // len bits being decoded
    unsigned int first;          // first code of length len
    unsigned int count;          // number of codes of length len
    uint32_t next32r = __brev(nextbits32(s));

    first = 0;
    for (len = 1; len <= MAXBITS; len++)
    {
        code = (next32r >> (32 - len)) - first;
        count = counts[len];
        if (code < count)       // if length len, return symbol
        {
            skipbits(s, len);
            return symbols[code];
        }
        symbols += count;                 // else update for next length
        first += count;
        first <<= 1;
    }
    return -10;                         // ran out of codes
}


/*
* Given the list of code lengths length[0..n-1] representing a canonical
* Huffman code for n symbols, construct the tables required to decode those
* codes.  Those tables are the number of codes of each length, and the symbols
* sorted by length, retaining their original order within each length.  The
* return value is zero for a complete code set, negative for an over-
* subscribed code set, and positive for an incomplete code set.  The tables
* can be used if the return value is zero or positive, but they cannot be used
* if the return value is negative.  If the return value is zero, it is not
* possible for decode() using that table to return an error--any stream of
* enough bits will resolve to a symbol.  If the return value is positive, then
* it is possible for decode() using that table to return an error for received
* codes past the end of the incomplete lengths.
*
* Not used by decode(), but used for error checking, count[0] is the number
* of the n symbols not in the code.  So n - count[0] is the number of
* codes.  This is useful for checking for incomplete codes that have more than
* one symbol, which is an error in a dynamic block.
*
* Assumption: for all i in 0..n-1, 0 <= length[i] <= MAXBITS
* This is assured by the construction of the length arrays in dynamic() and
* fixed() and is not verified by construct().
*
* Format notes:
*
* - Permitted and expected examples of incomplete codes are one of the fixed
*   codes and any code with a single symbol which in deflate is coded as one
*   bit instead of zero bits.  See the format notes for fixed() and dynamic().
*
* - Within a given code length, the symbols are kept in ascending order for
*   the code bits definition.
*/
__device__ int construct(inflate_state_s *s, int16_t *counts, int16_t *symbols, const int16_t *length, int n)
{
    int symbol;         // current symbol when stepping through length[]
    int len;            // current length when stepping through counts[]
    int left;           // number of possible codes left of current length
    int16_t *offs = s->u.scratch.offs;

    // count number of codes of each length
    for (len = 0; len <= MAXBITS; len++)
        counts[len] = 0;
    for (symbol = 0; symbol < n; symbol++)
        (counts[length[symbol]])++;   // assumes lengths are within bounds
    if (counts[0] == n)               // no codes!
        return 0;                     // complete, but decode() will fail

    // check for an over-subscribed or incomplete set of lengths
    left = 1;                           // one possible code of zero length
    for (len = 1; len <= MAXBITS; len++)
    {
        left <<= 1;                     // one more bit, double codes left
        left -= counts[len];            // deduct count from possible codes
        if (left < 0)
            return left;                // over-subscribed--return negative
    }                                   // left > 0 means incomplete

    // generate offsets into symbol table for each length for sorting
    offs[1] = 0;
    for (len = 1; len < MAXBITS; len++)
        offs[len + 1] = offs[len] + counts[len];

    // put symbols in table sorted by length, by symbol order within each length
    for (symbol = 0; symbol < n; symbol++)
        if (length[symbol] != 0)
            symbols[offs[length[symbol]]++] = symbol;

    // return zero for complete set, positive for incomplete set
    return left;
}



// permutation of code length codes
static const __device__ __constant__ uint8_t g_code_order[19 + 1] = { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 0xff };


// Dynamic block (custom huffman tables)
__device__ int init_dynamic(inflate_state_s *s)
{
    int nlen, ndist, ncode;             /* number of lengths in descriptor */
    int index;                          /* index of lengths[] */
    int err;                            /* construct() return value */
    int16_t *lengths = s->u.scratch.lengths;
    
    // get number of lengths in each table, check lengths
    nlen = getbits(s, 5) + 257;
    ndist = getbits(s, 5) + 1;
    ncode = getbits(s, 4) + 4;
    if (nlen > MAXLCODES || ndist > MAXDCODES)
    {
        return -3; // bad counts
    }
    // read code length code lengths (really), missing lengths are zero
    for (index = 0; index < ncode; index++)
        lengths[g_code_order[index]] = getbits(s, 3);
    for (; index < 19; index++)
        lengths[g_code_order[index]] = 0;

    // build huffman table for code lengths codes (use lencode temporarily)
    err = construct(s, s->lencnt, s->lensym, lengths, 19);
    if (err != 0)               // require complete code set here
        return -4;

    // read length/literal and distance code length tables
    index = 0;
    while (index < nlen + ndist)
    {
        int symbol = decode(s, s->lencnt, s->lensym);
        if (symbol < 0)
            return symbol;          // invalid symbol
        if (symbol < 16)                // length in 0..15
            lengths[index++] = symbol;
        else {                          // repeat instruction
            int len = 0;                // last length to repeat, assume repeating zeros
            if (symbol == 16) {         // repeat last length 3..6 times
                if (index == 0)
                    return -5;          // no last length!
                len = lengths[index - 1];       // last length
                symbol = 3 + getbits(s, 2);
            }
            else if (symbol == 17)      // repeat zero 3..10 times
                symbol = 3 + getbits(s, 3);
            else                        // == 18, repeat zero 11..138 times
                symbol = 11 + getbits(s, 7);
            if (index + symbol > nlen + ndist)
                return -6;              // too many lengths!
            while (symbol--)            // repeat last or zero symbol times
                lengths[index++] = len;
        }
    }

    // check for end-of-block code -- there better be one!
    if (lengths[256] == 0)
        return -9;

    // build huffman table for literal/length codes
    err = construct(s, s->lencnt, s->lensym, lengths, nlen);
    if (err && (err < 0 || nlen != s->lencnt[0] + s->lencnt[1]))
        return -7;      // incomplete code ok only for single length 1 code

    // build huffman table for distance codes
    err = construct(s, s->distcnt, s->distsym, &lengths[nlen], ndist);
    if (err && (err < 0 || ndist != s->distcnt[0] + s->distcnt[1]))
        return -8;      // incomplete code ok only for single length 1 code

    return 0;
}



/*
* Initializes a fixed codes block.
*
* Format notes:
*
* - This block type can be useful for compressing small amounts of data for
*   which the size of the code descriptions in a dynamic block exceeds the
*   benefit of custom codes for that block.  For fixed codes, no bits are
*   spent on code descriptions.  Instead the code lengths for literal/length
*   codes and distance codes are fixed.  The specific lengths for each symbol
*   can be seen in the "for" loops below.
*
* - The literal/length code is complete, but has two symbols that are invalid
*   and should result in an error if received.  This cannot be implemented
*   simply as an incomplete code since those two symbols are in the "middle"
*   of the code.  They are eight bits long and the longest literal/length\
*   code is nine bits.  Therefore the code must be constructed with those
*   symbols, and the invalid symbols must be detected after decoding.
*
* - The fixed distance codes also have two invalid symbols that should result
*   in an error if received.  Since all of the distance codes are the same
*   length, this can be implemented as an incomplete code.  Then the invalid
*   codes are detected while decoding.
*/
__device__ int init_fixed(inflate_state_s *s)
{
    int16_t *lengths = s->u.scratch.lengths;
    int symbol;

    // literal/length table
    for (symbol = 0; symbol < 144; symbol++)
        lengths[symbol] = 8;
    for (; symbol < 256; symbol++)
        lengths[symbol] = 9;
    for (; symbol < 280; symbol++)
        lengths[symbol] = 7;
    for (; symbol < FIXLCODES; symbol++)
        lengths[symbol] = 8;
    construct(s, s->lencnt, s->lensym, lengths, FIXLCODES);

    // distance table
    for (symbol = 0; symbol < MAXDCODES; symbol++)
        lengths[symbol] = 5;

    // build huffman table for distance codes
    construct(s, s->distcnt, s->distsym, lengths, MAXDCODES);

    return 0;
}


/*
* Decode literal/length and distance codes until an end-of-block code.
*
* Format notes:
*
* - Compressed data that is after the block type if fixed or after the code
*   description if dynamic is a combination of literals and length/distance
*   pairs terminated by and end-of-block code.  Literals are simply Huffman
*   coded bytes.  A length/distance pair is a coded length followed by a
*   coded distance to represent a string that occurs earlier in the
*   uncompressed data that occurs again at the current location.
*
* - Literals, lengths, and the end-of-block code are combined into a single
*   code of up to 286 symbols.  They are 256 literals (0..255), 29 length
*   symbols (257..285), and the end-of-block symbol (256).
*
* - There are 256 possible lengths (3..258), and so 29 symbols are not enough
*   to represent all of those.  Lengths 3..10 and 258 are in fact represented
*   by just a length symbol.  Lengths 11..257 are represented as a symbol and
*   some number of extra bits that are added as an integer to the base length
*   of the length symbol.  The number of extra bits is determined by the base
*   length symbol.  These are in the static arrays below, lens[] for the base
*   lengths and lext[] for the corresponding number of extra bits.
*
* - The reason that 258 gets its own symbol is that the longest length is used
*   often in highly redundant files.  Note that 258 can also be coded as the
*   base value 227 plus the maximum extra value of 31.  While a good deflate
*   should never do this, it is not an error, and should be decoded properly.
*
* - If a length is decoded, including its extra bits if any, then it is
*   followed a distance code.  There are up to 30 distance symbols.  Again
*   there are many more possible distances (1..32768), so extra bits are added
*   to a base value represented by the symbol.  The distances 1..4 get their
*   own symbol, but the rest require extra bits.  The base distances and
*   corresponding number of extra bits are below in the static arrays dist[]
*   and dext[].
*
* - Literal bytes are simply written to the output.  A length/distance pair is
*   an instruction to copy previously uncompressed bytes to the output.  The
*   copy is from distance bytes back in the output stream, copying for length
*   bytes.
*
* - Distances pointing before the beginning of the output data are not
*   permitted.
*
* - Overlapped copies, where the length is greater than the distance, are
*   allowed and common.  For example, a distance of one and a length of 258
*   simply copies the last byte 258 times.  A distance of four and a length of
*   twelve copies the last four bytes three times.  A simple forward copy
*   ignoring whether the length is greater than the distance or not implements
*   this correctly.  You should not use memcpy() since its behavior is not
*   defined for overlapped arrays.  You should not use memmove() or bcopy()
*   since though their behavior -is- defined for overlapping arrays, it is
*   defined to do the wrong thing in this case.
*/

// permutation of code length codes
static const __device__ __constant__ uint16_t g_lens[29] = { // Size base for length codes 257..285
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
static const __device__ __constant__ uint16_t g_lext[29] = { // Extra bits for length codes 257..285
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4,
    5, 5, 5, 5, 0 };

static const __device__ __constant__ uint16_t g_dists[30] = { // Offset base for distance codes 0..29
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577 };
static const __device__ __constant__ uint16_t g_dext[30] = { // Extra bits for distance codes 0..29
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11,
    12, 12, 13, 13 };



// Thread 0 only: decode bitstreams and output symbols into the symbol queue
__device__ void decode_symbols(inflate_state_s *s)
{
    uint32_t bitpos = s->bitpos;
    uint2 bitbuf = s->bitbuf;
    uint8_t *cur = s->cur;
    uint8_t *end = s->end;
    int32_t batch = 0;
    int32_t sym, batch_len;

    do
    {
        volatile uint32_t *b = &s->x.u.symqueue[batch * BATCH_SIZE];
        // Wait for the next batch entry to be empty
    #if ENABLE_PREFETCH
        // Wait for prefetcher to fetch a worst-case of 48 bits per symbol
        while ((*(volatile int32_t *)&s->pref.cur_p - (int32_t)(size_t)cur < BATCH_SIZE*6)
            || (s->x.batch_len[batch] != 0))
    #else
        while (s->x.batch_len[batch] != 0)
    #endif
        {
            NANOSLEEP(100);
        }
        batch_len = 0;
    #if ENABLE_PREFETCH
        if (cur + (bitpos>>3) >= end)
        {
            s->err = 1;
            break;
        }
    #endif
        // Inner loop decoding symbols
        do
        {
            uint32_t next32 = __funnelshift_rc(bitbuf.x, bitbuf.y, bitpos);  // nextbits32(s);
            uint32_t len;
            sym = s->u.lut.lenlut[next32 & ((1 << LOG2LENLUT) - 1)];
            if ((uint32_t)sym < (uint32_t)(0x100<<5))
            {
                // We can lookup a second symbol if this was a short literal
                len = sym & 0x1f;
                sym >>= 5;
                b[batch_len++] = sym;
                next32 >>= len;
                bitpos += len;
                sym = s->u.lut.lenlut[next32 & ((1 << LOG2LENLUT) - 1)];
            }
            if (sym > 0) // short symbol
            {
                len = sym & 0x1f;
                sym = ((sym >> 5) & 0x3ff) + ((next32 >> (sym >> 24)) & ((sym >> 16) & 0x1f));
            }
            else
            {
                // Slow length path
                uint32_t next32r = __brev(next32);
                const int16_t *symbols = &s->lensym[s->index_slow_len];
                unsigned int first = s->first_slow_len;
                int lext;
                #pragma unroll 1
                for (len = LOG2LENLUT+1; len <= MAXBITS; len++)
                {
                    unsigned int code = (next32r >> (32 - len)) - first;
                    unsigned int count = s->lencnt[len];
                    if (code < count)       // if length len, return symbol
                    {
                        sym = symbols[code];
                        break;
                    }
                    symbols += count;                 // else update for next length
                    first += count;
                    first <<= 1;
                }
                if (len > MAXBITS)
                {
                    s->err = -10;
                    sym = 256;
                    len = 0;
                }
                if (sym > 256)
                {
                    sym -= 257;
                    lext = g_lext[sym];
                    sym = 256 + g_lens[sym] + bfe(next32, len, lext);
                    len += lext;
                }
            }
            if (sym > 256)
            {
                int dist, dext;
                // skipbits(s, len) inlined - no limit check
                bitpos += len;
                if (bitpos >= 32)
                {
                    bitbuf.x = bitbuf.y;
                #if ENABLE_PREFETCH
                    bitbuf.y = *PREFETCH_ADDR32(s->pref, cur + 8);
                    cur += 4;
                #else
                    cur += 8;
                    bitbuf.y = (cur < end) ? *(const uint32_t *)cur : 0;
                    cur -= 4;
                #endif
                    bitpos &= 0x1f;
                }
                // get distance
                next32 = __funnelshift_rc(bitbuf.x, bitbuf.y, bitpos);  // nextbits32(s);
                dist = s->u.lut.distlut[next32 & ((1 << LOG2DISTLUT) - 1)];
                if (dist > 0)
                {
                    len = dist & 0x1f;
                    dext = bfe(dist, 20, 5);
                    dist = bfe(dist, 5, 15);
                    sym |= (dist + bfe(next32, len, dext)) << 16;
                    len += dext;
                }
                else
                {
                    uint32_t next32r = __brev(next32);
                    const int16_t *symbols = &s->distsym[s->index_slow_dist];
                    unsigned int first = s->first_slow_dist;
                    #pragma unroll 1
                    for (len = LOG2DISTLUT + 1; len <= MAXBITS; len++)
                    {
                        unsigned int code = (next32r >> (32 - len)) - first;
                        unsigned int count = s->distcnt[len];
                        if (code < count)       // if length len, return symbol
                        {
                            dist = symbols[code];
                            break;
                        }
                        symbols += count;                 // else update for next length
                        first += count;
                        first <<= 1;
                    }
                    if (len > MAXBITS)
                    {
                        s->err = -10;
                        sym = 256;
                        len = 0;
                    }
                    else
                    {
                        dext = g_dext[dist];
                        sym |= (g_dists[dist] + bfe(next32, len, dext)) << 16;
                        len += dext;
                    }
                }
            }
            // skipbits(s, len) inlined with added error check for reading past the end of the input buffer
            bitpos += len;
            if (bitpos >= 32)
            {
                bitbuf.x = bitbuf.y;
            #if ENABLE_PREFETCH
                bitbuf.y = *PREFETCH_ADDR32(s->pref, cur + 8);
                cur += 4;
            #else
                cur += 8;
                if (cur < end)
                {
                    bitbuf.y = *(const uint32_t *)cur;
                    cur -= 4;
                }
                else
                {
                    bitbuf.y = 0;
                    cur -= 4;
                    if (cur > end)
                    {
                        s->err = 1;
                        sym = 256;
                    }
                }
            #endif
                bitpos &= 0x1f;
            }
            if (sym == 256)
                break;
            b[batch_len++] = sym;
        } while (batch_len < BATCH_SIZE-1);
        s->x.batch_len[batch] = batch_len;
    #if ENABLE_PREFETCH
        ((volatile inflate_state_s *)s)->cur = cur;
    #endif
        if (batch_len != 0)
            batch = (batch + 1) & (BATCH_COUNT - 1);
    } while (sym != 256);

    while (s->x.batch_len[batch] != 0)
    {
        NANOSLEEP(150);
    }
    s->x.batch_len[batch] = -1;
    s->bitbuf = bitbuf;
    s->bitpos = bitpos;
#if !ENABLE_PREFETCH
    s->cur = cur;
#endif
}


// Build lookup tables for faster decode
// LUT format is symbols*16+length
__device__ void init_length_lut(inflate_state_s *s, int t)
{
    int32_t *lut = s->u.lut.lenlut;

    for (uint32_t bits = t; bits < (1 << LOG2LENLUT); bits += NUMTHREADS)
    {
        const int16_t *cnt = s->lencnt;
        const int16_t *symbols = s->lensym;
        int sym = -10 << 5;
        unsigned int first = 0;
        unsigned int rbits = __brev(bits) >> (32 - LOG2LENLUT);
        for (unsigned int len = 1; len <= LOG2LENLUT; len++)
        {
            unsigned int code = (rbits >> (LOG2LENLUT - len)) - first;
            unsigned int count = cnt[len];
            if (code < count)
            {
                sym = symbols[code];
                if (sym > 256)
                {
                    int lext = g_lext[sym - 257];
                    sym = (256 + g_lens[sym - 257]) | (((1 << lext) - 1) << (16-5)) | (len << (24-5));
                    len += lext;
                }
                sym = (sym << 5) | len;
                break;
            }
            symbols += count;                 // else update for next length
            first += count;
            first <<= 1;
        }
        lut[bits] = sym;
    }
    if (!t)
    {
        unsigned int first = 0;
        unsigned int index = 0;
        const int16_t *cnt = s->lencnt;
        for (unsigned int len = 1; len <= LOG2LENLUT; len++)
        {
            unsigned int count = cnt[len];
            index += count;
            first += count;
            first <<= 1;
        }
        s->first_slow_len = first;
        s->index_slow_len = index;
    }
}


// Build lookup tables for faster decode of distance symbol
// LUT format is symbols*16+length
__device__ void init_distance_lut(inflate_state_s *s, int t)
{
    int32_t *lut = s->u.lut.distlut;

    for (uint32_t bits = t; bits < (1 << LOG2DISTLUT); bits += NUMTHREADS)
    {
        const int16_t *cnt = s->distcnt;
        const int16_t *symbols = s->distsym;
        int sym = 0;
        unsigned int first = 0;
        unsigned int rbits = __brev(bits) >> (32 - LOG2DISTLUT);
        for (unsigned int len = 1; len <= LOG2DISTLUT; len++)
        {
            unsigned int code = (rbits >> (LOG2DISTLUT - len)) - first;
            unsigned int count = cnt[len];
            if (code < count)
            {
                int dist = symbols[code];
                int dext = g_dext[dist];
                sym = g_dists[dist] | (dext << 15);
                sym = (sym << 5) | len;
                break;
            }
            symbols += count;                 // else update for next length
            first += count;
            first <<= 1;
        }
        lut[bits] = sym;
    }
    if (!t)
    {
        unsigned int first = 0;
        unsigned int index = 0;
        const int16_t *cnt = s->distcnt;
        for (unsigned int len = 1; len <= LOG2DISTLUT; len++)
        {
            unsigned int count = cnt[len];
            index += count;
            first += count;
            first <<= 1;
        }
        s->first_slow_dist = first;
        s->index_slow_dist = index;
    }
}


// WARP1: process symbols and output uncompressed stream
__device__ void process_symbols(inflate_state_s *s, int t)
{
    uint8_t *out = s->out;
    const uint8_t *outend = s->outend;
    const uint8_t *outbase = s->outbase;
    int batch = 0;

    do
    {
        volatile uint32_t *b = &s->x.u.symqueue[batch * BATCH_SIZE];
        int batch_len, pos;
        int32_t symt;
        uint32_t lit_mask;

        if (t == 0)
        {
            while ((batch_len = s->x.batch_len[batch]) == 0)
            {
                NANOSLEEP(100);
            }
        }
        else
        {
            batch_len = 0;
        }
        batch_len = SHFL0(batch_len);
        if (batch_len < 0)
        {
            break;
        }

        symt = (t < batch_len) ? b[t] : 256;
        lit_mask = BALLOT(symt >= 256);
        pos = min((__ffs(lit_mask) - 1) & 0xff, 32);
        if (t == 0)
        {
            s->x.batch_len[batch] = 0;
        }
        if (t < pos && out+t < outend)
        {
            out[t] = symt;
        }
        out += pos;
        batch_len -= pos;
        while (batch_len > 0)
        {
            int dist, len, symbol;

            // Process a non-literal symbol
            symbol = SHFL(symt, pos);
            len = max((symbol & 0xffff) - 256, 0); // max should be unnecessary, but just in case
            dist = symbol >> 16;
            for (int i = t; i < len; i += 32)
            {
                const uint8_t *src = out + ((i >= dist) ? (i % dist) : i) - dist;
                uint8_t b = (src < outbase) ? 0 : *src;
                if (out + i < outend)
                {
                    out[i] = b;
                }
            }
            out += len;
            pos++;
            batch_len--;
            // Process subsequent literals, if any
            if (!((lit_mask >> pos) & 1))
            {
                len = min((__ffs(lit_mask >> pos) - 1) & 0xff, batch_len);
                symbol = SHFL(symt, (pos + t) & 0x1f);
                if (t < len && out + t < outend)
                {
                    out[t] = symbol;
                }
                out += len;
                pos += len;
                batch_len -= len;
            }
        }
        batch = (batch + 1) & (BATCH_COUNT - 1);
    } while (1);

    if (t == 0)
    {
        s->out = out;
    }
}


/*
* Initializes a stored block.
*
* Format notes:
*
* - After the two-bit stored block type (00), the stored block length and
*   stored bytes are byte-aligned for fast copying.  Therefore any leftover
*   bits in the byte that has the last bit of the type, as many as seven, are
*   discarded.  The value of the discarded bits are not defined and should not
*   be checked against any expectation.
*
* - The second inverted copy of the stored block length does not have to be
*   checked, but it's probably a good idea to do so anyway.
*
* - A stored block can have zero length.  This is sometimes used to byte-align
*   subsets of the compressed data for random access or partial recovery.
*/
__device__ int init_stored(inflate_state_s *s)
{
    uint32_t len, nlen;   // length of stored block

    // Byte align
    if (s->bitpos & 7)
    {
        skipbits(s,  8 - (s->bitpos & 7));
    }
    if (s->cur + (s->bitpos >> 3) >= s->end)
    {
        return 2; // Not enough input
    }
    // get length and check against its one's complement
    len = getbits(s, 16);
    nlen = getbits(s, 16);
    if (len != (nlen ^ 0xffff))
    {
        return -2; // didn't match complement!
    }
    if (s->cur + (s->bitpos >> 3) + len > s->end)
    {
        return 2; // Not enough input
    }
    s->stored_blk_len = len;

    // done with a valid stored block
    return 0;
}


// Copy bytes from stored block to destination
__device__ void copy_stored(inflate_state_s *s, int t)
{
    int len = s->stored_blk_len;
    uint8_t *cur = s->cur + (s->bitpos >> 3);
    uint8_t *out = s->out;
    uint8_t *outend = s->outend;
    uint8_t *cur4;
    int slow_bytes = min(len, (int)((16 - (size_t)out) & 0xf));
    int fast_bytes, bitpos;

    // Slow copy until output is 16B aligned
    if (slow_bytes)
    {
        for (int i = t; i < slow_bytes; i += NUMTHREADS)
        {
            if (out + i < outend)
            {
                out[i] = cur[i]; // Input range has already been validated in init_stored()
            }
        }
        cur += slow_bytes;
        out += slow_bytes;
        len -= slow_bytes;
    }
    fast_bytes = len;
    if (out < outend)
    {
        fast_bytes = (int)min((size_t)fast_bytes, (outend - out));
    }
    fast_bytes &= ~0xf;
    bitpos = ((int)(3 & (size_t)cur)) << 3;
    cur4 = cur - (bitpos >> 3);
    if (out < outend)
    {
        // Fast copy 16 bytes at a time
        for (int i = t*16; i < fast_bytes; i += NUMTHREADS*16)
        {
            uint4 u;
            u.x = *(const uint32_t *)(cur4 + i + 0*4);
            u.y = *(const uint32_t *)(cur4 + i + 1*4);
            u.z = *(const uint32_t *)(cur4 + i + 2*4);
            u.w = *(const uint32_t *)(cur4 + i + 3*4);
            if (bitpos != 0)
            {
                uint32_t v = (bitpos != 0) ? *(const uint32_t *)(cur4 + i + 4*4) : 0;
                u.x = __funnelshift_rc(u.x, u.y, bitpos);
                u.y = __funnelshift_rc(u.y, u.z, bitpos);
                u.z = __funnelshift_rc(u.z, u.w, bitpos);
                u.w = __funnelshift_rc(u.w, v, bitpos);
            }
            *(uint4 *)(out + i) = u;
        }
    }
    cur += fast_bytes;
    out += fast_bytes;
    len -= fast_bytes;
    // Slow copy for remaining bytes
    for (int i = t; i < len; i += NUMTHREADS)
    {
        if (out + i < outend)
        {
            out[i] = cur[i]; // Input range has already been validated in init_stored()
        }
    }
    out += len;
    __syncthreads();
    if (t == 0)
    {
        // Reset bitstream to end of block
        uint8_t *p = cur + len;
        uint32_t prefix_bytes = (uint32_t)(((size_t)p) & 3);
        p -= prefix_bytes;
        s->cur = p;
        s->bitbuf.x = (p < s->end) ? *(uint32_t *)p : 0;
        p += 4;
        s->bitbuf.y = (p < s->end) ? *(uint32_t *)p : 0;
        s->bitpos = prefix_bytes * 8;
        s->out = out;
    }
}


#if ENABLE_PREFETCH
__device__ void init_prefetcher(inflate_state_s *s, int t)
{
    if (t == 0)
    {
        s->pref.cur_p = s->cur;
        s->pref.run = 1;
    }
}


__device__ void prefetch_warp(volatile inflate_state_s *s, int t)
{
    const uint8_t *cur_p = s->pref.cur_p;
    const uint8_t *end = s->end;
    while (SHFL0((t == 0) ? s->pref.run : 0))
    {
        int32_t cur_lo = (int32_t)(size_t)cur_p;
        int do_pref = SHFL0((t == 0) ? (cur_lo - *(volatile int32_t *)&s->cur < PREFETCH_SIZE-32*4-4) : 0);
        if (do_pref)
        {
            const uint8_t *p = cur_p + 4*t;
            *PREFETCH_ADDR32(s->pref, p) = (p < end) ? *(const uint32_t *)p : 0;
            cur_p += 4*32;
            __threadfence_block();
            SYNCWARP();
            if (!t)
            {
                s->pref.cur_p = cur_p;
                __threadfence_block();
            }
        }
        else if (t == 0)
        {
            NANOSLEEP(150);
        }
    }
}
#endif // ENABLE_PREFETCH


#define GZ_FLG_FTEXT    0x01    // ASCII text hint
#define GZ_FLG_FHCRC    0x02    // Header CRC present
#define GZ_FLG_FEXTRA   0x04    // Extra fields present
#define GZ_FLG_FNAME    0x08    // Original file name present
#define GZ_FLG_FCOMMENT 0x10    // Comment present

__device__ int parse_gzip_header(const uint8_t *src, size_t src_size)
{
    int hdr_len = -1;
    
    if (src_size >= 18)
    {
        uint32_t sig = (src[0] << 16) | (src[1] << 8) | src[2];
        if (sig == 0x1f8b08) // 24-bit GZIP inflate signature {0x1f, 0x8b, 0x08}
        {
            uint32_t flags = src[3];
            hdr_len = 10;
            if (flags & GZ_FLG_FEXTRA) // Extra fields present
            {
                int xlen = src[hdr_len] | (src[hdr_len+1] << 8);
                hdr_len += xlen;
                if (hdr_len >= src_size)
                    return -1;
            }
            if (flags & GZ_FLG_FNAME) // Original file name present
            {
                // Skip zero-terminated string
                do {
                    if (hdr_len >= src_size)
                        return -1;
                } while (src[hdr_len++] != 0);
            }
            if (flags & GZ_FLG_FCOMMENT) // Comment present
            {
                // Skip zero-terminated string
                do {
                    if (hdr_len >= src_size)
                        return -1;
                } while (src[hdr_len++] != 0);
            }
            if (flags & GZ_FLG_FHCRC) // Header CRC present
            {
                hdr_len += 2;
            }
            if (hdr_len+8 >= src_size)
                hdr_len = -1;
        }
    }
    return hdr_len;
}



// blockDim {128,1,1}
__global__ void __launch_bounds__(NUMTHREADS)
inflate_kernel(gpu_inflate_input_s *inputs, gpu_inflate_status_s *outputs, int parse_hdr)
{
    __shared__ __align__(16) inflate_state_s state_g;

    int t = threadIdx.x;
    int z = blockIdx.x;
    inflate_state_s *state = &state_g;
    
    if (!t)
    {
        uint8_t *p = (uint8_t *)inputs[z].srcDevice;
        size_t src_size = inputs[z].srcSize;
        uint32_t prefix_bytes;
        // Parse header if needed
        state->err = 0;
        if (parse_hdr)
        {
            int hdr_len = parse_gzip_header(p, src_size);
            src_size = (src_size >= 8) ? src_size - 8 : 0;  // ignore footer
            if (hdr_len >= 0) {
                p += hdr_len;
                src_size -= hdr_len;
            } else {
                state->err = hdr_len;
            }
        }
        // Initialize shared state
        state->out = (uint8_t *)inputs[z].dstDevice;
        state->outbase = state->out;
        state->outend = state->out + inputs[z].dstSize;
        state->end = p + src_size;
        prefix_bytes = (uint32_t)(((size_t)p) & 3);
        p -= prefix_bytes;
        state->cur = p;
        state->bitbuf.x = (p < state->end) ? *(uint32_t *)p : 0;
        p += 4;
        state->bitbuf.y = (p < state->end) ? *(uint32_t *)p : 0;
        state->bitpos = prefix_bytes * 8;
    }
    __syncthreads();
    while (!state->err)
    {
        if (!t)
        {
            if (state->cur + (state->bitpos >> 3) >= state->end)
                state->err = 2;
            else
            {
                state->blast = getbits(state, 1);
                state->btype = getbits(state, 2);
                if (state->btype == 0)
                    state->err = init_stored(state);
                else if (state->btype == 1)
                    state->err = init_fixed(state);
                else if (state->btype == 2)
                    state->err = init_dynamic(state);
                else
                    state->err = -1; // Invalid block
            }
        }
        __syncthreads();
        if (!state->err && (state->btype == 1 || state->btype == 2))
        {
            init_length_lut(state, t);
            init_distance_lut(state, t);
        #if ENABLE_PREFETCH
            init_prefetcher(state, t);
        #endif
            if (t < BATCH_COUNT)
            {
                state->x.batch_len[t] = 0;
            }
            __syncthreads();
            // decode data until end-of-block code
            if (t < 1*32)
            {
                // WARP0
                if (!t)
                {
                    decode_symbols(state);
                #if ENABLE_PREFETCH
                    state->pref.run = 0;
                #endif
                }
            }
            else if (t < 2*32)
            {
                // WARP1
                process_symbols(state, t & 0x1f);
            }
        #if ENABLE_PREFETCH
            else if (t < 3*32)
            {
                // WARP3: Prefetcher
                prefetch_warp(state, t & 0x1f);
            }
        #endif
        }
        else if (!state->err && state->btype == 0)
        {
            copy_stored(state, t);
        }
        __syncthreads();
        if (state->blast)
            break;
    }

    if (!t)
    {
        if (state->err == 0 && state->cur + ((state->bitpos + 7) >> 3) > state->end)
        {
            // Read past the end of the input buffer
            state->err = 2;
        }
        else if (state->err == 0 && state->out > state->outend)
        {
            // Output buffer too small
            state->err = 1;
        }
        outputs[z].bytes_written = state->out - state->outbase;
        outputs[z].status = state->err;
        outputs[z].reserved = (int)(state->end - state->cur);
    }
}


// blockDim {1024,1,1}
__global__ void __launch_bounds__(1024)
copy_uncompressed_kernel(gpu_inflate_input_s *inputs)
{
    __shared__ const uint8_t * volatile src_g;
    __shared__ uint8_t * volatile dst_g;
    __shared__ uint32_t volatile copy_len_g;

    uint32_t t = threadIdx.x;
    uint32_t z = blockIdx.x;
    const uint8_t *src;
    uint8_t *dst;
    uint32_t len, src_align_bytes, src_align_bits, dst_align_bytes;

    if (!t) {
        src = reinterpret_cast<const uint8_t *>(inputs[z].srcDevice);
        dst = reinterpret_cast<uint8_t *>(inputs[z].dstDevice);
        len = min((uint32_t)inputs[z].srcSize, (uint32_t)inputs[z].dstSize);
        src_g = src;
        dst_g = dst;
        copy_len_g = len;
    }
    __syncthreads();
    src = src_g;
    dst = dst_g;
    len = copy_len_g;
    // Align output to 32-bit
    dst_align_bytes = 3 & -reinterpret_cast<intptr_t>(dst);
    if (dst_align_bytes != 0) {
        uint32_t align_len = min(dst_align_bytes, len);
        if (t < align_len) {
            dst[t] = src[t];
        }
        src += align_len;
        dst += align_len;
        len -= align_len;
    }
    src_align_bytes = (uint32_t)(3 & reinterpret_cast<uintptr_t>(src));
    src_align_bits = src_align_bytes << 3;
    while (len >= 32) {
        const uint32_t *src32 = reinterpret_cast<const uint32_t *>(src - src_align_bytes);
        uint32_t copy_cnt = min(len >> 2, 1024);
        if (t < copy_cnt) {
            uint32_t v = src32[t];
            if (src_align_bits != 0) {
                v = __funnelshift_r(v, src32[t + 1], src_align_bits);
            }
            reinterpret_cast<uint32_t *>(dst)[t] = v;
        }
        src += copy_cnt * 4;
        dst += copy_cnt * 4;
        len -= copy_cnt * 4;
    }
    if (t < len) {
        dst[t] = src[t];
    }
}



cudaError_t __host__ gpuinflate(gpu_inflate_input_s *inputs, gpu_inflate_status_s *outputs, int count, int parse_hdr, cudaStream_t stream)
{
    if (count > 0) {
        inflate_kernel <<< count, NUMTHREADS, 0, stream >>>(inputs, outputs, parse_hdr);
    }
    return cudaSuccess;
}


cudaError_t __host__ gpu_copy_uncompressed_blocks(gpu_inflate_input_s *inputs, int count, cudaStream_t stream)
{
    if (count > 0) {
        copy_uncompressed_kernel <<< count, 1024, 0, stream >>>(inputs);
    }
    return cudaSuccess;
}


} // namespace io
} // namespace cudf

