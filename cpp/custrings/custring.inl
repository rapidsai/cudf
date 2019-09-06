/*
* Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <memory.h>
#include <cmath>
#include <limits>

namespace custr
{

// convert string with numerical characters to number
__device__ inline long stol( const char* str, unsigned int bytes )
{
    const char* ptr = str;
    if( !ptr || !bytes )
        return 0; // probably should be an assert
    long value = 0;
    int sign = 1;
    if( *ptr == '-' || *ptr == '+' )
    {
        sign = (*ptr=='-' ? -1:1);
        ++ptr;
        --bytes;
    }
    for( unsigned int idx=0; idx < bytes; ++idx )
    {
        char chr = *ptr++;
        if( chr < '0' || chr > '9' )
            break;
        value = (value * 10) + (long)(chr - '0');
    }
    return value * (long)sign;
}

__device__ inline int stoi( const char* str, unsigned int bytes )
{
    return (int)stol(str,bytes);
}

__device__ inline unsigned long stoul( const char* str, unsigned int bytes )
{
    const char* ptr = str;
    if( !ptr || !bytes )
        return 0; // probably should be an assert
    unsigned long value = 0;
    for( unsigned int idx=0; idx < bytes; ++idx )
    {
        char chr = *ptr++;
        if( chr < '0' || chr > '9' )
            break;
        value = (value * 10) + (unsigned long)(chr - '0');
    }
    return value;
}

__device__ inline double stod( const char* str, unsigned int bytes )
{
    char* ptr = (char*)str;
    if( !ptr || !bytes )
        return 0.0; // probably should be an assert
    // special strings
    if( compare(str,bytes,"nan",3)==0 )
        return std::numeric_limits<double>::quiet_NaN();
    if( compare(str,bytes,"inf",3)==0 )
        return std::numeric_limits<double>::infinity();
    if( compare(str,bytes,"-inf",4)==0 )
        return -std::numeric_limits<double>::infinity();
    char* end = ptr + bytes;
    double sign = 1.0;
    if(*ptr == '-' || *ptr == '+')
    {
        sign = (*ptr == '-' ? -1 : 1);
        ++ptr;
    }
    unsigned long max_mantissa = 0x0FFFFFFFFFFFFF;
    unsigned long digits = 0;
    int exp_off = 0;
    bool decimal = false;
    while( ptr < end )
    {
        char ch = *ptr;
        if(ch == '.')
        {
            decimal = true;
            ++ptr;
            continue;
        }
        if(ch < '0' || ch > '9')
            break;
        if( digits > max_mantissa )
            exp_off += (int)!decimal;
        else
        {
            digits = (digits * 10L) + (unsigned long)(ch-'0');
            if( digits > max_mantissa )
            {
                digits = digits / 10L;
                exp_off += (int)!decimal;
            }
            else
                exp_off -= (int)decimal;
        }
        ++ptr;
    }
    // check for exponent char
    int exp10 = 0;
    int exp_sign = 1;
    if( ptr < end )
    {
        char ch = *ptr++;
        if( ch=='e' || ch=='E' )
        {
            if( ptr < end )
            {
                ch = *ptr++;
                if( ch=='-' || ch=='+' )
                    exp_sign = (ch=='-' ? -1 : 1);
                while( ptr < end )
                {
                    ch = *ptr++;
                    if(ch < '0' || ch > '9')
                        break;
                    exp10 = (exp10 * 10) + (int)(ch-'0');
                }
            }
        }
    }
    exp10 *= exp_sign;
    exp10 += exp_off;
    if( exp10 > 308 )
        return sign > 0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    else if( exp10 < -308 )
        return 0.0;
    double value = (double)digits * pow(10.0,(double)exp10);
    return (value * sign);
}

__device__ inline float stof( const char* str, unsigned int bytes )
{
    return (float)stod(str,bytes);
}

/**
 * Implementation based on
 * https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 * and adapted from
 * https://github.com/rapidsai/cudf/cpp/src/io/csv/type_conversion.cuh
 *
 *
 */
__device__ inline unsigned int hash( const char* str, unsigned int bytes )
{
    unsigned int seed = 31; // prime number
    unsigned int hash = 0;
    // previous hash function -- lots of collisions
    //for( unsigned int i = 0; i < bytes; i++ )
    //    hash = hash * seed + str[i];

    // new hash function starts here
    auto getblock32 = [] __device__(const unsigned int* p, int i) -> unsigned int {
      // Individual byte reads for possible unaligned accesses
      auto q = (const unsigned char*)(p + i);
      return q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
    };

    auto rotl32 = [] __device__(unsigned int x, char r) -> unsigned int {
      return (x << r) | (x >> (32 - r));
    };

    auto fmix32 = [] __device__(unsigned int h) -> unsigned int {
      h ^= h >> 16;
      h *= 0x85ebca6b;
      h ^= h >> 13;
      h *= 0xc2b2ae35;
      h ^= h >> 16;
      return h;
    };

    const int len = (int)bytes;
    const unsigned char* const data = (const unsigned char*)str;
    const int nblocks = len / 4;
    unsigned int h1 = seed;
    constexpr unsigned int c1 = 0xcc9e2d51;
    constexpr unsigned int c2 = 0x1b873593;
    //----------
    // body
    const unsigned int* const blocks = (const unsigned int*)(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
      unsigned int k1 = getblock32(blocks, i);
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }
    //----------
    // tail
    const unsigned char* tail = (const unsigned char*)(data + nblocks * 4);
    unsigned int k1 = 0;
    switch (len & 3) {
      case 3:
        k1 ^= tail[2] << 16;
      case 2:
        k1 ^= tail[1] << 8;
      case 1:
        k1 ^= tail[0];
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
    };
    //----------
    // finalization
    h1 ^= len;
    h1 = fmix32(h1);
    hash = h1;
    return hash;
}

// This could possibly be optimized using vectorized loading on the character arrays.
// 0	They compare equal
// <0	Either the value of the first character of this string that does not match is lower in the arg string,
//      or all compared characters match but the arg string is shorter.
// >0	Either the value of the first character of this string that does not match is greater in the arg string,
//      or all compared characters match but the arg string is longer.
__device__ inline int compare(const char* src, unsigned int src_bytes, const char* tgt, unsigned int tgt_bytes )
{
    const unsigned char* ptr1 = reinterpret_cast<const unsigned char*>(src);
    if( !ptr1 )
        return -1;
    const unsigned char* ptr2 = reinterpret_cast<const unsigned char*>(tgt);
    if( !ptr2 )
        return 1;
    unsigned int idx = 0;
    for(; (idx < src_bytes) && (idx < tgt_bytes); ++idx)
    {
        if(*ptr1 != *ptr2)
            return (int)*ptr1 - (int)*ptr2;
        ++ptr1;
        ++ptr2;
    }
    if( idx < src_bytes )
        return 1;
    if( idx < tgt_bytes )
        return -1;
    return 0;
}

//
__device__ inline void copy( char* dst, unsigned int bytes, const char* src )
{
    memcpy(dst,src,bytes);
}

} // end of custr namespace