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
    int sign = 1, size = (int)bytes;
    if( *ptr == '-' || *ptr == '+' )
    {
        sign = (*ptr=='-' ? -1:1);
        ++ptr;
        --size;
    }
    for( int idx=0; idx < size; ++idx )
    {
        char chr = *ptr++;
        if( chr < '0' || chr > '9' )
            break;
        value = (value * 10) + (long)(chr - '0');
    }
    return value * sign;
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
    int size = (int)bytes;
    for( int idx=0; idx < size; ++idx )
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
    unsigned int digits = 0;
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
        digits = (digits * 10) + (unsigned int)(ch-'0');
        exp_off -= (int)decimal;
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

__device__ inline int compare(const char* src, unsigned int sbytes, const char* tgt, unsigned int tbytes )
{
    const char* ptr1 = src;
    if( !ptr1 )
        return -1;
    const char* ptr2 = tgt;
    if( !ptr2 )
        return 1;
    unsigned int len1 = sbytes;
    unsigned int len2 = tbytes;
    unsigned int idx;
    for(idx = 0; (idx < len1) && (idx < len2); ++idx)
    {
        if (*ptr1 != *ptr2)
            return (unsigned int)*ptr1 - (unsigned int)*ptr2;
        ptr1++;
        ptr2++;
    }
    if( idx < len1 )
        return 1;
    if( idx < len2 )
        return -1;
    return 0;
}

//
__device__ inline int find( const char* sptr, unsigned int sz, const char* str, unsigned int bytes )
{
    if(!sptr || !str || (sz < bytes))
        return -1;
    unsigned int end = sz - bytes;
    char* ptr1 = (char*)sptr;
    char* ptr2 = (char*)str;
    for(int idx=0; idx < end; ++idx)
    {
        bool match = true;
        for( int jdx=0; jdx < bytes; ++jdx )
        {
            if(ptr1[jdx] == ptr2[jdx] )
                continue;
            match = false;
            break;
        }
        if( match )
            return idx; // chars_in_string(sptr,idx);
        ptr1++;
    }
    return -1;
}

__device__ inline int rfind( const char* sptr, unsigned int sz, const char* str, unsigned int bytes )
{
    if(!sptr || !str || (sz < bytes) )
        return -1;
    unsigned end = sz - bytes;
    char* ptr1 = (char*)sptr + end;
    char* ptr2 = (char*)str;
    for(int idx=0; idx < end; ++idx)
    {
        bool match = true;
        for( int jdx=0; jdx < bytes; ++jdx )
        {
            if(ptr1[jdx] == ptr2[jdx] )
                continue;
            match = false;
            break;
        }
        if( match )
            return sz - bytes - idx; //chars_in_string(sptr,end - idx);
        ptr1--; // go backwards
    }
    return -1;
}

//
__device__ inline void copy( char* dst, unsigned int bytes, const char* src )
{
    memcpy(dst,src,bytes);
}

}