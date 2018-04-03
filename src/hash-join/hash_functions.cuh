/* Copyright 2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#ifndef HASH_FUNCTIONS_CUH
#define HASH_FUNCTIONS_CUH

template <typename Key>
struct make_positive
{
    typedef Key argument_type;
    typedef size_t result_type;
    __forceinline__ 
    __host__ __device__ result_type operator()(const Key& key) const 
    {
        return key < 0 ? -1*key : key;
    }
};

//MurmurHash3_32 implementation from https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp 
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.
template <typename Key>
struct MurmurHash3_32
{
    typedef Key         argument_type;
    typedef uint32_t    result_type;
    
    __forceinline__ 
    __host__ __device__ 
    MurmurHash3_32() : m_seed( 0 ) {}
    
    __forceinline__ 
    __host__ __device__ uint32_t rotl32( uint32_t x, int8_t r ) const
    {
      return (x << r) | (x >> (32 - r));
    }
    
    __forceinline__ 
    __host__ __device__ uint32_t fmix32( uint32_t h ) const
    {
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }
    
    __forceinline__ 
    __host__ __device__ result_type operator()(const Key& key) const
    {
        constexpr int len = sizeof(argument_type);
        const uint8_t * const data = (const uint8_t*)&key;
        constexpr int nblocks = len / 4;

        uint32_t h1 = m_seed;

        constexpr uint32_t c1 = 0xcc9e2d51;
        constexpr uint32_t c2 = 0x1b873593;

        //----------
        // body

        const uint32_t * const blocks = (const uint32_t *)(data + nblocks*4);

        for(int i = -nblocks; i; i++)
        {
            uint32_t k1 = blocks[i];//getblock32(blocks,i);

            k1 *= c1;
            k1 = rotl32(k1,15);
            k1 *= c2;

            h1 ^= k1;
            h1 = rotl32(h1,13); 
            h1 = h1*5+0xe6546b64;
        }

        //----------
        // tail

        const uint8_t * tail = (const uint8_t*)(data + nblocks*4);

        uint32_t k1 = 0;

        switch(len & 3)
        {
            case 3: k1 ^= tail[2] << 16;
            case 2: k1 ^= tail[1] << 8;
            case 1: k1 ^= tail[0];
                    k1 *= c1; k1 = rotl32(k1,15); k1 *= c2; h1 ^= k1;
        };

        //----------
        // finalization

        h1 ^= len;

        h1 = fmix32(h1);

        return h1;
    }
private:
    const uint32_t m_seed;
};

template <typename Key>
using default_hash = make_positive<Key>;

#endif //HASH_FUNCTIONS_CUH
