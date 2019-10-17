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

#include <cstring>
#include <rmm/rmm.h>

// single unicode character to utf8 character
// used only by translate method
__host__ __device__ inline unsigned int u2u8( unsigned int unchr )
{
    unsigned int utf8 = 0;
    if( unchr < 0x00000080 )
        utf8 = unchr;
    else if( unchr < 0x00000800 )
    {
        utf8 =  (unchr << 2) & 0x1F00;
        utf8 |= (unchr & 0x3F);
        utf8 |= 0x0000C080;
    }
    else if( unchr < 0x00010000 )
    {
        utf8 =  (unchr << 4) & 0x0F0000;  // upper 4 bits
        utf8 |= (unchr << 2) & 0x003F00;  // next 6 bits
        utf8 |= (unchr & 0x3F);           // last 6 bits
        utf8 |= 0x00E08080;
    }
    else if( unchr < 0x00110000 ) // 3-byte unicode?
    {
        utf8 =  (unchr << 6) & 0x07000000;  // upper 3 bits
        utf8 |= (unchr << 4) & 0x003F0000;  // next 6 bits
        utf8 |= (unchr << 2) & 0x00003F00;  // next 6 bits
        utf8 |= (unchr & 0x3F);             // last 6 bits
        utf8 |= (unsigned)0xF0808080;
    }
    return utf8;
}

__host__ __device__ inline unsigned int u82u( unsigned int utf8 )
{
    unsigned int unchr = 0;
    if( utf8 < 0x00000080 )
        unchr = utf8;
    else if( utf8 < 0x0000E000 )
    {
        unchr =  (utf8 & 0x1F00) >> 2;
        unchr |= (utf8 & 0x003F);
    }
    else if( utf8 < 0x00F00000 )
    {
        unchr =  (utf8 & 0x0F0000) >> 4;
        unchr |= (utf8 & 0x003F00) >> 2;
        unchr |= (utf8 & 0x00003F);
    }
    else if( utf8 <= (unsigned)0xF8000000 )
    {
        unchr =  (utf8 & 0x03000000) >> 6;
        unchr |= (utf8 & 0x003F0000) >> 4;
        unchr |= (utf8 & 0x00003F00) >> 2;
        unchr |= (utf8 & 0x0000003F);
    }
    return unchr;
}

__device__ inline char* copy_and_incr( char*& dest, char* src, unsigned int bytes )
{
    memcpy(dest,src,bytes);
    dest += bytes;
    return dest;
}

__device__ inline char* copy_and_incr_both( char*& dest, char*& src, unsigned int bytes )
{
    memcpy(dest,src,bytes);
    dest += bytes;
    src += bytes;
    return dest;
}

template<typename T>
T* device_alloc(size_t count, cudaStream_t sid)
{
    T* buffer = nullptr;
    rmmError_t rerr = RMM_ALLOC(&buffer,count*sizeof(T),sid);
    if( rerr != RMM_SUCCESS )
    {
        if( rerr==RMM_ERROR_OUT_OF_MEMORY )
        {
            std::cerr.imbue(std::locale(""));
            std::cerr << "out of memory on alloc request of " << count << " elements of size " << sizeof(T) << " = " << (count*sizeof(T)) << " bytes\n";
        }
        std::ostringstream message;
        message << "allocate error " << rerr;
        throw std::runtime_error(message.str());
    }
    return buffer;
}
