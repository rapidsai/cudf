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

#ifndef __ORC__UTIL_HEADER__
#define __ORC__UTIL_HEADER__

#include "orc_types.h"
#include <vector>

/** ---------------------------------------------------------------------------*
* @brief define the chunk header of ORC format
* ---------------------------------------------------------------------------**/
struct chunk_header {
    const unsigned char header[3];
    bool isOriginal() const { return header[0] & 0x01; };
    unsigned int getSize() const {
        unsigned int val =
            ((unsigned int)header[0] >> 1) |
            ((unsigned int)header[1] << 7) |
            ((unsigned int)header[2] << 15);
        return val;
    };
};


// encode zigzag encoding
// fixed W bytes in __BIG__ endian format, which is zigzag encoded if they type is signed
// \return count of encoded bytes
template <class T>
inline orc_uint8 encode_zigzag(orc_byte* raw, T value)
{
    std::vector<orc_byte> d;
    do {
        orc_byte t = (value & 0xff);
        value >>= 8;
        d.push_back(t);
    } while (value);

    for (int i = 0; i < d.size(); i++) raw[d.size() - 1 - i] = d[i];
    return d.size();
}

// encode base value encoding
// fixed W bytes in __BIG__ endian format, which the signed bit is MSB if they type is signed
// \return count of encoded bytes
template <class T>
inline orc_uint8 encode_baseValueMSB(orc_byte* raw, T value) {
    return encode_zigzag(raw, value);    // if T is not signed, the value is same as unsigned encode zigzag
}


// Encode Base 128 Varint
// 1-8 bytes
// __LITTLE__ endian format using the low 7 bits of each byte
// the signed bit is LSB of first byte.
// \return count of encoded bytes
template <class T>
inline orc_uint8 encode_varint128(orc_byte* raw, T value)
{
    orc_uint8 byte_count = 0;
    do {
        orc_uint8 t = (value & 0x7f) | 0x80;
        value >>= 7;
        raw[byte_count++] = t;    // push last one
    } while (value);
    raw[byte_count - 1] &= 0x7f;    // end mask.

    return byte_count;
}

// -----------------------------------------------------------------------------------------

template <>
inline orc_byte encode_zigzag(orc_byte* raw, orc_sint64 value) {
    orc_uint64 convert = (value << 1) ^ (value >> 63);
    return encode_zigzag(raw, convert);
}

template <>
inline orc_byte encode_zigzag(orc_byte* raw, orc_sint32 value) {
    return encode_zigzag(raw, (orc_sint64)value);
}

template <>
inline orc_byte encode_varint128(orc_byte* raw, orc_sint64 value) {
    orc_uint64 convert = (value << 1) ^ (value >> 63);
    return encode_varint128(raw, convert);
}

template <>
inline orc_byte encode_varint128(orc_byte* raw, orc_sint32 value) {
    return encode_varint128(raw, (orc_sint64)value);
}

template <>
inline orc_byte encode_baseValueMSB(orc_byte* raw, orc_sint64 value) {
    std::vector<orc_byte> d;
    orc_byte t;
    orc_byte has_signbit = 0;

    if (value >> 63) {
        value = ~value;
        has_signbit = 0x80;
    }

    do {
        t = (value & 0xff);
        value >>= 8;
        d.push_back(t);
    } while (value);

    if (has_signbit) {
        if (has_signbit & d[d.size() - 1]) {
            d.push_back(has_signbit);
        }
        else {
            d[d.size() - 1] |= has_signbit;
        }

    }

    for (int i = 0; i < d.size(); i++) raw[d.size() - 1 - i] = d[i];

    return d.size();
}

template <>
inline orc_byte encode_baseValueMSB(orc_byte* raw, orc_sint32 value) {
    return encode_baseValueMSB(raw, (orc_sint64)value);
}


CudaOrcError_t AllocateAndCopyToDevice(void** dest, const void* src, size_t size);
CudaOrcError_t AllocateTemporaryBufffer(void** dest, size_t size);
CudaOrcError_t ReleaseTemporaryBufffer(void* dest, size_t size);

bool findGMToffsetFromRegion(int& gmtoffset, const char* region);

#endif // __ORC__UTIL_HEADER__
