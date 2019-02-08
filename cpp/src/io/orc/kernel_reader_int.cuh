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

#ifndef __ORC_KERNEL_READER_INT_H__
#define __ORC_KERNEL_READER_INT_H__

#include "kernel_reader.cuh"

namespace cudf {
namespace orc {

template <class TU, class  T_reader_input>
class stream_reader_int : public stream_reader<T_reader_input> {
public:
    __device__ stream_reader_int(const KernelParamCommon* kernParam)
        : stream_reader<T_reader_input>(reinterpret_cast<const KernelParamBase*>(kernParam))
    {};

    __device__ ~stream_reader_int() {};

    // get varint128 from buffer with offset, the value would be unzigzagged if TV is unsigned.
    // return byte count used by getVarint128() 
    template<class TV>
    __device__
        orc_uint8 getVarint128(TV *theValue, size_t offset)
    {
#if 1
        orc_uint8 p = this->getLocal(offset);

        orc_uint8 count = 1;
        orc_uint8 has_next = p & 0x80;
        *theValue = (p & 0x7f);
        orc_uint8 the_shift = 7;

        while (has_next) {
            p = this->getLocal(offset + count);
            has_next = (p & 0x80);
            TV val = (TV)(p & 0x7f);
            *theValue += (val << the_shift);
            count++;
            the_shift += 7;
        };

        UnZigzag(*theValue);
#else   // for debug use
        TV v;
        orc_uint8 p = this->getLocal(offset);

        orc_uint8 count = 1;
        orc_uint8 has_next = p & 0x80;
        v = (p & 0x7f);
        orc_uint8 the_shift = 7;

        DP0("value = 0x%x, ", p);
        while (has_next) {
            p = this->getLocal(offset + count);
            DP0("0x%x, ", p);
            has_next = (p & 0x80);
            TV val = (TV)(p & 0x7f);
            v += (val << the_shift);
            count++;
            the_shift += 7;
        };

        UnZigzag(v);

        DP0(": %ld, ", v);

        *theValue = v;
        DP0("\n");
#endif
        return count;
    }

    __device__ TU getBaseValueMSB(int BW, size_t offset)
    {
        orc_byte p = this->getLocal(offset);
        bool is_signed = p & 0x80;
        TU BaseValue = p;

        if (std::is_signed<TU>::value == true) {
            BaseValue &= 0x7f;
        }

        for (int i = 1; i < BW; i++) {
            BaseValue <<= 8;
            BaseValue += this->getLocal(offset + i);
        }

        if (std::is_signed<TU>::value == true) {
            if (is_signed) BaseValue |= ~BaseValue;
        }

        return BaseValue;
    }

    // this is only suppoted for bit_width = 1, 2, 4
    __device__ inline
        int getBitWidthValue(CudaThreadControl& ctc, int bit_width)
    {
        orc_byte current = this->getLocal(ctc.tid_per(bit_width));
        int value = ((current >> ctc.tid_rev_shift(bit_width)) & ctc.bitmask(bit_width));
        // the diff is always unsigned, no need to un-zigzag
        return value;
    }

    // for the cases bit_width = not any of (1, 2, 4), bit_width is upto 30 bits
    __device__ inline
        int getDeprecatedBitWidthValue(int local_index, int bit_width)
    {
        int start_bit = local_index * bit_width;
        int start_index = start_bit >> 3;
        int the_shift_bit = start_bit & 0x07;
        int the_end_bit = the_shift_bit + bit_width;
        int the_byte_width = 1 + ((the_end_bit - 1) >> 3);

        unsigned long value = getUnsignedBitSequence(the_byte_width, start_index);
#if (__CUDA_ARCH__ < 700)   // this is for pascal or maxwell.
        if (the_end_bit > 32) {
            int the_extra_bit = the_end_bit - 32;
            unsigned int hi = value / (4294967296);
            unsigned int lo = (value >> the_shift_bit);

            value = (hi << the_extra_bit) | lo;
            return value;
        }
        else {
            unsigned int lo = value;
            lo <<= (32 - the_end_bit);
            lo >>= (32 - bit_width);
            return lo;
        }
#else
        value <<= (64 - the_end_bit);
        value >>= (64 - bit_width);
        return static_cast<int>(value);
#endif
    }

    // return unsigned base value with fixed byte width
    __device__
        TU getUnsignedBaseValue(int BW, size_t offset)
    {
        TU value = 0;
        for (int i = 0; i < BW; i++) {
            value <<= 8;
            value += this->getLocal(offset + i);
        }
        return value;
    }

    // 
    __device__
        TU getUnsignedBitSequence(int BW, size_t offset)
    {
        TU value = 0;
        for (int i = 0; i < BW; i++) {
            value += (this->getLocal(offset + i) << ((BW - 1 - i) * 8));
        }
        return value;
    }
};

}   // namespace orc
}   // namespace cudf

#endif // __ORC_KERNEL_READER_INT_H__
