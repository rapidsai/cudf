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

#include "kernel_private.cuh"
#include "kernel_reader_int.cuh"

namespace cudf {
namespace orc {

#define RLEv2_Mode_ShortRepeat      0
#define RLEv2_Mode_Direct           1
#define RLEv2_Mode_PatchedBase      2
#define RLEv2_Mode_Delta            3

#define USE_LAMBDA

#if defined(_DEBUG)
//#define USER_UBER_KERNEL
//#define MODE_DETAIL
#endif

template <class T, class T_writer, class T_reader>
class ORCdecodeIntRLEv2Warp : public ORCdecodeCommon<T, T_writer, T_reader>
{
public:
    __device__ ORCdecodeIntRLEv2Warp<T, T_writer, T_reader>(KernelParamCommon* param)
        : ORCdecodeCommon<T, T_writer, T_reader>(param)
        , hasPresentStream(param->present ? true : false)
    {};

    __device__ void decode();

protected:
    __device__ void decodeShortRepeat(orc_byte BW, orc_byte repeat_count);
    __device__ void decodeDirect(orc_byte width, int length);
    __device__ void decodePatchedBase(orc_byte width, int length, int BW, int PW, int PGW, int PLL);
    __device__ void decodeDelta(orc_byte width, orc_uint32 length);

    __device__ void decodePatchedBaseWithPresent(orc_byte width, int length, int BW, int PW, int PGW, int PLL);

    template <class Function>
    __device__
    void fixedBitOperation(orc_byte width, int length, Function lambda_convert, bool do_write);

protected:
    bool hasPresentStream;
};


template <class T, class T_writer, class T_reader>
__device__ void ORCdecodeIntRLEv2Warp<T, T_writer, T_reader>::decode()
{
    this->stat.Init(4, 0);

    while (!this->reader.end() && !this->writer.end()) {
        orc_byte h = this->reader.getLocal(0);

        orc_byte mode = h >> 6;
        switch (mode) {
        case RLEv2_Mode_ShortRepeat:    // repearts 3-10 values.
        {
            orc_byte BW = ((h >> 3) & 0x07) + 1;                // (1 to 8 bytes)
            orc_byte repeat_count = ((h) & 0x07) + 3;            // (3 to 10 values)
            this->reader.add_offset(1);

            decodeShortRepeat(BW, repeat_count);
            this->stat.IncrementModeCount(RLEv2_Mode_ShortRepeat);
            break;
        }
        case RLEv2_Mode_Direct:
        {
            orc_byte h1 = this->reader.getLocal(1);
            orc_byte encoded_width = (h >> 1) & 0x1f;
            orc_byte width = get_decoded_width(encoded_width);    // 5 bits for encoded width (W) of values (1 to 64 bits) 
            int length = (((unsigned int)(h & 0x01) << 8) | h1) + 1;    // 9 bits for length(L) (1 to 512 values) 
            this->reader.add_offset(2);

            decodeDirect(width, length);
            this->stat.IncrementModeCount(RLEv2_Mode_Direct);
            break;
        }
        case RLEv2_Mode_PatchedBase:
        {
            orc_byte h1 = this->reader.getLocal(1);
            orc_byte h2 = this->reader.getLocal(2);
            orc_byte h3 = this->reader.getLocal(3);

            int width = get_decoded_width((h >> 1) & 0x1f);
            int length = (int(h & 0x01) << 8 | (h1)) + 1;    // 9 bits for length (L) (1 to 512 values)
            int BW = ((h2 >> 5) & 0x07) + 1;                // 3 bits for base value width (BW) (1 to 8 bytes)
            int PW = get_decoded_width(h2 & 0x1f);
            int PGW = ((h3>> 5) & 0x07) + 1;                    // 3 bits for patch gap width (PGW) (1 to 8 bits)
            int PLL = h3 & 0x1f;                            // 5 bits for patch list length (PLL) (0 to 31 patches)
            this->reader.add_offset(4);

            if (hasPresentStream) {
                decodePatchedBaseWithPresent(width, length, BW, PW, PGW, PLL);
            }
            else {
                decodePatchedBase(width, length, BW, PW, PGW, PLL);
            }
            this->stat.IncrementModeCount(RLEv2_Mode_PatchedBase);
            break;
        }
        case RLEv2_Mode_Delta:
        {
            orc_byte h1 = this->reader.getLocal(1);

            int encoded_width = (h >> 1) & 0x1f;
            int width = get_decoded_width(encoded_width);                // 5 bits for encoded width (W) of values (1 to 64 bits) 
            if (width == 1) width = 0;                                  // exceptional case for delta encoding.
            orc_uint32 length = ((orc_uint32(h & 0x01) << 8) | h1) + 1;    // 9 bits for length(L) (1 to 512 values)
            this->reader.add_offset(2);

            decodeDelta(width, length);
            this->stat.IncrementModeCount(RLEv2_Mode_Delta);
            break;
        }
        default:
            this->stat.SetReturnCode(1);
            break;
        }

    }    // while (!this->reader.end() && !this->writer.end())

    this->stat.SetDecodeCount(this->writer.get_decoded_count());
    this->stat.SetReadCount(this->reader.get_read_count());
    this->stat.Output();
};


//! short repeat (3-10 times) the base value
template <class T, class T_writer, class T_reader>
__device__ void ORCdecodeIntRLEv2Warp<T, T_writer, T_reader>::decodeShortRepeat(
    orc_byte BW,                //< the width of Base value to be repeated (1 to 8 bytes)
    orc_byte repeat_count        //< the count of repeats (3 to 10 values)
)
{
    T BaseValue = this->reader.getUnsignedBaseValue(BW, 0);
    UnZigzag(BaseValue);
#ifdef MODE_DETAIL
    DP0("RLEv2 short repeat: %d, %d, %d\n", repeat_count, BW, BaseValue)
#endif

    this->writer.expect(repeat_count);
    if (this->ctc.tid < repeat_count) {
        this->writer.write_local(BaseValue, this->ctc.tid);
    }
    this->writer.add_offset(repeat_count);
    this->reader.add_offset(BW);
};


// --------------------------------------------------------------------------------------------------

template <class T, class T_writer, class T_reader>
template <class Function>
__device__ inline
void ORCdecodeIntRLEv2Warp<T, T_writer, T_reader>::fixedBitOperation(orc_byte width, int length, Function lambda_convert, bool do_write)
{
    T value;    // need unzigzag
    int the_rest = this->ctc.getRestCount(length);

    int loop_count;
    switch (width) {
    case 1:    case 2:    case 4:
    {
        for (loop_count = 0; loop_count < this->ctc.getFullCount(length); loop_count++) {
            value = this->reader.getBitWidthValue(this->ctc, width);

            lambda_convert(value, blockDim.x * loop_count + this->ctc.tid);

            if (do_write) {
                this->writer.expect(blockDim.x);
                this->writer.write_local(value, this->ctc.tid);
                this->writer.add_offset(blockDim.x);
            }

            this->reader.add_offset(this->ctc.block_per(width));
        }

        if (do_write) this->writer.expect(the_rest);

        if (this->ctc.tid < the_rest) {
            value = this->reader.getBitWidthValue(this->ctc, width);

            lambda_convert(value, blockDim.x * loop_count + this->ctc.tid);

            if (do_write) this->writer.write_local(value, this->ctc.tid);
        }

        int read_offset = (((the_rest - 1) >> this->ctc.getBitIndex(width)) + 1);
        this->reader.add_offset(read_offset);

        if (do_write) this->writer.add_offset(the_rest);

        break;
    }
    case 8: case 16: case 24: case 32: case 40: case 48: case 56: case 64:
    {
        int byte_count = (width >> 3);
        for (loop_count = 0; loop_count < this->ctc.getFullCount(length); loop_count++) {
            value = this->reader.getUnsignedBaseValue(byte_count, byte_count * this->ctc.tid);
            lambda_convert(value, blockDim.x * loop_count + this->ctc.tid);

            if (do_write) {
                this->writer.expect(blockDim.x);
                this->writer.write_local(value, this->ctc.tid);
                this->writer.add_offset(blockDim.x);
            }
            this->reader.add_offset(byte_count * blockDim.x);
        }

        if (do_write) this->writer.expect(the_rest);

        if (this->ctc.tid < the_rest) {
            value = this->reader.getUnsignedBaseValue(byte_count, byte_count * this->ctc.tid);
            lambda_convert(value, blockDim.x * loop_count + this->ctc.tid);
            if (do_write)  this->writer.write_local(value, this->ctc.tid);
        }


        if (do_write)  this->writer.add_offset(the_rest);
        this->reader.add_offset(byte_count * the_rest);

        break;
    }
    // deprecated (but still alive) cases
    case 3:  case 5:  case 6:  case 7:
    case 9:  case 10: case 11: case 12: case 13: case 14: case 15:
    case 17: case 18: case 19: case 20: case 21: case 22: case 23:
    case 26: case 28: case 30:
    {
        for (loop_count = 0; loop_count < this->ctc.getFullCount(length); loop_count++) {
            int local_index = loop_count * blockDim.x + this->ctc.tid;
            value = this->reader.getDeprecatedBitWidthValue(local_index, width);

            lambda_convert(value, blockDim.x * loop_count + this->ctc.tid);

            if (do_write) {
                this->writer.expect(blockDim.x);
                this->writer.write_local(value, this->ctc.tid);
                this->writer.add_offset(blockDim.x);
            }
        }

        if (do_write) this->writer.expect(the_rest);

        if (this->ctc.tid < the_rest) {
            int local_index = loop_count * blockDim.x + this->ctc.tid;
            value = this->reader.getDeprecatedBitWidthValue(local_index, width);

            lambda_convert(value, blockDim.x * loop_count + this->ctc.tid);

            if (do_write) this->writer.write_local(value, this->ctc.tid);
        }

        int read_offset = ((length * width +7)>>3 );
        this->reader.add_offset(read_offset);

        if (do_write) this->writer.add_offset(the_rest);
        break;
    }
    default:
        PRINT0(" unsupported fixedBitOperation width : %d\n", width);
        this->stat.SetReturnCode(1);
        break;
    }

};

// --------------------------------------------------------------------------------------------------



//! read the fixed input_size bits data
template <class T, class T_writer, class T_reader>
__device__ void ORCdecodeIntRLEv2Warp<T, T_writer, T_reader>::decodeDirect(
    orc_byte width,     //< width (W) of values (1 to 64 bits) 
    int length            //< length(L) (1 to 512 values) 
)
{
#ifdef MODE_DETAIL
    DP0("Direct Mode : W=%d, L=%d \n", width, length);
#endif

    auto func = [](T& value, int index) {
        UnZigzag(value);
    };

    fixedBitOperation(width, length, func, true);
};


template <class T, class T_writer, class T_reader>
__device__ void ORCdecodeIntRLEv2Warp<T, T_writer, T_reader>::decodePatchedBase(
    orc_byte width,        //< width (W) of values (1 to 64 bits) 
    int length,        //< length (L) (1 to 512 values)
    int BW,            //< width of base value (BW) (1 to 8 bytes)
    int PW,            //< width of patch value (PW) (1 to 8 bytes)
    int PGW,        //< patch gap width (PGW) (1 to 8 bits)
    int PLL            //< patch list length (PLL) (0 to 31 patches)
)
{
    T BaseValue = this->reader.getBaseValueMSB(BW, 0);
    this->reader.add_offset(BW);

#ifdef MODE_DETAIL
    DP0("in... mode patch list.\n");
    DP0("W, L, BW = %d, %d, %d\n", width, length, BW);
    DP0("PW, PGW, PLL = %d, %d, %d\n", PW, PGW, PLL);

    DP0_INT(BaseValue);
#endif
    size_t start_index = this->writer.get_decoded_count();

    auto func = [&BaseValue](T& value, int index) {
        value += BaseValue;
    };

    fixedBitOperation(width, length, func, true);

    if(PLL){    // PLL can be 0.
        __syncthreads();

        // load patch list ( PLL * (PGW + PW) bits), and apply the patch
        int        PSW = PGW + PW;
        long    the_patch;
        int        the_gap = 0;

        // PLL <= 32, PLL fetch should be in a warp fetch.
        if (this->ctc.tid < PLL) {
            int bit_start = (PSW * this->ctc.tid) >> 3;
            int bit_end = (PSW * this->ctc.tid + PSW) >> 3;
            long    the_psw = 0;

            // Todo: have to care when 56 < PSW < 64, the patch may lie on 9 bytes...
            for (int i = bit_start; i <= bit_end; i++) {
                the_psw <<= 8;
                the_psw += this->reader.getLocal(i);
            }

            // cut off unnecessary part
            the_psw >>= (8 - ((PSW * this->ctc.tid + PSW) & 0x00000007));

            the_gap = (the_psw >> PW) & ~(0xffffffff << PGW);
            the_patch = the_psw & ~(0xffffffff << PW);
            the_patch <<= width;
        }

        // get accumlated gap value from top
        int sum = 0;
        int gap_from_top = GetAccumlatedDelta(the_gap, sum);

        if (this->ctc.tid < PLL && the_patch) {
            this->writer.add_value(the_patch, start_index + gap_from_top);
        }

        this->reader.add_offset((PSW *PLL + 7) >> 3);
    }
};

// this is specialized for decoding with present stream.
template <class T, class T_writer, class T_reader>
__device__ void ORCdecodeIntRLEv2Warp<T, T_writer, T_reader>::decodePatchedBaseWithPresent(
    orc_byte width,        //< width (W) of values (1 to 64 bits) 
    int length,        //< length (L) (1 to 512 values)
    int BW,            //< width of base value (BW) (1 to 8 bytes)
    int PW,            //< width of patch value (PW) (1 to 8 bytes)
    int PGW,        //< patch gap width (PGW) (1 to 8 bits)
    int PLL            //< patch list length (PLL) (0 to 31 patches)
)
{
    __shared__ T values[512];

    T BaseValue = this->reader.getBaseValueMSB(BW, 0);
    this->reader.add_offset(BW);

    int the_rest = this->ctc.getRestCount(length);

#ifdef MODE_DETAIL
    DP0("in... mode patch list.\n");
    DP0("W, L, BW = %d, %d, %d\n", width, length, BW);
    DP0("PW, PGW, PLL = %d, %d, %d\n", PW, PGW, PLL);

    DP0_INT(BaseValue);
#endif

    T* shared_pointer = values;
    auto func = [&BaseValue, shared_pointer](T& value, int index) {
        shared_pointer[index] = BaseValue + value;
    };

    fixedBitOperation(width, length, func, false);

    if (PLL) {    // PLL can be 0.

        // load patch list ( PLL * (PGW + PW) bits), and apply the patch
        int        PSW = PGW + PW;
        long    the_patch;
        int        the_gap = 0;

        // PLL <= 32, PLL fetch should be in a warp fetch.
        if (this->ctc.tid < PLL) {
            int bit_start = (PSW * this->ctc.tid) >> 3;
            int bit_end = (PSW * this->ctc.tid + PSW) >> 3;
            long    the_psw = 0;

            // Todo: have to care when 56 < PSW < 64, the patch may lie on 9 bytes...
            for (int i = bit_start; i <= bit_end; i++) {
                the_psw <<= 8;
                the_psw += this->reader.getLocal(i);
            }

            // cut off unnecessary part
            the_psw >>= (8 - ((PSW * this->ctc.tid + PSW) & 0x00000007));

            the_gap = (the_psw >> PW) & ~(0xffffffff << PGW);
            the_patch = the_psw & ~(0xffffffff << PW);
            the_patch <<= width;
        }

        // get accumlated gap value from top
        int sum = 0;
        int gap_from_top = GetAccumlatedDelta(the_gap, sum);

        __syncthreads();

        // add the gap onto the values in shared memory
        if (this->ctc.tid < PLL && the_patch) {
            values[gap_from_top] += the_patch;
        }

        this->reader.add_offset((PSW *PLL + 7) >> 3);
    }

    {   // wrtite the values in shared memory.
        int loop_count;
        for (loop_count = 0; loop_count < this->ctc.getFullCount(length); loop_count++) {
            this->writer.expect(blockDim.x);
            this->writer.write_local(values[blockDim.x * loop_count + this->ctc.tid], this->ctc.tid);
            this->writer.add_offset(blockDim.x);
        }

        this->writer.expect(the_rest);
        if (this->ctc.tid < the_rest) {
            this->writer.write_local(values[blockDim.x * loop_count + this->ctc.tid], this->ctc.tid);
        }
        this->writer.add_offset(the_rest);
    }
}

template <class T, class T_writer, class T_reader>
__device__ void ORCdecodeIntRLEv2Warp<T, T_writer, T_reader>::decodeDelta(
    orc_byte width,                //< width(W) of values(1 to 64 bits)
    orc_uint32 length        //< length(L) (1 to 512 values)
)
{
    T value;
    T BaseValue = 0;    // Base 128 Varint
    int base_count = this->reader.getVarint128(&BaseValue, 0);
    this->reader.add_offset(base_count);

    orc_sint32 DeltaValue;    // Base 128 __signed__  Varint, must be signed always.
    int count = this->reader.getVarint128(&DeltaValue, 0);
    this->reader.add_offset(count);

    int order_sign = (DeltaValue > 0) ? 1 : -1;

#ifdef MODE_DETAIL
    DP0("in... mode delta.\n");
    DP0("byte_count, length = %d, %d\n", width, length);
    int int_base = BaseValue;
    DP0("Base, Delta : %d, %d\n", int_base, DeltaValue);
    DP0("base_count: %d\n", base_count);
#endif

    // write 1st and 2nd values.
    int write_count = min(2, length);   // taking care if length == 1.
    this->writer.expect(write_count);
    if (this->ctc.tid < write_count)
    {
        T  value = this->ctc.tid ? BaseValue + DeltaValue : BaseValue;
        this->writer.write_local(value, this->ctc.tid);
    }
    this->writer.add_offset(write_count);

    if (length < 3)return;    // the length >= 3.
    length -= 2;

    int the_rest = this->ctc.getRestCount(length);
    int sum = 0;
    switch (width) {
    case 0:
    {
        int loop_count = 0;
        for (loop_count = 0; loop_count < this->ctc.getFullCount(length); loop_count++) {
            int index = loop_count * blockDim.x + this->ctc.tid + 2;
            value = BaseValue + DeltaValue * index;
            this->writer.expect(blockDim.x);
            this->writer.write_local(value, this->ctc.tid);
            this->writer.add_offset(blockDim.x);
        }

        this->writer.expect(the_rest);
        if (this->ctc.tid < the_rest) {
            int index = loop_count * blockDim.x + this->ctc.tid + 2;
            value = BaseValue + DeltaValue * index;
            this->writer.write_local(value, this->ctc.tid);
        }
        this->writer.add_offset(the_rest);
        break;
    }
    case 2: case 4: // delta mode won't have 1 bit values.
    {
        int diff_value;
        for (int i = 0; i < this->ctc.getFullCount(length); i++) {
            diff_value = this->reader.getBitWidthValue(this->ctc, width);
            // get the accumulataed value from thread id0.
            value = GetAccumlatedDelta(diff_value, sum);

            value = BaseValue + DeltaValue + value * order_sign;
            this->writer.expect(blockDim.x);
            this->writer.write_local(value, this->ctc.tid);

            this->writer.add_offset(blockDim.x);
            this->reader.add_offset(this->ctc.block_per(width));
        }

        this->writer.expect(the_rest);
        if (this->ctc.tid < the_rest) {
            diff_value = this->reader.getBitWidthValue(this->ctc, width);
        }else {
            diff_value = 0;
        }
        value = GetAccumlatedDelta(diff_value, sum);
        if (this->ctc.tid < the_rest) {
            value = BaseValue + DeltaValue + value * order_sign;
            this->writer.write_local(value, this->ctc.tid);
        }

        int read_offset = (((the_rest - 1) >> this->ctc.getBitIndex(width)) + 1);
        this->writer.add_offset(the_rest);
        this->reader.add_offset(read_offset);
        break;
    }
    case 8:    case 16: case 24: case 32: case 40: case 48: case 56: case 64:
    {
        orc_uint64 diff_value;
        int byte_count = width >> 3;
        for (int i = 0; i < this->ctc.getFullCount(length); i++) {
            diff_value = this->reader.getUnsignedBaseValue(byte_count, byte_count * this->ctc.tid);
            value = GetAccumlatedDelta(diff_value, sum);    // get the accumulataed value from thread id 0.

            value = BaseValue + DeltaValue + value * order_sign;
            this->writer.expect(blockDim.x);
            this->writer.write_local(value, this->ctc.tid);
            this->writer.add_offset(blockDim.x);
            this->reader.add_offset(byte_count * blockDim.x);
        }

        this->writer.expect(the_rest);
        if (this->ctc.tid < the_rest) {
            diff_value = this->reader.getUnsignedBaseValue(byte_count, byte_count * this->ctc.tid);
        }
        else {
            diff_value = 0;
        }
        value = GetAccumlatedDelta(diff_value, sum);
        if (this->ctc.tid < the_rest) {
            value = BaseValue + DeltaValue + value * order_sign;
            this->writer.write_local(value, this->ctc.tid);
        }

        this->writer.add_offset(the_rest);
        this->reader.add_offset(byte_count * the_rest);
        break;
    }
    // deprecated (but still alive) cases
    case 3:    case 5: case 6: case 7:
    case 9: case 10: case 11: case 12: case 13: case 14: case 15:
    case 17: case 18: case 19: case 20: case 21:  case 22: case 23:
    case 26: case 28: case 30:
    {
        orc_uint64 diff_value;
        int loop_count;
        for (loop_count = 0; loop_count < this->ctc.getFullCount(length); loop_count++) {
            int local_index = loop_count * blockDim.x + this->ctc.tid;
            diff_value = this->reader.getDeprecatedBitWidthValue(local_index, width);
            value = GetAccumlatedDelta(diff_value, sum);    // get the accumulataed value from thread id 0.
            value = BaseValue + DeltaValue + value * order_sign;

            this->writer.expect(blockDim.x);
            this->writer.write_local(value, this->ctc.tid);
            this->writer.add_offset(blockDim.x);
        }

        this->writer.expect(the_rest);

        if (this->ctc.tid < the_rest) {
            int local_index = loop_count * blockDim.x + this->ctc.tid;
            diff_value = this->reader.getDeprecatedBitWidthValue(local_index, width);
        }else {
            diff_value = 0;
        }

        value = GetAccumlatedDelta(diff_value, sum);
        if (this->ctc.tid < the_rest) {
            value = BaseValue + DeltaValue + value * order_sign;
            this->writer.write_local(value, this->ctc.tid);
        }

        int read_offset = ((length * width + 7) >> 3);
        this->reader.add_offset(read_offset);
        this->writer.add_offset(the_rest);
        break;
    }
    default:
        PRINT0(" unsupported decodeDelta width : %d", width);
        break;
    }
};


#ifndef USER_UBER_KERNEL

template <class T_decoder>
__global__ void kernel_integerRLEv2(KernelParamCommon param)
{
    T_decoder decoder(&param);
    decoder.decode();
}

// invoking kernel
template <typename T, class T_writer, class T_reader>
void cuda_integerRLEv2_entry(KernelParamCommon* param)
{
    const int num_threads = 32;

    kernel_integerRLEv2<ORCdecodeIntRLEv2Warp<T, T_writer, T_reader>> << <1, num_threads, 0, param->stream >> > (*param);
    ORC_DEBUG_KERNEL_CALL_CHECK();
}

// ----------------------------------------------------------
template <typename T, class T_writer>
void cuda_integerRLEv2_reader_select(KernelParamCommon* param)
{
    if (param->input) {
        using reader = stream_reader_int<T, data_reader_single_buffer>;
        cuda_integerRLEv2_entry<T, T_writer, reader>(param);
    }
    else {
        using reader = stream_reader_int<T, data_reader_multi_buffer>;
        cuda_integerRLEv2_entry<T, T_writer, reader>(param);
    }
}

template <typename T, class T_converter = ORCConverterBase<T>>
void cuda_integerRLEv2_writer_select(KernelParamCommon* param)
{
    if (param->present) {
        cuda_integerRLEv2_reader_select<T, data_writer_depends_warp<T, T_converter>>(param);
    }
    else {
        cuda_integerRLEv2_reader_select<T, data_writer<T, T_converter>>(param);
    }
}

template <typename T>
void cuda_integerRLEv2_converter_select(KernelParamCommon* param)
{
    if (param->convertType == OrcKernelConvertionType::GdfConvertNone) {
        cuda_integerRLEv2_writer_select<T, ORCConverterBase<T>>(param);
    }
    else {
        switch (param->convertType) {
        case OrcKernelConvertionType::GdfDate64:
            cuda_integerRLEv2_writer_select<T, ORCConverterGdfDate64<T>>(param);
            break;
        default:
            EXIT("unhandled convert type");
            break;
        }
    }
}

void cudaDecodeIntRLEv2(KernelParamCommon* param)
{
    switch (param->elementType) {
    case OrcElementType::Uint64:
        cuda_integerRLEv2_converter_select<orc_uint64>(param);
        break;
    case OrcElementType::Sint64:
        cuda_integerRLEv2_converter_select<orc_sint64>(param);
        break;
    case OrcElementType::Uint32:
        cuda_integerRLEv2_converter_select<orc_uint32>(param);
        break;
    case OrcElementType::Sint32:
        cuda_integerRLEv2_converter_select<orc_sint32>(param);
        break;
    case OrcElementType::Uint16:
        cuda_integerRLEv2_converter_select<orc_uint16>(param);
        break;
    case OrcElementType::Sint16:
        cuda_integerRLEv2_converter_select<orc_sint16>(param);
        break;
    default:
        EXIT("unhandled type");
        break;
    }

}

#else

template <class T, class T_converter = ORCConverterBase<T>>
__device__ void kernel_integerRLEv2_depends_convert(KernelParamCommon* kernParam)
{
    if (kernParam->present) {
        using T_writer = data_writer_depends_warp<T, T_converter>;
        if (kernParam->input) {
            using reader = stream_reader_int<T, data_reader_single_buffer>;
            ORCdecodeIntRLEv2Warp<T, T_writer, reader> decoder(kernParam);
            decoder.decode();
        }
        else {
            using reader = stream_reader_int<T, data_reader_multi_buffer>;
            ORCdecodeIntRLEv2Warp<T, T_writer, reader> decoder(kernParam);
            decoder.decode();
        }
    }
    else {
        using T_writer = data_writer<T, T_converter>;

        if (kernParam->input) {
            using reader = stream_reader_int<T, data_reader_single_buffer>;
            ORCdecodeIntRLEv2Warp<T, T_writer, reader> decoder(kernParam);
            decoder.decode();
        }
        else {
            using reader = stream_reader_int<T, data_reader_multi_buffer>;
            ORCdecodeIntRLEv2Warp<T, T_writer, reader> decoder(kernParam);
            decoder.decode();
        }
    }
}



template <typename T>
__global__ void kernel_integerRLEv2_depends(KernelParamCommon kernParam)
{
    if (kernParam.convertType == OrcKernelConvertionType::GdfConvertNone) {
        kernel_integerRLEv2_depends_convert<T, ORCConverterBase<T>>(&kernParam);
    }
    else {
        switch (kernParam.convertType) {
        case OrcKernelConvertionType::GdfDate64:
            kernel_integerRLEv2_depends_convert<T, ORCConverterGdfDate64<T>>(&kernParam);
        default:
            break;
        }
    }
}

void cudaDecodeIntRLEv2(KernelParamCommon* param)
{
    const int num_CTA = 32;

    switch (param->elementType) {
    case OrcElementType::Uint64:
        kernel_integerRLEv2_depends<orc_uint64> << <1, num_CTA, 0, param->stream >> >(*param);
        break;
    case OrcElementType::Sint64:
        kernel_integerRLEv2_depends<orc_sint64> << <1, num_CTA, 0, param->stream >> >(*param);
        break;
    case OrcElementType::Uint32:
        kernel_integerRLEv2_depends<orc_uint32> << <1, num_CTA, 0, param->stream >> >(*param);
        break;
    case OrcElementType::Sint32:
        kernel_integerRLEv2_depends<orc_sint32> << <1, num_CTA, 0, param->stream >> >(*param);
        break;
    case OrcElementType::Uint16:
        kernel_integerRLEv2_depends<orc_uint16> << <1, num_CTA, 0, param->stream >> >(*param);
        break;
    case OrcElementType::Sint16:
        kernel_integerRLEv2_depends<orc_sint16> << <1, num_CTA, 0, param->stream >> >(*param);
        break;
    default:
        EXIT("unhandled type");
        break;
    }

    ORC_DEBUG_KERNEL_CALL_CHECK();
}

#endif

}   // namespace orc
}   // namespace cudf