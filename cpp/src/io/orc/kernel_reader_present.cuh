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

#ifndef __ORC_KERNEL_READER_PRESENT_H__
#define __ORC_KERNEL_READER_PRESENT_H__

#include "kernel_reader.cuh"

//#define _INVESTIGATION

// work as single thread
class byte_reader_present_single : public byte_reader<orc_bitmap> {
public:
    __device__ byte_reader_present_single(const orc_bitmap* bitmap, size_t size_, int start_offset_, int end_count_)
        : byte_reader<orc_bitmap>(bitmap, ((size_ + 7) >> 3))
        , element_count(end_count_), local_start_offset(start_offset_), start_offset(start_offset_)
    {
        byte_reader<orc_bitmap>::size = ((size_ + 7) >> 3);
    };

    __device__ ~byte_reader_present_single() {};

    __device__ size_t expect(int count);

    __device__ void setExpectRange(int count) { expect_range = count; };
    __device__ int  getExpectRange() { return expect_range; };

protected:
    __device__ orc_bitmap getLocal_unbound(size_t offset) {
        size_t the_offset = offset + local_offset;
        if (the_offset >= size) {
            return 0x00;        // return 0 if the_offset is out of the range.
        }
        return *(top + the_offset);
    };

    size_t element_count;

    int local_start_offset;     //< start offset at local scope of expect()
    int end_offset;
    int start_offset;           //< start offset for this class 

private:
    int expect_range;           //< the range at expect()
};

__device__ inline size_t byte_reader_present_single::expect(int count)
{
    orc_bitmap bitmap;
    int the_bit_index;
    int next_offset = 0;

    if (count == 0) {
        setExpectRange(0);
        return 0;
    }

    while (1) {
        bitmap = getLocal_unbound(0);   // getLocal returns 0 if tid is out of range

        bitmap >>= local_start_offset;
        the_bit_index = find_nth_true_bit(bitmap, 1);
        if (the_bit_index != 8) break;

        // no true bit (the_bit_index == 8)
        this->add_offset(1);
        next_offset += (8 - local_start_offset);
        local_start_offset = 0;
    };

    next_offset += the_bit_index;

    local_start_offset += the_bit_index + 1;
    if (local_start_offset == 8) {
        this->add_offset(1);
        local_start_offset = 0;
    }

    setExpectRange(next_offset +1);
    return next_offset;
}

// --------------------------------------------------------------------------------------

class byte_reader_present_warp : public byte_reader_present_single {
public:
    __device__ byte_reader_present_warp(const orc_bitmap* bitmap, size_t size_, int start_offset_, int end_count_)
        : byte_reader_present_single(bitmap, size_, start_offset_, end_count_)
    {
    };

    __device__ ~byte_reader_present_warp() {};

    __device__ size_t expect(int count);
};

//! returns 0 if thread id larger than count
//! count must be in the range of [1, blockDim.x]
//! have to be called from all threads in the warp
__device__ inline size_t byte_reader_present_warp::expect(int count)
{
    __shared__ orc_bitmap bitmaps[32];
    __shared__ int accum_count[32];
#ifdef _INVESTIGATION
    __shared__ int full_offset[32];
#endif
    if (count == 0) {
        setExpectRange(0);
        return 0;
    }
    int tid = threadIdx.x;
    int sum = 0;
    int last_sum = 0;
    int offset_from_local = 0;  // the return value 
    int index;
//    bool break_all_bits_are_true = false;   // todo: implement it, perf tuning.

    while (sum < blockDim.x) {
        bitmaps[tid] = getLocal_unbound(tid);   // getLocal returns 0 if tid is out of range
        __syncwarp();   // sync after shared memory write

        if (local_start_offset)
        {
            if (tid == 0) { // fill 0 bit at first local_start_offset bits.
                (bitmaps[0] >>= local_start_offset) <<= local_start_offset;
            }
        }
        int bitcount = getBitCount(bitmaps[tid]);

        accum_count[tid] = GetAccumlatedDelta(bitcount, sum);
        __syncwarp();   // sync after shared memory write

        if (sum == last_sum) {
            add_offset(blockDim.x); // increment present reader's offset +total thread counts since all present stream in a warp are 0
            offset_from_local += blockDim.x * 8 - local_start_offset;
            local_start_offset = 0;
            continue;   // no valid bit count
        }

        // todo: tuning option; quick break if all bits are true.

        // find index
        if (last_sum <= tid && tid < sum && tid < count) {
            index = find_index(tid +1, accum_count);

            int index_diff = accum_count[index] - (tid);
            offset_from_local += index * 8 - local_start_offset + find_nth_true_bit_from_tail(bitmaps[index], index_diff);
#ifdef _INVESTIGATION
            full_offset[tid] = offset_from_local;
            __syncwarp();
#endif
        }

        if (sum >= count) {
            break;
        }
        last_sum = sum;
    }

    int updating_offset;
    int the_max_offset;
    int updating_start_offset;
    if (tid == count -1)
    {
        the_max_offset = offset_from_local +1;
        updating_start_offset =  (( offset_from_local + local_start_offset ) & 0x07)+1;
        updating_offset = index;
        if (updating_start_offset == 8) {    // increments
            updating_start_offset = 0;
            updating_offset++;
        }
    }

#ifdef _INVESTIGATION
    __syncwarp();
#endif

    // broadcast
    the_max_offset = __shfl_sync(0xffffffff, the_max_offset, count - 1);
    updating_offset = __shfl_sync(0xffffffff, updating_offset, count - 1);
    local_start_offset = __shfl_sync(0xffffffff, updating_start_offset, count - 1);

    setExpectRange(the_max_offset);
    add_offset(updating_offset);

#ifdef _INVESTIGATION
    printf("[%u] %d, %d, %d\n", threadIdx.x, the_max_offset, updating_offset, offset_from_local);
#endif

    return offset_from_local;
}


#endif // __ORC_KERNEL_READER_PRESENT_H__
