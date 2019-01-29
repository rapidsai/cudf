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

// unit_test.cpp : Defines the entry point for the console application.
//

#include "tests_common.h"
#include "io/orc/kernel_orc.cuh"
#include "io/orc/orc_util.hpp"

bool Do_test_cuda_present_bitmap_RLE(orc_bitmap* expected, int count, orc_byte* raw, int raw_count, int first_offset)
{
    cudaError_t cudaStatus;
    void* device_input;
    void* device_output;

    AllocateAndCopyToDevice(&device_input, raw, raw_count);

    size_t bitmap_byte_size = (count + first_offset + 7) >> 3;

    CudaFuncCall(cudaMallocManaged(&device_output, bitmap_byte_size));
    CudaFuncCall(cudaMemset(device_output, 0, bitmap_byte_size));   // fill zero!

    // ----------------------------------
    CudaOrcRetStats stats;
    int start_offset = first_offset >> 3;
    int start_index = first_offset & 0x7;

    // for now, decode as 1.
    KernelParamBitmap param;
    param.output = reinterpret_cast<orc_bitmap*>(device_output);;
    param.output_count = count;
    param.input = reinterpret_cast<const orc_byte*>(device_input);
    param.input_size = raw_count;
    param.start_id = start_index;

    param.parent = NULL;
    param.stat = (CudaOrcKernelRetStatsValue*)stats.getGpuAddr();

    cuda_booleanRLEbitmapDepends(&param);
    // ----------------------------------

    CudaFuncCall(cudaDeviceSynchronize());    // kernel launch failure will be trapped here.

    int ret = compare_arrays(expected, (orc_bitmap*)device_output, bitmap_byte_size);

    CudaFuncCall(cudaFree(device_input));
    CudaFuncCall(cudaFree(device_output));

    return (ret == -1);
}

// --------------------------------------------------------------------------------------------

void encode_bitmap_run(orc_bitmap* expected, orc_byte* raw, int& expected_count, int& raw_count,
    orc_bitmap value, int length, int& start_offset)
{
    assert(length < 128 + 3 && length > 2);

    orc_bitmap flipped_value = getBitOrderFlip(value);

    // write header
    raw[raw_count] = (length - 3);
    raw_count++;

    // wirte run data
    raw[raw_count] = value;
    raw_count++;

    // fill expected table
    if (start_offset == 0) {
        for (int i = 0; i < length; i++) {
            expected[expected_count + i] = flipped_value;
        }
        expected_count += length;
    }
    else {
        orc_bitmap firstValue = flipped_value << start_offset;
        orc_bitmap carriedValue = flipped_value >> (8 - start_offset);
        orc_bitmap bitmap = firstValue | carriedValue;

        if (expected_count) {
            expected[expected_count - 1] += firstValue;
        }
        else {
            expected[0] = firstValue;
            expected_count++;
        }

        for (int i = 0; i < length - 1; i++) {
            expected[expected_count + i] = bitmap;
        }

        expected_count += length;
        expected[expected_count - 1] = carriedValue;
    }
}

#define BITMAP_COUT count * 100

TEST(PresentBitmap, Run_offset0)
{
    const int count = 50;

    orc_byte raw[count];
    orc_bitmap* expected = new orc_bitmap[BITMAP_COUT];

    int expected_count = 0;
    int raw_count = 0;
    int start_offset = 0;

    // Todo literals
    encode_bitmap_run(expected, raw, expected_count, raw_count, 0xcf, 3, start_offset);
    encode_bitmap_run(expected, raw, expected_count, raw_count, 0xa7, 127 + 3, start_offset);
    encode_bitmap_run(expected, raw, expected_count, raw_count, 0x59, 20, start_offset);
    encode_bitmap_run(expected, raw, expected_count, raw_count, 0x35, 9, start_offset);

    assert(BITMAP_COUT > expected_count);
    assert(count > raw_count);

    Do_test_cuda_present_bitmap_RLE(expected, expected_count * 8, raw, raw_count, start_offset);

    delete expected;
}

TEST(PresentBitmap, Run_offsetN)
{
    const int count = 50;

    orc_byte raw[count];
    orc_bitmap* expected = new orc_bitmap[BITMAP_COUT];

    int expected_count = 0;
    int raw_count = 0;
    int start_offset = 3;

    // do encode
    encode_bitmap_run(expected, raw, expected_count, raw_count, 0xfa, 3, start_offset);
    encode_bitmap_run(expected, raw, expected_count, raw_count, 0x95, 20, start_offset);
    encode_bitmap_run(expected, raw, expected_count, raw_count, 0x9b, 127 + 3, start_offset);
    encode_bitmap_run(expected, raw, expected_count, raw_count, 0x57, 35, start_offset);

    assert(BITMAP_COUT > expected_count);
    assert(count > raw_count);

    Do_test_cuda_present_bitmap_RLE(expected, (expected_count - 1) * 8, raw, raw_count, start_offset);

    delete expected;
}



void encode_bitmap_literals(orc_bitmap* expected, orc_byte* raw, int& expected_count, int& raw_count,
    int length, int& start_offset)
{
    assert(length <= 128 && length > 0);

    // generate random source
    orc_bitmap* random_src = new orc_bitmap[length];
    set_random(random_src, length, (orc_bitmap)0x00, (orc_bitmap)0xff);    // expected is filled by random numbers

    // write header
    raw[raw_count] = orc_bitmap(-length);
    raw_count++;

    if (start_offset == 0) {

        // wirte literal data
        for (int i = 0; i < length; i++) {
            expected[expected_count] = random_src[i];
            raw[raw_count] = getBitOrderFlip(random_src[i]);
            raw_count++;
            expected_count++;
        }
    }
    else {
        orc_bitmap carriedValue = 0;
        if (expected_count) {   // get last value
            carriedValue = expected[expected_count - 1];    // get carried from last execution
            expected_count--;
        }

        for (int i = 0; i < length; i++) {
            orc_bitmap topValue = ((random_src[i] << start_offset) & 0xff);
            orc_bitmap bitmap = carriedValue | topValue;
            carriedValue = (random_src[i] >> (8 - start_offset)) & 0xff;

            expected[expected_count] = bitmap;
//            printf("[%d]: %d\n", expected_count, expected[expected_count]);
            expected_count++;

            raw[raw_count] = getBitOrderFlip(random_src[i]);
            raw_count++;
        }

        expected[expected_count] = carriedValue;
//        printf("[%d]: %d\n", expected_count, expected[expected_count]);
        expected_count++;
    }

    delete random_src;
}



TEST(PresentBitmap, Literal_offset0)
{
    const int count = 500;

    orc_bitmap* expected = new orc_bitmap[count];
    orc_byte raw[count + 10];

    int expected_count = 0;
    int raw_count = 0;
    int start_offset = 0;

    // do encode
    encode_bitmap_literals(expected, raw, expected_count, raw_count, 1, start_offset);
    encode_bitmap_literals(expected, raw, expected_count, raw_count, 20, start_offset);
    encode_bitmap_literals(expected, raw, expected_count, raw_count, 128, start_offset);
    encode_bitmap_literals(expected, raw, expected_count, raw_count, 32, start_offset);


    assert(BITMAP_COUT > expected_count);
    assert(count > raw_count);

    Do_test_cuda_present_bitmap_RLE(expected, (expected_count) * 8, raw, raw_count, start_offset);

    delete expected;
}


TEST(PresentBitmap, Literal_offsetN)
{
    const int count = 500;

    orc_bitmap* expected = new orc_bitmap[count];
    orc_byte raw[count + 10];


    int expected_count = 0;
    int raw_count = 0;
    int start_offset = 7;

    // do encode
    encode_bitmap_literals(expected, raw, expected_count, raw_count, 1, start_offset);
    encode_bitmap_literals(expected, raw, expected_count, raw_count, 22, start_offset);
    encode_bitmap_literals(expected, raw, expected_count, raw_count, 128, start_offset);
    encode_bitmap_literals(expected, raw, expected_count, raw_count, 45, start_offset);

    assert(BITMAP_COUT > expected_count);
    assert(count > raw_count);

    Do_test_cuda_present_bitmap_RLE(expected, (expected_count - 1) * 8, raw, raw_count, start_offset);

    delete expected;
}

TEST(PresentBitmap, Mixed_offsetN)
{
    const int count = 500;

    orc_bitmap* expected = new orc_bitmap[count];
    orc_byte raw[count + 10];

    int expected_count = 0;
    int raw_count = 0;
    int start_offset = 7;

    // do encode
    encode_bitmap_literals(expected, raw, expected_count, raw_count, 1, start_offset);
    encode_bitmap_run(expected, raw, expected_count, raw_count, 0xfa, 3, start_offset);
    encode_bitmap_literals(expected, raw, expected_count, raw_count, 22, start_offset);
    encode_bitmap_literals(expected, raw, expected_count, raw_count, 128, start_offset);
    encode_bitmap_run(expected, raw, expected_count, raw_count, 0x95, 20, start_offset);
    encode_bitmap_run(expected, raw, expected_count, raw_count, 0x9b, 127 + 3, start_offset);
    encode_bitmap_run(expected, raw, expected_count, raw_count, 0x57, 35, start_offset);
    encode_bitmap_literals(expected, raw, expected_count, raw_count, 45, start_offset);

    assert(BITMAP_COUT > expected_count);
    assert(count > raw_count);

    Do_test_cuda_present_bitmap_RLE(expected, (expected_count - 1) * 8, raw, raw_count, start_offset);

    delete expected;
}
