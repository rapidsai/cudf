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
#include "io/orc/orc_util.hpp"
#include "io/orc/kernel_orc.cuh"
#include "io/orc/kernel_util.cuh"

template <class T> bool Do_test_cuda_raw(T* expected, int count, const orc_byte* raw, int raw_count, 
    OrcElementType elementType, const orc_bitmap* present =NULL, size_t present_size = 0, OrcBufferArray* array= NULL)
{
    OrcKernelParameterHelper<T> helper;
    KernelParamCommon* param = helper.create(expected, count, raw, raw_count * sizeof(T), present, present_size, array);
    param->elementType = elementType;

    cuda_raw_data_depends(param);

    return helper.validate();
}


TEST(RawTest, single_char)
{
    const int count = 100;
    const char expected[] = "abcdefghijklmnopqrstuvwxyz";
    orc_byte raw[count * 10];

    int raw_count = sizeof(expected);

    for (int i = 0; i < raw_count; i++)raw[i] = expected[i];

    Do_test_cuda_raw(const_cast<char*>(expected), raw_count, raw, raw_count, OrcElementType::Uint8);
}

TEST(RawTest, single_char_with_present)
{
    const int count = 100;
    const char raw[] = "abcdefghijklmnopqrstuvwxyz";
    const orc_bitmap present[] = {0x00, 0xff, 0x1c, 0xfa, 0x73, 0x00, 0x00, 0x00, 
        0x00, 0x8d, 0x24, 0x56, 0x9b, 0xe0, 0xff, 0xff, 0xff};
    char expected[count * 10];

    int raw_count = sizeof(raw);
    int present_count = sizeof(present);

    int expected_count = present_encode(expected, raw, raw_count, present, present_count);

    Do_test_cuda_raw(const_cast<char*>(expected), expected_count, reinterpret_cast<const orc_byte*>(raw), raw_count, OrcElementType::Uint8, present, present_count);
}

TEST(RawTest, double_with_present)
{
    const int count = 100;
    double raw[count];
    double expected[count * 10];

    const orc_bitmap present[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0x01, 0xff, 0x1c, 0xfa, 0x73, 0x00, 0x00, 0x00,
        0x00, 0x8d, 0x24, 0x56, 0x9b, 0xe0,  0xff, 0xff, 0xff,  0x01, 0xff, 0x1c, 0xfa, 0x73, 0x00, 0x56, 0x9b, 0xe0,  0xff, 0xff};

    int raw_count = count;
    int present_count = sizeof(present);

    for (int i = 0; i < count; i++) {
        raw[i] = 0.05 + 0.1 * i;
    }

    int expected_count = present_encode(expected, raw, raw_count, present, present_count);

    assert(expected_count < count * 10);

    Do_test_cuda_raw(expected, expected_count, reinterpret_cast<const orc_byte*>(raw), raw_count, 
        OrcElementType::Float64, present, present_count);
}


TEST(RawTest, float_buffer_array)
{
    const int count = 1000;
    float raw[count];
    float expected[count];

    for (int i = 0; i < count; i++) {
        raw[i] = 0.05 + 0.1 * i;
        expected[i] = raw[i];
    }

    const int max_buffer_size = 30;
    OrcBuffer buf[max_buffer_size];

    OrcBufferArray holder;
    holder.buffers = & buf[0];

    int devision = 64;
    int total = 0;
    for (int i = 0; i < count; i++)
    {
        int size = devision;
        total += devision;
        if (total > count) {
            buf[i].bufferSize = ( size - (total - count) ) * sizeof(float);
            total = count;
            holder.numBuffers = i +1;
            break;
        }
        buf[i].bufferSize = size * sizeof(float);
    }

    Do_test_cuda_raw(expected, count, reinterpret_cast<const orc_byte*>(raw), count,
        OrcElementType::Float32, NULL, 0, &holder);
}