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

template <class T> bool Do_test_cuda_integerRLEv2_depends(T* expected, int count, orc_byte* raw, int raw_count,
    const orc_bitmap* present = NULL, size_t present_size = 0, OrcBufferArray* array = NULL)
{
    OrcKernelParameterHelper<T> helper;
    KernelParamCommon* param = helper.create(expected, count, raw, raw_count, present, present_size, array);
    param->elementType = std::is_same<orc_uint32, T>::value ? OrcElementType::Uint32 : OrcElementType::Sint32;

    cuda_integerRLEv2_Depends(param);

    return helper.validate();
}

template <class T> bool Do_test_cuda_integerRLEv2(T* expected, int count, orc_byte* raw, int raw_count)
{
    bool succeeded;

    succeeded = Do_test_cuda_integerRLEv2_depends(expected, count, raw, raw_count);

    size_t present_length;
    const orc_bitmap* present = gen_present(present_length, count);
    size_t expected_sparse_length = present_length * 8;
    T* expected_sparse = new T[present_length * 8];

    int the_length = present_encode(expected_sparse, expected, count, present, present_length);
    assert(the_length <= expected_sparse_length);

    succeeded &= Do_test_cuda_integerRLEv2_depends(expected_sparse, the_length, raw, raw_count, present, present_length);

    delete[] expected_sparse;
    free((void*)present);
    return succeeded;
}


// --------------------------------------------------------------------------------------------

void encode_short_reapeat(orc_sint32* expected, orc_byte* raw, int& expected_count, int& raw_count,
    orc_sint32 value, int length)
{
    // write valint128 and cal width of valint.
    int width = encode_zigzag(raw + raw_count + 1, value);

    assert(width < 9);
    assert(length < 11 && length > 2);

    // write header
    raw[raw_count] = (0 << 6) | ((width - 1) << 3) | (length - 3);
    raw_count += 1 + width;

    // fill expected table
    for (int i = 0; i < length; i++) {
        expected[expected_count + i] = value;
    }
    expected_count += length;
}


TEST(RLE_tests, Short_repeat)
{
    const int count = 50;

    orc_sint32 expected[count];
    orc_byte raw[count];

    int expected_count = 0;
    int raw_count = 0;

    encode_short_reapeat(expected, raw, expected_count, raw_count, 70, 5);
    encode_short_reapeat(expected, raw, expected_count, raw_count, 10000, 10);
    encode_short_reapeat(expected, raw, expected_count, raw_count, -64, 3);

    encode_short_reapeat(expected, raw, expected_count, raw_count, 1048576, 7);
    encode_short_reapeat(expected, raw, expected_count, raw_count, 128, 5);
    encode_short_reapeat(expected, raw, expected_count, raw_count, -45001, 7);

    assert(count > expected_count);
    assert(count > raw_count);

    Do_test_cuda_integerRLEv2(expected, expected_count, raw, raw_count);
}

TEST(RLE_tests, Direct_1bits)
{
    // mode direct 1 bit test
    const int count = 10;
    orc_uint32 expected[count] = { 1, 0, 1, 1,  1, 0, 0, 1,   1, 0 };
    orc_byte raw[count];

    int raw_count = 0;
    raw[raw_count] = (0x40) | (0 << 1);    raw_count++;
    raw[raw_count] = count - 1;
    for (int k = 0; k < count; k++) {
        if (0 == k % 8) {
            raw_count++;
            raw[raw_count] = 0;
        }
        raw[raw_count] <<= 1;
        raw[raw_count] |= expected[k];
    }
    if (count % 8) raw[raw_count] <<= (8 - count % 8);
    raw_count++;

    Do_test_cuda_integerRLEv2(expected, count, raw, raw_count);
}

TEST(RLE_tests, Direct_2bits)
{
    // mode direct 2 bit test
    const int count = 10;
    orc_uint32 expected[count] = { 3, 0, 2, 1,  1, 0, 3, 1,   1, 2 };
    orc_byte raw[count];

    int raw_count = 0;
    raw[raw_count] = (0x40) | (1 << 1);    raw_count++;
    raw[raw_count] = count - 1;
    for (int k = 0; k < count; k++) {
        if (0 == k % 4) {
            raw_count++;
            raw[raw_count] = 0;
        }
        raw[raw_count] <<= 2;
        raw[raw_count] |= expected[k];
    }
    if (count % 4) raw[raw_count] <<= (8 - (count % 4) * 2);
    raw_count++;

    Do_test_cuda_integerRLEv2(expected, count, raw, raw_count);
}

TEST(RLE_tests, Direct_4bits)
{
    // mode direct 4 bit test
    const int count = 10;
    orc_uint32 expected[count] = { 15, 0, 2, 1,  9, 10, 3, 12,   11, 7 };
    orc_byte raw[count];

    int raw_count = 0;
    raw[raw_count] = (0x40) | (3 << 1);    raw_count++;
    raw[raw_count] = count - 1;
    for (int k = 0; k < count; k++) {
        if (0 == k % 2) {
            raw_count++;
            raw[raw_count] = 0;
        }
        raw[raw_count] <<= 4;
        raw[raw_count] |= expected[k];
    }
    if (count % 2) raw[raw_count] <<= (8 - (count % 2) * 4);
    raw_count++;

    Do_test_cuda_integerRLEv2(expected, count, raw, raw_count);
}

void encode_and_test_multis(const int* expected_src, int* expected, orc_byte *raw, const int count, int src_mask, int num_bytes)
{
    const orc_uint8 encoded_witdh[] = { 7, 15, 23, 27 };

    // set up raw data
    int raw_count = 0;
    raw[raw_count] = (0x40) | (encoded_witdh[num_bytes - 1] << 1);    raw_count++;
    raw[raw_count] = count - 1;
    raw_count++;

    // convert expected_src -> expected by src_mask
    for (int i = 0; i < count; i++) {
        int val = expected_src[i];
        int zig = (val << 1) ^ (val >> 31);
        zig &= src_mask;
        int expects = zig >> 1;

        if (zig & 0x01) {
            expects = ~expects;
        }

        expected[i] = expects;

        type_convert *t = (type_convert *)&zig;
        for (int k = 0; k < num_bytes; k++) {
            raw[raw_count + k] = t->c[num_bytes - 1 - k];
        }

        raw_count += num_bytes;
    }

    Do_test_cuda_integerRLEv2(expected, count, raw, raw_count);
}

TEST(RLE_tests, Direct_multi_bytes)
{
    // multi bytes tests
    const int count = 10;
    orc_sint32 expected_src[count] = { 15, 129, 20000, 0,  2147483647 /* max int32 */, 1003012, -1, -2147483648 /* min int32 */,   -11, -7 };
    orc_sint32 expected[count];
    orc_byte raw[count * 4 + 4];

    // single byte
    encode_and_test_multis(expected_src, expected, raw, count, 0x000000ff, 1);

    // multi-bytes
    encode_and_test_multis(expected_src, expected, raw, count, 0x0000ffff, 2);
    encode_and_test_multis(expected_src, expected, raw, count, 0x00ffffff, 3);
    encode_and_test_multis(expected_src, expected, raw, count, 0x7fffffff, 4);  //! sign bit must be kept
}

TEST(RLE_tests, Delta_comp_2bit)
{
    const int count = 10;
    orc_uint32 expected[count] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };
    orc_byte raw[count * 2 + 10];

    int raw_count = 0;
    raw[raw_count] = (0xc0) | (3 << 1);    raw_count++;
    raw[raw_count] = count - 1;
    raw_count++;

    raw[raw_count] = 2;    // base value
    raw_count++;

    raw[raw_count] = 2;    // delta value (2 := valint of unsigned 1)
    raw_count++;

    for (int i = 2; i < count; i += 2) {
        raw[raw_count] = expected[i] - expected[i - 1];
        raw[raw_count] <<= 4;
        raw[raw_count] |= expected[i + 1] - expected[i];
        raw_count++;
    }

    Do_test_cuda_integerRLEv2(expected, count, raw, raw_count);
}


TEST(RLE_tests, PatchedBase)
{
    // official sample from https://orc.apache.org/specification/ORCv1/
    orc_uint32 expected[] = { 2030, 2000, 2020, 1000000, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120, 2130, 2140, 2150, 2160, 2170, 2180, 2190 };
    orc_byte raw[] = { 0x8e, 0x13, 0x2b, 0x21, 0x07, 0xd0, 0x1e, 0x00, 0x14, 0x70, 0x28, 0x32, 0x3c, 0x46, 0x50, 0x5a, 0x64, 0x6e, 0x78, 0x82, 0x8c, 0x96, 0xa0, 0xaa, 0xb4, 0xbe, 0xfc, 0xe8 };

    int raw_count = sizeof(raw);
    int count = sizeof(expected) / sizeof(orc_uint32);

    Do_test_cuda_integerRLEv2(expected, count, raw, raw_count);
}

