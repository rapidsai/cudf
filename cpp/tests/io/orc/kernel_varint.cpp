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

template <class T> bool Do_test_cuda_unbound_base128_varint(T* expected, int count, orc_byte* raw, int raw_count, 
    OrcElementType type, const orc_bitmap* present = NULL, size_t present_size = 0, OrcBufferArray* array = NULL)
{
    OrcKernelParameterHelper<T> helper;
    KernelParamCommon* param = helper.create(expected, count, raw, raw_count, present, present_size, array);
    param->elementType = type;

    cudaDecodeVarint(param);

    return helper.validate();
}

TEST(Varint128_tests, varint_test_singed)
{
    const int count = 100;
    orc_sint64 expected_const[] = { 0, -1, 2, -2, 63,  64, -65, -64, -10000, 6555500 };
    orc_sint64 expected[count];

    orc_byte raw[count * 10];

    for (int i = 0; i < 10; i++)expected[i] = expected_const[i];

    // generate random
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_int_distribution<orc_sint64> dist((orc_sint64)INT64_MIN, (orc_sint64)INT64_MAX);

    for (int i = 10; i < count; i++)expected[i] = dist(engine);

    // cpu convert
    int raw_count = 0;
    orc_byte* r = raw;

    for (int i = 0; i < count; i++) {
        int enc_count = encode_varint128(r, expected[i]);
        raw_count += enc_count;
        r += enc_count;
    }

    Do_test_cuda_unbound_base128_varint(expected, count, raw, raw_count, OrcElementType::Sint64);
}

TEST(Varint128_tests, varint_test_singed_present)
{
    const int count = 100;
    orc_sint64 expected_const[] = { 0, -1, 2, -2, 63,  64, -65, -64, -10000, 6555500 };
    orc_sint64 expected[count];
    orc_byte raw[count * 10];

    for (int i = 0; i < 10; i++)expected[i] = expected_const[i];

    // generate random
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_int_distribution<orc_sint64> dist((orc_sint64)INT64_MIN, (orc_sint64)INT64_MAX);

    for (int i = 10; i < count; i++)expected[i] = dist(engine);


    const orc_bitmap present[] = {
        0x00, 0xff, 0x1c, 0xfa, 0x73,  0x00, 0x00, 0x00, 0x00, 0x8d,
        0x24, 0x56, 0x9b, 0xe0, 0xff,  0xff, 0xff, 0x2c, 0xfa, 0x73,
        0x00, 0x56, 0x9b, 0xe0, 0xfa,  0x73, 0xff, 0x1c, 0x6a, 0x53,
        0xff, 0xff, 0xff, 0xff, 0xff,  0xff, 0xff, 0xff, 0xff, 0xff, };

    int present_count = sizeof(present);
    orc_sint64 expected_sparse[count*10];
    int expected_sparse_count = present_encode(expected_sparse, expected, count, present, present_count);
    assert(expected_sparse_count < count * 10);

    // cpu convert
    int raw_count = 0;
    orc_byte* r = raw;

    for (int i = 0; i < count; i++) {
        int enc_count = encode_varint128(r, expected[i]);
        raw_count += enc_count;
        r += enc_count;
    }

    Do_test_cuda_unbound_base128_varint(expected_sparse, expected_sparse_count, raw, raw_count, OrcElementType::Sint64, 
        present, present_count);
}


TEST(Varint128_tests, varint_test_unsinged)
{
    const int count = 100;
    orc_uint64 expected_const[] = { 0, 127, 128, 6555500 };
    orc_uint64 expected[count];

    orc_byte raw[count * 10];

    for (int i = 0; i < 4; i++)expected[i] = expected_const[i];

    // generate random
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_int_distribution<orc_uint64> dist((orc_uint64)0, (orc_uint64)UINT64_MAX);

    for (int i = 4; i < count; i++)expected[i] = dist(engine);

    // cpu convert
    int raw_count = 0;
    orc_byte* r = raw;

    for (int i = 0; i < count; i++) {
        int enc_count = encode_varint128(r, expected[i]);
        raw_count += enc_count;
        r += enc_count;
    }

    Do_test_cuda_unbound_base128_varint(expected, count, raw, raw_count, OrcElementType::Uint64);
}

// ----------------------------------------------------------------------------------

template <typename T>
void Do_Varint128_encode_test(T value, int expected_size, const orc_byte* expected)
{
    orc_byte raw[10];

    EXPECT_EQ(expected_size, encode_varint128(raw, value));
    int ret = compare_arrays(expected, (const orc_byte*)raw, expected_size);
    EXPECT_EQ(-1, ret);
}

TEST(Varint128_tests, Varint128encode)
{
#define EncTest(value, size, ...)  {const orc_byte expects[] = {__VA_ARGS__};  Do_Varint128_encode_test(value, size, expects); }

    // unsigned test
    EncTest(0UL, 1, 0x00);
    EncTest(1UL, 1, 0x01);
    EncTest(127UL, 1, 0x7f);
    EncTest(128UL, 2, 0x80, 0x01);
    EncTest(129UL, 2, 0x81, 0x01);
    EncTest(16383UL, 2, 0xff, 0x7f);
    EncTest(16384UL, 3, 0x80, 0x80, 0x01);
    EncTest(16385UL, 3, 0x81, 0x80, 0x01);

    // signed test
    EncTest(0, 1, 0x00);
    EncTest(-1, 1, 0x01);
    EncTest(1, 1, 0x02);
    EncTest(-2, 1, 0x03);
    EncTest(2, 1, 0x04);

    EncTest(63, 1, 0x7e);
    EncTest(64, 2, 0x80, 0x01);
    EncTest(-64, 1, 0x7f);
    EncTest(-65, 2, 0x81, 0x01);
}

