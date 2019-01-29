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

template <class T> bool Do_test_cuda_integerRLEv1_depends(T* expected, int count, orc_byte* raw, int raw_count,
    const orc_bitmap* present = NULL, size_t present_size = 0, OrcBufferArray* array = NULL)
{
    OrcKernelParameterHelper<T> helper;
    KernelParamCommon* param = helper.create(expected, count, raw, raw_count, present, present_size, array);
    param->elementType = std::is_same<orc_uint32, T>::value ? OrcElementType::Uint32 : OrcElementType::Sint32;

    cuda_integerRLEv1_Depends(param);

    return helper.validate();
}

template <class T> bool Do_test_cuda_integerRLEv1(T* expected, int count, orc_byte* raw, int raw_count)
{
    bool succeeded;

    succeeded = Do_test_cuda_integerRLEv1_depends(expected, count, raw, raw_count);

    size_t present_length;
    const orc_bitmap* present = gen_present(present_length, count);
    size_t expected_sparse_length = present_length * 8;
    T* expected_sparse = new T[present_length * 8];

    int the_length = present_encode(expected_sparse, expected, count, present, present_length);
    assert(the_length <= expected_sparse_length);

    succeeded &= Do_test_cuda_integerRLEv1_depends(expected_sparse, the_length, raw, raw_count, present, present_length);

    delete[] expected_sparse;
    free((void*)present);
    return succeeded;
}



// --------------------------------------------------------------------------------------------

void encode_rlev1_reapeat(orc_sint32* expected, orc_byte* raw, int& expected_count, int& raw_count,
    orc_sint32 value, orc_sint8 delta, int length)
{
    assert(length < 128 + 3 && length > 2);

    // write header
    raw[raw_count] = (length - 3);
    raw_count++;
    raw_copy(raw[raw_count], delta);
    raw_count++;

    raw_count += encode_varint128(&raw[raw_count], value);

    // fill expected table
    for (int i = 0; i < length; i++) {
        expected[expected_count + i] = value + delta * i;
    }
    expected_count += length;
}

void encode_rlev1_lieteral(orc_sint32* expected, orc_byte* raw, int& expected_count, int& raw_count, int length, int min, int max)
{
    assert(0 < length  && length <= 128);

    set_random(&expected[expected_count], length, min, max);

    // write header
    raw[raw_count] = -length;
    raw_count++;

    // fill expected table
    for (int i = 0; i < length; i++) {
        raw_count += encode_varint128(&raw[raw_count], expected[expected_count]);
        expected_count++;
    }
}

// --------------------------------------------------------------------------------------------

TEST(RLEv1_tests, v1_repeat)
{
    const int count = 500;

    orc_sint32 expected[count];
    orc_byte raw[count];

    int expected_count = 0;
    int raw_count = 0;

    encode_rlev1_reapeat(expected, raw, expected_count, raw_count, 49, -1, 3);
    encode_rlev1_reapeat(expected, raw, expected_count, raw_count, -589, 3, 5);
    encode_rlev1_reapeat(expected, raw, expected_count, raw_count, 64, 127, 127 + 3);
    encode_rlev1_reapeat(expected, raw, expected_count, raw_count, -64, -128, 19);

    assert(count > expected_count);
    assert(count > raw_count);

    Do_test_cuda_integerRLEv1(expected, expected_count, raw, raw_count);
}

TEST(RLEv1_tests, v1_litreral)
{
    const int count = 500;

    orc_sint32 expected[count];
    orc_byte raw[count];

    int expected_count = 0;
    int raw_count = 0;

    encode_rlev1_lieteral(expected, raw, expected_count, raw_count, 1, -64, 63);
    encode_rlev1_lieteral(expected, raw, expected_count, raw_count, 10, 0, 102300);
    encode_rlev1_lieteral(expected, raw, expected_count, raw_count, 128, -512, 512);

    assert(count > expected_count);
    assert(count > raw_count);

    Do_test_cuda_integerRLEv1(expected, expected_count, raw, raw_count);
}


TEST(RLEv1_tests, Mixed)
{
    const int count = 500;

    orc_sint32 expected[count];
    orc_byte raw[count];

    int expected_count = 0;
    int raw_count = 0;

    encode_rlev1_lieteral(expected, raw, expected_count, raw_count, 128, -512, 512);
    encode_rlev1_reapeat(expected, raw, expected_count, raw_count, 49, -1, 3);
    encode_rlev1_lieteral(expected, raw, expected_count, raw_count, 1, -64, 63);
    encode_rlev1_lieteral(expected, raw, expected_count, raw_count, 10, 0, 102300);
    encode_rlev1_reapeat(expected, raw, expected_count, raw_count, 64, 127, 127 + 3);
    encode_rlev1_reapeat(expected, raw, expected_count, raw_count, -589, 3, 5);
    encode_rlev1_reapeat(expected, raw, expected_count, raw_count, -64, -128, 19);

    assert(count > expected_count);
    assert(count > raw_count);

    Do_test_cuda_integerRLEv1(expected, expected_count, raw, raw_count);
}


TEST(RLEv1_tests, Literal_Reference)
{
    orc_uint32 expected[] = { 2, 3, 4, 7, 11 };
    orc_byte raw[] = { 0xfb, 0x02, 0x03, 0x04, 0x07, 0xb };

    int expected_count = 5;
    int raw_count = 6;

    Do_test_cuda_integerRLEv1(expected, expected_count, raw, raw_count);
}

TEST(RLEv1_tests, Run_Reference)
{
    orc_uint32 expected[200];
    orc_byte raw[] = { 0x61, 0x00, 0x07,  0x61, 0xff, 0x64 };

    std::fill_n(expected, 100, 7);
    for (int i = 0; i < 100; i++) { expected[i + 100] = 100 - i; }

    int expected_count = 200;
    int raw_count = 3  + 3;

    Do_test_cuda_integerRLEv1(expected, expected_count, raw, raw_count);
}
