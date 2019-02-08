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

#ifndef __TESTS_COMMON__HEADER__
#define __TESTS_COMMON__HEADER__

//#define ORC_CONVERT_TIMESTAMP_GMT

#include "io/orc/orc_read.h"
#include "io/orc/orc_debug.h"
#include "io/orc/orc_util.hpp"
#include "io/orc/kernel_orc.cuh"
#include "io/orc/kernel_util.cuh"
#include "cudf.h"


#include <iostream>
#include <random>

// all file reading test will be skipped if this flag is on
 //#define GDF_ORC_NO_FILE_TEST

#ifndef ORC_DEVELOP_MODE

#include "gtest/gtest.h"
#endif

using namespace cudf::orc;

gdf_error release_orc_read_arg(orc_read_arg* arg);

size_t present_count(const orc_bitmap* present, size_t present_size);

const orc_bitmap* gen_present(size_t& present_length, size_t valid_count);

union type_convert {
    int i;
    unsigned char c[4];
};

#ifndef ORC_DEVELOP_MODE
template<class T>
long compare_arrays(T* lhs, T* rhs, size_t count) {
    long ret = -1;
    for (size_t i = 0; i < count; i++) {
        bool match = (lhs[i] == rhs[i])? true : false;
        if (false == match){
            ret = long(i);
#if 0
            std::cout << "[" << i << "]: ";
            EXPECT_EQ(lhs[i], rhs[i]);
//            std::cout << "[" << i <<  "]: " <<  lhs[i] << ", " << rhs[i] << std::endl;
//            printf(" %d : (%d / %d)", i, lhs[i], rhs[i]);
#else
            break;
#endif
        }
    }

    EXPECT_EQ(-1, ret);

    return ret;
}
#endif


//! generate random values between min and max
template <typename T>
void set_random(T* expected, size_t count, T min, T max) {
    // if USE_RANDOM_DEVICE is on, the random seed will be different at each execution.
#ifdef USE_RANDOM_DEVICE    
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
#else   // the random seeds are guaranteed to be same at each execution 
    static int seed_gen = 0; seed_gen++;
    std::mt19937 engine(seed_gen);
#endif
    std::uniform_int_distribution<> dist(min, max);

    for (int i = 0; i < count; i++)expected[i] = dist(engine);
};

template< typename T>
void raw_copy(orc_byte& raw, T value) {
    memcpy((void*)&raw, (void*)&value, sizeof(T));
};


template<typename T>
bool isValidPresent(const orc_bitmap* present, T i) {
    int index = i >> 3;
    int bit = i & 0x07;

    return (present[index] >> bit) & 0x01;
}

template<typename T>
size_t present_encode(T* expected, const T* raw, size_t raw_count, const orc_bitmap* present, size_t present_size, T unref_value = 0)
{
    int expected_count = 0;

    for (int i = 0; i < raw_count; i++) {
        while (!isValidPresent(present, expected_count)) {
            expected[expected_count] = unref_value;   // fill 0 if the present[expected_count] is null
            expected_count++;
        }
        expected[expected_count] = raw[i];
        expected_count++;
    }

    return expected_count;
}


#ifdef ORC_CONVERT_TIMESTAMP_GMT 
#define ORC_PST_OFFSET 3600  *8
#define ORC_PDT_OFFSET 3600  *7
#else
#define ORC_PST_OFFSET 0
#define ORC_PDT_OFFSET 3600  * (-1)
#endif

// GMT+0
__device__ __host__ inline
orc_sint64 convertGdfTimestampMsGMT(int year, int month, int day, int hour = 0, int minute = 0, int second = 0, int millisec = 0)
{
    return convertGdfTimestampMs(year, month, day, hour, minute, second, millisec);
}

// local time adjusted during PDT (Pacific Daylight Time, GMT-7)
__device__ __host__ inline
orc_sint64 convertGdfTimestampMsPDT(int year, int month, int day, int hour = 0, int minute = 0, int second = 0, int millisec = 0)
{
    return convertGdfTimestampMs(year, month, day, hour, minute, second, millisec, ORC_PDT_OFFSET);
}

// local time adjusted during PDT (Pacific Standard Time, GMT-8)
__device__ __host__ inline
orc_sint64 convertGdfTimestampMsPST(int year, int month, int day, int hour = 0, int minute = 0, int second = 0, int millisec = 0)
{
    return convertGdfTimestampMs(year, month, day, hour, minute, second, millisec, ORC_PST_OFFSET);
}

template <class T>
class OrcKernelParameterHelper {
public:
    OrcKernelParameterHelper() :
        device_input(NULL), device_output(NULL), device_present(NULL), arr(NULL), expected(NULL), count(0), unreferenced_value(0)
    {};

    ~OrcKernelParameterHelper() { release(); };

public:
    KernelParamCommon* create(T* _expected, int _count, const orc_byte* raw, int raw_count,
        const orc_bitmap* present = NULL, size_t present_size = 0, OrcBufferArray* array = NULL)
    {
        expected = _expected;
        count = _count;
        if (array) {
            // device_input should be null
            construct(array, raw);
            param.bufferArray = *array;
        }
        else {
            AllocateAndCopyToDevice(&device_input, raw, raw_count);
        }

        CudaFuncCall(cudaMallocManaged(&device_output, count * sizeof(T)));


        param.output = reinterpret_cast<orc_byte*>(device_output);
        param.input = reinterpret_cast<const orc_byte*>(device_input);
        param.output_count = count;
        param.input_size = raw_count;
        param.start_id = 0;
        param.stat = 0;

        void* device_present = NULL;
        if (present) {
            AllocateAndCopyToDevice(&device_present, present, present_size);
            T* fill_point = reinterpret_cast<T*> (device_output);
            for (int i = 0; i < count; i++) {
                fill_point[i] = unreferenced_value;
            }

            // if max count of present stream is less than output count, cap the output count by present count.
            int the_count = present_count(present, present_size);
        }
        param.present = reinterpret_cast<const orc_bitmap*>(device_present);

        return getParam();
    };

    bool validate()
    {
        CudaFuncCall(cudaStatus = cudaDeviceSynchronize());

        int ret = compare_arrays(expected, reinterpret_cast<T*>(device_output), count);

        return (ret == -1);
    }


    void construct(OrcBufferArray* array, const orc_byte* raw) {
        arr = array;
        buf_arr_org = &arr->buffers[0];
        int offset = 0;

        for (int i = 0; i < arr->numBuffers; i++) {
            OrcBuffer* buf = &arr->buffers[i];
            AllocateAndCopyToDevice(
                reinterpret_cast<void**>(&buf->buffer),
                raw + offset, buf->bufferSize);

            offset += buf->bufferSize;
        }

        AllocateAndCopyToDevice(
            reinterpret_cast<void**>(&buf_arr),
            &arr->buffers[0], arr->numBuffers * sizeof(OrcBuffer));
        array->buffers = buf_arr;
    };

    void release() {
        CudaFuncCall(cudaFree(device_input));
        CudaFuncCall(cudaFree(device_output));
        CudaFuncCall(cudaFree(device_present));

        if (arr) {
            for (int i = 0; i < arr->numBuffers; i++) {
                CudaFuncCall(cudaFree(const_cast<orc_byte*>(buf_arr_org[i].buffer)));
            }

            CudaFuncCall(cudaFree(buf_arr));
        }
    }

    KernelParamCommon* getParam() {
        return &param;
    };

    void setUnrefValue(int val) { unreferenced_value = val; };

protected:
    cudaError_t cudaStatus;
    KernelParamCommon param;

    void* device_input;
    void* device_output;
    void* device_present;
    T*    expected;
    size_t count;

    OrcBufferArray* arr;
    OrcBuffer*      buf_arr;        // only gpu accessible
    OrcBuffer*      buf_arr_org;    // only cpu accessible

    int unreferenced_value;
};


#endif // __TESTS_COMMON__HEADER__


