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

// default converter: day + nanosec -> millisec
template <typename T_secondary>
__device__ orc_sint64 kernel_convert_to_millisec(const orc_sint64 sec, const T_secondary nano, const orc_sint64 adjustClock)
{
    T_secondary nano_e = nano;
    NanoConvert(nano_e);

    orc_sint64 sec_e = sec + adjustClock;
    if (sec_e < 0 && nano_e != 0) {
        // this is adjustment for known bug https://issues.apache.org/jira/browse/ORC-346
        sec_e -= 1;
    }
    orc_sint64 ret = sec_e * 1000 + ( nano_e / 1000000 );

    return ret;
}

template <typename T_secondary>
__global__ void kernel_convert_timestamp_full(KernelParamCoversion param)
{
    orc_sint64 *output = reinterpret_cast<orc_sint64*>(param.output);
    const orc_sint64 *data = reinterpret_cast<const orc_sint64*>(param.input);
    const T_secondary *secondary = reinterpret_cast<const T_secondary*>(param.secondary);

    size_t id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < param.data_count) {
        output[id] = kernel_convert_to_millisec(data[id], secondary[id], param.adjustClock);
    }

}

__global__ void kernel_convert_string_warp(KernelParamCoversion param)
{
    CudaThreadControl ctc;

    gdf_string *output = reinterpret_cast<gdf_string*>(param.output);                   // output gdf_string
    const char *data = reinterpret_cast<const char*>(param.input);                      // array of chars
    const orc_uint32 *length = reinterpret_cast<const orc_uint32*>(param.secondary);    // array of each length;

    int loop_count = ctc.getFullCount(param.data_count);
    int the_rest = ctc.getRestCount(param.data_count);
    int sum = 0;

    for (int i = 0; i < loop_count; i++) {
        orc_uint32 the_offset = sum + GetAccumlatedDelta(length[ctc.tid], sum) - length[ctc.tid];

        output[ctc.tid].first = (length[ctc.tid]) ? (data + the_offset) : NULL;
        output[ctc.tid].second = length[ctc.tid];

        output += blockDim.x;
        length += blockDim.x;
    }

    orc_uint32 the_len = 0;
    for (int i = 0; i < the_rest; i++) {
        the_len = length[ctc.tid];
    }
    orc_uint32 the_offset = sum + GetAccumlatedDelta(the_len, sum) - the_len;

    for (int i = 0; i < the_rest; i++) {
        output[ctc.tid].first = (the_len) ? (data + the_offset) : NULL;
        output[ctc.tid].second = the_len;
    }
}

__global__ void kernel_convert_string_dictionary_full(KernelParamCoversion param)
{
    gdf_string *output = reinterpret_cast<gdf_string*>(param.output);                   // output gdf_string
    const orc_uint32 *index = reinterpret_cast<const orc_uint32*>(param.input);         // array of dictionary index
    const gdf_string *dict = reinterpret_cast<const gdf_string*>(param.secondary);      // array of dictionary entry
    const orc_bitmap *bitm = reinterpret_cast<const orc_bitmap*>(param.null_bitmap);    // array of null_bitmap

    size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= param.data_count) return;

    if (bitm) {     // if null bitmap stream exists
        bool isExist = present_is_exist(bitm, id, param.start_id);
        if (isExist) {
            output[id].first = dict[index[id]].first;
            output[id].second = dict[index[id]].second;
        }
        else {
            output[id].first = NULL;
            output[id].second = 0;
        }
    }
    else {      // NO null bitmap stream
        output[id].first = dict[index[id]].first;
        output[id].second = dict[index[id]].second;
    }

}


// --------------------------------------------------------------------

void cuda_convert_depends_full_throttle(KernelParamCoversion* param)
{
    orc_uint32 block_size = 1024;
    int64_t block_dim = (param->data_count + block_size - 1) / block_size;
    if (block_size > param->data_count) block_size = param->data_count;

    switch (param->convertType)
    {
    case OrcKernelConvertionType::GdfTimestampUnit_ms:
#if (GDF_ORC_TIMESTAMP_NANO_PRECISION == 8)
        kernel_convert_timestamp_full<orc_uint64> << <block_dim, block_size >> > (*param);
#else
        kernel_convert_timestamp_full<orc_uint32> << <block_dim, block_size >> > (*param);
#endif
        break;
    case OrcKernelConvertionType::GdfString_dictionary:
        kernel_convert_string_dictionary_full << <block_dim, block_size >> > (*param);
        break;
    default:
        EXIT("unknown convert type!");
        break;
    }
    ORC_DEBUG_KERNEL_CALL_CHECK();
}

void cuda_convert_depends_warp(KernelParamCoversion* param)
{
    int block_size = 32;

    switch (param->convertType)
    {
    case OrcKernelConvertionType::GdfString_direct:
        kernel_convert_string_warp << <1, block_size >> > (*param);
        break;
    default:
        EXIT("unknown convert type!");
        break;
    }
    ORC_DEBUG_KERNEL_CALL_CHECK();
}


void cuda_convert_depends(KernelParamCoversion* param)
{
    switch (param->convertType)
    {
    case OrcKernelConvertionType::GdfTimestampUnit_ms:
    case OrcKernelConvertionType::GdfString_dictionary:
        cuda_convert_depends_full_throttle(param);
        break;

    case OrcKernelConvertionType::GdfString_direct:
        cuda_convert_depends_warp(param);
        break;
    default:
        EXIT("unknown convert type!");
        break;
    }
}