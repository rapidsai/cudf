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

namespace cudf {
namespace orc {


__device__ orc_bitmap getLvalue(int start)
{
    orc_bitmap value = 0xff;
    value >>= start;
    value <<= start;

    return value;
}

__device__ orc_bitmap getRvalue(int end)
{
    orc_bitmap value = 0xff;
    if (end == 0)return value;
    int offset = 8 - end;
    value <<= end;
    value >>= end;

    return value;
}

// for now, this kernel won't use atomic operation
// the caller should take care not to call present kernels simultaneously
__global__ void kernel_clear_present_add(KernelParamBitmap param)
{
    orc_bitmap *output = reinterpret_cast<orc_bitmap*>(param.output);

    size_t id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < param.output_count) {
        if (id == 0) {
            orc_bitmap value = getLvalue(param.start_id);
            output[id] += value;
        }
        else if (id == param.output_count - 1)
        {
            orc_bitmap value = getRvalue(param.end_id);
            output[id] += value;
        }
        else {
            output[id] = 0xff;
        }
    }
}

void cuda_clear_present(KernelParamBitmap* param)
{
    orc_uint32 block_size = 1024;
    int64_t block_dim = (param->output_count + block_size - 1) / block_size;
    if (block_size > param->output_count) block_size = param->output_count;

    kernel_clear_present_add << <block_dim, block_size, 0, param->stream >> > (*param);
}

// clearing present stream
void cudaClearPresent(KernelParamBitmap* param)
{
    if (param->start_id == 0 && param->end_id == 0) {
        // ToDo: the stream can also be skipped if the stream is the member of union and there is no data for the stream.
        cudaMemsetAsync(param->output, 0xff, param->output_count, param->stream);
    }
    else {
//        EXIT("not suported case.");
        // todo: copy from parent stream.
        cuda_clear_present(param);
    }

    ORC_DEBUG_KERNEL_CALL_CHECK();
}

}   // namespace orc
}   // namespace cudf
