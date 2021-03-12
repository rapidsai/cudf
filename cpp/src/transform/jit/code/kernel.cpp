/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

namespace cudf {
namespace transformation {
namespace jit {
namespace code {
const char* kernel_header =
  R"***(
    #pragma once

    // Include Jitify's cstddef header first
    #include <cstddef>

    #include <cuda/std/climits>
    #include <cuda/std/cstddef>
    #include <cuda/std/limits>

    #include <cudf/types.hpp>
    #include <cudf/wrappers/timestamps.hpp>

    struct Masked {
      int value;
      bool valid;
    };

  )***";

const char* kernel =
  R"***(

    template <typename TypeOut, typename TypeIn>
    __global__
    void kernel(cudf::size_type size,
                    TypeOut* out_data, TypeIn* in_data) {
        int tid = threadIdx.x;
        int blkid = blockIdx.x;
        int blksz = blockDim.x;
        int gridsz = gridDim.x;

        int start = tid + blkid * blksz;
        int step = blksz * gridsz;

        Masked m;
        m.value = 1;
        m.valid = true;

        for (cudf::size_type i=start; i<size; i+=step) {
          GENERIC_UNARY_OP(&out_data[i], in_data[i]);  
        }
    }
  )***";

const char* masked_binary_op_kernel = 
  R"***(
    

    template <typename TypeOut, typename TypeIn1, typename TypeIn2>
    __global__
    void kernel(cudf::size_type size,
                    TypeOut* out_data, TypeIn1* in_data1, TypeIn2 in_data2) {
        int tid = threadIdx.x;
        int blkid = blockIdx.x;
        int blksz = blockDim.x;
        int gridsz = gridDim.x;

        int start = tid + blkid * blksz;
        int step = blksz * gridsz;

        Masked m;

        for (cudf::size_type i=start; i<size; i+=step) {
          GENERIC_BINARY_OP(&out_data[i], in_data1[i], in_data2[i]);  
        }
    }
  )***";


}  // namespace code
}  // namespace jit
}  // namespace transformation
}  // namespace cudf
