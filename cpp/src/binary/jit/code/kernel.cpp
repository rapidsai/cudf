/*
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 *     Copyright 2018 Rommel Quintanilla <rommel@blazingdb.com>
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
namespace binops {
namespace jit {
namespace code {

const char* kernel =
R"***(
    #include <cstdint>
    #include "traits.h"
    #include "operation.h"

    template <typename TypeOut, typename TypeLhs, typename TypeRhs, typename TypeOpe>
    __global__
    void kernel_v_s(int size,
                    TypeOut* out_data, TypeLhs* lhs_data, TypeRhs* rhs_data,
                    uint32_t* out_valid, uint32_t* lhs_valid) {
        int tid = threadIdx.x;
        int blkid = blockIdx.x;
        int blksz = blockDim.x;
        int gridsz = gridDim.x;

        int start = tid + blkid * blksz;
        int step = blksz * gridsz;

        for (int i=start; i<size; i+=step) {
            out_data[i] = TypeOpe::template operate<TypeOut, TypeLhs, TypeRhs>(lhs_data[i], rhs_data[0]);

            if ((out_valid != nullptr) && (i % warpSize) == 0) {
                int index = i / warpSize;
                uint32_t lhs_valid_word = (lhs_valid != nullptr) ? lhs_valid[index] : 0xffffffff;
                out_valid[index] = lhs_valid_word;
            }
        }
    }

    template <typename TypeOut, typename TypeLhs, typename TypeRhs, typename TypeOpe>
    __global__
    void kernel_v_v(int size,
                    TypeOut* out_data, TypeLhs* lhs_data, TypeRhs* rhs_data,
                    uint32_t* out_valid, uint32_t* lhs_valid, uint32_t* rhs_valid) {
        int tid = threadIdx.x;
        int blkid = blockIdx.x;
        int blksz = blockDim.x;
        int gridsz = gridDim.x;

        int start = tid + blkid * blksz;
        int step = blksz * gridsz;

        for (int i=start; i<size; i+=step) {
            out_data[i] = TypeOpe::template operate<TypeOut, TypeLhs, TypeRhs>(lhs_data[i], rhs_data[i]);

            if ((out_valid != nullptr) && (i % warpSize) == 0) {
                int index = i / warpSize;
                uint32_t lhs_valid_word = (lhs_valid != nullptr) ? lhs_valid[index] : 0xffffffff;
                uint32_t rhs_valid_word = (rhs_valid != nullptr) ? rhs_valid[index] : 0xffffffff;
                out_valid[index] = lhs_valid_word & rhs_valid_word;
            }
        }
    }
)***";

} // namespace code
} // namespace jit
} // namespace binops
} // namespace cudf
