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

namespace gdf {
namespace binops {
namespace jit {
namespace code {

const char* kernel =
R"***(
    #include <cstdint>
    #include "traits.h"
    #include "operation.h"

    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
    __global__
    void kernel_v_s(int size,
                    TypeOut* out_data, TypeVax* vax_data, TypeVay* vay_data,
                    uint32_t* out_valid, uint32_t* vax_valid) {
        int tid = threadIdx.x;
        int blkid = blockIdx.x;
        int blksz = blockDim.x;
        int gridsz = gridDim.x;

        int start = tid + blkid * blksz;
        int step = blksz * gridsz;

        for (int i=start; i<size; i+=step) {
            out_data[i] = TypeOpe::template operate<TypeOut, TypeVax, TypeVay>(vax_data[i], vay_data[0]);

            if ((i % warpSize) == 0) {
                int index = i / warpSize;
                out_valid[index] = vax_valid[index];
            }
        }
    }

    template <typename TypeOut, typename TypeVax, typename TypeVay, typename TypeOpe>
    __global__
    void kernel_v_v(int size,
                    TypeOut* out_data, TypeVax* vax_data, TypeVay* vay_data,
                    uint32_t* out_valid, uint32_t* vax_valid, uint32_t* vay_valid) {
        int tid = threadIdx.x;
        int blkid = blockIdx.x;
        int blksz = blockDim.x;
        int gridsz = gridDim.x;

        int start = tid + blkid * blksz;
        int step = blksz * gridsz;

        for (int i=start; i<size; i+=step) {
            out_data[i] = TypeOpe::template operate<TypeOut, TypeVax, TypeVay>(vax_data[i], vay_data[i]);

            if ((i % warpSize) == 0) {
                int index = i / warpSize;
                out_valid[index] = vax_valid[index] & vay_valid[index];
            }
        }
    }
)***";

} // namespace code
} // namespace jit
} // namespace binops
} // namespace gdf
