/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
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
namespace experimental {
namespace binops {
namespace jit {
namespace code {

const char* kernel =
R"***(
    #include <cudf/types.hpp>
    #include <simt/limits>
    #include <cudf/wrappers/timestamps.hpp>
    #include "operation.h"

    template <typename TypeOut, typename TypeLhs, typename TypeRhs, typename TypeOpe>
    __global__
    void kernel_v_s(cudf::size_type size,
                    TypeOut* out_data, TypeLhs* lhs_data, TypeRhs* rhs_data) {
        int tid = threadIdx.x;
        int blkid = blockIdx.x;
        int blksz = blockDim.x;
        int gridsz = gridDim.x;

        int start = tid + blkid * blksz;
        int step = blksz * gridsz;

        for (cudf::size_type i=start; i<size; i+=step) {
            out_data[i] = TypeOpe::template operate<TypeOut, TypeLhs, TypeRhs>(lhs_data[i], rhs_data[0]);
        }
    }

    template <typename TypeOut, typename TypeLhs, typename TypeRhs, typename TypeOpe>
    __global__
    void kernel_v_v(cudf::size_type size,
                    TypeOut* out_data, TypeLhs* lhs_data, TypeRhs* rhs_data) {
        int tid = threadIdx.x;
        int blkid = blockIdx.x;
        int blksz = blockDim.x;
        int gridsz = gridDim.x;

        int start = tid + blkid * blksz;
        int step = blksz * gridsz;

        for (cudf::size_type i=start; i<size; i+=step) {
            out_data[i] = TypeOpe::template operate<TypeOut, TypeLhs, TypeRhs>(lhs_data[i], rhs_data[i]);
        }
    }
)***";

} // namespace code
} // namespace jit
} // namespace binops
} // namespace experimental
} // namespace cudf
