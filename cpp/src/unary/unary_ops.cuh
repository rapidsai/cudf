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

#ifndef UNARY_OPS_H
#define UNARY_OPS_H

#include <utilities/cudf_utils.h>
#include <utilities/column_utils.hpp>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <cudf/cudf.h>
#include <cudf/column/column_view.hpp>

namespace cudf {
namespace experimental {
namespace unary {

template<typename T, typename Tout, typename F>
struct Launcher {
    static
    void launch(cudf::column_view const& input,
                cudf::mutable_column_view& output,
                cudaStream_t stream = 0) {

        CUDF_EXPECTS(input.size() > 0,              "Launcher requires input size to be non-zero.");
        CUDF_EXPECTS(input.size() == output.size(), "Launcher requires input and output size to be equal.");

        thrust::transform(rmm::exec_policy(stream)->on(stream),
                          input.begin<T>(),
                          input.end<T>(),
                          output.begin<Tout>(),
                          F{});

        CUDA_CHECK_LAST();
    }
};

inline void handleChecksAndValidity(column_view const& input, mutable_column_view& output) {

    if (not input.nullable()) {
        if (not output.nullable())
            CUDA_TRY( cudaMemset(output.null_mask(), 0xff, gdf_num_bitmask_elements(input.size())));

        output.set_null_count(0);
    }
    else { // input.valid != nullptr
        CUDF_EXPECTS(output.nullable(), "Input column has valid mask but output column does not");

        // Validity mask transfer
        CUDA_TRY( cudaMemcpy(output.null_mask(), input.null_mask(),
                             gdf_num_bitmask_elements(input.size()),
                             cudaMemcpyDeviceToDevice));

        output.set_null_count(input.null_count());
    }
}

} // unary
} // namespace experimental
} // cudf

#endif // UNARY_OPS_H
