/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <memory>
#include <cudf/bin.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/types.hpp>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <stdio.h>
#include <vector>
#include <numeric>

namespace cudf {

namespace bin {

constexpr unsigned int MYNULL = 0xffffffff;

template <typename StrictWeakOrderingLeft, typename StrictWeakOrderingRight>
struct bin_finder
{
    bin_finder(
            const float *left_edges,
            const float *left_edges_end,
            const float *right_edges,
            StrictWeakOrderingLeft left_comp,
            StrictWeakOrderingRight right_comp
            )
        : m_left_edges(left_edges), m_left_edges_end(left_edges_end), m_right_edges(right_edges),
          m_left_comp(left_comp), m_right_comp(right_comp)
    {}

    __device__ unsigned int operator()(const float value) const
    {
        // TODO: Immediately return NULL for NULL values.
        auto bound = thrust::lower_bound(thrust::seq,
                m_left_edges, m_left_edges_end,
                value,
                m_left_comp);

        // First check if the input is actually contained in the interval; if not, assign MYNULL.
        if ((bound == m_left_edges) || (bound == m_left_edges_end))
            return MYNULL;

        // We must subtract 1 because lower bound returns the first index _greater than_ the value. This is
        auto index = bound - m_left_edges - 1;
        return (m_right_comp(value, m_right_edges[index])) ? index : MYNULL;
    }

    const float *m_left_edges;
    const float *m_left_edges_end;
    const float *m_right_edges;
    // TODO: Can I store these by reference? Don't think so since the argument
    // to lower_bound is not a ref, but I should check to be sure.
    StrictWeakOrderingLeft m_left_comp;
    StrictWeakOrderingRight m_right_comp;
};

// Bin the input by the edges in left_edges and right_edges.
std::unique_ptr<column> bin(column_view const& input, 
                            column_view const& left_edges,
                            inclusive left_inclusive,
                            column_view const& right_edges,
                            inclusive right_inclusive,
                            rmm::mr::device_memory_resource * mr)
{
    CUDF_EXPECTS(input.type() == left_edges.type(), "The input and edge columns must have the same types.");
    CUDF_EXPECTS(input.type() == right_edges.type(), "The input and edge columns must have the same types.");
    CUDF_EXPECTS(left_edges.size() == right_edges.size(), "The left and right edge columns must be of the same length.");

    // TODO: Figure out how to get these two template type from the input.
    auto output = cudf::make_numeric_column(data_type(type_id::UINT32), input.size());

    if ((left_inclusive == inclusive::YES) && (left_inclusive == inclusive::YES))
    {
        thrust::transform(thrust::device,
                input.begin<float>(), input.end<float>(),
                static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(),
                bin_finder<thrust::less_equal<float>, thrust::less_equal<float>>(
                    left_edges.begin<float>(), left_edges.end<float>(), right_edges.begin<float>(),
                    thrust::less_equal<float>(), thrust::less_equal<float>()
                    )
                );
    }
    else if ((left_inclusive == inclusive::YES) && (left_inclusive == inclusive::NO))
    {
        thrust::transform(thrust::device,
                input.begin<float>(), input.end<float>(),
                static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(),
                bin_finder<thrust::less_equal<float>, thrust::less<float>>(
                    left_edges.begin<float>(), left_edges.end<float>(), right_edges.begin<float>(),
                    thrust::less_equal<float>(), thrust::less<float>()
                    )
                );
    }
    else if ((left_inclusive == inclusive::NO) && (left_inclusive == inclusive::YES))
    {
        thrust::transform(thrust::device,
                input.begin<float>(), input.end<float>(),
                static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(),
                bin_finder<thrust::less<float>, thrust::less_equal<float>>(
                    left_edges.begin<float>(), left_edges.end<float>(), right_edges.begin<float>(),
                    thrust::less<float>(), thrust::less_equal<float>()
                    )
                );
    }
    else if ((left_inclusive == inclusive::NO) && (left_inclusive == inclusive::NO))
    {
        thrust::transform(thrust::device,
                input.begin<float>(), input.end<float>(),
                static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(),
                bin_finder<thrust::less<float>, thrust::less<float>>(
                    left_edges.begin<float>(), left_edges.end<float>(), right_edges.begin<float>(),
                    thrust::less<float>(), thrust::less<float>()
                    )
                );
    }


    //unsigned int *tmp = (unsigned int *) malloc(10 * sizeof(unsigned int));
    //cudaError_t err = cudaMemcpy(tmp, static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(), 10 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //fprintf(stderr, "The values of the output are %d, %d, %d.\n", tmp[0], tmp[1], tmp[2]);

    return output;
}
}  // namespace bin
}  // namespace cudf
