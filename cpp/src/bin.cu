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

/// Kernel for accumulation.
// TODO: Need to template a lot of these types.
// Note that the two comparators will always be called with an input value as
// the first argument, i.e. inclusivity in bin i will be determined by
// `left_comp(value, left_edges[i]) && right_comp(value, right_edges[i])`
template <typename StrictWeakOrderingLeft, typename StrictWeakOrderingRight>
__global__ void accumulateKernel(
        const float *values, unsigned int num_values,
        const float *left_edges,
        const float *right_edges,
        unsigned int *counts, unsigned int num_bins,
        StrictWeakOrderingLeft left_comp,
        StrictWeakOrderingRight right_comp)
{
    // Assume a set of blocks each containing a single thread for now.
    unsigned int step = static_cast<unsigned int>(num_values / gridDim.x);
    unsigned int lower_bound = blockIdx.x * step;
    unsigned int upper_bound = lower_bound + step;

    // For the final bin, need to do a min then a max because the calculated upper bound could either be:
    // 1. Exactly num_values, in which case the min/max will be no-ops.
    // 2. Larger than num_values, in which case the min will give num_values and the max will be a no-op.
    // 3. Smaller than num_values, in which case the min will be a no-op and max will bring back up to num_values.
    if ((blockIdx.x + 1) == gridDim.x)
        upper_bound = max(min(upper_bound, num_values), num_values);

    for (unsigned int i = lower_bound; i < upper_bound; ++i)
    {
        float value = values[i];

        // Pre-filter anything that isn't within the range. These can always
        // use strict inequality checks because even if one of the boundaries
        // should be excluded that will be handled by the checks below.
		if (value < left_edges[0] || value > right_edges[num_bins - 1])
        {
			return;
		}

        // Perform a binary search to determine the bin.
		unsigned int high = num_bins - 1;
		unsigned int low = 0;
		while (high - low > 1) {
			unsigned int mid = (high + low) / 2;
			if (left_comp(value, left_edges[mid]))
            {
				low = mid;
			}
            else
            {
				high = mid;
			}
		}
        if (right_comp(value, right_edges[low]))
        {
            counts[i] = low;
        }
    }
}

template <typename StrictWeakOrderingLeft, typename StrictWeakOrderingRight>
struct bin_finder_new
{
    bin_finder_new(
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
        auto bound = thrust::lower_bound(thrust::seq,
                m_left_edges, m_left_edges_end,
                value,
                m_left_comp);
        // First check if the input is actually contained in the interval; if not, assign MYNULL.
        // TODO: Use the proper comparator here.
        auto index = bound - m_left_edges;
        if (m_right_comp(value, m_right_edges[index]))
        {
            // TODO: Fix case where the input is less than all elements, so subtracting one will give a negative.
            // We must subtract 1 because lower bound's behavior is shifted by 1.
            return index - 1;
        }
        else
        {
            return MYNULL;
        }
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
    // TODO: Use the comparator
    auto output = cudf::make_numeric_column(data_type(type_id::UINT32), input.size());

    if ((left_inclusive == inclusive::YES) && (left_inclusive == inclusive::YES))
    {
        thrust::transform(thrust::device,
                input.begin<float>(), input.end<float>(),
                static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(),
                bin_finder_new<thrust::less_equal<float>, thrust::less_equal<float>>(
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
                bin_finder_new<thrust::less_equal<float>, thrust::less<float>>(
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
                bin_finder_new<thrust::less<float>, thrust::less_equal<float>>(
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
                bin_finder_new<thrust::less<float>, thrust::less<float>>(
                    left_edges.begin<float>(), left_edges.end<float>(), right_edges.begin<float>(),
                    thrust::less<float>(), thrust::less<float>()
                    )
                );
    }


    //unsigned int *tmp = (unsigned int *) malloc(10 * sizeof(unsigned int));
    //cudaError_t err = cudaMemcpy(tmp, static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(), 10 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //fprintf(stderr, "The values of the output are %d, %d, %d.\n", tmp[0], tmp[1], tmp[2]);

    // Run the kernel for accumulation.
    //if ((left_inclusive == inclusive::YES) && (left_inclusive == inclusive::YES))
    //{
    //    accumulateKernel<<<256, 1>>>(
    //            input.begin<float>(), input.size(),
    //            left_edges.begin<float>(),
    //            right_edges.begin<float>(),
    //            static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(),
    //            left_edges.size(),
    //            thrust::greater_equal<float>(),
    //            thrust::less_equal<float>());
    //}
    //else if ((left_inclusive == inclusive::YES) && (left_inclusive == inclusive::NO))
    //{
    //    accumulateKernel<<<256, 1>>>(
    //            input.begin<float>(), input.size(),
    //            left_edges.begin<float>(),
    //            right_edges.begin<float>(),
    //            static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(),
    //            left_edges.size(),
    //            thrust::greater_equal<float>(),
    //            thrust::less<float>());
    //}
    //else if ((left_inclusive == inclusive::NO) && (left_inclusive == inclusive::YES))
    //{
    //    accumulateKernel<<<256, 1>>>(
    //            input.begin<float>(), input.size(),
    //            left_edges.begin<float>(),
    //            right_edges.begin<float>(),
    //            static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(),
    //            left_edges.size(),
    //            thrust::greater<float>(),
    //            thrust::less_equal<float>());
    //}
    //else if ((left_inclusive == inclusive::NO) && (left_inclusive == inclusive::NO))
    //{
    //    accumulateKernel<<<256, 1>>>(
    //            input.begin<float>(), input.size(),
    //            left_edges.begin<float>(),
    //            right_edges.begin<float>(),
    //            static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(),
    //            left_edges.size(),
    //            thrust::greater<float>(),
    //            thrust::less<float>());
    //}

    return output;
}
}  // namespace bin
}  // namespace cudf
