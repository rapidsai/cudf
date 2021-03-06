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
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cudf/types.hpp>

namespace cudf {

namespace bin {

/// Kernel for accumulation. Note that hist_min/hist_max could be inferred from
//the bins, but accessing them once outside the kernel ensures increases
//likelihood of caching for performance.
// TODO: Need to template a lot of these types.
__global__ void accumulateKernel(float hist_min, float hist_max,
        float *values, unsigned int num_values,
        float *left_edges, unsigned int num_edges,
        float *right_edges,
        unsigned int *counts, unsigned int num_bins)
{

    // Assume a set of blocks each containing a single thread for now.
    unsigned int step = static_cast<unsigned int>(num_values / gridDim.x);
    unsigned int lower_bound = blockIdx.x * step;

    // Need to do a min then a max because the calculated upper bound could either be:
    // 1. Exactly num_values, in which case the min/max will be no-ops.
    // 2. Larger than num_values, in which case the min will give num_values and the max will be a no-op.
    // 3. Smaller than num_values, in which case the min will be a no-op and max will bring back up to num_values.
    unsigned int upper_bound = lower_bound + step;
    if ((blockIdx.x + 1) == gridDim.x)
        upper_bound = max(min(upper_bound, num_values), num_values);

    for (unsigned int i = lower_bound; i < upper_bound; ++i)
    {
        float value = values[i];

        // TODO: Currently this operates on a half-open interval [bin_hist_min,
        // bin_hist_max) for consistency with division operations in C++. The
        // left/right-inclusive checks need to be used to determine this.
		if (value < hist_min || value >= hist_max)
        {
			return;
		}

        // Perform a binary search to determine the bin.
		unsigned int high = num_bins - 1;
		unsigned int low = 0;
		while (high - low > 1) {
			unsigned int mid = (high + low) / 2;
			if (left_edges[mid] <= value)
            {
				low = mid;
			}
            else
            {
				high = mid;
			}
		}

        // Avoid overflow.
        if (low == num_bins)
        {
            --low;
        }
        atomicAdd(&(counts[low]), 1);
    }
}

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
    thrust::device_vector<float> dev_input(input.size(), 0);
    thrust::device_vector<float> dev_left_edges(left_edges.size(), 0);
    thrust::device_vector<float> dev_right_edges(right_edges.size(), 0);
    thrust::device_vector<unsigned int> dev_counts(left_edges.size(), 0);

    thrust::copy(input.begin<float>(), input.end<float>(), dev_input.begin());
    thrust::copy(left_edges.begin<float>(), left_edges.end<float>(), dev_left_edges.begin());
    thrust::copy(right_edges.begin<float>(), right_edges.end<float>(), dev_right_edges.begin());

    // Run the kernel for accumulation.
    accumulateKernel<<<256, 1>>>(
            // TODO: Figure out how to get these template parameters from the inputs.
            *(left_edges.begin<float>()), *(right_edges.end<float>() - 1),
            thrust::raw_pointer_cast(dev_input.data()), input.size(),
            thrust::raw_pointer_cast(dev_left_edges.data()), left_edges.size(),
            thrust::raw_pointer_cast(dev_right_edges.data()),
            thrust::raw_pointer_cast(dev_counts.data()), left_edges.size());

    auto output = cudf::make_numeric_column(data_type(type_id::UINT32), left_edges.size());

    // TODO: Figure out if creating a mutable view of output is the best way to pass it to thrust here.
    thrust::copy(dev_counts.begin(), dev_counts.end(), static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>());

    return output;
}
}  // namespace bin
}  // namespace cudf
