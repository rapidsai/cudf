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

// TODO: Clean up includes when all debugging is done.
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <memory>
#include <cudf/binning/bin.hpp>
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
#include <cudf/copying.hpp>

namespace cudf {

namespace bin {

constexpr unsigned int MYNULL = 0xffffffff;

template <typename StrictWeakOrderingLeft, typename StrictWeakOrderingRight>
struct bin_finder
{
    bin_finder(
            const float *left_edges,
            const float *left_edges_end,
            const float *right_edges
            )
        : m_left_edges(left_edges), m_left_edges_end(left_edges_end), m_right_edges(right_edges),
          m_left_comp(StrictWeakOrderingLeft()), m_right_comp(StrictWeakOrderingRight())
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

        // We must subtract 1 because lower bound returns the first index _greater than_ the value.
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
template <typename StrictWeakOrderingLeft, typename StrictWeakOrderingRight>
std::unique_ptr<column> bin_internal(column_view const& input, 
                            column_view const& left_edges,
                            column_view const& right_edges,
                            rmm::mr::device_memory_resource * mr)
{
    // TODO: Add check that edge sizes are > 0.
    CUDF_EXPECTS(input.type() == left_edges.type(), "The input and edge columns must have the same types.");
    CUDF_EXPECTS(input.type() == right_edges.type(), "The input and edge columns must have the same types.");
    CUDF_EXPECTS(left_edges.size() == right_edges.size(), "The left and right edge columns must be of the same length.");

    // Handle empty inputs.
    if (input.is_empty()) {
        // TODO: Determine what output type actually makes sense here, it
        // probably shouldn't be empty_like but instead of some numeric type.
        return empty_like(input);
    }

    // TODO: Figure out how to get these two template type from the input.
    auto output = cudf::make_numeric_column(input.type(), input.size());

    thrust::transform(thrust::device,
            input.begin<float>(), input.end<float>(),
            static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(),
            bin_finder<StrictWeakOrderingLeft, StrictWeakOrderingRight>(
                left_edges.begin<float>(), left_edges.end<float>(), right_edges.begin<float>()
                )
            );

    //unsigned int *tmp = (unsigned int *) malloc(10 * sizeof(unsigned int));
    //cudaError_t err = cudaMemcpy(tmp, static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(), 10 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //fprintf(stderr, "The values of the output are %d, %d, %d.\n", tmp[0], tmp[1], tmp[2]);

    return output;
}



template <typename T>
constexpr inline auto is_supported_bin_type()
{
  return (cudf::is_numeric<T>() && not std::is_same<T, bool>::value); // || cudf::is_fixed_point<T>();
}


struct bin_type_dispatcher {
    template <typename T, typename... Args>
    std::enable_if_t<not is_supported_bin_type<T>(), std::unique_ptr<column>> operator()(
            Args&&... args)
    {
        CUDF_FAIL("Type not support for cudf::bin");
    }

    template <typename T>
    std::enable_if_t<is_supported_bin_type<T>(), std::unique_ptr<column>> operator()(
            column_view const& input, 
            column_view const& left_edges,
            inclusive left_inclusive,
            column_view const& right_edges,
            inclusive right_inclusive,
            rmm::mr::device_memory_resource * mr)
    {
        switch (left_inclusive)
        {
            case inclusive::YES:
                switch (right_inclusive)
                {
                    case inclusive::YES:
                        // TODO: Don't pick float by default, dispatch as needed.
                        return bin_internal<thrust::less_equal<float>, thrust::less_equal<float> >(input, left_edges, right_edges, mr);
                    case inclusive::NO:
                        return bin_internal<thrust::less_equal<float>, thrust::less<float> >(input, left_edges, right_edges, mr);
                }
            case inclusive::NO:
                switch (right_inclusive)
                {
                    case inclusive::YES:
                        return bin_internal<thrust::less<float>, thrust::less_equal<float> >(input, left_edges, right_edges, mr);
                    case inclusive::NO:
                        return bin_internal<thrust::less<float>, thrust::less<float> >(input, left_edges, right_edges, mr);
                }
            default:
                CUDF_FAIL("Undefined rounding method");
        }
    }
};



// Bin the input by the edges in left_edges and right_edges.
std::unique_ptr<column> bin(column_view const& input, 
                            column_view const& left_edges,
                            inclusive left_inclusive,
                            column_view const& right_edges,
                            inclusive right_inclusive,
                            rmm::mr::device_memory_resource * mr)
{
    // TODO: Add check that edge sizes are > 0.
    CUDF_EXPECTS(input.type() == left_edges.type(), "The input and edge columns must have the same types.");
    CUDF_EXPECTS(input.type() == right_edges.type(), "The input and edge columns must have the same types.");
    CUDF_EXPECTS(left_edges.size() == right_edges.size(), "The left and right edge columns must be of the same length.");

    // Handle empty inputs.
    if (input.is_empty()) {
        // TODO: Determine what output type actually makes sense here, it
        // probably shouldn't be empty_like but instead of some numeric type.
        return empty_like(input);
    }

    return type_dispatcher(
            input.type(), bin_type_dispatcher{}, input, left_edges, left_inclusive, right_edges, right_inclusive, mr);
}
}  // namespace bin
}  // namespace cudf
