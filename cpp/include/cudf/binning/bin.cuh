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

#pragma once

#include <cudf/column/column_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/binning/bin.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/types.hpp>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
//#include <stdio.h>
#include <cudf/copying.hpp>

namespace cudf {

namespace bin {

template <typename T, typename StrictWeakOrderingLeft, typename StrictWeakOrderingRight>
struct bin_finder
{
    bin_finder(
            const T *left_edges,
            const T *left_edges_end,
            const T *right_edges
            )
        : m_left_edges(left_edges), m_left_edges_end(left_edges_end), m_right_edges(right_edges),
          m_left_comp(StrictWeakOrderingLeft()), m_right_comp(StrictWeakOrderingRight())
    {}

    __device__ size_type operator()(const T value) const
    {
        // Immediately return NULL for NULL values.
        if (value == NULL)
            return NULL;

        auto bound = thrust::lower_bound(thrust::seq,
                m_left_edges, m_left_edges_end,
                value,
                m_left_comp);

        // Exit early and return NULL for values not within the interval.
        if ((bound == m_left_edges) || (bound == m_left_edges_end))
            return NULL;

        // We must subtract 1 because lower bound returns the first index
        // _greater than_ the value. This is safe because bound == m_left edges
        // would already have triggered a NULL return above.
        auto index = bound - m_left_edges - 1;
        return (m_right_comp(value, m_right_edges[index])) ? index : NULL;
    }

    const T *m_left_edges{};  // Pointer to the beginning of the device data containing left bin edges.
    const T *m_left_edges_end{};  // Pointer to the end of the device data containing left bin edges.
    const T *m_right_edges{};  // Pointer to the beginning of the device data containing right bin edges.
    // TODO: Can I implement these as static members rather than making an instance on construction?
    StrictWeakOrderingLeft m_left_comp; // Comparator used for left edges.
    StrictWeakOrderingRight m_right_comp; // Comparator used for left edges.
};


// Bin the input by the edges in left_edges and right_edges.
template <typename T, typename StrictWeakOrderingLeft, typename StrictWeakOrderingRight>
std::unique_ptr<column> bin_internal(column_view const& input, 
                            column_view const& left_edges,
                            column_view const& right_edges,
                            rmm::mr::device_memory_resource * mr)
{
    // TODO: Determine if UINT32 is the output type that we want. Is there a
    // way to map (at compile time) from size_type to the largest value in type_id
    // that can hold this?
    auto output = cudf::make_numeric_column(data_type(type_id::UINT32), input.size());

    thrust::transform(thrust::device,
            input.begin<T>(), input.end<T>(),
            static_cast<cudf::mutable_column_view>(*output).begin<size_type>(),
            bin_finder<T, StrictWeakOrderingLeft, StrictWeakOrderingRight>(
                left_edges.begin<T>(), left_edges.end<T>(), right_edges.begin<T>()
                )
            );

    //unsigned int *tmp = (unsigned int *) malloc(10 * sizeof(unsigned int));
    //cudaError_t err = cudaMemcpy(tmp, static_cast<cudf::mutable_column_view>(*output).begin<unsigned int>(), 10 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //fprintf(stderr, "The values of the output are %d, %d, %d.\n", tmp[0], tmp[1], tmp[2]);

    return output;
}



// TODO: Figure out how this is instantiated for export to Python. It makes
// sense that C++ tests will compile and run, but we need explicit template
// instantiations somewhere to make this available to Python.
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
        // Using a switch statement might be more appropriate for an enum, but it's far more verbose in this case.
        if ((left_inclusive == inclusive::YES) && (right_inclusive == inclusive::YES))
            return bin_internal<T, thrust::less_equal<T>, thrust::less_equal<T> >(input, left_edges, right_edges, mr);
        if ((left_inclusive == inclusive::YES) && (right_inclusive == inclusive::NO))
            return bin_internal<T, thrust::less_equal<T>, thrust::less<T> >(input, left_edges, right_edges, mr);
        if ((left_inclusive == inclusive::NO) && (right_inclusive == inclusive::YES))
            return bin_internal<T, thrust::less<T>, thrust::less_equal<T> >(input, left_edges, right_edges, mr);
        if ((left_inclusive == inclusive::NO) && (right_inclusive == inclusive::NO))
            return bin_internal<T, thrust::less<T>, thrust::less<T> >(input, left_edges, right_edges, mr);

        CUDF_FAIL("Undefined inclusive setting.");
    }
};

}  // namespace bin
}  // namespace cudf
