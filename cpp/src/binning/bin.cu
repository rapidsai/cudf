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
#include <cudf/binning/bin.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/binning/bin.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/types.hpp>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <cudf/copying.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/detail/iterator.cuh>
#include <thrust/tuple.h>
#include <thrust/pair.h>

namespace cudf {

// Sentinel used to indicate that an input value should be placed in the null
// bin.
// NOTE: In theory if a user decided to specify 2^31 bins this would fail. We
// could make this an error in Python, but that is such a crazy edge case...
constexpr size_type NULL_VALUE{-1};

namespace detail {
namespace {

template <typename T, typename StrictWeakOrderingLeft, typename StrictWeakOrderingRight>
struct bin_finder
{
    bin_finder(
            const T *left_edges,
            const T *left_edges_end,
            const T *right_edges
            )
        : m_left_edges(left_edges), m_left_edges_end(left_edges_end), m_right_edges(right_edges)
    {}

    __device__ size_type operator()(thrust::pair<T, bool> input_value) const
    {
        if (!input_value.second)
            return NULL_VALUE;

        const T value = input_value.first;
        auto bound = thrust::lower_bound(thrust::seq,
                m_left_edges, m_left_edges_end,
                value,
                m_left_comp);

        // Exit early and return NULL for values not within the interval.
        if ((bound == m_left_edges) || (bound == m_left_edges_end))
            return NULL_VALUE;

        // We must subtract 1 because lower bound returns the first index
        // _greater than_ the value. This is safe because bound == m_left edges
        // would already have triggered a NULL return above.
        auto index = bound - m_left_edges - 1;
        return (m_right_comp(value, m_right_edges[index])) ? index : NULL_VALUE;
    }

    const T *m_left_edges{};  // Pointer to the beginning of the device data containing left bin edges.
    const T *m_left_edges_end{};  // Pointer to the end of the device data containing left bin edges.
    const T *m_right_edges{};  // Pointer to the beginning of the device data containing right bin edges.
    // TODO: Can I implement these as static members rather than making an instance on construction?
    StrictWeakOrderingLeft m_left_comp{}; // Comparator used for left edges.
    StrictWeakOrderingRight m_right_comp{}; // Comparator used for left edges.
};


struct filter_null_sentinel
{
    __device__ bool operator()(size_type i)
    {
        return (i != NULL_VALUE);
    }
};

/// Bin the input by the edges in left_edges and right_edges.
template <typename T, typename StrictWeakOrderingLeft, typename StrictWeakOrderingRight, bool InputIsNullable>
std::unique_ptr<column> bin(column_view const& input, 
                            column_view const& left_edges,
                            column_view const& right_edges,
                            rmm::mr::device_memory_resource * mr)
{
    auto output = cudf::make_numeric_column(data_type(type_to_id<size_type>()), input.size());
    // TODO: Figure out why auto doesn't work here (or rather, why this works
    // but then calls to begin/end later fail). I imagine suitable addition of
    // a "template" keyword will fix it.
    cudf::mutable_column_view output_mutable_view = output->mutable_view();

    thrust::transform(thrust::device,
            column_device_view::create(input)->pair_begin<T, InputIsNullable>(),
            column_device_view::create(input)->pair_end<T, InputIsNullable>(),
            output_mutable_view.begin<size_type>(),
            bin_finder<T, StrictWeakOrderingLeft, StrictWeakOrderingRight>(
                left_edges.begin<T>(), left_edges.end<T>(), right_edges.begin<T>()
                )
            );

    auto mask_and_count = cudf::detail::valid_if(
        output_mutable_view.begin<size_type>(),
        output_mutable_view.end<size_type>(),
        filter_null_sentinel()
        );

    output->set_null_mask(mask_and_count.first, mask_and_count.second);
    return output;
}

// TODO: Figure out how this is instantiated for export to Python.  We need
// explicit template instantiations (or some automatic template metaprogramming
// solution) somewhere to make this available to Python.
template <typename T>
constexpr inline auto is_supported_bin_type()
{
  // TODO: Determine what other types (such as fixed point numbers) should be
  // supported, and whether any of them (like strings) require special
  // handling.
  return (cudf::is_numeric<T>() && not std::is_same<T, bool>::value); // || cudf::is_fixed_point<T>();
}

}  // anonymous namespace
}  // namespace detail


/// Functor suitable for use with type_dispatcher that exploits SFINAE to call the appropriate detail::bin method.
struct bin_type_dispatcher {
    template <typename T, typename... Args>
    std::enable_if_t<not detail::is_supported_bin_type<T>(), std::unique_ptr<column>> operator()(
            Args&&... args)
    {
        CUDF_FAIL("Type not support for cudf::bin");
    }

    template <typename T>
    std::enable_if_t<detail::is_supported_bin_type<T>(), std::unique_ptr<column>> operator()(
            column_view const& input, 
            column_view const& left_edges,
            inclusive left_inclusive,
            column_view const& right_edges,
            inclusive right_inclusive,
            rmm::mr::device_memory_resource * mr)
    {
        // Using a switch statement might be more appropriate for an enum, but it's far more verbose in this case.
        if (input.nullable())
        {
            if ((left_inclusive == inclusive::YES) && (right_inclusive == inclusive::YES))
                return detail::bin<T, thrust::less_equal<T>, thrust::less_equal<T>, true>(input, left_edges, right_edges, mr);
            if ((left_inclusive == inclusive::YES) && (right_inclusive == inclusive::NO))
                return detail::bin<T, thrust::less_equal<T>, thrust::less<T>, true>(input, left_edges, right_edges, mr);
            if ((left_inclusive == inclusive::NO) && (right_inclusive == inclusive::YES))
                return detail::bin<T, thrust::less<T>, thrust::less_equal<T>, true>(input, left_edges, right_edges, mr);
            if ((left_inclusive == inclusive::NO) && (right_inclusive == inclusive::NO))
                return detail::bin<T, thrust::less<T>, thrust::less<T>, true>(input, left_edges, right_edges, mr);
        }
        else
        {
            if ((left_inclusive == inclusive::YES) && (right_inclusive == inclusive::YES))
                return detail::bin<T, thrust::less_equal<T>, thrust::less_equal<T>, false>(input, left_edges, right_edges, mr);
            if ((left_inclusive == inclusive::YES) && (right_inclusive == inclusive::NO))
                return detail::bin<T, thrust::less_equal<T>, thrust::less<T>, false>(input, left_edges, right_edges, mr);
            if ((left_inclusive == inclusive::NO) && (right_inclusive == inclusive::YES))
                return detail::bin<T, thrust::less<T>, thrust::less_equal<T>, false>(input, left_edges, right_edges, mr);
            if ((left_inclusive == inclusive::NO) && (right_inclusive == inclusive::NO))
                return detail::bin<T, thrust::less<T>, thrust::less<T>, false>(input, left_edges, right_edges, mr);
        }

        CUDF_FAIL("Undefined inclusive setting.");
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
    // TODO: Check if we want an empty set of bins to be an error, or if it should just return NULL for all bins.
    CUDF_EXPECTS(input.type() == left_edges.type(), "The input and edge columns must have the same types.");
    CUDF_EXPECTS(input.type() == right_edges.type(), "The input and edge columns must have the same types.");
    CUDF_EXPECTS(left_edges.size() == right_edges.size(), "The left and right edge columns must be of the same length.");

    // Handle empty inputs.
    if (input.is_empty()) {
        return cudf::make_numeric_column(data_type(type_to_id<size_type>()), 0);
    }

    return type_dispatcher(
            input.type(), bin_type_dispatcher{}, input, left_edges, left_inclusive, right_edges, right_inclusive, mr);
}
}  // namespace cudf
