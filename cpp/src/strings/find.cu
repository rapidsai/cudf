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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/traits.hpp>
#include "./utilities.hpp"

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform.h>

namespace
{

/**
 * @brief This does the find (or rfind) operation on each string returning the
 * the character position of the first occurrence of the target string within
 * the [start,stop) range.
 *
 * No checking is done to prevent overflow if the position value does not fit
 * in the output type. In this case, the position value is truncated by
 * casting it to an IntegerType.
 */
template <typename IntegerType, typename FindFunction>
struct find_fn
{
    cudf::column_device_view d_strings;
    FindFunction pfn; // accepts string, target, start, stop and returns position
    cudf::string_view d_target;  // string to search for
    cudf::size_type start, stop; // character position range to search for within each string [start,stop)

    __device__ IntegerType operator()( cudf::size_type idx )
    {
        IntegerType position = 0;
        if( !d_strings.is_null(idx) )
            position = static_cast<IntegerType>(pfn(d_strings.element<cudf::string_view>(idx),d_target,start,stop));
        return position;
    }
};

// Dispatcher allows filling in any integer type as the output column.
struct dispatch_find_fn
{
    /**
     * @brief Utility to return integer column indicating the postion of
     * target string within each string in a strings column.
     *
     * Null string entries return corresponding null output column entries.
     *
     * @tparam FindFunction Returns integer character position value given a string and target.
     *
     * @param strings Strings column to search for target.
     * @param target String to search for in each string in the strings column.
     * @param start First character position to start the search.
     * @param stop Last character position (exclusive) to end the search.
     * @param pfn Strings instance for this operation.
     * @param mr Resource for allocating device memory.
     * @param stream Stream to use for kernel calls.
     * @return New integer column with character position values.
     */
    template <typename IntegerType, typename FindFunction, std::enable_if_t<std::is_integral<IntegerType>::value>* = nullptr>
    std::unique_ptr<cudf::column> operator()( cudf::strings_column_view const& strings,
                                              std::string const& target,
                                              cudf::size_type start, cudf::size_type stop,
                                              FindFunction& pfn,
                                              rmm::mr::device_memory_resource* mr,
                                              cudaStream_t stream ) const
    {
        CUDF_EXPECTS( !target.empty(), "Parameter target must not be empty.");
        CUDF_EXPECTS( start >= 0, "Parameter start must be positive integer or zero.");
        if( (stop) > 0 && (start >stop) )
            CUDF_FAIL( "Parameter start must be less than stop.");
        //
        auto target_ptr = cudf::strings::detail::string_from_host(target.c_str(), stream);
        auto d_target = *target_ptr;
        auto strings_column = cudf::column_device_view::create(strings.parent(),stream);
        auto d_column = *strings_column;
        // copy the null mask
        auto strings_count = strings.size();
        rmm::device_buffer null_mask;
        cudf::size_type null_count = d_column.null_count();
        if( d_column.nullable() )
            null_mask = rmm::device_buffer( d_column.null_mask(),
                                            cudf::bitmask_allocation_size_bytes(strings_count),
                                            stream, mr);
        // create output column
        auto results = std::make_unique<cudf::column>( cudf::data_type{cudf::experimental::type_to_id<IntegerType>()},
            strings_count, rmm::device_buffer(strings_count * sizeof(IntegerType), stream, mr),
            null_mask, null_count);
        auto results_view = results->mutable_view();
        auto d_results = results_view.data<IntegerType>();
        // set the position values by evaluating the passed function
        thrust::transform( rmm::exec_policy(stream)->on(stream),
            thrust::make_counting_iterator<cudf::size_type>(0),
            thrust::make_counting_iterator<cudf::size_type>(strings_count),
            d_results, find_fn<IntegerType,FindFunction>{d_column, pfn, d_target, start, stop});
        //
        results->set_null_count(null_count);
        return results;
    }

    template <typename IntegerType, typename FindFunction, std::enable_if_t<not std::is_integral<IntegerType>::value>* = nullptr>
    std::unique_ptr<cudf::column> operator()( cudf::strings_column_view const& strings,
                                              std::string const& target,
                                              cudf::size_type start, cudf::size_type stop,
                                              FindFunction& pfn,
                                              rmm::mr::device_memory_resource* mr,
                                              cudaStream_t stream ) const
    {
        CUDF_FAIL("Output type must be integral type.");
    }
};

} // namespace

namespace cudf
{
namespace strings
{
namespace detail
{
std::unique_ptr<cudf::column> find( cudf::strings_column_view const& strings,
                                    std::string const& target,
                                    cudf::size_type start=0, cudf::size_type stop=-1,
                                    cudf::data_type output_type = cudf::data_type{INT32},
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                    cudaStream_t stream=0 )
{
    auto pfn = [] __device__ (string_view d_string, string_view d_target,
                              size_type start, size_type stop) {
        size_type length = d_string.length();
        size_type end = (stop < 0) || (stop > length) ? length : stop;
        return d_string.find( d_target, start, end-start );
    };

    return cudf::experimental::type_dispatcher( output_type, dispatch_find_fn{},
                                                strings, target, start, stop, pfn, mr, stream);
}

std::unique_ptr<cudf::column> rfind( cudf::strings_column_view const& strings,
                                     std::string const& target,
                                     cudf::size_type start=0, cudf::size_type stop=-1,
                                     cudf::data_type output_type = cudf::data_type{INT32},
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                     cudaStream_t stream=0 )
{
    auto pfn = [] __device__ (string_view d_string, string_view d_target,
                              size_type start, size_type stop) {
        size_type length = d_string.length();
        size_type end = (stop < 0) || (stop > length) ? length : stop;
        return d_string.rfind( d_target, start, end-start );
    };

    return cudf::experimental::type_dispatcher( output_type, dispatch_find_fn{},
                                                strings, target, start, stop, pfn, mr, stream);
}

} // namespace detail

// external APIs

std::unique_ptr<cudf::column> find( strings_column_view const& strings,
                                    std::string const& target,
                                    size_type start, size_type stop,
                                    cudf::data_type output_type,
                                    rmm::mr::device_memory_resource* mr)
{
    return detail::find( strings, target, start, stop, output_type, mr );
}

std::unique_ptr<cudf::column> rfind( strings_column_view const& strings,
                                     std::string const& target,
                                     size_type start, size_type stop,
                                     cudf::data_type output_type,
                                     rmm::mr::device_memory_resource* mr)
{
    return detail::rfind( strings, target, start, stop, output_type, mr );
}

namespace
{

/**
 * @brief Utility to return a bool column indicating the presence of
 * a given target string in a strings column.
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam BoolFunction Return bool value given two strings.
 *
 * @param strings Column of strings to check for target.
 * @param target UTF-8 encoded string to check in strings column.
 * @param pfn Returns bool value if target is found in the given string.
 * @param mr Resource for allocating device memory.
 * @param stream Stream to use for kernel calls.
 * @return New INT8 column with character position values.
 */
template <typename BoolFunction>
std::unique_ptr<cudf::column> contains_fn( cudf::strings_column_view const& strings,
                                           std::string const& target,
                                           BoolFunction pfn,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream )
{
    auto strings_count = strings.size();
    if( strings_count == 0 )
        return cudf::make_numeric_column( data_type{INT8}, 0 );

    CUDF_EXPECTS( !target.empty(), "Parameter target must not be empty.");
    auto target_ptr = cudf::strings::detail::string_from_host(target.c_str(),stream);
    auto d_target = *target_ptr;
    auto strings_column = cudf::column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;

    // copy the null mask
    rmm::device_buffer null_mask;
    cudf::size_type null_count = d_column.null_count();
    if( d_column.nullable() )
        null_mask = rmm::device_buffer( d_column.null_mask(),
                                        cudf::bitmask_allocation_size_bytes(strings_count),
                                        stream, mr);
    // create output column
    // TODO make this bool8 type
    auto results = std::make_unique<cudf::column>( cudf::data_type{cudf::INT8}, strings_count,
        rmm::device_buffer(strings_count * sizeof(int8_t), stream, mr),
        null_mask, null_count);
    auto results_view = results->mutable_view();
    auto d_results = results_view.data<int8_t>();
    // set the values but evaluating the passed function
    thrust::transform( rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<cudf::size_type>(0),
        thrust::make_counting_iterator<cudf::size_type>(strings_count),
        d_results,
        [d_column, pfn, d_target] __device__ (cudf::size_type idx) {
            int8_t result = 0;
            if( !d_column.is_null(idx) )
                result = static_cast<int8_t>(pfn(d_column.element<cudf::string_view>(idx), d_target));
            return result;
        });
    results->set_null_count(null_count);
    return results;
}

} // namespace

namespace detail
{

std::unique_ptr<cudf::column> contains( cudf::strings_column_view const& strings,
                                        std::string const& target,
                                        rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                        cudaStream_t stream=0 )
{
    auto pfn = [] __device__ (string_view d_string, string_view d_target) {
        return d_string.find( d_target )>=0;
    };

    return contains_fn( strings, target, pfn, mr, stream );
}

std::unique_ptr<cudf::column> starts_with( cudf::strings_column_view const& strings,
                                           std::string const& target,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                           cudaStream_t stream=0 )
{
    auto pfn = [] __device__ (cudf::string_view d_string, cudf::string_view d_target) {
        return d_string.find( d_target )==0;
    };
    return contains_fn( strings, target, pfn, mr, stream );
}

std::unique_ptr<cudf::column> ends_with( cudf::strings_column_view const& strings,
                                         std::string const& target,
                                         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                         cudaStream_t stream=0 )
{

    auto pfn = [] __device__ (string_view d_string, string_view d_target) {
        auto str_length = d_string.length();
        auto tgt_length = d_target.length();
        if( str_length <= tgt_length )
            return false;
        return d_string.find( d_target, str_length - tgt_length )>=0;
    };

    return contains_fn( strings, target, pfn, mr, stream );
}

} // namespace detail

// external APIs

std::unique_ptr<cudf::column> contains( strings_column_view const& strings,
                                        std::string const& target,
                                        rmm::mr::device_memory_resource* mr )
{
    return detail::contains( strings, target, mr );
}

std::unique_ptr<cudf::column> starts_with( strings_column_view const& strings,
                                           std::string const& target,
                                           rmm::mr::device_memory_resource* mr )
{
    return detail::starts_with( strings, target, mr );
}

std::unique_ptr<cudf::column> ends_with( strings_column_view const& strings,
                                         std::string const& target,
                                         rmm::mr::device_memory_resource* mr )
{
    return detail::ends_with( strings, target, mr );
}

} // namespace strings
} // namespace cudf
