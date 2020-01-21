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
#include "./utilities.hpp"

#include <thrust/transform.h>

namespace cudf
{
namespace strings
{
namespace detail
{
namespace
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
template <typename FindFunction>
std::unique_ptr<column> find_fn( strings_column_view const& strings,
                                 string_scalar const& target,
                                 size_type start, size_type stop,
                                 FindFunction& pfn,
                                 rmm::mr::device_memory_resource* mr,
                                 cudaStream_t stream )
{
    CUDF_EXPECTS( target.is_valid() && target.size()>0, "Parameter target must not be empty.");
    CUDF_EXPECTS( start >= 0, "Parameter start must be positive integer or zero.");
    if( (stop) > 0 && (start >stop) )
        CUDF_FAIL( "Parameter start must be less than stop.");
    //
    auto d_target = string_view(target.data(),target.size());
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column;
    auto strings_count = strings.size();
    // create output column
    auto results = make_numeric_column( data_type{INT32}, strings_count,
        copy_bitmask( strings.parent(), stream, mr ), strings.null_count(), stream, mr);
    auto results_view = results->mutable_view();
    auto d_results = results_view.data<int32_t>();
    // set the position values by evaluating the passed function
    thrust::transform( rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count),
        d_results, [d_strings, pfn, d_target, start, stop] __device__ (size_type idx) {
            int32_t position = -1;
            if( !d_strings.is_null(idx) )
                position = static_cast<int32_t>(pfn(d_strings.element<string_view>(idx),d_target,start,stop));
            return position;
        });
    results->set_null_count(strings.null_count());
    return results;
}

} // namespace

std::unique_ptr<column> find( strings_column_view const& strings,
                              string_scalar const& target,
                              size_type start=0, size_type stop=-1,
                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                              cudaStream_t stream=0 )
{
    auto pfn = [] __device__ (string_view d_string, string_view d_target,
                              size_type start, size_type stop) {
        size_type length = d_string.length();
        size_type end = (stop < 0) || (stop > length) ? length : stop;
        return d_string.find( d_target, start, end-start );
    };

    return find_fn( strings, target, start, stop, pfn, mr, stream);
}

std::unique_ptr<column> rfind( strings_column_view const& strings,
                               string_scalar const& target,
                               size_type start=0, size_type stop=-1,
                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                               cudaStream_t stream=0 )
{
    auto pfn = [] __device__ (string_view d_string, string_view d_target,
                              size_type start, size_type stop) {
        size_type length = d_string.length();
        size_type end = (stop < 0) || (stop > length) ? length : stop;
        return d_string.rfind( d_target, start, end-start );
    };

    return find_fn( strings, target, start, stop, pfn, mr, stream);
}

} // namespace detail

// external APIs

std::unique_ptr<column> find( strings_column_view const& strings,
                              string_scalar const& target,
                              size_type start, size_type stop,
                              rmm::mr::device_memory_resource* mr)
{
    return detail::find( strings, target, start, stop, mr );
}

std::unique_ptr<column> rfind( strings_column_view const& strings,
                               string_scalar const& target,
                               size_type start, size_type stop,
                               rmm::mr::device_memory_resource* mr)
{
    return detail::rfind( strings, target, start, stop, mr );
}

namespace detail
{
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
 * @return New BOOL8 column.
 */
template <typename BoolFunction>
std::unique_ptr<column> contains_fn( strings_column_view const& strings,
                                     string_scalar const& target,
                                     BoolFunction pfn,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream )
{
    auto strings_count = strings.size();
    if( strings_count == 0 )
        return make_numeric_column( data_type{BOOL8}, 0 );

    CUDF_EXPECTS( target.is_valid() && target.size()>0, "Parameter target must not be empty.");
    auto d_target = string_view( target.data(), target.size());
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column;
    // create output column
    auto results = make_numeric_column( data_type{BOOL8}, strings_count,
        copy_bitmask( strings.parent(), stream, mr ), strings.null_count(), stream, mr);
    auto results_view = results->mutable_view();
    auto d_results = results_view.data<experimental::bool8>();
    // set the bool values but evaluating the passed function
    thrust::transform( rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count),
        d_results,
        [d_strings, pfn, d_target] __device__ (size_type idx) {
            experimental::bool8 result = 0;
            if( !d_strings.is_null(idx) )
                result = static_cast<experimental::bool8>(pfn(d_strings.element<string_view>(idx), d_target));
            return result;
        });
    results->set_null_count(strings.null_count());
    return results;
}

} // namespace


std::unique_ptr<column> contains( strings_column_view const& strings,
                                  string_scalar const& target,
                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                  cudaStream_t stream=0 )
{
    auto pfn = [] __device__ (string_view d_string, string_view d_target) {
        return d_string.find( d_target )>=0;
    };

    return contains_fn( strings, target, pfn, mr, stream );
}

std::unique_ptr<column> starts_with( strings_column_view const& strings,
                                     string_scalar const& target,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                     cudaStream_t stream=0 )
{
    auto pfn = [] __device__ (string_view d_string, string_view d_target) {
        return d_string.find( d_target )==0;
    };
    return contains_fn( strings, target, pfn, mr, stream );
}

std::unique_ptr<column> ends_with( strings_column_view const& strings,
                                   string_scalar const& target,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                   cudaStream_t stream=0 )
{

    auto pfn = [] __device__ (string_view d_string, string_view d_target) {
        auto str_length = d_string.length();
        auto tgt_length = d_target.length();
        if( str_length < tgt_length )
            return false;
        return d_string.find( d_target, str_length - tgt_length )>=0;
    };

    return contains_fn( strings, target, pfn, mr, stream );
}

} // namespace detail

// external APIs

std::unique_ptr<column> contains( strings_column_view const& strings,
                                  string_scalar const& target,
                                  rmm::mr::device_memory_resource* mr )
{
    return detail::contains( strings, target, mr );
}

std::unique_ptr<column> starts_with( strings_column_view const& strings,
                                     string_scalar const& target,
                                     rmm::mr::device_memory_resource* mr )
{
    return detail::starts_with( strings, target, mr );
}

std::unique_ptr<column> ends_with( strings_column_view const& strings,
                                   string_scalar const& target,
                                   rmm::mr::device_memory_resource* mr )
{
    return detail::ends_with( strings, target, mr );
}

} // namespace strings
} // namespace cudf
