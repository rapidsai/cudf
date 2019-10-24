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

#include <rmm/thrust_rmm_allocator.h>

namespace
{

/**
 * @brief Utility to build int32 column from output of the
 * provided FindFunction.
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam FindFunction Returns int32 position value given a string_view.
 * @param pfn Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @param stream Stream to use for kernel calls.
 * @return New INT32 column with character position values.
 */
template<typename FindFunction>
std::unique_ptr<cudf::column> find_fn( cudf::strings_column_view strings,
                                       FindFunction& pfn,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                       cudaStream_t stream=0 )
{
    auto strings_count = strings.size();
    auto execpol = rmm::exec_policy(stream);
    auto strings_column = cudf::column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;

    // copy the null mask
    rmm::device_buffer null_mask;
    cudf::size_type null_count = d_column.null_count();
    if( d_column.nullable() )
        null_mask = rmm::device_buffer( d_column.null_mask(),
                                        gdf_valid_allocation_size(strings_count),
                                        stream, mr);
    // create output column
    auto results = std::make_unique<cudf::column>( cudf::data_type{cudf::INT32}, strings_count,
        rmm::device_buffer(strings_count * sizeof(int32_t), stream, mr),
        null_mask, null_count);
    auto results_view = results->mutable_view();
    auto d_results = results_view.data<int32_t>();
    // set the values but evaluating the passed function
    thrust::transform( execpol->on(stream),
        thrust::make_counting_iterator<cudf::size_type>(0),
        thrust::make_counting_iterator<cudf::size_type>(strings_count),
        d_results,
        [d_column, pfn] __device__ (cudf::size_type idx) {
            if( d_column.is_null(idx) )
                return 0;
            return pfn(d_column.element<cudf::string_view>(idx));
        });
    results->set_null_count(null_count);
    return results;
}

/**
 * @brief Utility to build bool column from output of the
 * provided BoolFunction.
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam BoolFunction Returns bool value given a string_view.
 * @param pfn Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @param stream Stream to use for kernel calls.
 * @return New INT32 column with character position values.
 */
template <typename BoolFunction>
std::unique_ptr<cudf::column> contains_fn( cudf::strings_column_view strings,
                                           BoolFunction pfn,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                           cudaStream_t stream=0 )
{
    auto strings_count = strings.size();
    auto execpol = rmm::exec_policy(stream);
    auto strings_column = cudf::column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;

    // copy the null mask
    rmm::device_buffer null_mask;
    cudf::size_type null_count = d_column.null_count();
    if( d_column.nullable() )
        null_mask = rmm::device_buffer( d_column.null_mask(),
                                        gdf_valid_allocation_size(strings_count),
                                        stream, mr);

    // create output column -- TODO make this bool8
    auto results = std::make_unique<cudf::column>( cudf::data_type{cudf::INT8}, strings_count,
        rmm::device_buffer(strings_count * sizeof(int8_t), stream, mr),
        null_mask, null_count);
    auto results_view = results->mutable_view();
    auto d_results = results_view.data<int8_t>();
    // set the values but evaluating the passed function
    thrust::transform( execpol->on(stream),
        thrust::make_counting_iterator<cudf::size_type>(0),
        thrust::make_counting_iterator<cudf::size_type>(strings_count),
        d_results,
        [d_column, pfn] __device__ (cudf::size_type idx) {
            if( d_column.is_null(idx) )
                return (int8_t)0;
            return (int8_t)pfn(d_column.element<cudf::string_view>(idx));
        });
    results->set_null_count(null_count);
    return results;
}
} // namespace

namespace cudf
{
namespace strings
{

// APIs
std::unique_ptr<cudf::column> find( strings_column_view strings,
                                    const char* target,
                                    int32_t start, int32_t stop,
                                    rmm::mr::device_memory_resource* mr )
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return detail::make_empty_strings_column(mr);
    CUDF_EXPECTS( target!=nullptr, "Parameter target must not be null.");
    CUDF_EXPECTS( start >= 0, "Parameter start must be positive integer or zero.");
    if( (stop) > 0 && (start >stop) )
        CUDF_FAIL( "Parameter start must be less than stop.");

    auto target_ptr = detail::string_from_host(target);
    auto d_target = *target_ptr;

    auto pfn = [d_target, start, stop] __device__ (const cudf::string_view& d_str) {
        size_type length = d_str.length();
        size_type end = (stop < 0) || (stop > length) ? length : stop;
        return d_str.find( d_target, start, end-start );
    };

    return find_fn( strings, pfn, mr);
}

std::unique_ptr<cudf::column> rfind( strings_column_view strings,
                                     const char* target,
                                     int32_t start, int32_t stop,
                                     rmm::mr::device_memory_resource* mr )
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return detail::make_empty_strings_column(mr);
    CUDF_EXPECTS( target!=nullptr, "Parameter target must not be null.");
    CUDF_EXPECTS( start >= 0, "Parameter start must be positive integer or zero.");
    if( (stop) > 0 && (start >stop) )
        CUDF_FAIL( "Parameter start must be less than stop.");

    auto target_ptr = detail::string_from_host(target);
    auto d_target = *target_ptr;

    auto pfn = [d_target, start, stop] __device__ (const cudf::string_view& d_str) {
        size_type length = d_str.length();
        size_type end = (stop < 0) || (stop > length) ? length : stop;
        return d_str.rfind( d_target, start, end-start );
    };

    return find_fn( strings, pfn, mr);
}

std::unique_ptr<cudf::column> contains( strings_column_view strings,
                                        const char* target,
                                        rmm::mr::device_memory_resource* mr )
{
    auto strings_count = strings.size();
    if( strings_count == 0 )
        return detail::make_empty_strings_column(mr);
    CUDF_EXPECTS( target!=nullptr, "Parameter target must not be null.");

    auto target_ptr = detail::string_from_host(target);
    auto d_target = *target_ptr;

    auto pfn = [d_target] __device__ (const cudf::string_view& d_str) {
        return d_str.find( d_target )>=0;
    };

    return contains_fn( strings, pfn, mr );
}

std::unique_ptr<cudf::column> starts_with( strings_column_view strings,
                                           const char* target,
                                           rmm::mr::device_memory_resource* mr )
{
    auto strings_count = strings.size();
    if( strings_count == 0 )
        return detail::make_empty_strings_column(mr);
    CUDF_EXPECTS( target!=nullptr, "Parameter target must not be null.");

    auto target_ptr = detail::string_from_host(target);
    auto d_target = *target_ptr;

    auto pfn = [d_target] __device__ (const cudf::string_view& d_str) {
        return d_str.find( d_target )==0;
    };

    return contains_fn( strings, pfn, mr );
}

std::unique_ptr<cudf::column> ends_with( strings_column_view strings,
                                         const char* target,
                                         rmm::mr::device_memory_resource* mr )
{
    auto strings_count = strings.size();
    if( strings_count == 0 )
        return detail::make_empty_strings_column(mr);
    CUDF_EXPECTS( target!=nullptr, "Parameter target must not be null.");

    auto target_ptr = detail::string_from_host(target);
    auto d_target = *target_ptr;

    auto pfn = [d_target] __device__ (const cudf::string_view& d_str) {
        auto str_length = d_str.length();
        auto tgt_length = d_target.length();
        if( str_length <= tgt_length )
            return false;
        return d_str.find( d_target, str_length - tgt_length )>=0;
    };

    return contains_fn( strings, pfn, mr );
}

} // namespace strings
} // namespace cudf
