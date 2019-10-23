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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/convert.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/traits.hpp>
#include "./utilities.hpp"
#include "./utilities.cuh"

#include <rmm/thrust_rmm_allocator.h>

namespace cudf
{
namespace strings
{
namespace detail
{

namespace
{

__device__ int64_t string_to_integer( const string_view& d_str )
{
    int64_t value = 0;
    size_type bytes = d_str.size_bytes();
    const char* ptr = d_str.data();
    int sign = 1;
    if( *ptr == '-' || *ptr == '+' )
    {
        sign = (*ptr=='-' ? -1:1);
        ++ptr;
        --bytes;
    }
    for( size_type idx=0; idx < bytes; ++idx )
    {
        char chr = *ptr++;
        if( chr < '0' || chr > '9' )
            break;
        value = (value * 10) + static_cast<int64_t>(chr - '0');
    }
    return value * static_cast<int64_t>(sign);
}

} // namespace

std::unique_ptr<cudf::column> to_integers( strings_column_view strings,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                           cudaStream_t stream = 0)
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);

    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(), stream);
    auto d_column = *strings_column;

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
    // set the values
    thrust::transform( execpol->on(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count),
        d_results,
        [d_column] __device__ (size_type idx) {
            if( d_column.is_null(idx) )
                return int32_t(0);
            return static_cast<int32_t>(string_to_integer(d_column.element<cudf::string_view>(idx)));
        });
    results->set_null_count(null_count);
    return results;
}

namespace
{

template <typename IntegerType>
struct integer_to_string_size_fn
{
    column_device_view d_column;

    __device__ size_type operator()(size_type idx)
    {
        if( d_column.is_null(idx) )
            return 0;
        IntegerType value = d_column.element<IntegerType>(idx);
        if( value==0 )
            return 1;
        bool sign = value < 0;
        if( sign )
            value = -value;
        size_type digits = 0;
        while( value > 0 )
        {
            ++digits;
            value = value/10;
        }
        return digits + static_cast<size_type>(sign);
    }
};

template <typename IntegerType>
struct integer_to_string_fn
{
    column_device_view d_column;
    const int32_t* d_offsets;
    char* d_chars;

    __device__ void operator()(size_type idx)
    {
        if( d_column.is_null(idx) )
            return;
        IntegerType value = d_column.element<IntegerType>(idx);
        char* d_buffer = d_chars + d_offsets[idx];
        if( value==0 )
        {
            memcpy( d_buffer, "0", 1);
            return;
        }
        bool sign = value < 0;
        if( sign )
            value = -value;
        char* ptr = d_buffer;
        while( value > 0 )
        {
            *ptr++ = '0' + (value % 10);
            value = value/10;
        }
        if( sign )
            *ptr++ = '-';
        size_type length = static_cast<size_type>(ptr-d_buffer);
        // numbers are backwards, reverse the string
        for( size_type j=0; j<(length/2); ++j )
        {
            char ch1 = d_buffer[j];
            char ch2 = d_buffer[length-j-1];
            d_buffer[j] = ch2;
            d_buffer[length-j-1] = ch1;
        }
    }
};

struct dispatch_from_integers_fn
{
    template <typename IntegerType, std::enable_if_t<std::is_integral<IntegerType>::value>* = nullptr>
    std::unique_ptr<cudf::column> operator()( column_view& integers,
                                              rmm::mr::device_memory_resource* mr,
                                              cudaStream_t stream ) const noexcept
    {
        size_type strings_count = integers.size();
        auto execpol = rmm::exec_policy(0);
        auto column = column_device_view::create(integers, stream);
        auto d_column = *column;

        // copy the null mask
        rmm::device_buffer null_mask;
        cudf::size_type null_count = d_column.null_count();
        if( d_column.nullable() )
            null_mask = rmm::device_buffer( d_column.null_mask(),
                                            gdf_valid_allocation_size(strings_count),
                                            stream, mr);
        // build offsets column
        auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<int32_t>(0),
            integer_to_string_size_fn<IntegerType>{d_column} );
        auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                                        offsets_transformer_itr+strings_count,
                                                        mr, stream);
        auto offsets_view = offsets_column->view();
        auto d_new_offsets = offsets_view.template data<int32_t>();

        // build chars column
        size_type bytes = thrust::device_pointer_cast(d_new_offsets)[strings_count];
        auto chars_column = create_chars_child_column( strings_count, null_count, bytes, mr, stream );
        auto chars_view = chars_column->mutable_view();
        auto d_chars = chars_view.template data<char>();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<cudf::size_type>(0), strings_count,
            integer_to_string_fn<IntegerType>{d_column, d_new_offsets, d_chars});
        //
        return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                                   null_count, std::move(null_mask), stream, mr);
    }

    template <typename T, std::enable_if_t<not std::is_integral<T>::value>* = nullptr>
    std::unique_ptr<cudf::column> operator()(column_view&, rmm::mr::device_memory_resource*, cudaStream_t) const noexcept
    {
        // suppress 'throw will always call terminate() [-Wterminate]'
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wterminate"
        CUDF_FAIL("Values must be integral type.");
        #pragma GCC diagnostic pop        
    }
};

}

std::unique_ptr<cudf::column> from_integers( column_view integers,
                                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                             cudaStream_t stream = 0)
{
    size_type strings_count = integers.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);

    return cudf::experimental::type_dispatcher(integers.type(),
                dispatch_from_integers_fn{},
                integers, mr, stream );
}

} // namespace detail

std::unique_ptr<cudf::column> to_integers( strings_column_view strings,
                                           rmm::mr::device_memory_resource* mr )
{
    return detail::to_integers(strings,mr);
}


std::unique_ptr<cudf::column> from_integers( column_view integers,
                                             rmm::mr::device_memory_resource* mr )
{
    return detail::from_integers(integers,mr);
}

} // namespace strings
} // namespace cudf
