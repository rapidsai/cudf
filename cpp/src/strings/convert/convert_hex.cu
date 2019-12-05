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
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/traits.hpp>
#include <strings/utilities.hpp>
#include <strings/utilities.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf
{
namespace strings
{
namespace detail
{
namespace
{

/**
 * @brief Converts hex strings into an integers.
 *
 * Used by the dispatch method to convert to different integer types.
 */
template <typename IntegerType>
struct hex_to_integer_fn
{
    column_device_view const strings_column;

    /**
     * @brief Converts a single hex string into an integer.
     *
     * Non-hexadecimal characters are ignored.
     * This means it can handle "0x01A23" and "1a23".
     *
     * Overflow of the int64 type is not detected.
     */
    __device__ int64_t string_to_integer( string_view const& d_str )
    {
        int64_t result = 0, base = 1;
        const char* str = d_str.data();
        size_type index = d_str.size_bytes();
        while( index-- > 0 )
        {
            char ch = str[index];
            if( ch >= '0' && ch <= '9' )
            {
                result += static_cast<int64_t>(ch-48) * base;
                base *= 16;
            }
            else if( ch >= 'A' && ch <= 'Z' )
            {
                result += static_cast<int64_t>(ch-55) * base;
                base *= 16;
            }
            else if( ch >= 'a' && ch <= 'z' )
            {
                result += static_cast<int64_t>(ch-87) * base;
                base *= 16;
            }
        }
        return result;
    }

    __device__ IntegerType operator()(size_type idx)
    {
        if( strings_column.is_null(idx) )
            return static_cast<IntegerType>(0);
        // the cast to IntegerType will create predictable results
        // for integers that are larger than the IntegerType can hold
        return static_cast<IntegerType>(string_to_integer(strings_column.element<string_view>(idx)));
    }
};

/**
 * @brief The dispatch functions for converting strings to integers.
 *
 * The output_column is expected to be one of the integer types only.
 */
struct dispatch_hex_to_integers_fn
{
    template <typename IntegerType, std::enable_if_t<std::is_integral<IntegerType>::value>* = nullptr>
    void operator()( column_device_view const& strings_column,
                     mutable_column_view& output_column,
                     cudaStream_t stream ) const
    {
        auto d_results = output_column.data<IntegerType>();
        thrust::transform( rmm::exec_policy(stream)->on(stream),
            thrust::make_counting_iterator<size_type>(0),
            thrust::make_counting_iterator<size_type>(strings_column.size()),
            d_results, hex_to_integer_fn<IntegerType>{strings_column});
    }
    // non-integral types throw an exception
    template <typename T, std::enable_if_t<not std::is_integral<T>::value>* = nullptr>
    void operator()(column_device_view const&, mutable_column_view&, cudaStream_t) const
    {
        CUDF_FAIL("Output for hex_to_integers must be an integral type.");
    }
};

template <>
void dispatch_hex_to_integers_fn::operator()<experimental::bool8>(column_device_view const&, mutable_column_view&, cudaStream_t) const
{
    CUDF_FAIL("Output for hex_to_integers must not be a boolean type.");
}

} // namespace


// This will convert a strings column into any integer column type.
std::unique_ptr<column> hex_to_integers( strings_column_view const& strings,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                         cudaStream_t stream = 0)
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return make_empty_column( output_type );
    auto strings_column = column_device_view::create(strings.parent(), stream);
    auto d_strings = *strings_column;
    // create integer output column copying the strings null-mask
    auto results = make_numeric_column( output_type, strings_count,
        copy_bitmask( strings.parent(), stream, mr), strings.null_count(), stream, mr);
    auto results_view = results->mutable_view();
    // fill output column with integers
    experimental::type_dispatcher( output_type, dispatch_hex_to_integers_fn{},
                                   d_strings, results_view, stream );
    results->set_null_count(strings.null_count());
    return results;
}

} // namespace detail

// external API
std::unique_ptr<column> hex_to_integers( strings_column_view const& strings,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr)
{
    return detail::hex_to_integers(strings, output_type, mr );
}

} // namespace strings
} // namespace cudf
