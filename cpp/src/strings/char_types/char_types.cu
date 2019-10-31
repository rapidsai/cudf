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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/char_types/char_types.hpp>
#include "../utilities.hpp"
#include "../utilities.cuh"


namespace cudf
{
namespace strings
{
namespace detail
{
//
std::unique_ptr<cudf::column> all_characters_of_type( strings_column_view const& strings,
                                                      string_character_types types,
                                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                                      cudaStream_t stream = 0)
{
    auto strings_count = strings.size();
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_column = *strings_column;

    // copy the null mask
    rmm::device_buffer null_mask;
    cudf::size_type null_count = d_column.null_count();
    if( d_column.nullable() ) // copy null_mask
        null_mask = rmm::device_buffer( d_column.null_mask(),
                                        bitmask_allocation_size_bytes(strings_count),
                                        stream, mr);

    // create output column
    // TODO: use BOOL8 type here when available
    auto results = std::make_unique<cudf::column>( cudf::data_type{cudf::INT8}, strings_count,
        rmm::device_buffer(strings_count * sizeof(int8_t), stream, mr),
        null_mask, null_count);
    auto results_view = results->mutable_view();
    auto d_results = results_view.data<int8_t>();
    //
    auto d_flags = detail::get_character_flags_table();
    // set the output values by checking the character types
    thrust::transform(rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count),
        d_results,
        [d_column, d_flags, types, d_results] __device__(size_type idx){
            if( d_column.is_null(idx) )
                return static_cast<int8_t>(0);
            auto d_str = d_column.element<string_view>(idx);
            bool check = !d_str.empty(); // positive result requires at least one character
            for( auto itr = d_str.begin(); check && (itr != d_str.end()); ++itr )
            {
                auto code_point = detail::utf8_to_codepoint(*itr);
                // lookup flags in table by code-point
                auto flag = code_point <= 0x00FFFF ? d_flags[code_point] : 0;
                check = (types & flag) > 0;
            }
            return static_cast<int8_t>(check);
        });
    //
    results->set_null_count(null_count);
    return results;
}

} // namespace detail

std::unique_ptr<cudf::column> all_characters_of_type( strings_column_view const& strings,
                                                      string_character_types types,
                                                      rmm::mr::device_memory_resource* mr)
{
    return detail::all_characters_of_type(strings, types, mr);
}

} // namespace strings
} // namespace cudf
