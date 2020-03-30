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
#include <cudf/strings/translate.hpp>
#include <strings/utilities.hpp>
#include <strings/utilities.cuh>

#include <algorithm>
#include <thrust/find.h>

namespace cudf
{
namespace strings
{
namespace detail
{

using translate_table = thrust::pair<char_utf8,char_utf8>;

namespace
{

/**
 * @brief This is the translate functor for replacing individual characters
 * in each string.
 */
struct translate_fn
{
    column_device_view const d_strings;
    rmm::device_vector<translate_table>::iterator table_begin;
    rmm::device_vector<translate_table>::iterator table_end;
    int32_t const* d_offsets{};
    char* d_chars{};

    __device__ size_type operator()(size_type idx)
    {
        if( d_strings.is_null(idx) )
            return 0;
        string_view d_str = d_strings.element<string_view>(idx);
        size_type bytes = d_str.size_bytes();
        char* out_ptr = d_offsets ? d_chars + d_offsets[idx] : nullptr;
        for( auto chr : d_str )
        {
            auto entry = thrust::find_if( thrust::seq, table_begin, table_end,
                [chr] __device__ ( auto const& te ) { return te.first==chr; } );
            if( entry != table_end )
            {
                bytes -= bytes_in_char_utf8(chr);
                chr = static_cast<translate_table>(*entry).second;
                if( chr ) // if null, skip the character
                    bytes += bytes_in_char_utf8(chr);
            }
            if( chr && out_ptr )
                out_ptr += from_char_utf8(chr,out_ptr);
        }
        return bytes;
    }
};

} // namespace

//
std::unique_ptr<column> translate( strings_column_view const& strings,
                                   std::vector<std::pair<char_utf8,char_utf8>> const& chars_table,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                   cudaStream_t stream = 0 )
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);

    size_type table_size = static_cast<size_type>(chars_table.size());
    // convert input table
    thrust::host_vector<translate_table> htable(table_size);
    std::transform( chars_table.begin(), chars_table.end(), htable.begin(),
        [] ( auto entry ) { return translate_table{entry.first,entry.second}; });
    // copy translate table to device memory
    rmm::device_vector<translate_table> table(htable);

    auto execpol = rmm::exec_policy(stream);
    auto strings_column = column_device_view::create(strings.parent(), stream);
    auto d_strings = *strings_column;
    // create null mask
    rmm::device_buffer null_mask = copy_bitmask( strings.parent(), stream, mr );
    // create offsets column
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<int32_t>(0),
         translate_fn{d_strings,table.begin(),table.end()});
    auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                                    offsets_transformer_itr+strings_count,
                                                    mr, stream);
    auto d_offsets = offsets_column->view().data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
    auto chars_column = strings::detail::create_chars_child_column( strings_count, strings.null_count(), bytes, mr, stream );
    auto d_chars = chars_column->mutable_view().data<char>();
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream), thrust::make_counting_iterator<cudf::size_type>(0), strings_count,
        translate_fn{d_strings,table.begin(),table.end(),d_offsets,d_chars});
    //
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               strings.null_count(), std::move(null_mask), stream, mr);
}

} // namespace detail

// external APIs

std::unique_ptr<column> translate( strings_column_view const& strings,
                                   std::vector<std::pair<uint32_t,uint32_t>> const& chars_table,
                                   rmm::mr::device_memory_resource* mr )
{
    return detail::translate(strings,chars_table);
}

} // namespace strings
} // namespace cudf
