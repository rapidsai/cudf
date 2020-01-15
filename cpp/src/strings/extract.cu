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

#include <cudf/null_mask.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/char_types/char_types.hpp>
#include <strings/utilities.hpp>
#include <strings/regex/regex.cuh>


namespace cudf
{
namespace strings
{
namespace detail
{

using string_index_pair = thrust::pair<const char*,size_type>;

namespace
{

/**
 * @brief This functor handles extracting strings by applying the compiled regex pattern
 * and creating string_index_pairs for all the substrings.
 *
 * @tparam stack_size Correlates to the regex instructions state to maintain for each string.
 *         Each instruction requires a fixed amount of overhead data.
 */
template<size_t stack_size>
struct extract_fn
{
    reprog_device prog;
    column_device_view d_strings;
    size_type column_index;

    __device__ string_index_pair operator()(size_type idx)
    {
        u_char data1[stack_size], data2[stack_size];
        prog.set_stack_mem(data1,data2);
        if( d_strings.is_null(idx) )
            return string_index_pair{nullptr,0};
        string_view d_str = d_strings.element<string_view>(idx);
        string_index_pair result{nullptr,0};
        int32_t begin = 0, end = d_str.length();
        if( (prog.find(idx,d_str,begin,end) > 0) &&
            (prog.extract(idx,d_str,begin,end,column_index) > 0) )
        {
            auto offset = d_str.byte_offset(begin);
            result = string_index_pair{ d_str.data() + offset, d_str.byte_offset(end)-offset };
        }
        return result;
    }
};

} // namespace

//
std::unique_ptr<experimental::table> extract( strings_column_view const& strings,
                                              std::string const& pattern,
                                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                              cudaStream_t stream = 0)
{
    auto strings_count = strings.size();
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column;

    // compile regex into device object
    auto prog = reprog_device::create(pattern,get_character_flags_table(),strings_count,stream);
    auto d_prog = *prog;
    // extract should include groups
    int groups = d_prog.group_counts();
    CUDF_EXPECTS( groups > 0, "Group indicators not found in regex pattern");

    // build a result column for each group
    std::vector<std::unique_ptr<column>> results;
    auto execpol = rmm::exec_policy(stream);
    auto regex_insts = d_prog.insts_counts();
    for( int32_t column_index=0; column_index < groups; ++column_index )
    {
        rmm::device_vector<string_index_pair> indices(strings_count);
        string_index_pair* d_indices = indices.data().get();

        if( (regex_insts > MAX_STACK_INSTS) || (regex_insts <= RX_SMALL_INSTS) )
            thrust::transform(execpol->on(stream),
                thrust::make_counting_iterator<size_type>(0),
                thrust::make_counting_iterator<size_type>(strings_count),
                d_indices, extract_fn<RX_STACK_SMALL>{d_prog, d_strings, column_index});
        else if( regex_insts <= RX_MEDIUM_INSTS )
            thrust::transform(execpol->on(stream),
                thrust::make_counting_iterator<size_type>(0),
                thrust::make_counting_iterator<size_type>(strings_count),
                d_indices, extract_fn<RX_STACK_MEDIUM>{d_prog, d_strings, column_index});
        else
            thrust::transform(execpol->on(stream),
                thrust::make_counting_iterator<size_type>(0),
                thrust::make_counting_iterator<size_type>(strings_count),
                d_indices, extract_fn<RX_STACK_LARGE>{d_prog, d_strings, column_index});
        //
        auto column = make_strings_column(indices,stream,mr);
        results.emplace_back(std::move(column));
    }
    return std::make_unique<experimental::table>(std::move(results));
}

} // namespace detail

// external API

std::unique_ptr<experimental::table> extract( strings_column_view const& strings,
                                              std::string const& pattern,
                                              rmm::mr::device_memory_resource* mr)
{
    return detail::extract(strings, pattern, mr);
}

} // namespace strings
} // namespace cudf
