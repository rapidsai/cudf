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
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/replace_re.hpp>
#include <strings/utilities.hpp>
#include <strings/utilities.cuh>
#include <strings/regex/regex.cuh>


namespace cudf
{
namespace strings
{
namespace detail
{
namespace
{

/**
 * @brief This functor handles replacing strings by applying the compiled regex patterns
 * and inserting the corresponding new string within the matched range of characters.
 *
 * The logic includes computing the size of each string and also writing the output.
 *
 * The stack is used to keep progress on evaluating the regex instructions on each string.
 * So the size of the stack is in proportion to the number of instructions in the given regex pattern.
 *
 * There are three call types based on the number of regex instructions in the given pattern.
 * Small to medium instruction lengths can use the stack effectively though smaller executes faster.
 * Longer patterns require global memory. Shorter patterns are common in data cleaning.
 *
 */
template<size_t stack_size>
struct replace_multi_regex_fn
{
    column_device_view const d_strings;
    reprog_device* progs; // array of regex progs
    size_type number_of_patterns;
    column_device_view const d_repls; // replacment strings
    const int32_t* d_offsets{}; // these are null when
    char* d_chars{};            // only computing size

    __device__ size_type operator()(size_type idx)
    {
        if( d_strings.is_null(idx) )
            return 0;
        u_char data1[stack_size];
        u_char data2[stack_size];
        string_view d_str = d_strings.element<string_view>(idx);
        auto nchars = d_str.length();     // number of characters in input string
        auto nbytes = d_str.size_bytes(); // number of bytes in input string
        const char* in_ptr = d_str.data(); // input pointer (i)
        char* out_ptr = nullptr;  // running output pointer (o)
        if( d_offsets )
            out_ptr = d_chars + d_offsets[idx];
        size_type lpos = 0;
        size_type ch_pos = 0;
        while( ch_pos < nchars )
        {
            for( size_type ptn_idx=0; ptn_idx < number_of_patterns; ++ptn_idx )
            {
                reprog_device prog = progs[ptn_idx];
                prog.set_stack_mem(data1,data2);
                size_type begin = ch_pos, end = ch_pos+1;
                if( prog.find(idx,d_str,begin,end) > 0 )
                {
                    string_view d_repl = d_repls.size() > 1 ?
                                         d_repls.element<string_view>(ptn_idx) :
                                         d_repls.element<string_view>(0);
                    auto spos = d_str.byte_offset(begin);
                    auto epos = d_str.byte_offset(end);
                    nbytes += d_repl.size_bytes() - (epos - spos);
                    if( out_ptr )
                    {
                        out_ptr = copy_and_increment(out_ptr,in_ptr+lpos,spos-lpos);
                        out_ptr = copy_string(out_ptr, d_repl);
                        lpos = epos;
                    }
                    ch_pos = end - 1;
                    break; // go to next character position
                }
            }
            ++ch_pos;
        }
        if( out_ptr ) // copy the remainder
            memcpy(out_ptr, in_ptr+lpos, d_str.size_bytes()-lpos);
        return nbytes;
    }
};

} // namespace

//
std::unique_ptr<column> replace_re( strings_column_view const& strings,
                                    std::vector<std::string> const& patterns,
                                    strings_column_view const& repls,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                    cudaStream_t stream = 0)
{
    auto strings_count = strings.size();
    if( strings_count==0 )
        return make_empty_strings_column(mr,stream);
    if( patterns.empty() ) // no patterns; just return a copy
        return std::make_unique<column>(strings.parent());

    CUDF_EXPECTS( !repls.has_nulls(), "Parameter repls must not have any nulls");

    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column;
    auto repls_column = column_device_view::create(repls.parent(),stream);
    auto d_repls = *repls_column;
    auto d_flags = get_character_flags_table();
    // compile regexes into device objects
    size_type regex_insts = 0;
    std::vector<std::unique_ptr<reprog_device, std::function<void(reprog_device*)> > > h_progs;
    rmm::device_vector<reprog_device> progs;
    for( auto itr = patterns.begin(); itr != patterns.end(); ++itr )
    {
        auto prog = reprog_device::create(*itr,d_flags,strings_count,stream);
        auto insts = prog->insts_counts();
        if( insts > regex_insts )
            regex_insts = insts;
        progs.push_back(*prog);
        h_progs.emplace_back(std::move(prog));
    }
    auto d_progs = progs.data().get();

    // copy null mask
    auto null_mask = copy_bitmask(strings.parent());
    auto null_count = strings.null_count();

    // create child columns
    std::pair< std::unique_ptr<column>, std::unique_ptr<column> > children(nullptr,nullptr);
    // Each invocation is predicated on the stack size which is dependent on the number of regex instructions
    if( (regex_insts > MAX_STACK_INSTS) || (regex_insts <= RX_SMALL_INSTS) )
        children = make_strings_children(
            replace_multi_regex_fn<RX_STACK_SMALL>{d_strings,d_progs,static_cast<size_type>(progs.size()),d_repls},
            strings_count, null_count, mr, stream);
    else if( regex_insts <= RX_MEDIUM_INSTS )
        children = make_strings_children(
            replace_multi_regex_fn<RX_STACK_MEDIUM>{d_strings,d_progs,static_cast<size_type>(progs.size()),d_repls},
            strings_count, null_count, mr, stream);
    else
        children = make_strings_children(
            replace_multi_regex_fn<RX_STACK_LARGE>{d_strings,d_progs,static_cast<size_type>(progs.size()),d_repls},
            strings_count, null_count, mr, stream);
    //
    return make_strings_column(strings_count, std::move(children.first), std::move(children.second),
                               null_count, std::move(null_mask), stream, mr);
}

} // namespace detail

// external API

std::unique_ptr<column> replace_re( strings_column_view const& strings,
                                    std::vector<std::string> const& patterns,
                                    strings_column_view const& repls,
                                    rmm::mr::device_memory_resource* mr )
{
    return detail::replace_re(strings, patterns, repls, mr);
}

} // namespace strings
} // namespace cudf
