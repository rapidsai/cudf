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
 * @brief This functor handles replacing strings by applying the compiled regex pattern
 * and inserting the new string within the matched range of characters.
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
struct replace_regex_fn
{
    column_device_view const d_strings;
    reprog_device prog;
    string_view const d_repl;
    size_type maxrepl;
    const int32_t* d_offsets{}; // these are null when
    char* d_chars{};            // only computing size

    __device__ size_type operator()(size_type idx)
    {
        if( d_strings.is_null(idx) )
            return 0;
        u_char data1[stack_size];
        u_char data2[stack_size];
        prog.set_stack_mem(data1,data2);
        string_view d_str = d_strings.element<string_view>(idx);
        auto mxn = maxrepl;
        auto nchars = d_str.length();     // number of characters in input string
        auto nbytes = d_str.size_bytes(); // number of bytes in input string
        if( mxn < 0 )
            mxn = nchars; // max possible replaces for this string
        const char* in_ptr = d_str.data(); // input pointer (i)
        char* out_ptr = nullptr;  // running output pointer (o)
        if( d_offsets )
            out_ptr = d_chars + d_offsets[idx];
        size_type lpos = 0;
        size_type begin = 0;
        size_type end = nchars;  // working vars
        // copy input to output replacing strings as we go
        while( mxn-- > 0 ) // maximum number of replaces
        {
            if( prog.find(idx,d_str,begin,end) <= 0 )
                break; // no more matches
            auto spos = d_str.byte_offset(begin); // get offset for these
            auto epos = d_str.byte_offset(end);   // character position values
            nbytes += d_repl.size_bytes() - (epos - spos); // compute new size
            if( out_ptr )                                                      // replace
            {                                                                  // i:bbbbsssseeee
                out_ptr = copy_and_increment(out_ptr,in_ptr+lpos,spos-lpos);   // o:bbbb
                out_ptr = copy_string(out_ptr, d_repl);                        // o:bbbbrrrrrr
                                                                               //  out_ptr ---^
                lpos = epos;                                                   // i:bbbbsssseeee
            }                                                                  //  in_ptr --^
            begin = end;
            end = nchars;
        }
        if( out_ptr ) // copy the remainder
            memcpy(out_ptr, in_ptr+lpos, d_str.size_bytes()-lpos);             // o:bbbbrrrrrreeee
        return nbytes;
    }
};

} // namespace

//
std::unique_ptr<column> replace_re( strings_column_view const& strings,
                                    std::string const& pattern,
                                    string_scalar const& repl = string_scalar(""),
                                    size_type maxrepl = -1,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                    cudaStream_t stream = 0)
{
    auto strings_count = strings.size();
    if( strings_count==0 )
        return make_empty_strings_column(mr,stream);

    CUDF_EXPECTS( repl.is_valid(), "Parameter repl must be valid");
    string_view d_repl( repl.data(), repl.size() );

    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column;
    // compile regex into device object
    auto prog = reprog_device::create(pattern,get_character_flags_table(),strings_count,stream);
    auto d_prog = *prog;
    auto regex_insts = d_prog.insts_counts();

    // copy null mask
    auto null_mask = copy_bitmask(strings.parent());
    auto null_count = strings.null_count();

    // create child columns
    std::pair< std::unique_ptr<column>, std::unique_ptr<column> > children(nullptr,nullptr);
    // Each invocation is predicated on the stack size which is dependent on the number of regex instructions
    if( (regex_insts > MAX_STACK_INSTS) || (regex_insts <= RX_SMALL_INSTS) )
        children = make_strings_children(replace_regex_fn<RX_STACK_SMALL>{d_strings,d_prog,d_repl,maxrepl},
                                         strings_count, null_count, mr, stream);
    else if( regex_insts <= RX_MEDIUM_INSTS )
        children = make_strings_children(replace_regex_fn<RX_STACK_MEDIUM>{d_strings,d_prog,d_repl,maxrepl},
                                         strings_count, null_count, mr, stream);
    else
        children = make_strings_children(replace_regex_fn<RX_STACK_LARGE>{d_strings,d_prog,d_repl,maxrepl},
                                         strings_count, null_count, mr, stream);
    //
    return make_strings_column(strings_count, std::move(children.first), std::move(children.second),
                               null_count, std::move(null_mask), stream, mr);
}

} // namespace detail

// external API

std::unique_ptr<column> replace_re( strings_column_view const& strings,
                                    std::string const& pattern,
                                    string_scalar const& repl,
                                    size_type maxrepl,
                                    rmm::mr::device_memory_resource* mr )
{
    return detail::replace_re(strings, pattern, repl, maxrepl, mr);
}

} // namespace strings
} // namespace cudf
