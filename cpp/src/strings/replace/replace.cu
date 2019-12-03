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
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include "../utilities.hpp"
#include "../utilities.cuh"

namespace cudf
{
namespace strings
{
namespace detail
{
namespace
{

/**
 * @brief Used as template parameter to divide size calculation from
 * the actual string operation within a function.
 *
 * Useful when most of the logic is identical for both passes.
 */
enum class two_pass
{
    SIZE_ONLY = 0, ///< calculate the size only
    EXECUTE_OP     ///< run the string operation
};

/**
 * @brief Function logic for the replace API.
 *
 * This will perform a replace operation on each string.
 */
template <two_pass Pass=two_pass::SIZE_ONLY>
struct replace_fn
{
    column_device_view const d_strings;
    string_view const d_target;
    string_view const d_repl;
    int32_t max_repl;
    const int32_t* d_offsets{};
    char* d_chars{};

    __device__ size_type operator()(size_type idx)
    {
        if( d_strings.is_null(idx) )
            return 0; // null string
        string_view d_str = d_strings.element<string_view>(idx);
        auto max_n = max_repl;
        if( max_n < 0 )
            max_n = d_str.length(); // max possible replacements
        char* out_ptr = nullptr;
        if( Pass==two_pass::EXECUTE_OP )
            out_ptr = d_chars + d_offsets[idx];
        const char* in_ptr = d_str.data();
        size_type bytes = d_str.size_bytes();
        auto position = d_str.find(d_target);
        size_type last_pos = 0;
        while( (position >= 0) && (max_n > 0) )
        {
            if( Pass==two_pass::SIZE_ONLY )
                bytes += d_repl.size_bytes() - d_target.size_bytes();
            else // EXECUTE_OP
            {
                size_type curr_pos = d_str.byte_offset(position);
                out_ptr = copy_and_increment(out_ptr, in_ptr + last_pos, curr_pos - last_pos); // copy left
                out_ptr = copy_string(out_ptr, d_repl); // copy repl
                last_pos = curr_pos + d_target.size_bytes();
            }
            position = d_str.find(d_target, position + d_target.size_bytes() );
            --max_n;
        }
        if( Pass==two_pass::EXECUTE_OP ) // copy whats left (or right depending on your point of view)
            memcpy(out_ptr, in_ptr + last_pos, d_str.size_bytes() - last_pos);
        return bytes;
    }
};

} // namespace


//
std::unique_ptr<column> replace( strings_column_view const& strings,
                                 string_scalar const& target,
                                 string_scalar const& repl,
                                 int32_t maxrepl = -1,
                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                 cudaStream_t stream = 0 )
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);
    CUDF_EXPECTS( repl.is_valid(), "Parameter repl must be valid.");
    CUDF_EXPECTS( target.is_valid(), "Parameter target must be valid.");
    CUDF_EXPECTS( target.size()>0, "Parameter target must not be empty string.");

    string_view d_target(target.data(),target.size());
    string_view d_repl(repl.data(),repl.size());

    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column;

    // copy the null mask
    rmm::device_buffer null_mask = copy_bitmask( strings.parent(), stream, mr);
    // build offsets column
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<int32_t>(0),
        replace_fn<two_pass::SIZE_ONLY>{d_strings, d_target, d_repl, maxrepl} );
    auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                       offsets_transformer_itr+strings_count,
                                       mr, stream);
    auto d_offsets = offsets_column->view().data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
    auto chars_column = create_chars_child_column( strings_count, strings.null_count(), bytes, mr, stream );
    auto d_chars = chars_column->mutable_view().data<char>();
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count,
        replace_fn<two_pass::EXECUTE_OP>{d_strings, d_target, d_repl, maxrepl, d_offsets, d_chars} );
    //
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               strings.null_count(), std::move(null_mask), stream, mr);
}

namespace
{
/**
 * @brief Function logic for the replace_slice API.
 *
 * This will perform a replace_slice operation on each string.
 */
template <two_pass Pass=two_pass::SIZE_ONLY>
struct replace_slice_fn
{
    column_device_view const d_strings;
    string_view const d_repl;
    size_type start, stop;
    const int32_t* d_offsets{};
    char* d_chars{};

    __device__ size_type operator()(size_type idx)
    {
        if( d_strings.is_null(idx) )
            return 0; // null string
        string_view d_str = d_strings.element<string_view>(idx);
        auto length = d_str.length();
        char* out_ptr = nullptr;
        if( Pass==two_pass::EXECUTE_OP )
            out_ptr = d_chars + d_offsets[idx];
        const char* in_ptr = d_str.data();
        size_type bytes = d_str.size_bytes();
        size_type begin = ((start < 0) || (start > length) ? length : start);
        size_type end = ((stop < 0) || (stop > length) ? length : stop);
        begin = d_str.byte_offset(begin);
        end = d_str.byte_offset(end);
        bytes += d_repl.size_bytes() - (end - begin);
        if( Pass==two_pass::EXECUTE_OP )
        {
            out_ptr = copy_and_increment( out_ptr, in_ptr, begin );
            out_ptr = copy_string( out_ptr, d_repl );
            out_ptr = copy_and_increment( out_ptr, in_ptr + end, d_str.size_bytes() - end );
        }
        return bytes;
    }
};

}

std::unique_ptr<column> replace_slice( strings_column_view const& strings,
                                       string_scalar const& repl,
                                       size_type start, size_type stop = -1,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                       cudaStream_t stream = 0)
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);
    CUDF_EXPECTS( repl.is_valid(), "Parameter repl must be valid.");
    if( stop > 0 )
        CUDF_EXPECTS( start <= stop, "Parameter start must be less than or equal to stop.");

    string_view d_repl(repl.data(),repl.size());

    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column;

    // copy the null mask
    rmm::device_buffer null_mask = copy_bitmask( strings.parent(), stream, mr);
    // build offsets column
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<int32_t>(0),
        replace_slice_fn<two_pass::SIZE_ONLY>{d_strings, d_repl, start, stop} );
    auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                       offsets_transformer_itr+strings_count,
                                       mr, stream);
    auto offsets_view = offsets_column->view();
    auto d_offsets = offsets_view.data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
    auto chars_column = create_chars_child_column( strings_count, strings.null_count(), bytes, mr, stream );
    auto chars_view = chars_column->mutable_view();
    auto d_chars = chars_view.data<char>();
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count,
        replace_slice_fn<two_pass::EXECUTE_OP>{d_strings, d_repl, start, stop, d_offsets, d_chars} );
    //
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               strings.null_count(), std::move(null_mask), stream, mr);
}

namespace
{

/**
 * @brief Function logic for the replace_multi API.
 *
 * This will perform the multi-replace operation on each string.
 */
template <two_pass Pass=two_pass::SIZE_ONLY>
struct replace_multi_fn
{
    column_device_view const d_strings;
    column_device_view const d_targets;
    column_device_view const d_repls;
    const int32_t* d_offsets{};
    char* d_chars{};

    __device__ size_type operator()(size_type idx)
    {
        if( d_strings.is_null(idx) )
            return 0;
        string_view d_str = d_strings.element<string_view>(idx);
        char* out_ptr = nullptr;
        if( Pass==two_pass::EXECUTE_OP )
            out_ptr = d_chars + d_offsets[idx];
        const char* in_ptr = d_str.data();
        size_type size = d_str.size_bytes();
        size_type bytes = size, spos = 0, lpos = 0;
        while( spos < size )
        {   // check each character against each target
            for( int tgt_idx=0; tgt_idx < d_targets.size(); ++tgt_idx )
            {
                string_view d_tgt = d_targets.element<string_view>(tgt_idx);
                if( (d_tgt.size_bytes() <= (size-spos)) && // check fit
                    (d_tgt.compare(in_ptr+spos, d_tgt.size_bytes())==0) ) // does it match
                {   // found one
                    string_view d_repl;
                    if( d_repls.size()==1 )
                        d_repl = d_repls.element<string_view>(0);
                    else
                        d_repl = d_repls.element<string_view>(tgt_idx);
                    if( Pass==two_pass::SIZE_ONLY )
                        bytes += d_repl.size_bytes() - d_tgt.size_bytes();
                    else
                    {
                        out_ptr = copy_and_increment(out_ptr,in_ptr+lpos,spos-lpos);
                        out_ptr = copy_string(out_ptr,d_repl);
                        lpos = spos + d_tgt.size_bytes();
                    }
                    spos += d_tgt.size_bytes()-1;
                    break;
                }
            }
            ++spos;
        }
        if( Pass==two_pass::EXECUTE_OP ) // copy remainder
            memcpy(out_ptr,in_ptr+lpos,size-lpos);
        return bytes;
    }
};

}

std::unique_ptr<column> replace( strings_column_view const& strings,
                                 strings_column_view const& targets,
                                 strings_column_view const& repls,
                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                 cudaStream_t stream = 0)
{
    auto strings_count = strings.size();
    if( strings_count==0 )
        return make_empty_strings_column(mr,stream);
    CUDF_EXPECTS( ((targets.size() > 0) && (targets.null_count()==0)), "Parameters targets must not be empty and must not have nulls");
    CUDF_EXPECTS( ((repls.size() > 0) && (repls.null_count()==0)), "Parameters repls must not be empty and must not have nulls");
    if( repls.size() > 1 )
        CUDF_EXPECTS( repls.size()==targets.size(), "Sizes for targets and repls must match");

    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column;
    auto targets_column = column_device_view::create(targets.parent(),stream);
    auto d_targets = *targets_column;
    auto repls_column = column_device_view::create(repls.parent(),stream);
    auto d_repls = *repls_column;

    // copy the null mask
    rmm::device_buffer null_mask = copy_bitmask( strings.parent(), stream, mr);
    // build offsets column
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<int32_t>(0),
        replace_multi_fn<two_pass::SIZE_ONLY>{d_strings, d_targets, d_repls} );
    auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                       offsets_transformer_itr+strings_count,
                                       mr, stream);
    auto d_offsets = offsets_column->view().data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
    auto chars_column = create_chars_child_column( strings_count, strings.null_count(), bytes, mr, stream );
    auto d_chars = chars_column->mutable_view().data<char>();
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count,
        replace_multi_fn<two_pass::EXECUTE_OP>{d_strings, d_targets, d_repls, d_offsets, d_chars} );
    //
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               strings.null_count(), std::move(null_mask), stream, mr);
}

std::unique_ptr<column> replace_nulls( strings_column_view const& strings,
                                       string_scalar const& repl = string_scalar(""),
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                       cudaStream_t stream = 0 )
{
    size_type strings_count = strings.size();
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);
    CUDF_EXPECTS( repl.is_valid(), "Parameter repl must be valid.");

    string_view d_repl(repl.data(),repl.size());

    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column;

    // build offsets column
    auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<int32_t>(0),
        [d_strings, d_repl] __device__ (size_type idx) {
            return d_strings.is_null(idx) ? d_repl.size_bytes() : d_strings.element<string_view>(idx).size_bytes();
        } );
    auto offsets_column = make_offsets_child_column(offsets_transformer_itr,
                                       offsets_transformer_itr+strings_count,
                                       mr, stream);
    auto d_offsets = offsets_column->view().data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
    auto chars_column = strings::detail::create_chars_child_column( strings_count, strings.null_count(), bytes, mr, stream );
    auto d_chars = chars_column->mutable_view().data<char>();
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count,
        [d_strings, d_repl, d_offsets, d_chars] __device__ (size_type idx) {
            string_view d_str = d_repl;
            if( !d_strings.is_null(idx) )
                d_str = d_strings.element<string_view>(idx);
            memcpy( d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes() );
        });
    //
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               0, rmm::device_buffer{}, stream, mr);
}


} // namespace detail

// external API

std::unique_ptr<column> replace( strings_column_view const& strings,
                                 string_scalar const& target,
                                 string_scalar const& repl,
                                 int32_t maxrepl,
                                 rmm::mr::device_memory_resource* mr)
{
    return detail::replace(strings, target, repl, maxrepl, mr );
}

std::unique_ptr<column> replace_slice( strings_column_view const& strings,
                                       string_scalar const& repl,
                                       size_type start, size_type stop,
                                       rmm::mr::device_memory_resource* mr)
{
    return detail::replace_slice(strings, repl, start, stop, mr);
}

std::unique_ptr<column> replace( strings_column_view const& strings,
                                 strings_column_view const& targets,
                                 strings_column_view const& repls,
                                 rmm::mr::device_memory_resource* mr)
{
    return detail::replace(strings, targets, repls, mr);
}

std::unique_ptr<column> replace_nulls( strings_column_view const& strings,
                                       string_scalar const& repl,
                                       rmm::mr::device_memory_resource* mr)
{
    return detail::replace_nulls(strings, repl, mr);
}

} // namespace strings
} // namespace cudf
