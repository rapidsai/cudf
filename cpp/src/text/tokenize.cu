/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/text/tokenize.hpp>
#include <cudf/utilities/error.hpp>

#include <vector>
#include <thrust/transform.h>
#include <thrust/logical.h>

namespace cudf
{
namespace nvtext
{
namespace detail
{

using string_index_pair = thrust::pair<const char*,size_type>;

namespace
{

// common code for tokenize, token-counters, normalize functions
// with characters or whitespace delimiter
struct base_tokenator
{
    string_view d_delimiter;

    __device__ bool is_delimiter(char_utf8 chr)
    {
        return d_delimiter.empty() ? (chr <= ' ') : // whitespace check
               thrust::any_of( thrust::seq, d_delimiter.begin(), d_delimiter.end(),
                               [chr] __device__ (char_utf8 c) {return c==chr;});
    }

    __device__ bool next_token( string_view const& d_str, bool& spaces,
                                string_view::const_iterator& itr,
                                size_type& spos, size_type& epos )
    {
        if( spos >= d_str.length() )
            return false;
        for( ; itr != d_str.end(); ++itr )
        {
            char_utf8 ch = *itr;
            if( spaces == is_delimiter(ch) )
            {
                if( spaces )
                    spos = itr.position()+1;
                else
                    epos = itr.position()+1;
                continue;
            }
            spaces = !spaces;
            if( spaces )
            {
                epos = itr.position();
                break;
            }
        }
        return spos < epos;
    }
};

// used for counting tokens and tokenizing characters or whitespace
struct tokenator_fn : base_tokenator
{
    column_device_view d_strings;
    size_type* d_offsets{};
    string_index_pair* d_tokens{};

    tokenator_fn( column_device_view& d_strings, string_view& d_delimiter,
                  size_type* d_offsets=nullptr,
                  string_index_pair* d_tokens=nullptr )
    : base_tokenator{d_delimiter},
      d_strings(d_strings), d_offsets(d_offsets), d_tokens(d_tokens) {}

    __device__ size_type operator()(size_type idx)
    {
        if( d_strings.is_null(idx) )
            return 0;
        string_view d_str = d_strings.element<string_view>(idx);
        string_index_pair* d_str_tokens = d_tokens ? d_tokens + d_offsets[idx] : nullptr;
        bool spaces = true;
        size_type spos = 0;
        size_type epos = d_str.length();
        size_type token_idx = 0;
        auto itr = d_str.begin();
        while( next_token(d_str,spaces,itr,spos,epos) )
        {
            if( d_str_tokens )
            {
                int spos_bo = d_str.byte_offset(spos); // convert char pos
                int epos_bo = d_str.byte_offset(epos); // to byte offset
                d_str_tokens[token_idx] = string_index_pair{ d_str.data() + spos_bo,
                                                            (epos_bo-spos_bo) };
            }
            spos = epos + 1;
            epos = d_str.length();
            ++itr;
            ++token_idx;
        }
        return token_idx;
    }
};

template<typename TokenCounter>
std::unique_ptr<column> token_count_fn( size_type strings_count, TokenCounter tcfn,
                                        rmm::mr::device_memory_resource* mr,
                                        cudaStream_t stream )
{
    // create output column
    auto token_counts = make_numeric_column( data_type{INT32}, strings_count, UNALLOCATED, stream, mr);
    auto d_token_counts = token_counts->mutable_view().data<int32_t>();
    // add the counts to the column
    thrust::transform( rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count),
        d_token_counts, tcfn );
    return token_counts;
}

// common pattern for tokenize functions
template<typename Tokenizer>
std::unique_ptr<column> tokenize_fn( size_type strings_count, Tokenizer ttfn,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream )
{
    auto execpol = rmm::exec_policy(stream);
    // get the number of tokens in each string
    auto token_counts = token_count_fn( strings_count, ttfn, mr, stream );
    auto d_token_counts = token_counts->view();
    // create token-index offsets from the counts
    rmm::device_vector<int32_t> token_offsets(strings_count+1);
    thrust::inclusive_scan( execpol->on(stream),
                            d_token_counts.template begin<int32_t>(),
                            d_token_counts.template end<int32_t>(),
                            token_offsets.begin()+1 );
    cudaMemsetAsync( token_offsets.data().get(), 0, sizeof(int32_t), stream );
    auto total_tokens = token_offsets.back();
    // build a list of pointers to each token
    rmm::device_vector<string_index_pair> tokens(total_tokens);
    // now go get the tokens
    ttfn.d_offsets = token_offsets.data().get();
    ttfn.d_tokens = tokens.data().get();
    thrust::for_each_n(execpol->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count, ttfn );
    // create the strings column using the tokens pointers
    return make_strings_column(tokens,stream,mr);
}

} // namespace

std::unique_ptr<column> tokenize( strings_column_view const& strings,
                                  string_scalar const& delimiter = string_scalar(""),
                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                  cudaStream_t stream = 0 )
{
    CUDF_EXPECTS( delimiter.is_valid(), "Parameter delimiter must be valid");
    string_view d_delimiter( delimiter.data(), delimiter.size() );
    auto strings_column = column_device_view::create(strings.parent(),stream);
    return tokenize_fn( strings.size(), tokenator_fn{*strings_column,d_delimiter}, mr, stream );
}

std::unique_ptr<column> token_count( strings_column_view const& strings,
                                     string_scalar const& delimiter = string_scalar(""),
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                     cudaStream_t stream = 0 )
{
    CUDF_EXPECTS( delimiter.is_valid(), "Parameter delimiter must be valid");
    string_view d_delimiter( delimiter.data(), delimiter.size() );
    auto strings_column = column_device_view::create(strings.parent(),stream);
    return token_count_fn( strings.size(), tokenator_fn{*strings_column,d_delimiter}, mr, stream );
}

namespace
{

// delimiters' iterator = delimiterator
using delimiterator = cudf::column_device_view::const_iterator<string_view>;

// handles token counting and tokenizing
struct multi_delimiter_tokenizer_fn
{
    column_device_view d_strings;
    delimiterator delimiters_begin;
    delimiterator delimiters_end;
    size_type* d_offsets{};
    string_index_pair* d_tokens{};

    multi_delimiter_tokenizer_fn( column_device_view& d_strings,
                                  delimiterator delimiters_begin,
                                  delimiterator delimiters_end,
                                  size_type* d_offsets=nullptr,
                                  string_index_pair* d_tokens=nullptr )
    : d_strings(d_strings), delimiters_begin(delimiters_begin), delimiters_end(delimiters_end),
      d_offsets(d_offsets), d_tokens(d_tokens) {}

    __device__ size_type operator()(size_type idx)
    {
        if( d_strings.is_null(idx) )
            return 0;
        string_view d_str = d_strings.element<string_view>(idx);
        auto d_str_tokens = d_tokens ? d_tokens + d_offsets[idx] : nullptr;
        auto data_ptr = d_str.data();
        auto curr_ptr = data_ptr;
        size_type last_pos = 0, token_idx = 0;
        while( curr_ptr < data_ptr + d_str.size_bytes() )
        {
            string_view sub_str(curr_ptr,static_cast<size_type>(data_ptr + d_str.size_bytes() - curr_ptr));
            size_type increment_bytes = 1;
            // look for delimiter at current position
            auto itr_find = thrust::find_if( thrust::seq, delimiters_begin, delimiters_end,
                [sub_str]__device__(string_view const& d_delim) {
                    return !d_delim.empty() && (d_delim.size_bytes() <= sub_str.size_bytes()) &&
                           d_delim.compare(sub_str.data(),d_delim.size_bytes())==0;
                });
            if( itr_find != delimiters_end )
            {   // found delimiter
                auto token_size = static_cast<size_type>((curr_ptr - data_ptr) - last_pos);
                if( token_size > 0 ) // we only care about non-zero sized tokens
                {
                    if( d_str_tokens )
                        d_str_tokens[token_idx] = string_index_pair{ data_ptr + last_pos, token_size };
                    ++token_idx;
                }
                increment_bytes = (*itr_find).size_bytes();
                last_pos = (curr_ptr - data_ptr) + increment_bytes; // point past delimiter
            }
            curr_ptr += increment_bytes; // move on to the next byte
        }
        if( last_pos < d_str.size_bytes() ) // left-over tokens
        {
            if( d_str_tokens )
                d_str_tokens[token_idx] = string_index_pair{ data_ptr + last_pos, d_str.size_bytes() - last_pos };
            ++token_idx;
        }
        return token_idx; // this is the number of tokens found for this string
    }
};

}

std::unique_ptr<column> tokenize( strings_column_view const& strings,
                                  strings_column_view const& delimiters,
                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                  cudaStream_t stream = 0 )
{
    CUDF_EXPECTS( delimiters.size()>0, "Parameter delimiters must not be empty");
    CUDF_EXPECTS( !delimiters.has_nulls(), "Parameter delimiters must not have nulls");
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto delimiters_column = column_device_view::create(delimiters.parent(),stream);
    return tokenize_fn( strings.size(),
                        multi_delimiter_tokenizer_fn{*strings_column,
                                                     delimiters_column->begin<string_view>(),
                                                     delimiters_column->end<string_view>()},
                        mr, stream );
}

std::unique_ptr<column> token_count( strings_column_view const& strings,
                                     strings_column_view const& delimiters,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                     cudaStream_t stream = 0 )
{
    CUDF_EXPECTS( delimiters.size()>0, "Parameter delimiters must not be empty");
    CUDF_EXPECTS( !delimiters.has_nulls(), "Parameter delimiters must not have nulls");
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto delimiters_column = column_device_view::create(delimiters.parent(),stream);
    return token_count_fn( strings.size(),
                           multi_delimiter_tokenizer_fn{*strings_column,
                                                        delimiters_column->begin<string_view>(),
                                                        delimiters_column->end<string_view>()},
                           mr, stream );
}

} // namespace detail

// external APIs

std::unique_ptr<column> tokenize( strings_column_view const& strings,
                                  string_scalar const& delimiter,
                                  rmm::mr::device_memory_resource* mr )
{
    return detail::tokenize( strings, delimiter, mr );
}

std::unique_ptr<column> tokenize( strings_column_view const& strings,
                                  strings_column_view const& delimiters,
                                  rmm::mr::device_memory_resource* mr )
{
    return detail::tokenize( strings, delimiters, mr );
}

std::unique_ptr<column> token_count( strings_column_view const& strings,
                                     string_scalar const& delimiter,
                                     rmm::mr::device_memory_resource* mr)
{
    return detail::token_count( strings, delimiter, mr );
}

std::unique_ptr<column> token_count( strings_column_view const& strings,
                                     strings_column_view const& delimiters,
                                     rmm::mr::device_memory_resource* mr)
{
    return detail::token_count( strings, delimiters, mr );
}

} // namespace strings
} // namespace cudf
