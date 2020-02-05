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
#include <cudf/text/ngrams_tokenize.hpp>
#include <cudf/utilities/error.hpp>
#include <strings/utilities.hpp>
#include <strings/utilities.cuh>
#include <text/utilities/tokenize_ops.cuh>

#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace cudf
{
namespace nvtext
{
namespace detail
{
namespace
{
//
using position_pair = thrust::pair<int32_t,int32_t>;

/**
 * @brief This records the byte positions of each token within each string.
 *
 * The position values are recorded to prevent costly locating tokens in
 * a string to generate ngrams from it.
 * Most of the work is done in the base class to locate the tokens.
 * This functor records the byte positions in the d_token_positions member.
 */
struct string_tokens_positions_fn : base_tokenator
{
    column_device_view d_strings;
    int32_t const* d_token_offsets;
    position_pair* d_token_positions;

    string_tokens_positions_fn( column_device_view& d_strings,
        string_view& d_delimiter, int32_t const* d_token_offsets,
        position_pair* d_token_positions )
        : base_tokenator{d_delimiter}, d_strings(d_strings),
          d_token_offsets(d_token_offsets), d_token_positions(d_token_positions) {}

    __device__ void operator()(size_type idx)
    {
        if( d_strings.is_null(idx) )
            return;
        string_view d_str = d_strings.element<string_view>(idx);
        bool spaces = true;
        size_type spos = 0;
        size_type epos = d_str.length();
        size_type token_index = 0;
        auto token_positions = d_token_positions + d_token_offsets[idx];
        auto itr = d_str.begin();
        while( next_token(d_str,spaces,itr,spos,epos) )
        {
            token_positions[token_index++] =
                thrust::make_pair(d_str.byte_offset(spos),  // convert char pos
                                  d_str.byte_offset(epos)); // to byte offset
            spos = epos + 1;
            ++itr;
        }
    }
};

/**
 * @brief Generate the ngrams for each string.
 *
 * The ngrams for each string are placed contiguously within the section of memory
 * assigned for the string. At the same time, the size of each ngram is recorded
 * in order to build the output offsets column.
 *
 * This functor can be called to compute the size of memory needed to write out
 * each set of ngrams per string. Once the memory offsets (d_chars_offsets) are
 * set and the output memory is allocated (d_chars), the ngrams for each string
 * can be generated into the output buffer.
 */
struct ngram_builder_fn
{
    column_device_view d_strings;
    string_view d_separator;  // separator to place between 'grams
    size_type ngrams;  // number of ngrams to generate
    int32_t const* d_token_offsets;   // offsets for token position for each string
    position_pair const* d_token_positions;  // token positions for each string
    int32_t const* d_chars_offsets{}; // offsets for each string's ngrams
    char* d_chars{};  // write ngram strings to here
    int32_t const* d_ngram_offsets{}; // offsets for sizes of each string's ngrams
    int32_t* d_ngram_sizes{}; // write ngram sizes to here

    __device__ size_type operator()(size_type idx)
    {
        if( d_strings.is_null(idx) )
            return 0;
        string_view d_str = d_strings.element<string_view>(idx);
        auto token_positions = d_token_positions + d_token_offsets[idx];
        auto token_count = d_token_offsets[idx+1] - d_token_offsets[idx];
        size_type nbytes = 0;  // total number of output bytes needed for this string
        size_type ngram_index = 0;
        char* out_ptr = d_chars ? d_chars + d_chars_offsets[idx] : nullptr;
        int32_t* d_sizes = d_ngram_sizes ? d_ngram_sizes + d_ngram_offsets[idx] : nullptr;
        for( size_type token_index = (ngrams-1); token_index < token_count; ++token_index )
        {
            size_type length = 0; // calculate size of each ngram in bytes
            for( size_type n = (ngrams-1); n >= 0; --n ) // sliding window of tokens
            {
                position_pair item = token_positions[token_index-n];
                length += item.second - item.first;
                if( out_ptr )
                    out_ptr = strings::detail::copy_and_increment(out_ptr, d_str.data() + item.first,
                                                                  item.second - item.first);
                if( n > 0 )
                {   // copy separator (except for the last one)
                    if( out_ptr )
                        out_ptr = strings::detail::copy_string( out_ptr, d_separator );
                    length += d_separator.size_bytes();
                }
            }
            if( d_sizes )
                d_sizes[ngram_index++] = length;
            nbytes += length;
        }
        return nbytes;
    }
};


} // namespace

// detail APIs

//
std::unique_ptr<column> ngrams_tokenize( strings_column_view const& strings,
                                         size_type ngrams = 2,
                                         string_scalar const& delimiter = string_scalar(""),
                                         string_scalar const& separator = string_scalar{"_"},
                                         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                         cudaStream_t stream = 0 )
{
    CUDF_EXPECTS( delimiter.is_valid(), "Parameter delimiter must be valid");
    string_view d_delimiter( delimiter.data(), delimiter.size() );
    CUDF_EXPECTS( separator.is_valid(), "Parameter separator must be valid");
    string_view d_separator( separator.data(), separator.size() );

    CUDF_EXPECTS( ngrams >=1, "Parameter ngrams should be an integer value of 1 or greater");
    if( ngrams==1 ) // this is just a straight tokenize
        return tokenize(strings,delimiter,mr); // TODO: call the stream version

    auto execpol = rmm::exec_policy(stream);
    auto strings_count = strings.size();
    auto strings_column = column_device_view::create(strings.parent(),stream);
    auto d_strings = *strings_column;

    // Example for comments with ngrams=2
    // ["a bb ccc","dd e"] => ["a_bb", "bb_ccc", "dd_e"]

    // first, get the number of tokens per string to get the token-offsets
    // Ex. token-counts = [3,2]; token-offsets = [0,3,5]
    rmm::device_vector<int32_t> token_offsets(strings_count+1);
    auto d_token_offsets = token_offsets.data().get();
    thrust::transform_inclusive_scan( rmm::exec_policy(stream)->on(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count),
        d_token_offsets+1,
        tokenator_fn{d_strings,d_delimiter},
        thrust::plus<int32_t>());
    CUDA_TRY(cudaMemsetAsync( d_token_offsets, 0, sizeof(int32_t), stream ));
    auto total_tokens = token_offsets[strings_count];  // Ex. 5 tokens

    // get the token positions (in bytes) per string
    // Ex. start/end pairs: [(0,1),(2,4),(5,8), (0,2),(3,4)]
    rmm::device_vector<position_pair> token_positions(total_tokens);
    auto d_token_positions = token_positions.data().get();
    thrust::for_each_n( execpol->on(stream), thrust::make_counting_iterator<size_type>(0), strings_count,
        string_tokens_positions_fn{d_strings, d_delimiter, d_token_offsets, d_token_positions});

    // compute the number of ngrams per string to get the total number of ngrams to generate
    // Ex. ngram-counts = [2,1]; ngram-offsets = [0,2,3]; total = 3 bigrams
    rmm::device_vector<int32_t> ngram_offsets(strings_count+1);
    auto d_ngram_offsets = ngram_offsets.data().get();
    thrust::transform_inclusive_scan( execpol->on(stream), thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count), d_ngram_offsets+1,
        [d_token_offsets, ngrams] __device__ (size_type idx)
        {
            auto token_count = d_token_offsets[idx+1] - d_token_offsets[idx];
            //printf("%d:token_count=%d\n",(int)idx,(int)token_count);
            return (token_count >= ngrams) ? token_count - ngrams + 1 : 0;
        }, thrust::plus<int32_t>());
    CUDA_TRY(cudaMemsetAsync( d_ngram_offsets, 0, sizeof(int32_t), stream));
    auto total_ngrams = ngram_offsets[strings_count];

    // Compute the total size of the ngrams for each string (not for each ngram)
    // Ex. 2 bigrams in 1st string total to 10 bytes; 1 bigram in 2nd string is 4 bytes
    //     => sizes = [10,4]; offsets = [0,10,14]
    //
    // This produces a set of offsets for the output memory where we can build adjacent
    // ngrams for each string.
    // Ex. bigram for first string produces 2 bigrams ("a_bb","bb_ccc") which
    // is build in memory like this: "a_bbbb_ccc"
    rmm::device_vector<int32_t> chars_offsets(strings_count+1); // output memory offsets
    auto d_chars_offsets = chars_offsets.data().get();          // per input string
    thrust::transform_inclusive_scan( execpol->on(stream), thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count), d_chars_offsets +1,
        ngram_builder_fn{d_strings, d_separator, ngrams, d_token_offsets, d_token_positions},
        thrust::plus<int32_t>() );
    CUDA_TRY(cudaMemsetAsync( d_chars_offsets, 0, sizeof(int32_t), stream));
    auto output_chars_size = chars_offsets[strings_count];  // Ex. 14 output bytes total

    rmm::device_vector<int32_t> ngram_sizes(total_ngrams);  // size in bytes of each
    auto d_ngram_sizes = ngram_sizes.data().get();          // ngram to generate

    // build chars column
    auto chars_column = strings::detail::create_chars_child_column( strings_count, 0, output_chars_size, mr, stream );
    auto d_chars = chars_column->mutable_view().data<char>();
    // Generate the ngrams into the chars column data buffer.
    // The ngram_builder_fn functor also fills the d_ngram_sizes vector with the
    // size of each ngram. This will be used to build the official output offsets column
    // pointing to each generated ngram.
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int32_t>(0), strings_count,
        ngram_builder_fn{d_strings, d_separator, ngrams, d_token_offsets, d_token_positions,
                         d_chars_offsets, d_chars, d_ngram_offsets, d_ngram_sizes});
    // build the offsets column -- convert the ngram sizes into offsets
    auto offsets_column = make_numeric_column(
          data_type{INT32}, total_ngrams + 1, mask_state::UNALLOCATED, stream, mr);
    auto d_offsets = offsets_column->mutable_view().data<int32_t>();
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           d_ngram_sizes, d_ngram_sizes + total_tokens,
                           d_offsets+1);
    CUDA_TRY(cudaMemsetAsync(d_offsets, 0, sizeof(int32_t), stream));
    chars_column->set_null_count(0);
    offsets_column->set_null_count(0);
    // create the output strings column
    return make_strings_column(total_ngrams, std::move(offsets_column), std::move(chars_column),
                               0, rmm::device_buffer{}, stream, mr);
}

} // namespace detail

// external APIs

std::unique_ptr<column> ngrams_tokenize( strings_column_view const& strings,
                                         size_type ngrams,
                                         string_scalar const& delimiter,
                                         string_scalar const& separator,
                                         rmm::mr::device_memory_resource* mr )
{
    return detail::ngrams_tokenize( strings, ngrams, delimiter, separator, mr );
}

} // namespace nvtext
} // namespace cudf
