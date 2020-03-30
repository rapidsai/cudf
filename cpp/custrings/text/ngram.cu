/*
* Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include "nvstrings/NVStrings.h"
#include "nvstrings/NVText.h"

#include "../custring_view.cuh"
#include "../util.h"

//
NVStrings* NVText::create_ngrams(NVStrings& strs, unsigned int ngrams, const char* separator )
{
    if( ngrams==0 )
        ngrams = 2;
    if( separator==nullptr )
        separator = "";
    unsigned int count = strs.size();
    if( count==0 )
        return strs.copy();

    auto execpol = rmm::exec_policy(0);
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);

    // first let's remove any nulls or empty strings
    auto end = thrust::remove_if(execpol->on(0), d_strings, d_strings + count,
        [] __device__ ( custring_view* ds ) { return (ds==nullptr) || ds->empty(); } );
    count = (unsigned int)(end - d_strings); // new count
    if( count <= ngrams )
        return strs.join(separator,""); // this not quite right if there are nulls we removed
    if( ngrams==1 )
        return strs.copy(); // same with this one; need method to create NVStrings from custring_views

    custring_view* d_separator = custring_from_host(separator);

    // compute size of new strings
    unsigned int ngrams_count = count - ngrams +1;
    rmm::device_vector<size_t> sizes(ngrams_count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), ngrams_count,
        [d_strings, ngrams, d_separator, d_sizes] __device__(unsigned int idx) {
            size_t size = 0;
            for( unsigned int n=0; n < ngrams; ++n )
            {
                custring_view* dstr = d_strings[n+idx];
                size += dstr->size();
                if( (n+1) < ngrams )
                    size += d_separator->size();
            }
            d_sizes[idx] = size;
        });

    size_t bufsize = thrust::reduce(execpol->on(0), d_sizes, d_sizes+ngrams_count );
    rmm::device_vector<char> buffer(bufsize);
    char* d_buffer = buffer.data().get();
    rmm::device_vector<size_t> offsets(ngrams_count,0);
    thrust::exclusive_scan( execpol->on(0), sizes.begin(), sizes.end(), offsets.begin() );
    size_t* d_offsets = offsets.data().get();
    // build the memory and a list of pointers
    rmm::device_vector< thrust::pair<const char*,size_t> > results(ngrams_count);
    thrust::pair<const char*,size_t>* d_results = results.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), ngrams_count,
        [d_strings, d_separator, ngrams, d_offsets, d_buffer, d_results] __device__(unsigned int idx) {
            char* buffer = d_buffer + d_offsets[idx];
            char* sptr = buffer;
            size_t length = 0;
            for( unsigned int n=0; n < ngrams; ++n )
            {
                custring_view* dstr = d_strings[n+idx];
                unsigned int bytes = dstr->size();
                length += bytes;
                memcpy( sptr, dstr->data(), bytes );
                sptr += bytes;
                if( (n+1) >= ngrams )
                    continue;
                bytes = d_separator->size();
                length += bytes;
                memcpy( sptr, d_separator->data(), bytes );
                sptr += bytes;
            }
            d_results[idx].first = buffer;
            d_results[idx].second = length;
        });
    //
    RMM_FREE(d_separator,0);
    // build strings object from results elements
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_results,ngrams_count);
}

//
using position_pair = thrust::pair<int32_t,int32_t>;
using index_pair = thrust::pair<const char*,size_t>;

// This class walks a string looking for specified delimiter character(s).
// It will automatically ignore adjacent delimiters (i.e. different than split).
// The next_token method returns character start position (spos) and end
// position (epos) between delimiter runs identifying each token.
// An iterator is used to retrieve each utf8 character to be checked.
// The spaces parameter identifies a run of delimiters (or not delimiters).
struct base_string_tokenize
{
    custring_view_array d_strings;
    custring_view* d_delimiter;
    custring_view* d_separator;

    __device__ bool is_delimiter(Char ch)
    {
        if( !d_delimiter )
            return (ch <= ' '); // all ascii whitespace
        return d_delimiter->find(ch)>=0;
    }

    __device__ bool next_token( custring_view* dstr, bool& spaces,
                                custring_view::iterator& itr, int& spos, int& epos )
    {
        if( spos >= dstr->chars_count() )
            return false;
        for( ; itr != dstr->end(); ++itr )
        {
            Char ch = *itr;
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

// Count the number of tokens within a string.
struct string_token_counter_fn : base_string_tokenize
{
    string_token_counter_fn( custring_view_array d_strings,
        custring_view* d_delimiter, custring_view* d_separator)
        : base_string_tokenize{d_strings, d_delimiter, d_separator} {}

    __device__ int32_t operator()(custring_view* dstr)
    {
        bool spaces = true;
        int nchars = dstr->chars_count();
        int spos = 0, epos = nchars;
        int32_t token_count = 0;
        auto itr = dstr->begin();
        while( next_token(dstr,spaces,itr,spos,epos) )
        {
            ++token_count;
            spos = epos + 1;
            epos = nchars;
            ++itr;
        }
        return token_count;
    }
};

// Record the byte positions of each token within each string.
struct string_tokens_positions_fn : base_string_tokenize
{
    const int32_t* d_token_offsets;
    position_pair* d_token_positions;

    string_tokens_positions_fn( custring_view_array d_strings,
        custring_view* d_delimiter, custring_view* d_separator,
        const int32_t* d_token_offsets, position_pair* d_token_positions )
        : base_string_tokenize{d_strings, d_delimiter, d_separator},
          d_token_offsets(d_token_offsets), d_token_positions(d_token_positions) {}

    __device__ void operator()(unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        bool spaces = true;
        int nchars = dstr->chars_count();
        int spos = 0, epos = nchars, token_index = 0;
        auto token_positions = d_token_positions + d_token_offsets[idx];
        auto itr = dstr->begin();
        while( next_token(dstr,spaces,itr,spos,epos) )
        {
            token_positions[token_index++] =
                thrust::make_pair(dstr->byte_offset_for(spos),  // convert char pos
                                  dstr->byte_offset_for(epos)); // to byte offset
            spos = epos + 1;
            epos = nchars;
            ++itr;
        }
    }
};

// Compute the size of each ngram that will be created for each string.
// Adjacent token position-pairs are used to calculate the total ngram sizes.
struct ngram_sizes_fn
{
    custring_view* d_separator;
    int32_t ngrams; // always >=2
    const int32_t* d_token_offsets;
    const position_pair* d_token_positions;
    const int32_t* d_ngram_offsets;

    __device__ int32_t operator()(unsigned int idx)
    {
        auto token_positions = d_token_positions + d_token_offsets[idx];
        auto token_count = d_token_offsets[idx+1] - d_token_offsets[idx];
        int32_t bytes = 0;
        for( int token_index = (ngrams-1); token_index < token_count; ++token_index )
        {
            int32_t length = 0;
            for( int n = (ngrams-1); n >= 0; --n ) // sliding window of tokens
            {
                position_pair item = token_positions[token_index-n];
                length += item.second - item.first; // size of this token in bytes
                length += d_separator->size(); // add size of the separator
            }
            length -= d_separator->size(); // remove trailing separator
            bytes += length;
        }
        return bytes;
    }
};

// Build the ngrams for each string.
// The ngrams for each string are placed contiguously within the section of memory
// assigned for the string. And an index_pair is recorded for each ngram.
struct ngram_builder_fn
{
    custring_view_array d_strings;
    custring_view* d_separator;
    int32_t ngrams;
    const int32_t* d_token_offsets;
    const position_pair* d_token_positions;
    const int32_t* d_ngram_offsets;
    const int32_t* d_chars_offsets;
    char* d_chars;  // write ngram strings to here
    index_pair* d_indices; // output ngram index-pairs here

    __device__ void operator()(int32_t idx)
    {
        custring_view* dstr = d_strings[idx];
        auto token_positions = d_token_positions + d_token_offsets[idx];
        auto token_count = d_token_offsets[idx+1] - d_token_offsets[idx];
        int ngram_index = 0;
        char* out_ptr = d_chars + d_chars_offsets[idx];
        auto indices = d_indices + d_ngram_offsets[idx];
        for( int token_index = (ngrams-1); token_index < token_count; ++token_index )
        {
            int32_t length = 0;
            auto ngram_ptr = out_ptr;
            for( int n = (ngrams-1); n >= 0; --n )
            {
                position_pair item = token_positions[token_index-n];
                out_ptr = copy_and_incr(out_ptr, dstr->data() + item.first, item.second - item.first);
                length += item.second - item.first;
                if( n > 0 )
                {   // copy separator (except for the last one)
                    out_ptr = copy_and_incr( out_ptr, d_separator->data(), d_separator->size() );
                    length += d_separator->size();
                }
            }
            indices[ngram_index++] = index_pair{ngram_ptr,length};
        }
    }
};

// This will create ngrams for each string and not across strings.
NVStrings* NVText::ngrams_tokenize(NVStrings const& strs, const char* delimiter, int32_t ngrams, const char* separator )
{
    auto count = strs.size();
    if( count==0 )
        return strs.copy();
    if( ngrams==1 )
        return NVText::tokenize(strs,delimiter);
    if( ngrams==0 ) // default is 2
        ngrams = 2;
    if( !separator ) // no separator specified
        separator = "";

    auto execpol = rmm::exec_policy(0);
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);

    // first remove any nulls or empty strings
    auto end = thrust::remove_if(execpol->on(0), d_strings, d_strings + count,
        [] __device__ ( custring_view* dstr ) { return (dstr==nullptr) || dstr->empty(); } );
    count = (unsigned int)(end - d_strings); // new count

    custring_view* d_separator = custring_from_host(separator);
    custring_view* d_delimiter = custring_from_host(delimiter);

    // Example for comments with ngrams=2
    // ["a bb ccc","dd e"] => ["a_bb", "bb_ccc", "dd_e"]

    // get the number of tokens per string
    // token-counts = [3,2]; token-offsets = [0,3,5]
    rmm::device_vector<int32_t> token_offsets(count+1);
    auto d_token_offsets = token_offsets.data().get();
    thrust::transform_inclusive_scan( execpol->on(0), strings.begin(), strings.end(),
        d_token_offsets+1, string_token_counter_fn{d_strings,d_delimiter,d_separator},
        thrust::plus<int32_t>());
    cudaMemset( d_token_offsets, 0, sizeof(int32_t) );
    int32_t total_tokens = token_offsets[count];  // 5

    // get the token positions per string
    // => [(0,1),(2,4),(5,8), (0,2),(3,4)]
    rmm::device_vector<position_pair> token_positions(total_tokens);
    auto d_token_positions = token_positions.data().get();
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int32_t>(0), count,
        string_tokens_positions_fn{d_strings, d_delimiter, d_separator, d_token_offsets, d_token_positions});

    // compute the number of ngrams per string
    // ngram-counts = [2,1]; ngram-offsets = [0,2,3]
    rmm::device_vector<int32_t> ngram_offsets(count+1);
    auto d_ngram_offsets = ngram_offsets.data().get();
    thrust::transform_inclusive_scan( execpol->on(0), thrust::make_counting_iterator<int32_t>(0),
        thrust::make_counting_iterator<int32_t>(count), d_ngram_offsets+1,
        [d_token_offsets, ngrams] __device__ (int32_t idx)
        {
            auto token_count = d_token_offsets[idx+1] - d_token_offsets[idx];
            return (token_count >= ngrams) ? token_count - ngrams + 1 : 0;
        }, thrust::plus<int32_t>());
    cudaMemset( d_ngram_offsets, 0, sizeof(int32_t) );
    int32_t total_ngrams = ngram_offsets[count]; // 3

    // compute the size of the ngrams for each string
    // sizes = [10,4]; offsets = [0,10,14]
    // 2 bigrams in 1st string total to 10 bytes; 1 bigram in 2nd string is 4 bytes
    rmm::device_vector<int32_t> chars_offsets(count+1);
    auto d_chars_offsets = chars_offsets.data().get();
    thrust::transform_inclusive_scan( execpol->on(0), thrust::make_counting_iterator<int32_t>(0),
        thrust::make_counting_iterator<int32_t>(count), d_chars_offsets +1,
        ngram_sizes_fn{d_separator, ngrams, d_token_offsets, d_token_positions,
                       d_ngram_offsets},//, d_ngram_sizes},
        thrust::plus<int32_t>() );
    cudaMemset( d_chars_offsets, 0, sizeof(int32_t));

    // create output buffer size
    auto output_chars_size = chars_offsets[count];  // 14
    rmm::device_vector<char> output_buffer(output_chars_size);
    auto d_output_buffer = output_buffer.data().get();
    // build the ngrams in the output buffer
    rmm::device_vector<index_pair> results(total_ngrams);
    auto d_results = results.data().get();
    thrust::for_each_n( execpol->on(0), thrust::make_counting_iterator<int32_t>(0), count,
        ngram_builder_fn{d_strings, d_separator, ngrams, d_token_offsets, d_token_positions,
                         d_ngram_offsets, d_chars_offsets, d_output_buffer, d_results});

    RMM_FREE(d_separator,0);
    RMM_FREE(d_delimiter,0);
    // build strings object from results indices
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_results,total_ngrams);
}
