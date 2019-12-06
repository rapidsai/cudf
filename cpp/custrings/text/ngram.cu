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
    size_t* d_offsets;

    __device__ bool is_delimiter(Char ch)
    {
        if( !d_delimiter )
            return (ch <= ' '); // all ascii whitespace
        return d_delimiter->find(ch)>=0;
    }

    __device__ bool next_token( custring_view* dstr, bool& spaces, custring_view::iterator& itr, int& spos, int& epos )
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

struct ngram_token_counter_fn : base_string_tokenize
{
    custring_view_array d_strings;
    custring_view* d_separator;

    __device__ int32_t operator()(unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        if( !dstr )
            return 0;
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

struct ngram_counts_fn : base_string_tokenize
{
    int32_t ngrams; // always >=2 
    __device__ int32_t operator()(int32_t count)
    {
        return count ? count - ngrams + 1 : 0;
    }
};

struct ngram_tokens_positions_fn : base_string_tokenize
{
    custring_view_array d_strings;
    custring_view* d_separator;
    int32_t* d_token_offsets;
    thrust::pair<int32_t,int32_t>* d_token_positions;

    __device__ void operator()(unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        if( !dstr )
            return;
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

struct ngram_sizes_fn : base_string_tokenize
{
    using position_pair = thrust::pair<int32_t,int32_t>;
    custring_view_array d_strings;
    custring_view* d_separator;
    int32_t ngrams; // always >=2 
    int32_t* d_token_counts;
    int32_t* d_token_offsets;
    position_pair* d_token_positions;

    __device__ int32_t operator()(unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        if( !dstr )
            return 0;
        int token_count = d_token_counts[idx];
        int32_t bytes = 0;
        for( int token_index = 0; token_index < token_count; ++token_index )
        {
            int32_t length = 0;
            for( int n = 0; n < ngrams; ++n )
            {
                if( token_index >= n )
                {
                    position_pair item = d_token_positions[token_index-n];
                    length += item.second - item.first;
                    length += d_separator->size();
                }
            }
            if( length > 0 )
                bytes += length - d_separator->size();
        }
        return bytes;
    }
};

NVStrings* NVText::ngrams_tokenize(NVStrings const& strs, const char* delimiter, unsigned int ngrams, const char* separator )
{
    if( ngrams==0 )
        ngrams = 2;
    if( separator==nullptr )
        separator = "";
    unsigned int count = strs.size();
    if( count==0 )
        return strs.copy();
    if( ngrams==1 )
        return NVText::tokenize(strs,delimiter);

    auto execpol = rmm::exec_policy(0);
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);

    // first let's remove any nulls or empty strings
    auto end = thrust::remove_if(execpol->on(0), d_strings, d_strings + count,
        [] __device__ ( custring_view* dstr ) { return (dstr==nullptr) || dstr->empty(); } );
    count = (unsigned int)(end - d_strings); // new count

    custring_view* d_separator = custring_from_host(separator);

    // compute size of new strings

}
