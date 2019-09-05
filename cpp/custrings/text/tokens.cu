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
#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include "NVStrings.h"
#include "NVText.h"

#include "../custring_view.cuh"
#include "../custring.cuh"
#include "../util.h"

// This base class walks a string looking for specified delimiter character(s).
// It will automatically ignore adjacent delimiters (different than split).
// The next_token method returns character start position (spos) and end
// position (epos) between delimiter runs identifying each token.
// An iterator is used to retrieve each utf8 character to be checked.
// The spaces parameter identifies a run of delimiters (or not delimiters).
struct base_tokenator
{
    custring_view* d_delimiter{nullptr};

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

//
struct tokenize_fn : base_tokenator
{
    custring_view_array d_strings;
    size_t* d_counts;
    size_t* d_offsets;
    thrust::pair<const char*,size_t>* d_tokens;

    tokenize_fn( custring_view_array d_strings, custring_view* d_delimiter, size_t* d_counts, size_t* d_offsets=nullptr, thrust::pair<const char*,size_t>* d_tokens=nullptr )
    : base_tokenator{d_delimiter}, d_strings(d_strings), d_counts(d_counts), d_offsets(d_offsets), d_tokens(d_tokens) {}

    __device__ void operator()(unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        if( !dstr )
            return;
        thrust::pair<const char*,size_t>* dstr_tokens = nullptr;
        if( d_tokens )
        {
            if( d_counts[idx]==0 )
                return;
            dstr_tokens = d_tokens + d_offsets[idx];
        }
        bool spaces = true;
        int nchars = dstr->chars_count();
        int spos = 0, epos = nchars, tidx = 0;
        auto itr = dstr->begin();
        while( next_token(dstr,spaces,itr,spos,epos) )
        {
            if( dstr_tokens )
            {
                int spos_bo = dstr->byte_offset_for(spos); // convert char pos
                int epos_bo = dstr->byte_offset_for(epos); // to byte offset
                dstr_tokens[tidx].first = dstr->data() + spos_bo;
                dstr_tokens[tidx].second = (epos_bo-spos_bo);
            }
            spos = epos + 1;
            epos = nchars;
            ++itr;
            ++tidx;
        }
        d_counts[idx] = tidx;
    }
};

NVStrings* NVText::tokenize(NVStrings& strs, const char* delimiter)
{
    auto execpol = rmm::exec_policy(0);
    custring_view* d_delimiter = custring_from_host(delimiter);

    unsigned int count = strs.size();
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);
    // count how many tokens in each string
    rmm::device_vector<size_t> counts(count,0);
    size_t* d_counts = counts.data().get();

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        tokenize_fn(d_strings,d_delimiter,d_counts));

    // compute the total number of tokens
    size_t tokens_count = thrust::reduce(execpol->on(0), counts.begin(), counts.end());
    // create token-index offsets
    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan( execpol->on(0), counts.begin(), counts.end(), offsets.begin() );
    // build a list of pointers to each token
    rmm::device_vector< thrust::pair<const char*,size_t> > tokens(tokens_count);
    thrust::pair<const char*,size_t>* d_tokens = tokens.data().get();

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        tokenize_fn(d_strings, d_delimiter, d_counts, d_offsets, d_tokens));
    //
    RMM_FREE(d_delimiter,0);
    // build strings object from tokens elements
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_tokens,tokens_count);
}

// same but with multiple delimiters
NVStrings* NVText::tokenize(NVStrings& strs, NVStrings& delims)
{
    unsigned int delims_count = delims.size();
    if( delims_count==0 )
        return NVText::tokenize(strs);
    auto execpol = rmm::exec_policy(0);
    rmm::device_vector<custring_view*> delimiters(delims_count,nullptr);
    custring_view** d_delimiters = delimiters.data().get();
    delims.create_custring_index(d_delimiters);

    unsigned int count = strs.size();
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);

    // count how many tokens in each string
    rmm::device_vector<size_t> counts(count,0);
    size_t* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiters, delims_count, d_counts] __device__(unsigned int idx){
            custring_view* d_string = d_strings[idx];
            if( !d_string )
                return;
            int tokens = 1;
            const char* sptr = d_string->data();
            const char* eptr = sptr + d_string->size();
            while( sptr < eptr )
            {
                int incr = 1;
                for( int didx=0; didx < delims_count; ++didx )
                {
                    custring_view* d_delim = d_delimiters[didx];
                    if( !d_delim || d_delim->empty() )
                        continue;
                    if( d_delim->compare(sptr,d_delim->size()) !=0 )
                        continue;
                    ++tokens;
                    incr = d_delim->size();
                    break;
                }
                sptr += incr;
            }
            d_counts[idx] = tokens;
        });

    // compute the total number of tokens
    size_t tokens_count = thrust::reduce(execpol->on(0), counts.begin(), counts.end());
    // create token-index offsets
    rmm::device_vector<size_t> offsets(count,0);
    thrust::exclusive_scan( execpol->on(0), counts.begin(), counts.end(), offsets.begin() );
    size_t* d_offsets = offsets.data().get();
    // build a list of pointers to each token
    rmm::device_vector< thrust::pair<const char*,size_t> > tokens(tokens_count);
    thrust::pair<const char*,size_t>* d_tokens = tokens.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiters, delims_count, d_counts, d_offsets, d_tokens] __device__(unsigned int idx) {
            custring_view* d_string = d_strings[idx];
            if( !d_string )
                return;
            size_t token_count = d_counts[idx];
            if( token_count==0 )
                return;
            auto dstr_tokens = d_tokens + d_offsets[idx];
            const char* data = d_string->data();
            const char* sptr = data;
            auto size = d_string->size();
            const char* eptr = sptr + size;
            int spos = 0, tidx = 0;
            while( sptr < eptr )
            {
                int incr = 1;
                for( int didx=0; didx < delims_count; ++didx )
                {
                    custring_view* d_delim = d_delimiters[didx];
                    if( !d_delim || d_delim->empty() )
                        continue;
                    if( d_delim->compare(sptr,d_delim->size()) !=0 )
                        continue;
                    // found delimiter
                    dstr_tokens[tidx].first = data + spos;
                    dstr_tokens[tidx].second = ((sptr - data) - spos);
                    ++tidx;
                    incr = d_delim->size();
                    spos = (sptr - data) + incr;
                    break;
                }
                sptr += incr;
            }
            if( (tidx < token_count) && (spos < size) )
            {
                dstr_tokens[tidx].first = data + spos;
                dstr_tokens[tidx].second = size - spos;
            }
        });
    // remove any empty strings -- occurs if two delimiters are next to each other
    auto end = thrust::remove_if(execpol->on(0), d_tokens, d_tokens + tokens_count,
        [] __device__ ( thrust::pair<const char*,size_t> w ) { return w.second==0; } );
    unsigned int nsize = (unsigned int)(end - d_tokens); // new token count
    //
    // build strings object from tokens elements
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_tokens,nsize);
}

//
NVStrings* NVText::unique_tokens(NVStrings& strs, const char* delimiter)
{
    auto execpol = rmm::exec_policy(0);
    custring_view* d_delimiter = custring_from_host(delimiter);

    unsigned int count = strs.size();
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);
    // count how many tokens in each string
    rmm::device_vector<size_t> counts(count,0);
    size_t* d_counts = counts.data().get();

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        tokenize_fn(d_strings,d_delimiter,d_counts));

    // compute the total number of tokens
    size_t tokens_count = thrust::reduce(execpol->on(0), counts.begin(), counts.end());
    // create token-index offsets
    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan( execpol->on(0), counts.begin(), counts.end(), offsets.begin() );
    // build a list of pointers to each token
    rmm::device_vector< thrust::pair<const char*,size_t> > tokens(tokens_count);
    thrust::pair<const char*,size_t>* d_tokens = tokens.data().get();

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        tokenize_fn(d_strings, d_delimiter, d_counts, d_offsets, d_tokens));
    //
    RMM_FREE(d_delimiter,0);

    thrust::sort( execpol->on(0), d_tokens, d_tokens + tokens_count,
        [] __device__ ( thrust::pair<const char*,size_t>& lhs, thrust::pair<const char*,size_t>& rhs) {
            return custr::compare(lhs.first,(unsigned)lhs.second,rhs.first,(unsigned)rhs.second)<0;
        });
    thrust::pair<const char*,size_t>* newend = thrust::unique(execpol->on(0), d_tokens, d_tokens + tokens_count,
        [] __device__ ( thrust::pair<const char*,size_t> lhs, thrust::pair<const char*,size_t> rhs ) {
            return custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second)==0;
        });
    unsigned int newsize = (unsigned int)(newend - d_tokens); // new size
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_tokens,newsize);
}

// Your basic token counter
struct nvtext_token_counter : base_tokenator
{
    custring_view_array d_strings;
    unsigned int* d_counts;
    //
    nvtext_token_counter( custring_view_array d_strings, custring_view* d_delimiter, unsigned int* d_counts )
    : base_tokenator{d_delimiter}, d_strings(d_strings), d_counts(d_counts) {}
    //
    __device__ void operator()(unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        unsigned int token_count = 0;
        if( dstr )
        {
            bool spaces = true;
            int nchars = dstr->chars_count();
            int spos = 0, epos = nchars;
            auto itr = dstr->begin();
            while( next_token(dstr,spaces,itr,spos,epos) )
            {
                ++token_count;
                spos = epos + 1; // setup
                epos = nchars;   // for next
                ++itr;           // token
            }
        }
        d_counts[idx] = token_count;
    }
};

// return a count of the number of tokens for each string when applying the specified delimiter
unsigned int NVText::token_count( NVStrings& strs, const char* delimiter, unsigned int* results, bool bdevmem )
{
    auto execpol = rmm::exec_policy(0);
    custring_view* d_delimiter = custring_from_host(delimiter);

    unsigned int count = strs.size();
    unsigned int* d_counts = results;
    if( !bdevmem )
        d_counts = device_alloc<unsigned int>(count,0);

    // count how many strings per string
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        nvtext_token_counter{d_strings,d_delimiter,d_counts});
    //
    if( !bdevmem )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_counts,count*sizeof(unsigned int),cudaMemcpyDeviceToHost))
        RMM_FREE(d_counts,0);
    }
    RMM_FREE(d_delimiter,0);
    return 0;
}

//
struct tokens_counts_fn : base_tokenator
{
    custring_view_array d_strings;
    custring_view_array d_tokens;
    unsigned int token_count;
    int* d_token_indexes;
    unsigned int* d_results;

    tokens_counts_fn( custring_view_array d_strings, custring_view_array d_tokens, unsigned int token_count,
                      int* d_token_indexes, custring_view* d_delimiter, unsigned int* d_results )
    : base_tokenator{d_delimiter}, d_strings(d_strings), d_tokens(d_tokens), token_count(token_count),
                     d_token_indexes(d_token_indexes), d_results(d_results) {}

    __device__ int match_tokens( custring_view* dstr, int spos_bo, int epos_bo )
    {
        int length = epos_bo - spos_bo;
        for( int tidx=0; tidx < token_count; ++tidx )
        {
            custring_view* d_token = d_tokens[tidx];
            if( d_token &&
                (length==d_token->size()) &&
                (d_token->compare(dstr->data()+spos_bo,length)==0) )
            {
                return tidx;
            }
        }
        return -1;
    }

    __device__ int match_sorted_tokens( custring_view* dstr, int spos_bo, int epos_bo )
    {
        int left = 0, right = token_count -1;
        int length = epos_bo - spos_bo;
        while( left <= right )
        {
            int tidx = (left + right)/2;
            custring_view* d_token = d_tokens[tidx];
            int cmp = (d_token ? d_token->compare(dstr->data()+spos_bo,length) : -1);
            if( cmp < 0 )
                left = tidx + 1;
            else if( cmp > 0 )
                right = tidx - 1;
            else
                return d_token_indexes[tidx];
        }
        return -1;
    }

    __device__ void operator()(unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        unsigned int* dresult = d_results + (idx*token_count);
        // initialize to zero
        for( int tidx=0; tidx < token_count; ++tidx )
            dresult[tidx] = 0;
        if( !dstr )
            return;
        bool spaces = true;
        int nchars = dstr->chars_count();
        int spos = 0, epos = nchars;
        auto itr = dstr->begin();
        while( next_token(dstr,spaces,itr,spos,epos) )
        {
            int spos_bo = dstr->byte_offset_for(spos); // convert char pos
            int epos_bo = dstr->byte_offset_for(epos); // to byte offset
            // check against all the tokens
            int tidx = match_sorted_tokens(dstr,spos_bo,epos_bo);
            if( tidx >= 0 )
                ++dresult[tidx];
            spos = epos + 1;
            epos = nchars;
            ++itr;
        }
    }
};

unsigned int NVText::tokens_counts( NVStrings& strs, NVStrings& tkns, const char* delimiter, unsigned int* results, bool todevice )
{
    unsigned int count = strs.size();
    unsigned int tcount = tkns.size();
    if( results==0 || count==0 || tcount==0 )
        return 0;
    //
    auto execpol = rmm::exec_policy(0);
    unsigned int* d_results = results;
    if( !todevice )
        d_results = device_alloc<unsigned int>(tcount*count,0);
    custring_view* d_delimiter = custring_from_host(delimiter);

    // get the strings
    rmm::device_vector<custring_view*> strings(count,nullptr);
    rmm::device_vector<custring_view*> tokens(tcount,nullptr);
    custring_view** d_strings = strings.data().get();
    custring_view** d_tokens = tokens.data().get();
    strs.create_custring_index(d_strings);
    tkns.create_custring_index(d_tokens);
    // sort the tokens
    rmm::device_vector<int> token_indexes(tcount);
    thrust::sequence(execpol->on(0), token_indexes.begin(), token_indexes.end());
    int* d_token_indexes = token_indexes.data().get();
    thrust::sort_by_key(execpol->on(0), d_tokens, d_tokens+tcount, d_token_indexes,
        [] __device__( custring_view*& lhs, custring_view*& rhs ) {
            if( lhs==0 || rhs==0 )
                return (rhs!=0); // null < non-null
            return lhs->compare(*rhs)<0;
        });

    // count the tokens
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        tokens_counts_fn(d_strings, d_tokens, tcount, d_token_indexes, d_delimiter, d_results) );
    // done
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_results,sizeof(unsigned int)*count*tcount,cudaMemcpyDeviceToHost))
        RMM_FREE(d_results,0);
    }
    return 0;
}

struct replace_tokens_fn : base_tokenator
{
    custring_view_array d_strings;
    custring_view_array d_tokens;
    unsigned int token_count;
    custring_view_array d_repls;
    unsigned int repl_count;
    custring_view* d_delimiter;
    size_t* d_offsets;
    bool bcompute_size_only;
    char* d_buffer;
    thrust::pair<const char*,size_t>* d_indexes;

    replace_tokens_fn( custring_view_array d_strings, custring_view_array d_tokens, unsigned int token_count,
                       custring_view_array d_repls, unsigned int repl_count, custring_view* d_delimiter,
                       size_t* d_offsets, bool bcompute_size_only=true, char* d_buffer=nullptr,
                       thrust::pair<const char*,size_t>* d_indexes=nullptr )
    : base_tokenator{d_delimiter}, d_strings(d_strings), d_tokens(d_tokens), token_count(token_count),
      d_repls(d_repls), repl_count(repl_count), d_offsets(d_offsets), bcompute_size_only(bcompute_size_only),
      d_buffer(d_buffer), d_indexes(d_indexes) {}
    //
    __device__ void operator()(unsigned int idx)
    {
        if( !bcompute_size_only )
        {
            d_indexes[idx].first = nullptr;   // initialize to
            d_indexes[idx].second = 0;        // null string
        }
        custring_view* dstr = d_strings[idx];
        if( !dstr )
            return;
        char* sptr = dstr->data();  // input buffer
        char* buffer = nullptr;     // output buffer
        if( !bcompute_size_only )
            buffer = d_buffer + d_offsets[idx];
        char* optr = buffer; // running output pointer
        int nbytes = dstr->size(), nchars = dstr->chars_count();
        int lpos = 0, spos = 0, epos = nchars;
        bool spaces = true;
        auto itr = dstr->begin();
        while( next_token(dstr,spaces,itr,spos,epos) )
        {
            int spos_bo = dstr->byte_offset_for(spos); // convert char pos
            int epos_bo = dstr->byte_offset_for(epos); // to byte offset
            // check against all the tokens
            for( int tidx=0; tidx < token_count; ++tidx )
            {
                custring_view* d_token = d_tokens[tidx];
                int length = epos_bo - spos_bo;
                if( d_token &&
                    (length==d_token->size()) &&
                    (d_token->compare(dstr->data()+spos_bo,length)==0) )
                {
                    custring_view* d_repl = (repl_count==1 ? d_repls[0] : d_repls[tidx]);
                    nbytes += (d_repl ? d_repl->size():0) - length;
                    if( !bcompute_size_only )
                    {
                        copy_and_incr(optr,sptr+lpos,spos_bo-lpos);
                        if( d_repl )
                            copy_and_incr(optr,d_repl->data(),d_repl->size());
                        lpos = epos_bo;
                    }
                    itr = custring_view::iterator(*dstr,epos);
                    break;
                }
            }
            spos = epos + 1;
            epos = nchars;
            itr++;
        }
        // set result
        if( bcompute_size_only )
            d_offsets[idx] = nbytes;
        else
        {
            memcpy( optr, sptr+lpos, dstr->size()-lpos );
            d_indexes[idx].first = buffer;
            d_indexes[idx].second = nbytes;
        }
    }
};

NVStrings* NVText::replace_tokens(NVStrings& strs, NVStrings& tgts, NVStrings& repls, const char* delimiter)
{
    if( strs.size()==0 || tgts.size()==0 )
        return strs.copy();
    if( (repls.size() > 1) && (repls.size()!=tgts.size()) )
        throw std::runtime_error("replace-tokens tokens and replacements must have the same number of strings");

    auto execpol = rmm::exec_policy(0);
    // go get the strings for all the parameters
    unsigned int count = strs.size();
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);
    unsigned int token_count = tgts.size();
    rmm::device_vector<custring_view*> tokens(token_count,nullptr);
    custring_view** d_tokens = tokens.data().get();
    tgts.create_custring_index(d_tokens);
    unsigned int repl_count = repls.size();
    rmm::device_vector<custring_view*> repl_strings(repl_count,nullptr);
    custring_view** d_repls = repl_strings.data().get();
    repls.create_custring_index(d_repls);

    custring_view* d_delimiter = custring_from_host(delimiter);

    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    char* d_buffer = nullptr;

    // calculate size of the output, allocate and then do the operation
    enum scan_and_operate { scan, operate };
    auto op = scan;
    while(true)
    {
        // 1st pass just calculates; 2nd pass will do the replace
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            replace_tokens_fn{d_strings, d_tokens, token_count, d_repls, repl_count, d_delimiter, d_offsets, (op==scan), d_buffer, d_indexes} );
        if( op==operate )
            break; // done after 2nd pass
        op = operate;
        // allocate memory for the output
        size_t buffer_size = thrust::reduce(execpol->on(0), d_offsets, d_offsets+count);
        if( buffer_size==0 )
            return nullptr;
        d_buffer = device_alloc<char>(buffer_size,0);
        // convert lengths to offsets
        thrust::exclusive_scan( execpol->on(0), offsets.begin(), offsets.end(), offsets.begin() );
    }

    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}

// Kernel operator for normalizing whitespace
struct normalize_spaces_fn : base_tokenator
{
    custring_view_array d_strings;
    size_t* d_offsets;
    bool bcompute_size_only{true};
    char* d_buffer;
    thrust::pair<const char*,size_t>* d_indexes;

    normalize_spaces_fn( custring_view_array d_strings, size_t* d_offsets,
                         bool bcompute_size_only=true, char* d_buffer=nullptr,
                         thrust::pair<const char*,size_t>* d_indexes=nullptr)
    : d_strings(d_strings), d_offsets(d_offsets), bcompute_size_only(bcompute_size_only),
      d_buffer(d_buffer), d_indexes(d_indexes) {}
    //
    __device__ void operator()(unsigned int idx)
    {
        if( !bcompute_size_only )
        {
            d_indexes[idx].first = nullptr;   // initialize to
            d_indexes[idx].second = 0;        // null string
        }
        custring_view* dstr = d_strings[idx];
        if( !dstr )
            return;
        char* sptr = dstr->data();  // input buffer
        char* buffer = nullptr;     // output buffer
        if( !bcompute_size_only )
            buffer = d_buffer + d_offsets[idx];
        char* optr = buffer; // running output pointer
        int nbytes = 0, spos = 0, epos = dstr->chars_count();
        bool spaces = true;
        auto itr = dstr->begin();
        while( next_token(dstr,spaces,itr,spos,epos) )
        {
            int spos_bo = dstr->byte_offset_for(spos); // convert char pos
            int epos_bo = dstr->byte_offset_for(epos); // to byte offset
            nbytes += epos_bo - spos_bo + 1; // include space per token
            if( !bcompute_size_only )
            {
                if( optr != buffer )
                    copy_and_incr(optr,(char*)" ",1); // add just one space
                copy_and_incr(optr,sptr+spos_bo,epos_bo-spos_bo); // copy token
            }
            spos = epos + 1;
            epos = dstr->chars_count();
            itr++; // skip the first whitespace
        }
        // set result (remove extra space for last token)
        if( bcompute_size_only )
            d_offsets[idx] = (nbytes ? nbytes-1:0);
        else
        {
            d_indexes[idx].first = buffer;
            d_indexes[idx].second = (nbytes ? nbytes-1:0);
        }
    }
};

// Replaces a run of whitespace with a single space character.
// Also trims whitespace from the beginning and end of each string.
NVStrings* NVText::normalize_spaces(NVStrings& strs)
{
    if( strs.size()==0 )
        return strs.copy();

    auto execpol = rmm::exec_policy(0);
    // go get the strings for all the parameters
    unsigned int count = strs.size();
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);
    // create working variables/memory
    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    char* d_buffer = nullptr;

    // calculate size of the output, allocate and then do the operation
    enum scan_and_operate { scan, operate };
    auto op = scan;
    while(true)
    {
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            normalize_spaces_fn{d_strings, d_offsets, (op==scan), d_buffer, d_indexes} );
        if( op==operate )
            break; // done after 2nd pass
        op = operate;
        // allocate memory for the output
        size_t buffer_size = thrust::reduce(execpol->on(0), d_offsets, d_offsets+count);
        if( buffer_size==0 )
            return nullptr;
        d_buffer = device_alloc<char>(buffer_size,0);
        // convert lengths to offsets
        thrust::exclusive_scan( execpol->on(0), offsets.begin(), offsets.end(), offsets.begin() );
    }
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}
