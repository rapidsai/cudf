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
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include "nvstrings/NVStrings.h"
#include "nvstrings/NVText.h"

#include "../custring_view.cuh"
#include "../custring.cuh"
#include "../util.h"

typedef custring_view** custring_view_array;

//
static custring_view* custring_from_host( const char* str )
{
    if( !str )
        return nullptr;
    unsigned int length = (unsigned int)strlen(str);
    unsigned int bytes = custring_view::alloc_size(str,length);
    custring_view* d_str = reinterpret_cast<custring_view*>(device_alloc<char>(bytes,0));
    custring_view::create_from_host(d_str,str,length);
    return d_str;
}


// common token counter for all split methods
struct nvtext_token_counter
{
    custring_view_array d_strings;
    custring_view* d_delimiter;
    size_t* d_counts;
    //
    nvtext_token_counter(custring_view_array d_strings, custring_view* d_delimiter, size_t* d_counts)
    : d_strings(d_strings), d_delimiter(d_delimiter), d_counts(d_counts) {}
    __device__ void operator()(unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        if( dstr )
            d_counts[idx] = dstr->split_size(d_delimiter->data(),d_delimiter->size(),0,-1);
    }
};

// special-case token counter for whitespace delimiter
// leading and trailing and duplicate delimiters are ignored
struct nvtext_whitespace_token_counter
{
    custring_view_array d_strings;
    size_t* d_counts;

    // count the 'words' only between non-whitespace characters
    nvtext_whitespace_token_counter(custring_view_array d_strings, size_t* d_counts)
    : d_strings(d_strings), d_counts(d_counts) {}
    __device__ void operator()(unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        if( !dstr )
            return;
        int dcount = 0;
        bool spaces = true;
        custring_view::iterator itr = dstr->begin();
        while( itr != dstr->end() )
        {
            Char ch = *itr;
            if( spaces == (ch <= ' ') )
                itr++;
            else
            {
                dcount += (int)spaces;
                spaces = !spaces;
            }
        }
        if( dcount==0 )
            dcount = 1; // always allow empty string
        d_counts[idx] = dcount;
    }
};

// return a set of tokens of all the strings in the target instance
// order needs to preserved so row-split like operation is required
NVStrings* NVText::tokenize(NVStrings& strs)
{
    auto execpol = rmm::exec_policy(0);
    unsigned int count = strs.size();
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);
    // count how many tokens in each string
    rmm::device_vector<size_t> counts(count,0);
    size_t* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        nvtext_whitespace_token_counter(d_strings,d_counts));

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
        [d_strings, d_counts, d_offsets, d_tokens] __device__(unsigned int idx) {
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            size_t token_count = d_counts[idx];
            if( token_count==0 )
                return;
            auto dstr_tokens = d_tokens + d_offsets[idx];
            int spos = 0, nchars = dstr->chars_count();
            int epos = nchars, tidx = 0;
            bool spaces = true;
            //for( int tidx=0; tidx < token_count; ++tidx )
            for( auto itr=dstr->begin(); itr!=dstr->end(); itr++ )
            {
                Char ch = *itr;
                if( spaces == (ch <= ' ') )
                {
                    if( spaces )
                        spos = itr.position()+1;
                    else
                        epos = itr.position()+1;
                    continue;
                }
                if( !spaces ) //
                {
                    epos = itr.position();
                    int spos_bo = dstr->byte_offset_for(spos); // convert char pos
                    int epos_bo = dstr->byte_offset_for(epos); // to byte offset
                    dstr_tokens[tidx].first = dstr->data() + spos_bo;
                    dstr_tokens[tidx].second = (epos_bo-spos_bo);
                    ++tidx;
                    spos = epos + 1;
                    epos = nchars;
                }
                spaces = !spaces;
            }
            if( spos < nchars )
            {
                int spos_bo = dstr->byte_offset_for(spos);
                dstr_tokens[tidx].first = dstr->data() + spos_bo;
                dstr_tokens[tidx].second = (dstr->size()-spos_bo);
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

// same with specific delimiter
NVStrings* NVText::tokenize(NVStrings& strs, const char* delimiter)
{
    if( delimiter==nullptr || *delimiter==0 )
        return NVText::tokenize(strs);
    auto execpol = rmm::exec_policy(0);
    unsigned int delim_length = (unsigned int)strlen(delimiter);
    unsigned int delim_size = custring_view::alloc_size(delimiter,delim_length);
    custring_view* d_delimiter = reinterpret_cast<custring_view*>(device_alloc<char>(delim_size,0));
    custring_view::create_from_host(d_delimiter,delimiter,delim_length);

    unsigned int count = strs.size();
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);
    // count how many tokens in each string
    rmm::device_vector<size_t> counts(count,0);
    size_t* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        nvtext_token_counter(d_strings,d_delimiter,d_counts));

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
        [d_strings, d_delimiter, d_counts, d_offsets, d_tokens] __device__(unsigned int idx) {
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            size_t token_count = d_counts[idx];
            if( token_count==0 )
                return;
            auto dstr_tokens = d_tokens + d_offsets[idx];
            int spos = 0, nchars = dstr->chars_count();
            int epos = nchars;
            for( int tidx=0; tidx < token_count; ++tidx )
            {
                epos = dstr->find(*d_delimiter,spos);
                if( epos < 0 )
                    epos = nchars;
                int spos_bo = dstr->byte_offset_for(spos); // convert char pos
                int epos_bo = dstr->byte_offset_for(epos); // to byte offset
                dstr_tokens[tidx].first = dstr->data() + spos_bo;
                dstr_tokens[tidx].second = (epos_bo-spos_bo);
                // position past the delimiter
                spos = epos + d_delimiter->chars_count();
            }
        });
    // remove any empty strings -- occurs if two delimiters are next to each other
    auto end = thrust::remove_if(execpol->on(0), d_tokens, d_tokens + tokens_count,
        [] __device__ ( thrust::pair<const char*,size_t> w ) { return w.second==0; } );
    unsigned int nsize = (unsigned int)(end - d_tokens); // new token count
    //
    RMM_FREE(d_delimiter,0);
    // build strings object from tokens elements
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_tokens,nsize);
}

// same but with multiple delimiters
NVStrings* NVText::tokenize(NVStrings& strs, NVStrings& delims )
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

// return unique set of tokens within all the strings using the specified delimiter
NVStrings* NVText::unique_tokens(NVStrings& strs, const char* delimiter )
{
    int bytes = (int)strlen(delimiter);
    auto execpol = rmm::exec_policy(0);
    char* d_delimiter = device_alloc<char>(bytes,0);
    CUDA_TRY( cudaMemcpyAsync(d_delimiter,delimiter,bytes,cudaMemcpyHostToDevice))

    // need to count how many output strings per string
    unsigned int count = strs.size();
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);
    rmm::device_vector<int> counts(count,0);
    int* d_counts = counts.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, d_counts] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_counts[idx] = dstr->split_size(d_delimiter,bytes,0,-1);
        });

    int columnsCount = *thrust::max_element(execpol->on(0), counts.begin(), counts.end() );

    // build an index for each column and then sort/unique it
    rmm::device_vector< thrust::pair<const char*,size_t> > vocab;
    for( int col=0; col < columnsCount; ++col )
    {
        // first, build a vector of pair<char*,int>'s' for each column
        // each pair points to a string for this column for each row
        rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
        thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
        thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
            [d_strings, col, d_delimiter, bytes, d_counts, d_indexes] __device__(unsigned int idx){
                custring_view* dstr = d_strings[idx];
                d_indexes[idx].first = nullptr;   // initialize to
                d_indexes[idx].second = 0;  // null string
                if( !dstr )
                    return;
                // dcount already accounts for the maxsplit value
                int dcount = d_counts[idx];
                if( col >= dcount )
                    return; // passed the end for this string
                // skip delimiters until we reach this column
                int spos = 0, nchars = dstr->chars_count();
                int epos = nchars;
                for( int c=0; c < (dcount-1); ++c )
                {
                    epos = dstr->find(d_delimiter,bytes,spos);
                    if( epos < 0 )
                    {
                        epos = nchars;
                        break;
                    }
                    if( c==col )  // found our column
                        break;
                    spos = epos + bytes;
                    epos = nchars;
                }
                // this will be the string for this column
                if( spos < epos )
                {
                    spos = dstr->byte_offset_for(spos); // convert char pos
                    epos = dstr->byte_offset_for(epos); // to byte offset
                    d_indexes[idx].first = dstr->data() + spos;
                    d_indexes[idx].second = (epos-spos);
                }
            });
        //cudaError_t err = cudaDeviceSynchronize();
        //if( err != cudaSuccess )
        //{
        //    fprintf(stderr,"unique_tokens:col=%d\n",col);
        //    printCudaError(err);
        //}
        // add column values to vocab list
        vocab.insert(vocab.end(),indexes.begin(),indexes.end());
        //printf("vocab size = %lu\n",vocab.size());
        thrust::pair<const char*,size_t>* d_vocab = vocab.data().get();
        // sort the list
        thrust::sort(execpol->on(0), d_vocab, d_vocab + vocab.size(),
            [] __device__( thrust::pair<const char*,size_t>& lhs, thrust::pair<const char*,size_t>& rhs ) {
                if( lhs.first==0 || rhs.first==0 )
                    return lhs.first==0; // non-null > null
                return custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second) < 0;
            });
        // unique the list
        thrust::pair<const char*,size_t>* newend = thrust::unique(execpol->on(0), d_vocab, d_vocab + vocab.size(),
            [] __device__ ( thrust::pair<const char*,size_t> lhs, thrust::pair<const char*,size_t> rhs ) {
                if( lhs.first==rhs.first )
                    return true;
                if( lhs.second != rhs.second )
                    return false;
                return custr::compare(lhs.first,(unsigned int)lhs.second,rhs.first,(unsigned int)rhs.second)==0;
            });
        // truncate list to the unique set
        // the above unique() call does an implicit dev-sync
        vocab.resize((size_t)(newend - d_vocab));
    }
    // remove the inevitable 'null' token
    thrust::pair<const char*,size_t>* d_vocab = vocab.data().get();
    auto end = thrust::remove_if(execpol->on(0), d_vocab, d_vocab + vocab.size(), [] __device__ ( thrust::pair<const char*,size_t> w ) { return w.first==0; } );
    unsigned int vsize = (unsigned int)(end - d_vocab); // may need new size
    // done
    RMM_FREE(d_delimiter,0);
    // build strings object from vocab elements
    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_vocab,vsize);
}

// return a count of the number of tokens for each string when applying the specified delimiter
unsigned int NVText::token_count( NVStrings& strs, const char* delimiter, unsigned int* results, bool bdevmem )
{
    int bytes = (int)strlen(delimiter);
    auto execpol = rmm::exec_policy(0);
    char* d_delimiter = device_alloc<char>(bytes,0);
    CUDA_TRY( cudaMemcpyAsync(d_delimiter,delimiter,bytes,cudaMemcpyHostToDevice))

    unsigned int count = strs.size();
    unsigned int* d_counts = results;
    if( !bdevmem )
        d_counts = device_alloc<unsigned int>(count,0);

    // count how many strings per string
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_delimiter, bytes, d_counts] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            int tc = 0;
            if( dstr )
                tc = dstr->empty() ? 0 : dstr->split_size(d_delimiter,bytes,0,-1);
            d_counts[idx] = tc;
        });
    //
    if( !bdevmem )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_counts,count*sizeof(unsigned int),cudaMemcpyDeviceToHost))
        RMM_FREE(d_counts,0);
    }
    RMM_FREE(d_delimiter,0);
    return 0;
}

// return boolean value for each token if found in the provided strings
unsigned int NVText::contains_strings( NVStrings& strs, NVStrings& tkns, bool* results, bool todevice )
{
    unsigned int count = strs.size();
    unsigned int tcount = tkns.size();
    if( results==0 || count==0 || tcount==0 )
        return 0;
    //
    auto execpol = rmm::exec_policy(0);
    bool* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<bool>(tcount*count,0);

    //
    rmm::device_vector<custring_view*> strings(count,nullptr);
    rmm::device_vector<custring_view*> tokens(tcount,nullptr);
    custring_view** d_strings = strings.data().get();
    custring_view** d_tokens = tokens.data().get();
    strs.create_custring_index(d_strings);
    tkns.create_custring_index(d_tokens);

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_tokens, tcount, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            for( int jdx=0; jdx < tcount; ++jdx )
            {
                custring_view* dtgt = d_tokens[jdx];
                d_rtn[(idx*tcount)+jdx] = ((dstr && dtgt) ? dstr->find(*dtgt) : -2) >=0 ;
            }
        });
    //
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count*tcount,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return 0;
}

// return the number of occurrences of each string within a set of strings
// this will fill in the provided memory as a matrix:
//           'aa'  'bbb'  'c' ...
// "aaaabc"    2     0     1
// "aabbcc"    1     0     2
// "abbbbc"    0     1     1
// ...
unsigned int NVText::strings_counts( NVStrings& strs, NVStrings& tkns, unsigned int* results, bool todevice )
{
    unsigned int count = strs.size();
    unsigned int tcount = tkns.size();
    if( results==0 || count==0 || tcount==0 )
        return 0;
    //
    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<unsigned int>(tcount*count,0);

    //
    rmm::device_vector<custring_view*> strings(count,nullptr);
    rmm::device_vector<custring_view*> tokens(tcount,nullptr);
    custring_view** d_strings = strings.data().get();
    custring_view** d_tokens = tokens.data().get();
    strs.create_custring_index(d_strings);
    tkns.create_custring_index(d_tokens);

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_tokens, tcount, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            for( int jdx=0; jdx < tcount; ++jdx )
            {
                custring_view* dtgt = d_tokens[jdx];
                int fnd = 0;
                if( dstr && dtgt )
                {
                    int pos = dstr->find(*dtgt);
                    while( pos >= 0 )
                    {
                        pos = dstr->find(*dtgt,pos+dtgt->chars_count());
                        ++fnd;
                    }
                }
                d_rtn[(idx*tcount)+jdx] = fnd;
            }
        });
    //
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(unsigned int)*count*tcount,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return 0;
}

// return the number of occurrences of each string within a set of strings
// this will fill in the provided memory as a matrix:
//              'aa'  'bbb'  'c' ...
// "aa aa b c"    2     0     1
// "aa bb c c"    1     0     2
// "a bbb ccc"    0     1     0
// ...
unsigned int NVText::tokens_counts( NVStrings& strs, NVStrings& tkns, const char* delimiter, unsigned int* results, bool todevice )
{
    unsigned int count = strs.size();
    unsigned int tcount = tkns.size();
    if( results==0 || count==0 || tcount==0 )
        return 0;
    //
    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<unsigned int>(tcount*count,0);
    int dellen = (int)strlen(delimiter);
    char* d_delimiter = device_alloc<char>(dellen,0);
    CUDA_TRY( cudaMemcpyAsync(d_delimiter,delimiter,dellen,cudaMemcpyHostToDevice))

    //
    rmm::device_vector<custring_view*> strings(count,nullptr);
    rmm::device_vector<custring_view*> tokens(tcount,nullptr);
    custring_view** d_strings = strings.data().get();
    custring_view** d_tokens = tokens.data().get();
    strs.create_custring_index(d_strings);
    tkns.create_custring_index(d_tokens);

    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_tokens, tcount, d_delimiter, dellen, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            for( int jdx=0; jdx < tcount; ++jdx )
            {
                custring_view* dtgt = d_tokens[jdx];
                int fnd = 0;
                if( dstr && dtgt )
                {
                    int pos = dstr->find(*dtgt);
                    while( pos >= 0 )
                    {
                        int epos = pos + dtgt->chars_count();
                        if( ((pos==0) || (dstr->find(d_delimiter,dellen,pos-1)==(pos-1))) &&
                            ((epos>=dstr->chars_count()) || (dstr->find(d_delimiter,dellen,epos)==epos)) )
                            ++fnd;
                        pos = dstr->find(*dtgt,pos+dtgt->chars_count());
                    }
                }
                d_rtn[(idx*tcount)+jdx] = fnd;
            }
        });
    //
    if( !todevice )
    {   // copy result back to host
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(unsigned int)*count*tcount,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return 0;
}

struct replace_tokens_fn
{
    custring_view_array d_strings;
    custring_view_array d_tokens;
    unsigned int token_count;
    custring_view_array d_repls;
    unsigned int repl_count;
    custring_view* d_delimiter;
    size_t* d_offsets;
    bool bcompute_size_only{true};
    char* d_buffer;
    thrust::pair<const char*,size_t>* d_indexes;

    __device__ bool is_delimiter(Char ch)
    {
        if( !d_delimiter )
            return (ch <= ' ');
        for( auto itr = d_delimiter->begin(); itr != d_delimiter->end(); itr++ )
            if( (*itr)==ch )
                return true;
        return false;
    }

    __device__ bool next_token( custring_view* dstr, bool& spaces, custring_view::iterator& itr, int& spos, int& epos )
    {
        if( spos >= dstr->chars_count() )
            return false;
        for( ; itr != dstr->end(); itr++ )
        {
            Char ch = *itr;
            if( spaces == is_delimiter(ch) ) // (ch <= ' ')
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
        return true;
    }

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

    // first, calculate size of the output
    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        replace_tokens_fn{d_strings, d_tokens, token_count, d_repls, repl_count, d_delimiter, d_offsets} );

    size_t buffer_size = thrust::reduce(execpol->on(0), d_offsets, d_offsets+count);
    if( buffer_size==0 )
        return nullptr;
    char* d_buffer = device_alloc<char>(buffer_size,0);
    thrust::exclusive_scan( execpol->on(0), offsets.begin(), offsets.end(), offsets.begin() );
    // build the output strings
    rmm::device_vector< thrust::pair<const char*,size_t> > indexes(count);
    thrust::pair<const char*,size_t>* d_indexes = indexes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        replace_tokens_fn{d_strings, d_tokens, token_count, d_repls, repl_count, d_delimiter, d_offsets, false, d_buffer, d_indexes} );

    return NVStrings::create_from_index((std::pair<const char*,size_t>*)d_indexes,count);
}

// Documentation here: https://www.cuelogic.com/blog/the-levenshtein-algorithm
// And here: https://en.wikipedia.org/wiki/Levenshtein_distances
struct editdistance_levenshtein_algorithm
{
    custring_view** d_strings; // trying match
    custring_view* d_tgt;      // match with this
    custring_view** d_tgts;    // or these
    short* d_buffer;           // compute buffer
    size_t* d_offsets;         // locate sub-buffer
    unsigned int* d_results;   // edit-distances

    // single string
    editdistance_levenshtein_algorithm( custring_view** strings, custring_view* tgt, short* buffer, size_t* offsets, unsigned int* results )
    : d_strings(strings), d_tgt(tgt), d_tgts(0), d_buffer(buffer), d_offsets(offsets), d_results(results) {}

    // multiple strings
    editdistance_levenshtein_algorithm( custring_view** strings, custring_view** tgts, short* buffer, size_t* offsets, unsigned int* results )
    : d_strings(strings), d_tgt(0), d_tgts(tgts), d_buffer(buffer), d_offsets(offsets), d_results(results) {}

    __device__ void operator() (unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        short* buf = (short*)d_buffer + d_offsets[idx];
        custring_view* dtgt = d_tgt;
        if( !d_tgt )
            dtgt = d_tgts[idx];
        d_results[idx] = compute_distance(dstr,dtgt,buf);
    }

    __device__ unsigned int compute_distance( custring_view* dstr, custring_view* dtgt, short* buf )
    {
        if( !dstr || dstr->empty() )
            return dtgt ? dtgt->chars_count() : 0;
        if( !dtgt || dtgt->empty() )
            return dstr->chars_count();
        //
        custring_view* strA = dstr;
        custring_view* strB = dtgt;
        int lenA = (int)dstr->chars_count();
        int lenB = (int)dtgt->chars_count();
        if( lenA > lenB )
        {
            lenB = lenA;
            lenA = dtgt->chars_count();
            strA = dtgt;
            strB = dstr;
        }
        //
        short* line2 = buf;
        short* line1 = line2 + lenA;
        short* line0 = line1 + lenA;
        int range = lenA + lenB - 1;
        for (int i = 0; i < range; i++)
        {
            short* tmp = line2;
            line2 = line1;
            line1 = line0;
            line0 = tmp;

            for(int x = (i < lenB ? 0 : i - lenB + 1); (x < lenA) && (x < i+1); x++)
            {
                int y = i - x;
                short u = y > 0 ? line1[x] : x + 1;
                short v = x > 0 ? line1[x - 1] : y + 1;
                short w;
                if((x > 0) && (y > 0))
                    w = line2[x - 1];
                else if(x > y)
                    w = x;
                else
                    w = y;
                u++; v++;
                Char c1 = strA->at(x);
                Char c2 = strB->at(y);
                if(c1 != c2)
                    w++;
                short value = u;
                if(v < value)
                    value = v;
                if(w < value)
                    value = w;
                line0[x] = value;
            }
        }
        return (unsigned int)line0[lenA-1];
    }
};

unsigned int NVText::edit_distance( distance_type algo, NVStrings& strs, const char* str, unsigned int* results, bool bdevmem )
{
    if( algo != levenshtein || str==0 || results==0 )
        throw std::invalid_argument("invalid algorithm");
    unsigned int count = strs.size();
    if( count==0 )
        return 0; // nothing to do
    auto execpol = rmm::exec_policy(0);
    unsigned int len = strlen(str);
    unsigned int alcsz = custring_view::alloc_size(str,len);
    custring_view* d_tgt = reinterpret_cast<custring_view*>(device_alloc<char>(alcsz,0));
    custring_view::create_from_host(d_tgt,str,len);

    // setup results vector
    unsigned int* d_rtn = results;
    if( !bdevmem )
        d_rtn = device_alloc<unsigned int>(count,0);

    // get the string pointers
    rmm::device_vector<custring_view*> strings(count,nullptr);
    custring_view** d_strings = strings.data().get();
    strs.create_custring_index(d_strings);

    // calculate the size of the compute-buffer: 6 * length of string
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_tgt, d_sizes] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                return;
            int len = dstr->chars_count();
            if( d_tgt->chars_count() < len )
                len = d_tgt->chars_count();
            d_sizes[idx] = len * 3;
        });
    //
    size_t bufsize = thrust::reduce(execpol->on(0), d_sizes, d_sizes+count );
    rmm::device_vector<short> buffer(bufsize,0);
    short* d_buffer = buffer.data().get();
    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0), sizes.begin(), sizes.end(), offsets.begin() );
    // compute edit distance
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        editdistance_levenshtein_algorithm(d_strings, d_tgt, d_buffer, d_offsets, d_rtn));
    //
    if( !bdevmem )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,count*sizeof(unsigned int),cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    RMM_FREE(d_tgt,0);
    return 0;
}

unsigned int NVText::edit_distance( distance_type algo, NVStrings& strs1, NVStrings& strs2, unsigned int* results, bool bdevmem )
{
    if( algo != levenshtein )
        throw std::invalid_argument("invalid algorithm");
    unsigned int count = strs1.size();
    if( count != strs2.size() )
        throw std::invalid_argument("sizes must match");
    if( count==0 )
        return 0; // nothing to do

    // setup results vector
    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !bdevmem )
        d_rtn = device_alloc<unsigned int>(count,0);

    // get the string pointers
    rmm::device_vector<custring_view*> strings1(count,nullptr);
    custring_view** d_strings1 = strings1.data().get();
    strs1.create_custring_index(d_strings1);
    rmm::device_vector<custring_view*> strings2(count,nullptr);
    custring_view** d_strings2 = strings2.data().get();
    strs2.create_custring_index(d_strings2);

    // calculate the size of the compute-buffer: 6 * length of string
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings1, d_strings2, d_sizes] __device__(unsigned int idx){
            custring_view* dstr1 = d_strings1[idx];
            custring_view* dstr2 = d_strings2[idx];
            if( !dstr1 || !dstr2 )
                return;
            int len1 = dstr1->chars_count();
            int len = dstr2->chars_count();
            if( len1 < len )
                len = len1;
            d_sizes[idx] = len * 3;
        });
    //
    size_t bufsize = thrust::reduce(execpol->on(0), d_sizes, d_sizes+count );
    rmm::device_vector<short> buffer(bufsize,0);
    short* d_buffer = buffer.data().get();
    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0), sizes.begin(), sizes.end(), offsets.begin() );
    // compute edit distance
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        editdistance_levenshtein_algorithm(d_strings1, d_strings2, d_buffer, d_offsets, d_rtn));
    //
    if( !bdevmem )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,count*sizeof(unsigned int),cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return 0;
}

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

    unsigned int sep_length = (unsigned int)strlen(separator);
    unsigned int sep_size = custring_view::alloc_size(separator,sep_length);
    custring_view* d_separator = reinterpret_cast<custring_view*>(device_alloc<char>(sep_size,0));
    custring_view::create_from_host(d_separator,separator,sep_length);

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
