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

#include <exception>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/error_utils.hpp>

#include "nvstrings/NVStrings.h"

#include "./NVStringsImpl.h"
#include "../custring_view.cuh"
#include "../custring.cuh"
#include "../util.h"

//
int NVStrings::hash(unsigned int* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<unsigned int>(count,0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = custr::hash(dstr->data(),dstr->size());//dstr->hash();
            else
                d_rtn[idx] = 0;
        });
    //
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(float)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

//
int NVStrings::stoi(int* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    int* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<int>(count,0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->stoi();
            else
                d_rtn[idx] = 0;
        });
    //
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(int)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

//
int NVStrings::stol(long* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    long* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<long>(count,0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->stol();
            else
                d_rtn[idx] = 0L;
        });
    //
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(long)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

//
int NVStrings::stof(float* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    float* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<float>(count,0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                d_rtn[idx] = (float)0;
            else if( (dstr->compare("NaN",3)==0) )
                d_rtn[idx] = std::numeric_limits<float>::quiet_NaN();
            else if( (dstr->compare("Inf",3)==0) )
                d_rtn[idx] = std::numeric_limits<float>::infinity();
            else if( (dstr->compare("-Inf",4)==0) )
                d_rtn[idx] = -std::numeric_limits<float>::infinity();
            else
                d_rtn[idx] = dstr->stof();

        });
    //
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(float)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

//
int NVStrings::stod(double* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    double* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<double>(count,0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr )
                d_rtn[idx] = 0.0;
            else if( (dstr->compare("NaN",3)==0) )
                d_rtn[idx] = std::numeric_limits<double>::quiet_NaN();
            else if( (dstr->compare("Inf",3)==0) )
                d_rtn[idx] = std::numeric_limits<double>::infinity();
            else if( (dstr->compare("-Inf",4)==0) )
                d_rtn[idx] = -std::numeric_limits<double>::infinity();
            else
                d_rtn[idx] = dstr->stod();
        });
    //
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(double)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

//
int NVStrings::htoi(unsigned int* results, bool todevice)
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !todevice )
        d_rtn = device_alloc<unsigned int>(count,0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr || dstr->empty() )
            {
                d_rtn[idx] = 0;
                return;
            }
            long result = 0, base = 1;
            const char* str = dstr->data();
            int len = dstr->size()-1;
            for( int i=len; i >= 0; --i )
            {
                char ch = str[i];
                if( ch >= '0' && ch <= '9' )
                {
                    result += (long)(ch-48) * base;
                    base *= 16;
                }
                else if( ch >= 'A' && ch <= 'Z' )
                {
                    result += (long)(ch-55) * base;
                    base *= 16;
                }
                else if( ch >= 'a' && ch <= 'z' )
                {
                    result += (long)(ch-87) * base;
                    base *= 16;
                }
            }
            d_rtn[idx] = (unsigned int)result;
        });
    //
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !todevice )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(unsigned int)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

// build strings from given integers
NVStrings* NVStrings::itos(const int* values, unsigned int count, const unsigned char* nullbitmask, bool bdevmem)
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::itos values or count invalid");
    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    int* d_values = (int*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        d_values = device_alloc<int>(count,0);
        CUDA_TRY( cudaMemcpyAsync(d_values,values,count*sizeof(int),cudaMemcpyHostToDevice))
        if( nullbitmask )
        {
            d_nulls = device_alloc<unsigned char>(((count+7)/8),0);
            CUDA_TRY( cudaMemcpyAsync(d_nulls,nullbitmask,((count+7)/8)*sizeof(unsigned char),cudaMemcpyHostToDevice))
        }
    }

    // compute size of memory we'll need
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_values, d_nulls, d_sizes] __device__ (unsigned int idx) {
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
            {
                d_sizes[idx] = 0;
                return;
            }
            int value = d_values[idx];
            int size = custring_view::ltos_size(value);
            d_sizes[idx] = ALIGN_SIZE(size);
        });

    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // build strings from integers
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    custring_view_array d_strings = rtn->pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_buffer, d_offsets, d_nulls, d_values, d_strings] __device__(unsigned int idx){
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) ) // from arrow spec
            {
                d_strings[idx] = nullptr;
                return;
            }
            char* str = d_buffer + d_offsets[idx];
            int value = d_values[idx];
            d_strings[idx] = custring_view::ltos(value,str);
        });
    //
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}

NVStrings* NVStrings::ltos(const long* values, unsigned int count, const unsigned char* nullbitmask, bool bdevmem)
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::ltos values or count invalid");
    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    long* d_values = (long*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        d_values = device_alloc<long>(count,0);
        CUDA_TRY( cudaMemcpyAsync(d_values,values,count*sizeof(long),cudaMemcpyHostToDevice))
        if( nullbitmask )
        {
            d_nulls = device_alloc<unsigned char>(((count+7)/8),0);
            CUDA_TRY( cudaMemcpyAsync(d_nulls,nullbitmask,((count+7)/8)*sizeof(unsigned char),cudaMemcpyHostToDevice))
        }
    }

    // compute size of memory we'll need
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_values, d_nulls, d_sizes] __device__ (unsigned int idx) {
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
            {
                d_sizes[idx] = 0;
                return;
            }
            long value = d_values[idx];
            int size = custring_view::ltos_size(value);
            d_sizes[idx] = ALIGN_SIZE(size);
        });

    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // build strings from integers
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    custring_view_array d_strings = rtn->pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_buffer, d_offsets, d_nulls, d_values, d_strings] __device__(unsigned int idx){
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) ) // from arrow spec
            {
                d_strings[idx] = nullptr;
                return;
            }
            char* str = d_buffer + d_offsets[idx];
            long value = d_values[idx];
            d_strings[idx] = custring_view::ltos(value,str);
        });
    //
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}

struct ftos_converter
{
    // significant digits is independent of scientific notation range
    // digits more than this may require using long values instead of ints
    const unsigned int significant_digits = 10;
    // maximum power-of-10 that will fit in 32-bits
    const unsigned int nine_digits = 1000000000; // 1x10^9
    // Range of numbers here is for normalizing the value.
    // If the value is above or below the following limits, the output is converted to
    // scientific notation in order to show (at most) the number of significant digits.
    const double upper_limit = 1000000000; // max is 1x10^9
    const double lower_limit = 0.0001; // printf uses scientific notation below this
    // Tables for doing normalization: converting to exponent form
    // IEEE double float has maximum exponent of 305 so these should cover everthing
    const double upper10[9]  = { 10, 100, 10000, 1e8,  1e16,  1e32,  1e64,  1e128,  1e256 };
    const double lower10[9]  = { .1, .01, .0001, 1e-8, 1e-16, 1e-32, 1e-64, 1e-128, 1e-256 };
    const double blower10[9] = { 1.0, .1, .001,  1e-7, 1e-15, 1e-31, 1e-63, 1e-127, 1e-255 };

    // utility for quickly converting known integer range to character array
    __device__ char* int2str( int value, char* output )
    {
        if( value==0 )
        {
            *output++ = '0';
            return output;
        }
        char buffer[10]; // should be big-enough for 10 significant digits
        char* ptr = buffer;
        while( value > 0 )
        {
            *ptr++ = (char)('0' + (value % 10));
            value /= 10;
        }
        while( ptr != buffer )
            *output++ = *--ptr;  // 54321 -> 12345
        return output;
    }

    //
    // dissect value into parts
    // return decimal_places
    __device__ int dissect_value( double value, unsigned int& integer, unsigned int& decimal, int& exp10 )
    {
        // dissect float into parts
        int decimal_places = significant_digits-1;
        // normalize step puts value between lower-limit and upper-limit
        // by adjusting the exponent up or down
        exp10 = 0;
        if( value > upper_limit )
        {
            int fx = 256;
            for( int idx=8; idx >= 0; --idx )
            {
                if( value >= upper10[idx] )
                {
                    value *= lower10[idx];
                    exp10 += fx;
                }
                fx = fx >> 1;
            }
        }
        else if( (value > 0.0) && (value < lower_limit) )
        {
            int fx = 256;
            for( int idx=8; idx >= 0; --idx )
            {
                if( value < blower10[idx] )
                {
                    value *= upper10[idx];
                    exp10 -= fx;
                }
                fx = fx >> 1;
            }
        }
        //
        unsigned int max_digits = nine_digits;
        integer = (unsigned int)value;
        for( unsigned int i=integer; i >= 10; i/=10 )
        {
            --decimal_places;
            max_digits /= 10;
        }
        double remainder = (value - (double)integer) * (double)max_digits;
        //printf("remainder=%g,value=%g,integer=%u,sd=%u\n",remainder,value,integer,max_digits);
        decimal = (unsigned int)remainder;
        remainder -= (double)decimal;
        //printf("remainder=%g,decimal=%u\n",remainder,decimal);
        decimal += (unsigned int)(2.0*remainder);
        if( decimal >= max_digits )
        {
            decimal = 0;
            ++integer;
            if( exp10 && (integer >= 10) )
            {
                ++exp10;
                integer = 1;
            }
        }
        //
        while( (decimal % 10)==0 && (decimal_places > 0) )
        {
            decimal /= 10;
            --decimal_places;
        }
        return decimal_places;
    }

    //
    // Converts value to string into output
    // Output need not be more than significant_digits+7
    // 7 = 1 sign, 1 decimal point, 1 exponent ('e'), 1 exponent-sign, 3 digits for exponent
    //
    __device__ int float_to_string( double value, char* output )
    {
        // check for valid value
        if( std::isnan(value) )
        {
            memcpy(output,"NaN",3);
            return 3;
        }
        bool bneg = false;
        if( value < 0.0 )
        {
            value = -value;
            bneg = true;
        }
        if( std::isinf(value) )
        {
            if( bneg )
                memcpy(output,"-Inf",4);
            else
                memcpy(output,"Inf",3);
            return bneg ? 4 : 3;
        }

        // dissect value into components
        unsigned int integer = 0, decimal = 0;
        int exp10 = 0;
        int decimal_places = dissect_value(value,integer,decimal,exp10);
        //
        // now build the string from the
        // components: sign, integer, decimal, exp10, decimal_places
        //
        // sign
        char* ptr = output;
        if( bneg )
            *ptr++ = '-';
        // integer
        ptr = int2str(integer,ptr);
        // decimal
        *ptr++ = '.';
        if( decimal_places )
        {
            char buffer[10];
            char* pb = buffer;
            while( decimal_places-- )
            {
                *pb++ = (char)('0' + (decimal % 10));
                decimal /= 10;
            }
            while( pb != buffer )  // reverses the digits
                *ptr++ = *--pb;    // e.g. 54321 -> 12345
        }
        else
            *ptr++ = '0'; // always include at least .0
        // exponent
        if( exp10 )
        {
            *ptr++ = 'e';
            if( exp10 < 0 )
            {
                *ptr++ ='-';
                exp10 = -exp10;
            }
            else
                *ptr++ ='+';
            if( exp10 < 10 )
                *ptr++ = '0'; // extra zero-pad
            ptr = int2str(exp10,ptr);
        }
        // done
        //*ptr = 0; // null-terminator

        return (int)(ptr-output);
    }

    // need to compute how much memory is needed to
    // hold the output string (not including null)
    __device__ int compute_ftos_size( double value )
    {
        if( std::isnan(value) )
            return 3; // NaN
        bool bneg = false;
        if( value < 0.0 )
        {
            value = -value;
            bneg = true;
        }
        if( std::isinf(value) )
            return 3 + (int)bneg; // Inf

        // dissect float into parts
        unsigned int integer = 0, decimal = 0;
        int exp10 = 0;
        int decimal_places = dissect_value(value,integer,decimal,exp10);
        // now count up the components
        // sign
        int count = (int)bneg;
        // integer
        count += (int)(integer==0);
        while( integer > 0 )
        {
            integer /= 10;
            ++count;
        } // log10(integer)
        // decimal
        ++count; // decimal point
        if( decimal_places )
            count += decimal_places;
        else
            ++count; // always include .0
        // exponent
        if( exp10 )
        {
            count += 2; // 'e±'
            if( exp10 < 0 )
                exp10 = -exp10;
            count += (int)(exp10<10); // padding
            while( exp10 > 0 )
            {
                exp10 /= 10;
                ++count;
            } // log10(exp10)
        }

        return count;
    }
};

// build strings from given floats
NVStrings* NVStrings::ftos(const float* values, unsigned int count, const unsigned char* nullbitmask, bool bdevmem)
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::ftos values or count invalid");
    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    float* d_values = (float*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        d_values = device_alloc<float>(count,0);
        CUDA_TRY( cudaMemcpyAsync(d_values,values,count*sizeof(float),cudaMemcpyHostToDevice))
        if( nullbitmask )
        {
            d_nulls = device_alloc<unsigned char>(((count+7)/8),0);
            CUDA_TRY( cudaMemcpyAsync(d_nulls,nullbitmask,((count+7)/8)*sizeof(unsigned char),cudaMemcpyHostToDevice))
        }
    }

    // compute size of memory we'll need
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_values, d_nulls, d_sizes] __device__ (unsigned int idx) {
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
            {
                d_sizes[idx] = 0;
                return;
            }
            float value = d_values[idx];
            ftos_converter fts;
            int bytes = fts.compute_ftos_size((double)value);
            int size = custring_view::alloc_size(bytes,bytes);
            d_sizes[idx] = ALIGN_SIZE(size);
        });

    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // build strings from integers
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    custring_view_array d_strings = rtn->pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_buffer, d_offsets, d_nulls, d_values, d_strings] __device__(unsigned int idx){
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) ) // from arrow spec
            {
                d_strings[idx] = nullptr;
                return;
            }
            char* str = d_buffer + d_offsets[idx];
            float value = d_values[idx];
            ftos_converter fts;
            int len = fts.float_to_string((double)value,str);
            d_strings[idx] = custring_view::create_from(str,str,len);
        });
    //
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}

// build strings from given doubles
NVStrings* NVStrings::dtos(const double* values, unsigned int count, const unsigned char* nullbitmask, bool bdevmem)
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::dtos values or count invalid");
    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    double* d_values = (double*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        d_values = device_alloc<double>(count,0);
        CUDA_TRY( cudaMemcpyAsync(d_values,values,count*sizeof(double),cudaMemcpyHostToDevice))
        if( nullbitmask )
        {
            d_nulls = device_alloc<unsigned char>(((count+7)/8),0);
            CUDA_TRY( cudaMemcpyAsync(d_nulls,nullbitmask,((count+7)/8)*sizeof(unsigned char),cudaMemcpyHostToDevice))
        }
    }

    // compute size of memory we'll need
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_values, d_nulls, d_sizes] __device__ (unsigned int idx) {
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
            {
                d_sizes[idx] = 0;
                return;
            }
            double value = d_values[idx];
            ftos_converter fts;
            int bytes = fts.compute_ftos_size(value);
            int size = custring_view::alloc_size(bytes,bytes);
            d_sizes[idx] = ALIGN_SIZE(size);
        });

    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // build strings from integers
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    custring_view_array d_strings = rtn->pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_buffer, d_offsets, d_nulls, d_values, d_strings] __device__(unsigned int idx){
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) ) // from arrow spec
            {
                d_strings[idx] = nullptr;
                return;
            }
            char* str = d_buffer + d_offsets[idx];
            double value = d_values[idx];
            ftos_converter fts;
            int len = fts.float_to_string(value,str);
            d_strings[idx] = custring_view::create_from(str,str,len);
        });
    //
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}

// convert IPv4 to integer
int NVStrings::ip2int( unsigned int* results, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;
    auto execpol = rmm::exec_policy(0);
    unsigned int* d_rtn = results;
    if( !bdevmem )
        d_rtn = device_alloc<unsigned int>(count,0);

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( !dstr || dstr->empty() )
            {
                d_rtn[idx] = 0;
                return; // empty or null string
            }
            int tokens = dstr->split_size(".",1,0,-1);
            if( tokens != 4 )
            {
                d_rtn[idx] = 0;
                return; // invalid format
            }
            unsigned int vals[4] = {0,0,0,0};
            const char* str = dstr->data();
            int len = dstr->size(), iv = 0;
            for( int i=0; (i < len) && (iv < 4); ++i )
            {
                char ch = str[i];
                if( ch >= '0' && ch <= '9' )
                {
                    vals[iv] *= 10;
                    vals[iv] += (unsigned int)(ch-'0');
                }
                else if( ch=='.' )
                    ++iv;
            }
            unsigned int result = (vals[0] * 16777216) + (vals[1] * 65536) + (vals[2] * 256) + vals[3];
            d_rtn[idx] = result;
        });
    //
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !bdevmem )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(unsigned int)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

NVStrings* NVStrings::int2ip( const unsigned int* values, unsigned int count, const unsigned char* nullbitmask, bool bdevmem )
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::int2ip values or count invalid");
    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    unsigned int* d_values = (unsigned int*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        d_values = device_alloc<unsigned int>(count,0);
        CUDA_TRY( cudaMemcpyAsync(d_values,values,count*sizeof(unsigned int),cudaMemcpyHostToDevice))
        if( nullbitmask )
        {
            d_nulls = device_alloc<unsigned char>(((count+7)/8),0);
            CUDA_TRY( cudaMemcpyAsync(d_nulls,nullbitmask,((count+7)/8)*sizeof(unsigned char),cudaMemcpyHostToDevice))
        }
    }

    // compute size of memory we'll need
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_values, d_nulls, d_sizes] __device__ (unsigned int idx) {
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
            {
                d_sizes[idx] = 0;
                return;
            }
            unsigned int ipnum = d_values[idx];
            int bytes = 3; // 3 dots: xxx.xxx.xxx.xxx
            for( int j=0; j < 4; ++j )
            {
                unsigned int value = (ipnum & 255)+1; // don't want log(0)
                bytes += (int)log10((double)value)+1; // number of base10 digits
                ipnum = ipnum >> 8;
            }
            int size = custring_view::alloc_size(bytes,bytes);
            d_sizes[idx] = ALIGN_SIZE(size);
        });

    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // build strings from integers
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    custring_view_array d_strings = rtn->pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_buffer, d_offsets, d_nulls, d_values, d_strings] __device__(unsigned int idx){
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
            {
                d_strings[idx] = nullptr;
                return;
            }
            unsigned int ipnum = d_values[idx];
            char* str = d_buffer + d_offsets[idx];
            char* ptr = str;
            for( int j=0; j < 4; ++j )
            {
                int value = ipnum & 255;
                do {
                    char ch = '0' + (value % 10);
                    *ptr++ = ch;
                    value = value/10;
                } while( value > 0 );
                if( j < 3 )
                    *ptr++ = '.';
                ipnum = ipnum >> 8;
            }
            int len = (int)(ptr-str);
            for( int j=0; j<(len/2); ++j )
            {
                char ch1 = str[j];
                char ch2 = str[len-j-1];
                str[j] = ch2;
                str[len-j-1] = ch1;
            }
            d_strings[idx] = custring_view::create_from(str,str,len);
        });
    //
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}

int NVStrings::to_bools( bool* results, const char* true_string, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;

    auto execpol = rmm::exec_policy(0);
    // copy parameter to device memory
    char* d_true = nullptr;
    int d_len = 0;
    if( true_string )
    {
        d_len = (int)strlen(true_string);
        d_true = device_alloc<char>(d_len+1,0);
        CUDA_TRY( cudaMemcpyAsync(d_true,true_string,d_len+1,cudaMemcpyHostToDevice))
    }
    //
    bool* d_rtn = results;
    if( !bdevmem )
        d_rtn = device_alloc<bool>(count,0);

    // set the values
    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_strings, d_true, d_len, d_rtn] __device__(unsigned int idx){
            custring_view* dstr = d_strings[idx];
            if( dstr )
                d_rtn[idx] = dstr->compare(d_true,d_len)==0;
            else
                d_rtn[idx] = (d_true==0); // let null be a thing
        });
    //
    // calculate the number of falses (to include nulls too)
    int falses = thrust::count(execpol->on(0),d_rtn,d_rtn+count,false);
    if( !bdevmem )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(bool)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    if( d_true )
        RMM_FREE(d_true,0);
    return (int)count-falses;
}

NVStrings* NVStrings::create_from_bools(const bool* values, unsigned int count, const char* true_string, const char* false_string, const unsigned char* nullbitmask, bool bdevmem)
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::create_from_bools values or count invalid");
    if( true_string==0 || false_string==0 )
        throw std::invalid_argument("nvstrings::create_from_bools false and true strings must not be null");

    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    int d_len_true = strlen(true_string);
    char* d_true = device_alloc<char>(d_len_true+1,0);
    CUDA_TRY( cudaMemcpyAsync(d_true,true_string,d_len_true+1,cudaMemcpyHostToDevice))
    int d_as_true = custring_view::alloc_size(true_string,d_len_true);
    int d_len_false = strlen(false_string);
    char* d_false = device_alloc<char>(d_len_false+1,0);
    CUDA_TRY( cudaMemcpyAsync(d_false,false_string,d_len_false+1,cudaMemcpyHostToDevice))
    int d_as_false = custring_view::alloc_size(false_string,d_len_false);

    bool* d_values = (bool*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        d_values = device_alloc<bool>(count,0);
        CUDA_TRY( cudaMemcpyAsync(d_values,values,count*sizeof(bool),cudaMemcpyHostToDevice))
        if( nullbitmask )
        {
            d_nulls = device_alloc<unsigned char>(((count+7)/8),0);
            CUDA_TRY( cudaMemcpyAsync(d_nulls,nullbitmask,((count+7)/8)*sizeof(unsigned char),cudaMemcpyHostToDevice))
        }
    }

    // compute size of memory we'll need
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_values, d_nulls, d_as_true, d_as_false, d_sizes] __device__ (unsigned int idx) {
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
            {
                d_sizes[idx] = 0;
                return;
            }
            bool value = d_values[idx];
            int size = value ? d_as_true : d_as_false;
            d_sizes[idx] = ALIGN_SIZE(size);
        });

    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // build strings of booleans
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    custring_view_array d_strings = rtn->pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_buffer, d_offsets, d_nulls, d_values, d_true, d_len_true, d_false, d_len_false, d_strings] __device__(unsigned int idx){
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) ) // from arrow spec
            {
                d_strings[idx] = nullptr; // null string
                return;
            }
            char* buf = d_buffer + d_offsets[idx];
            bool value = d_values[idx];
            if( value )
                d_strings[idx] = custring_view::create_from(buf,d_true,d_len_true);
            else
                d_strings[idx] = custring_view::create_from(buf,d_false,d_len_false);
        });
    //
    if( !bdevmem )
        RMM_FREE(d_values,0);
    RMM_FREE(d_true,0);
    RMM_FREE(d_false,0);
    return rtn;
}
