/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef __ORC_KERNEL_UTIL_H__
#define __ORC_KERNEL_UTIL_H__

#include "kernel_private_common.cuh"

// inclusive scan + running total in a warp.
// collect diff values from previous thread and return the accumulated diff values from thread 0,
// and (value at last thread + sum) in sum.
// input (a, b, c, d,...) => output (a, a+b, a+b+c, a+b+c+d, ...  a+b+c+d+...),
// sum -> a+b+c+d+... + sum
template <class T>
__device__ inline T GetAccumlatedDelta(T diff_value, int& sum) {
    const int num_threads = 32;
    int accum_mask = 0x01;
    T local_sum = diff_value;

    for (int i = 1; i < num_threads; i *= 2) {
        T neighbor = __shfl_xor_sync(0xffffffff, local_sum, i, num_threads);
        local_sum += neighbor;

        if (threadIdx.x & accum_mask)diff_value += neighbor;
        accum_mask <<= 1;
    }

    sum += local_sum;

    return diff_value;
}

__device__ __host__ inline orc_uint8 get_decoded_width(orc_uint8 encoded_width) {
    const int direct_encode_tbl[] = { 26, 28, 30, 32, 40, 48, 56, 64 };

    orc_uint8 width = (encoded_width < 24) ? encoded_width + 1 : direct_encode_tbl[encoded_width - 24];

    return width;
}
template <class T_signed, class T_unsigned>
__device__ __host__ inline void _private_unzigzag(T_signed& varint)
{
    T_unsigned *val = reinterpret_cast<T_unsigned*>(&varint);
    int signed_bit = *val & 0x00000001;
    *val >>= 1;
    if (signed_bit) *val = ~(*val);
}

//template<class T, EnableUnSigned<T> = nullptr >
template<class T>
__device__ __host__ inline void UnZigzag(T& varint)
{
    // if T is unsigned, do nothing.
}

// decode zigzag encoding. The LSB is the sign bit.
template <>
__device__ __host__ inline void UnZigzag(orc_sint64& varint)
{
    _private_unzigzag<orc_sint64, orc_uint64>(varint);
}

template <>
__device__ __host__ inline void UnZigzag(orc_sint32& varint)
{
    _private_unzigzag<orc_sint32, orc_uint32>(varint);
}

template <>
__device__ __host__ inline void UnZigzag(orc_sint16& varint)
{
    _private_unzigzag<orc_sint16, orc_uint16>(varint);
}

template <>
__device__ __host__ inline void UnZigzag(orc_sint8& varint)
{
    _private_unzigzag<orc_sint8, orc_uint8>(varint);
}


// fixed width Big endian, Raw or negative value if MSB=1.
__device__ __host__ inline orc_uint64 GetBaseValue(const orc_byte* p, int BW, bool is_signed)
{
    orc_uint64 BaseValue = (int)*p;
    orc_uint64 the_sign;
    if (is_signed) {    // if the output value is expected as singed value, the MSB is the sign bit.
        the_sign = (BaseValue & 0x80) ? 0x80000000 : 0;
        BaseValue &= 0x7f;
    }
    p++;
    while (--BW){
        orc_uint64 val = (orc_uint64)*p;
        BaseValue <<= 8;
        BaseValue += val;
        p++;
    } ;
    
    if (is_signed && the_sign) {
        BaseValue |= ~BaseValue;
    }

    return BaseValue;
}


// fixed width Big endian, Raw or negative value if MSB=1.
template <class T>
__device__ __host__ inline orc_uint8 GetVarint128(T *theValue, const unsigned char*p)
{
    orc_uint8 count = 1;
    orc_uint8 has_next = *p & 0x80;
    *theValue = *p & 0x7f;
    orc_uint8 the_shift = 7;

    while (has_next ){
        p++;
        has_next = *p & 0x80;
        T val = (T)(*p & 0x7f);
        *theValue += (val << the_shift);
        count++;
        the_shift += 7;
    };

    UnZigzag(*theValue);

    return count;
}

__device__ __host__ inline int getBitCount(orc_byte val) {
    int count = 0;
    for (int i = 0; i < 8; i++) {
        count += (val >> i) & 0x01;
    }
    return count;
}

__device__ __host__ inline orc_byte getBitOrderFlip(orc_byte val) {
    orc_byte ret = 0;
    for (int i = 0; i < 8; i++) {
        ret += (((val >> i) & 0x01) << (7 - i));
    }
    return ret;
}


/**
* do binary search the range where x is in.
* \param arr is an integer array, the values are sorted, but sometimes it has duplicated values.
* x is in range [1, arr[r]].
* e.g. arr[] = {0, 3, 8, 9, 13, 20} and l=0, r= 5,  return 2 if x=8; 4 if x=11.
* e.g. arr[] = {0, 0, 0, 3, 8, 9, 9, 9, 13, 20}, l=0, r=9, return 3 if x=0; 5 if x=9
* return n between [l, r]
**/
__device__ __host__ inline int binarySearchRange(int* arr, int l, int r, int x)
{
    if (r - l <= 1) {
        int lo = arr[l];
        if (x <= lo) return l;

        int hi = arr[r];
        if (lo < x && x <= hi)return r;
    }

    if (r >= l)
    {
        int mid = l + (r - l) / 2;

        // If the element is present at the middle  
        if (arr[mid] == x) {
            if (mid == 0)return 0;
            if (arr[mid] != arr[mid-1]) return mid;
        }

        // If element is smaller than mid, then  
        // it can only be present in left subarray 
        if (arr[mid] >= x)
        {
            return binarySearchRange(arr, l, mid, x);
        }
        else {
            // Else the element can only be present 
            // in right subarray 
            return binarySearchRange(arr, mid + 1, r, x);
        }
    }

    // We reach here when element is not  
    // present in array 
    return -1;
}

__device__ __host__ inline int find_index(int index, int* counts)
{
    return binarySearchRange(counts, 0, blockDim.x - 1, index);
}

__device__ __host__ inline int find_nth_true_bit(orc_bitmap bitmap, int count)
{
    int i;
    int true_count = 0;
    for (i = 0; i < 8; i++) {
        true_count += ( (bitmap >> i) & 0x01 );
        if (true_count == count)break;
    }

    // todo: the case if i == 8

    return i;
}

__device__ __host__ inline int find_nth_true_bit_from_tail(orc_bitmap bitmap, int count)
{
    int i;
    int true_count = 0;
    for (i = 7; i >= 0; i--) {
        true_count += ( (bitmap >> i) & 0x01 );
        if (true_count == count)break;
    }

    // todo: the case if i == -1

    return i;
}

__device__ __host__ inline
bool present_is_exist(const orc_bitmap* present, size_t index, int start_id)
{
    int id = index + start_id;
    
    return ((present[id >> 3] >> (id & 0x07))  & 0x01);
}


/**
* Convert a date into a unix epoch 
*/
__device__ __host__ inline
orc_sint64 convertDateToUnixEpoch(int year, int month, int day)
{
    year -= month <= 2;
    const orc_sint64 era = (year >= 0 ? year : year - 399) / 400;
    int yoe = static_cast<unsigned>(year - era * 400);      // [0, 399]
    int doy = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;  // [0, 365]
    orc_sint64 doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;         // [0, 146096]
    orc_sint64 days_e = era * 146097 + static_cast<orc_sint64>(doe) - 719468;

    return days_e;
}

/**
* Convert a date (DD/MM/YYYY) into a Orc timestamp Date (from number of seconds after 1 January 2015)
*/
__device__ __host__ inline
orc_sint64 convertDateToOrcTimestampDate(int year, int month, int day, int hour=0, int minute=0, int second=0)
{
    // get unix epic date
    orc_sint64 days_e = convertDateToUnixEpoch(year, month, day);

#if 0
    orc_sint64 days_orc_start = convertDateToUnixEpoch(2015, 1, 1);
#else
    orc_sint64 days_orc_start = 16436;  // := convertDateToUnixEpoch(2015, 1, 1);
#endif
    
    // now convert to seconds
    orc_sint64 t = (days_e - days_orc_start) * 24 * 60 * 60 + hour * 60 * 60 + minute * 60 + second;

    return t;
}

__device__ __host__ inline
orc_sint64 convertGdfTimestampMs(int year, int month, int day, int hour = 0, int minute = 0, int second = 0, int millisec=0, int adjustSec = 0)
{
    // get unix epic date
    orc_sint64 days_e = convertDateToUnixEpoch(year, month, day);

    // now convert to milli seconds
    orc_sint64 t = ( (days_e  * 24 * 60 * 60) + (hour * 60 * 60 ) + (minute * 60 )+ second + adjustSec) * 1000 + millisec;

    return t;
}


/**
* Convert a date (DD/MM/YYYY) into a date64
*/
__device__ __host__ inline
orc_sint64 convertDateToGdfDate64(int year, int month, int day)
{
    // get unix epic date
    orc_sint64 days_e = convertDateToUnixEpoch(year, month, day);

    // now convert to milli seconds
    orc_sint64 t = (days_e * 24) * 60 * 60 * 1000;

    return t;
}

/**
* Convert a date (DD/MM/YYYY) into a Orc Timestamp Date (the secondary stream of timestamp)
*/
__device__ __host__ inline
orc_uint64 convertDateToOrcTimestampTime(int ms, int us, int ns)
{
    orc_uint64 total_ns = ms * 1000 * 1000 + us * 1000 + ns;

    if (total_ns == 0)return 0;

    orc_uint64 degits = 10000000;
    int degits_count = 7;
    while ( (total_ns % degits ) != 0 && degits_count)
    {
        degits_count--;
        degits /= 10;
    }

    orc_uint64 total_ns_orc = ((total_ns / degits) << 3) + degits_count;

    return total_ns_orc;
}


#endif // __ORC_KERNEL_UTIL_H__
