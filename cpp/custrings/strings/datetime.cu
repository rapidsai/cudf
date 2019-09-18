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
#include <map>
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
#include "../util.h"

// used to index values in a timeparts array
#define TP_YEAR        0
#define TP_MONTH       1
#define TP_DAY         2
#define TP_HOUR        3
#define TP_MINUTE      4
#define TP_SECOND      5
#define TP_SUBSECOND   6
#define TP_TZ_MINUTES  7
#define TP_ARRAYSIZE   8

struct DTFormatItem
{
    bool item_type;    // 1=specifier, 0=literal
    char specifier;    // specifier
    short length;      // item length in bytes
    char literal;      // pass-thru character // this should be a Char

    static DTFormatItem new_specifier(char fmt, short len)
    {
        DTFormatItem item{true,fmt,len,0};
        return item;
    }
    static DTFormatItem new_delimiter(char ch)
    {
        DTFormatItem item{false,0,1,ch};
        return item;
    }
};

struct DTProgram
{
    size_t count;
    DTFormatItem* items;
};

struct DTFormatCompiler
{
    std::vector<DTFormatItem> items;
    const char* format;
    size_t length;
    std::string template_string;
    NVStrings::timestamp_units units;
    DTProgram* d_prog;
    DTFormatItem* d_items;

    std::map<char,short> specifiers = {
        {'a',0}, {'A',0},
        {'w',1},
        {'b',0}, {'B',0},
        {'Y',4},{'y',2}, {'m',2}, {'d',2},
        {'H',2},{'I',2},{'M',2},{'S',2},{'f',6},
        {'p',2},{'z',5},{'Z',3},
        {'j',3},{'U',2},{'W',2}
    };

    DTFormatCompiler( const char* format, size_t length, NVStrings::timestamp_units units )
    : format(format), length(length), units(units), d_prog(nullptr), d_items(nullptr) {}

    ~DTFormatCompiler()
    {
        if( !d_prog )
            RMM_FREE(d_prog,0);
        if( !d_items )
            RMM_FREE(d_items,0);
    }

    DTProgram* compile_to_device()
    {
        //printf("dtc: format=[%s],%ld\n",format,length);
        const char* str = format;
        while( length > 0 )
        {
            char ch = *str++;
            length--;
            if( ch!='%' )
            {
                // this should be a Char
                items.push_back(DTFormatItem::new_delimiter(ch));
                template_string.append(ch,1);
                continue;
            }
            if( length==0 )
                throw std::invalid_argument("unfinished specifier");

            ch = *str++;
            length--;
            if( ch=='%' )
            {   // escaped %
                items.push_back(DTFormatItem::new_delimiter(ch));
                template_string.append(ch,1);
                continue;
            }
            if( specifiers.find(ch)==specifiers.end() )
            {
                fprintf(stderr,"specifier %c unrecognized\n",ch);
                throw std::invalid_argument("invalid specifier");
            }

            short flen = specifiers[ch];
            if( ch=='f' )
            {
                if( units==NVStrings::timestamp_units::ms )
                    flen = 3;
                else if( units==NVStrings::timestamp_units::ns )
                    flen = 9;
            }
            items.push_back(DTFormatItem::new_specifier(ch,flen));
            template_string.append(ch,flen);
        }
        // create in device memory
        size_t buffer_size = items.size() * sizeof(DTFormatItem);
        d_items = reinterpret_cast<DTFormatItem*>(device_alloc<char>(buffer_size,0));
        CUDA_TRY( cudaMemcpyAsync(d_items, items.data(), buffer_size, cudaMemcpyHostToDevice))
        DTProgram hprog{items.size(),d_items};
        d_prog = reinterpret_cast<DTProgram*>(device_alloc<char>(sizeof(DTProgram),0));
        CUDA_TRY( cudaMemcpyAsync(d_prog,&hprog,sizeof(DTProgram),cudaMemcpyHostToDevice))
        return d_prog;
    }

    // call valid only after compile
    size_t string_length() { return template_string.size(); }
    const char* string_template() { return template_string.c_str(); }

    size_t size() { return items.size(); }
};


// this parses date/time characters into long timestamp
struct parse_datetime
{
    custring_view_array d_strings;
    unsigned long* d_timestamps;
    NVStrings::timestamp_units units;
    DTProgram* d_prog;

    parse_datetime( DTProgram* prog, custring_view_array strings, NVStrings::timestamp_units units, unsigned long* results )
    : d_prog(prog), d_strings(strings), d_timestamps(results), units(units) {}

    // could use the custring::stoi but this should be faster since we know the data limits
    __device__ int str2int( const char* str, unsigned int bytes )
    {
        const char* ptr = str;
        int value = 0;
        for( unsigned int idx=0; idx < bytes; ++idx )
        {
            char chr = *ptr++;
            if( chr < '0' || chr > '9' )
                break;
            value = (value * 10) + (int)(chr - '0');
        }
        return value;
    }

    // only supports ascii
    __device__ int strcmp_ignore_case( const char* str1, const char* str2, size_t len )
    {
        for( size_t idx=0; idx < len; ++idx )
        {
            char ch1 = *str1;
            if( ch1 >= 'a' && ch1 <= 'z' )
                ch1 = ch1 - 'a' + 'A';
            char ch2 = *str2;
            if( ch2 >= 'a' && ch2 <= 'z' )
                ch2 = ch2 - 'a' + 'A';
            if( ch1==ch2 )
                continue;
            return (int)(ch1 - ch2);
        }
        return 0;
    }

    // walk the prog to read the datetime string
    // return 0 if all ok
    __device__ int parse_into_parts( custring_view* d_string, int* timeparts )
    {
        unsigned int count = d_prog->count;
        DTFormatItem* items = d_prog->items;
        const char* ptr = d_string->data();
        unsigned int length = d_string->size();
        for( unsigned int idx=0; idx < count; ++idx )
        {
            DTFormatItem item = items[idx];
            int slen = (int)item.length;
            //printf("%d:%c=%d\n",(int)fmt.ftype,ch,(int)slen);
            if(item.item_type==false)
            {
                // consume fmt.len bytes from datetime
                // could also check ch matches and throw exception if it does not
                ptr += slen;
                length -= slen;
                continue;
            }
            if( length < slen )
                return 1;

            // special logic for each specifier
            switch(item.specifier)
            {
                case 'Y':
                    timeparts[TP_YEAR] = str2int(ptr,slen);
                    break;
                case 'y':
                    timeparts[TP_YEAR] = str2int(ptr,slen)+1900;
                    break;
                case 'm':
                    timeparts[TP_MONTH] = str2int(ptr,slen);
                    break;
                case 'd':
                case 'j':
                    timeparts[TP_DAY] = str2int(ptr,slen);
                    break;
                case 'H':
                case 'I':
                    timeparts[TP_HOUR] = str2int(ptr,slen);
                    break;
                case 'M':
                    timeparts[TP_MINUTE] = str2int(ptr,slen);
                    break;
                case 'S':
                    timeparts[TP_SECOND] = str2int(ptr,slen);
                    break;
                case 'f':
                    timeparts[TP_SUBSECOND] = str2int(ptr,slen);
                    break;
                case 'p':
                    if( timeparts[TP_HOUR] <= 12 && strcmp_ignore_case(ptr,"PM",2)==0 ) // strncasecmp
                        timeparts[TP_HOUR] += 12;
                    break;
                case 'z':
                {
                    int sign = *ptr=='-' ? -1:1;
                    int hh = str2int(ptr+1,2);
                    int mm = str2int(ptr+3,2);
                    // ignoring the rest for now
                    // slen has how many chars we should read
                    timeparts[TP_TZ_MINUTES] = sign * ((hh*60)+mm);
                    break;
                }
                case 'Z':
                    if( strcmp_ignore_case(ptr,"UTC",3)!=0 )
                        return 2;
                    break; // only recognize UTC
                default:
                    return 3;
            }
            //printf(">>%d:%d\n",part,timeparts[part]);
            ptr += slen;
            length -= slen;
        }
        return 0;
    }

    __device__ long timestamp_from_parts( int* timeparts, NVStrings::timestamp_units units )
    {
        int year = timeparts[TP_YEAR];
        if( units==NVStrings::timestamp_units::years )
            return year - 1970;
        int month = timeparts[TP_MONTH];
        if( units==NVStrings::timestamp_units::months )
            return ((year-1970) * 12) + (month-1); // months are 1-12, need to 0-base it here
        int day = timeparts[TP_DAY];
        // The months are shifted so that March is the starting month and February
        // (possible leap day in it) is the last month for the linear calculation
        year -= (month <= 2) ? 1 : 0;
        // date cycle repeats every 400 years (era)
        const int erasInDays = 146097;
        const int erasInYears = (erasInDays / 365);
        const int era = (year >= 0 ? year : year - 399) / erasInYears;
        const int yoe = year - era * erasInYears;
        const int doy = month==0 ? day : ((153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1);
        const int doe = (yoe * 365) + (yoe / 4) - (yoe / 100) + doy;
        int days = (era * erasInDays) + doe - 719468; // 719468 = days from 0000-00-00 to 1970-03-01
        if( units==NVStrings::timestamp_units::days )
            return days;

        int tzadjust = timeparts[TP_TZ_MINUTES]; // in minutes
        int hour = timeparts[TP_HOUR];
        if( units==NVStrings::timestamp_units::hours )
            return (days*24L) + hour + (tzadjust/60);

        int minute = timeparts[TP_MINUTE];
        if( units==NVStrings::timestamp_units::minutes )
            return (long)(days * 24L * 60L) + (hour * 60L) + minute + tzadjust;

        int second = timeparts[TP_SECOND];
        long timestamp = (days * 24L * 3600L) + (hour * 3600L) + (minute * 60L) + second + (tzadjust*60);
        if( units==NVStrings::timestamp_units::seconds )
            return timestamp;

        int subsecond = timeparts[TP_SUBSECOND];
        if( units==NVStrings::timestamp_units::ms )
            timestamp *= 1000L;
        else if( units==NVStrings::timestamp_units::us )
            timestamp *= 1000000L;
        else if( units==NVStrings::timestamp_units::ns )
            timestamp *= 1000000000L;
        timestamp += subsecond;
        return timestamp;
    }

     __device__ void operator()(unsigned int idx)
    {
        custring_view* dstr = d_strings[idx];
        if( (dstr==0) || dstr->empty() )
        {
            d_timestamps[idx] = 0;
            return;
        }
        //
        int timeparts[TP_ARRAYSIZE] = {0,1,1}; // month and day are 1-based
        if( parse_into_parts(dstr,timeparts) )
            d_timestamps[idx] = 0;
        else
            d_timestamps[idx] = timestamp_from_parts(timeparts,units);
    }
};

// convert date format into timestamp long integer
int NVStrings::timestamp2long( const char* format, timestamp_units units, unsigned long* results, bool bdevmem )
{
    unsigned int count = size();
    if( count==0 || results==0 )
        return -1;
    auto execpol = rmm::exec_policy(0);
    unsigned long* d_rtn = results;
    if( !bdevmem )
        d_rtn = device_alloc<unsigned long>(count,0);

    if( format==0 )
        format = "%Y-%m-%dT%H:%M:%SZ";
    size_t length = strlen(format);
    DTFormatCompiler dtc(format,length,units);
    DTProgram* prog = dtc.compile_to_device();

    custring_view** d_strings = pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        parse_datetime(prog,d_strings,units,d_rtn));
    //
    int zeros = thrust::count(execpol->on(0),d_rtn,d_rtn+count,0);
    if( !bdevmem )
    {
        CUDA_TRY( cudaMemcpyAsync(results,d_rtn,sizeof(unsigned long)*count,cudaMemcpyDeviceToHost))
        RMM_FREE(d_rtn,0);
    }
    return (int)count-zeros;
}

// converts long timestamp into date-time string
struct datetime_formatter
{
    unsigned long* d_timestamps;
    custring_view_array d_strings;
    unsigned char* d_nulls;
    char* d_buffer;
    size_t* d_offsets;
    NVStrings::timestamp_units units;
    DTProgram* d_prog;

    datetime_formatter( DTProgram* prog, NVStrings::timestamp_units units, char* buffer, size_t* offsets, unsigned char* nulls, unsigned long* timestamps, custring_view_array strings)
    : d_timestamps(timestamps), d_buffer(buffer), d_offsets(offsets), d_nulls(nulls), d_strings(strings), units(units), d_prog(prog) {}

    __device__ void dissect_timestamp( long timestamp, int* timeparts )
    {
        if( units==NVStrings::timestamp_units::years )
        {
            timeparts[TP_YEAR] = (int)timestamp + 1970;
            timeparts[TP_MONTH] = 1;
            timeparts[TP_DAY] = 1;
            return;
        }

        if( units==NVStrings::timestamp_units::months )
        {
            int month = timestamp % 12;
            int year = (timestamp / 12) + 1970;
            timeparts[TP_YEAR] = year;
            timeparts[TP_MONTH] = month +1; // months start at 1 and not 0
            timeparts[TP_DAY] = 1;
            return;
        }

        // first, convert to days so we can handle months, leap years, etc.
        int days = (int)timestamp;
        if( units==NVStrings::timestamp_units::hours )
            days = (int)(timestamp / 24L);
        else if( units==NVStrings::timestamp_units::minutes )
            days = (int)(timestamp / 1440L);  // 24*60
        else if( units==NVStrings::timestamp_units::seconds )
            days = (int)(timestamp / 86400L); // 24*60*60
        else if( units==NVStrings::timestamp_units::ms )
            days = (int)(timestamp / 86400000L);
        else if( units==NVStrings::timestamp_units::us )
            days = (int)(timestamp / 86400000000L);
        else if( units==NVStrings::timestamp_units::ns )
            days = (int)(timestamp / 86400000000000L);
        days = days + 719468; // 719468 is days between 0000-00-00 and 1970-01-01

        const int daysInEra = 146097; // (400*365)+97
        const int daysInCentury = 36524; // (100*365) + 24;
        const int daysIn4Years = 1461; // (4*365) + 1;
        const int daysInYear = 365;
        // day offsets for each month:   Mar Apr May June July  Aug  Sep  Oct  Nov  Dec  Jan  Feb
        const int monthDayOffset[] = { 0, 31, 61, 92, 122, 153, 184, 214, 245, 275, 306, 337, 366 };

        // code logic handles leap years in chunks: 400y,100y,4y,1y
        int year = 400 * (days / daysInEra);
        days = days % daysInEra;
        int leapy = days / daysInCentury;
        days = days % daysInCentury;
        if( leapy==4 )
        {   // landed exactly on a leap century
            days += daysInCentury;
            --leapy;
        }
        year += 100 * leapy;
        year += 4 * (days / daysIn4Years);
        days = days % daysIn4Years;
        leapy = days / daysInYear;
        days = days % daysInYear;
        if( leapy==4 )
        {   // landed exactly on a leap year
            days += daysInYear;
            --leapy;
        }
        year += leapy;

        //
        int month = 12;
        for( int idx=0; idx < month; ++idx )
        {   // find the month
            if( days < monthDayOffset[idx+1] )
            {
                month = idx;
                break;
            }
        }
        int day = days - monthDayOffset[month] +1; // compute day of month
        if( month >= 10 )
            ++year;
        month = ((month + 2) % 12) +1; // adjust Jan-Mar offset

        timeparts[TP_YEAR] = year;
        timeparts[TP_MONTH] = month;
        timeparts[TP_DAY] = day;
        if( units==NVStrings::timestamp_units::days )
            return;

        // done with date
        // now work on time
        long hour = timestamp, minute = timestamp, second = timestamp;
        if( units==NVStrings::timestamp_units::hours )
        {
            timeparts[TP_HOUR] = (int)(hour % 24);
            return;
        }
        hour = hour / 60;
        if( units==NVStrings::timestamp_units::minutes )
        {
            timeparts[TP_HOUR] = (int)(hour % 24);
            timeparts[TP_MINUTE] = (int)(minute % 60);
            return;
        }
        hour = hour / 60;
        minute = minute / 60;
        if( units==NVStrings::timestamp_units::seconds )
        {
            timeparts[TP_HOUR] = (int)(hour % 24);
            timeparts[TP_MINUTE] = (int)(minute % 60);
            timeparts[TP_SECOND] = (int)(second % 60);
            return;
        }
        hour = hour / 1000;
        minute = minute / 1000;
        second = second / 1000;
        if( units==NVStrings::timestamp_units::ms )
        {
            timeparts[TP_HOUR] = (int)(hour % 24);
            timeparts[TP_MINUTE] = (int)(minute % 60);
            timeparts[TP_SECOND] = (int)(second % 60);
            timeparts[TP_SUBSECOND] = (int)(timestamp % 1000);
            return;
        }
        hour = hour / 1000;
        minute = minute / 1000;
        second = second / 1000;
        if( units==NVStrings::timestamp_units::us )
        {
            timeparts[TP_HOUR] = (int)(hour % 24);
            timeparts[TP_MINUTE] = (int)(minute % 60);
            timeparts[TP_SECOND] = (int)(second % 60);
            timeparts[TP_SUBSECOND] = (int)(timestamp % 1000000);
            return;
        }
        hour = hour / 1000;
        minute = minute / 1000;
        second = second / 1000;
        timeparts[TP_HOUR] = (int)(hour % 24);
        timeparts[TP_MINUTE] = (int)(minute % 60);
        timeparts[TP_SECOND] = (int)(second % 60);
        timeparts[TP_SUBSECOND] = (int)(timestamp % 1000000000);
    }

    // utility to create 0-padded integers (up to 4 bytes)
    __device__ char* int2str( char* str, int len, int val )
    {
        char tmpl[9] = {'0','0','0','0','0','0','0','0','0'};
        char* ptr = tmpl;
        while( val > 0 )
        {
            int digit = val % 10;
            *ptr++ = '0' + digit;
            val = val / 10;
        }
        ptr = tmpl + len-1;
        while( len > 0 )
        {
            *str++ = *ptr--;
            --len;
        }
        return str;
    }

    __device__ char* format_from_parts( int* timeparts, char* ptr )
    {
        size_t count = d_prog->count;
        DTFormatItem* d_items = d_prog->items;
        for( size_t idx=0; idx < count; ++idx )
        {
            DTFormatItem item = d_items[idx];
            int slen = (int)item.length;
            //printf("%d:%c=%d\n",(int)fmt.ftype,ch,(int)slen);
            if(item.item_type==false)
            {
                *ptr++ = item.literal;
                continue;
            }
            // special logic for each specifier
            switch(item.specifier)
            {
                case 'Y':
                    ptr = int2str(ptr,slen,timeparts[TP_YEAR]);
                    break;
                case 'y':
                    ptr = int2str(ptr,slen,timeparts[TP_YEAR]-1900);
                    break;
                case 'm':
                    ptr = int2str(ptr,slen,timeparts[TP_MONTH]);
                    break;
                case 'd':
                case 'j':
                    ptr = int2str(ptr,slen,timeparts[TP_DAY]);
                    break;
                case 'H':
                    ptr = int2str(ptr,slen,timeparts[TP_HOUR]);
                    break;
                case 'I':
                    ptr = int2str(ptr,slen,timeparts[TP_HOUR] % 12);
                    break;
                case 'M':
                    ptr = int2str(ptr,slen,timeparts[TP_MINUTE]);
                    break;
                case 'S':
                    ptr = int2str(ptr,slen,timeparts[TP_SECOND]);
                    break;
                case 'f':
                    ptr = int2str(ptr,slen,timeparts[TP_SUBSECOND]);
                    break;
                case 'p':
                    if( timeparts[TP_HOUR] <= 12 )
                        memcpy(ptr,"AM",2);
                    else
                        memcpy(ptr,"PM",2);
                    ptr += 2;
                    break;
                case 'z':
                    break; // do nothing for this one
                case 'Z':
                    memcpy(ptr,"UTC",3);
                    ptr += 3;
                    break;
                default:
                    break;
            }
        }
        return ptr;
    }

    __device__ void operator()( unsigned int idx )
    {
        if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            d_strings[idx] = nullptr;
            return;
        }
        long timestamp = d_timestamps[idx];
        int timeparts[TP_ARRAYSIZE] = {0};
        dissect_timestamp(timestamp,timeparts);
        // convert to characters
        char* str = d_buffer + d_offsets[idx];
        char* ptr = format_from_parts(timeparts,str);
        int len = (int)(ptr - str);
        d_strings[idx] = custring_view::create_from(str,str,len);
    }
};


NVStrings* NVStrings::long2timestamp( const unsigned long* values, unsigned int count, timestamp_units units, const char* format, const unsigned char* nullbitmask, bool bdevmem )
{
    if( values==0 || count==0 )
        throw std::invalid_argument("nvstrings::long2timestamp values or count invalid");
    auto execpol = rmm::exec_policy(0);
    NVStrings* rtn = new NVStrings(count);

    unsigned long* d_values = (unsigned long*)values;
    unsigned char* d_nulls = (unsigned char*)nullbitmask;
    if( !bdevmem )
    {
        d_values = device_alloc<unsigned long>(count,0);
        CUDA_TRY( cudaMemcpyAsync(d_values,values,count*sizeof(unsigned long),cudaMemcpyHostToDevice))
        if( nullbitmask )
        {
            d_nulls = device_alloc<unsigned char>(((count+7)/8),0);
            CUDA_TRY( cudaMemcpyAsync(d_nulls,nullbitmask,((count+7)/8)*sizeof(unsigned char),cudaMemcpyHostToDevice))
        }
    }

    if( format==0 )
        format = "%Y-%m-%dT%H:%M:%SZ";
    size_t length = strlen(format);
    DTFormatCompiler dtc(format,length,units);
    DTProgram* prog = dtc.compile_to_device();

    // compute size of memory we'll need
    // each string will be the same size with the length
    int d_size = custring_view::alloc_size(dtc.string_template(),dtc.string_length());
    // we only need to account for any null strings
    rmm::device_vector<size_t> sizes(count,0);
    size_t* d_sizes = sizes.data().get();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        [d_values, d_size, d_nulls, d_sizes] __device__ (unsigned int idx) {
            if( d_nulls && ((d_nulls[idx/8] & (1 << (idx % 8)))==0) )
                d_sizes[idx] = 0;
            else
                d_sizes[idx] = ALIGN_SIZE(d_size);
        });

    rmm::device_vector<size_t> offsets(count,0);
    size_t* d_offsets = offsets.data().get();
    thrust::exclusive_scan(execpol->on(0),sizes.begin(),sizes.end(),offsets.begin());
    // build iso8601 strings from timestamps
    char* d_buffer = rtn->pImpl->createMemoryFor(d_sizes);
    custring_view_array d_strings = rtn->pImpl->getStringsPtr();
    thrust::for_each_n(execpol->on(0), thrust::make_counting_iterator<unsigned int>(0), count,
        datetime_formatter(prog, units,d_buffer, d_offsets, d_nulls, d_values, d_strings));
    //
    if( !bdevmem )
        RMM_FREE(d_values,0);
    return rtn;
}

