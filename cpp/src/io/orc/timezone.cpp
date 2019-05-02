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
#include <fstream>
#include <algorithm>
#include "timezone.h"

// NOTE: Assumes little-endian platform
#ifdef _MSC_VER
#define bswap_32(v)    _byteswap_ulong(v)
#define bswap_64(v)    _byteswap_uint64(v)
#else
#define bswap_32(v)    __builtin_bswap32(v)
#define bswap_64(v)    __builtin_bswap64(v)
#endif

#define TZIF_MAGIC      (('T' << 0) | ('Z' << 8) | ('i' << 16) | ('f' << 24))

#define ORC_UTC_OFFSET  1420099200  // Seconds from Jan 1st, 1970 to Jan 1st, 2015

#pragma pack(push, 1)
 /**
  * @brief 32-bit TZif header
  **/
struct tzif_hdr_s
{
    uint32_t magic;         // "TZif"
    uint8_t version;        // 0:version1, '2':version2, '3':version3
    uint8_t reserved15[15]; // unused, reserved for future use
    uint32_t isutccnt;      // number of UTC/local indicators contained in the body
    uint32_t isstdcnt;      // number of standard/wall indicators contained in the body
    uint32_t leapcnt;       // number of leap second records contained in the body
    uint32_t timecnt;       // number of transition times contained in the body
    uint32_t typecnt;       // number of local time type Records contained in the body - MUST NOT be zero
    uint32_t charcnt;       // total number of octets used by the set of time zone designations contained in the body
};

struct localtime_type_record_s
{
    int32_t utcoff;         // number of seconds to be added to UTC in order to determine local time
    uint8_t isdst;          // 0:standard time, 1:Daylight Savings Time (DST)
    uint8_t desigidx;       // index into the series of time zone designation characters
};

struct dst_transition_s
{
    int type;               // Transition type ('J','M' or day)
    int month;              // Month of transition
    int week;               // Week of transition
    int day;                // Day of transition
    int time;               // Time of day
};
#pragma pack(pop)


/**
 * @brief Parse a name from the poxiz TZ string
 *
 * @param[in] cur current position in TZ string
 * @param[in] end end of TZ string
 *
 * @return position after parsing the name
 **/
static const uint8_t *posix_parse_name(const uint8_t *cur, const uint8_t *end)
{
    if (cur < end)
    {
        int c = *cur;
        if (c == '<')
        {
            cur++;
            while (cur < end)
            {
                if (*cur++ == '>')
                {
                    break;
                }
            }
        }
        else
        {
            while ((c < '0' || c > '9') && (c != '-') && (c != '+') && (c != ','))
            {
                if (++cur >= end)
                {
                    break;
                }
                c = *cur;
            }
        }
    }
    return cur;
}

/**
 * @brief Parse a number from the poxiz TZ string
 *
 * @param[in] cur current position in TZ string
 * @param[in] end end of TZ string
 * @param[out] pval pointer to result
 *
 * @return position after parsing the number
 **/
static const uint8_t *posix_parse_number(const uint8_t *cur, const uint8_t *end, int64_t *pval)
{
    int64_t v = 0;
    while (cur < end)
    {
        uint32_t c = *cur - '0';
        if (c > 9u)
        {
            break;
        }
        v = v * 10 + c;
        cur++;
    }
    *pval = v;
    return cur;
}

/**
 * @brief Parse a UTC offset from the poxiz TZ string
 *
 * @param[in] cur current position in TZ string
 * @param[in] end end of TZ string
 * @param[out] putcoff pointer to UTC offset
 *
 * @return position after parsing the UTC offset
 **/
static const uint8_t *posix_parse_offset(const uint8_t *cur, const uint8_t *end, int64_t *putcoff)
{
    int64_t v = 0;
    if (cur < end)
    {
        int64_t scale = 60*60;
        int sign = *cur;
        cur += (sign == '-' || sign == '+');
        cur = posix_parse_number(cur, end, &v);
        v *= scale;
        while (cur < end && scale > 1 && *cur == ':')
        {
            int64_t v2;
            cur = posix_parse_number(cur + 1, end, &v2);
            scale /= 60;
            v += v2 * scale;
        }
        if (sign == '-')
        {
            v = -v;
        }
    }
    *putcoff = v;
    return cur;
}

/**
 * @brief Parse a DST transition time from the poxiz TZ string
 *
 * @param[in] cur current position in TZ string
 * @param[in] end end of TZ string
 * @param[out] ptrans pointer to resulting transition
 *
 * @return position after parsing the transition
 **/
static const uint8_t *posix_parse_transition(const uint8_t *cur, const uint8_t *end, dst_transition_s *ptrans)
{
    int type = 0;
    int month = 0;
    int week = 0;
    int day = 0;
    int time = 2 * 60 * 60;
    if (cur + 2 <= end && *cur == ',')
    {
        int64_t v;
        type = cur[1];
        cur += (type == 'M' || type == 'J') ? 2 : 1;
        if (type == 'M')
        {
            cur = posix_parse_number(cur, end, &v);
            month = (int)v;
            if (cur < end && *cur == '.')
            {
                cur = posix_parse_number(cur + 1, end, &v);
                week = (int)v;
                if (cur < end && *cur == '.')
                {
                    cur = posix_parse_number(cur + 1, end, &v);
                    day = (int)v;
                }
            }
        }
        else
        {
            cur = posix_parse_number(cur, end, &v);
            day = (int)v;
        }
        if (cur < end && *cur == '/')
        {
            cur = posix_parse_offset(cur + 1, end, &v);
            time = (int)v;
        }
    }
    ptrans->type = type;
    ptrans->month = month;
    ptrans->week = week;
    ptrans->day = day;
    ptrans->time = time;
    return cur;
}

/**
 * @brief Check if a year is a leap year
 *
 * @param[in] year year
 *
 * @return 1 if leap year, zero otherwise
 **/
static int IsLeapYear(uint32_t year)
{
    return ((year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0)));
}

/**
 * @brief Return the number of days in a month
 *
 * @param[in] month month (1..12)
 * @param[in] is_leap 1 if leap year
 *
 * @return number of days in the month
 **/
static int DaysInMonth(int month, int is_leap)
{
    return (month == 2) ? 28 + is_leap : (30 + ((0x55aa >> month) & 1));
}

/**
 * @brief Convert a daylight saving transition time to a number of seconds
 *
 * @param[in] trans transition day information
 * @param[in] year year of transition
 *
 * @return transition time in seconds from the beginning of the year
 **/
static int64_t GetTransitionTime(const dst_transition_s *trans, int year)
{
    int64_t t = trans->time;
    int day = trans->day;

    if (trans->type == 'M')
    {
        static uint8_t month_lut[12] = {1,4,4,0,2,5,0,3,6,1,4,6};
        static uint8_t century_lut[4] = {6,4,2,0};
        int is_leap = IsLeapYear(year);
        int month = std::min(std::max(trans->month, 1), 12);
        int week = std::min(std::max(trans->week, 1), 52);
        // Compute day of week
        int adjustedMonth = (month + 9) % 12 + 1;
        int adjustedYear = year - (month <= 2);
        int dayOfWeek = ((26 * adjustedMonth - 2) / 10
                      + 1 + (adjustedYear % 100) + (adjustedYear % 100) / 4
                      + (adjustedYear / 400) - 2 * (adjustedYear / 100)) % 7;
        if (dayOfWeek < 0)
        {
            dayOfWeek += 7;
        }
        day -= dayOfWeek;
        if (day < 0)
        {
            day += 7;
        }
        while (week > 1 && day + 7 < DaysInMonth(month, is_leap))
        {
            week--;
            day += 7;
        }
        for (int m = 1; m < month; m++)
        {
            day += DaysInMonth(m, is_leap);
        }
    }
    else if (trans->type == 'J')
    {
        day += (day > 60 && IsLeapYear(year));
    }
    return t + day * 24 * 60 * 60;
}


/**
 * @brief Creates a transition table to convert ORC timestanps to UTC
 *
 * @param[out] table output table (2 int64_t per transition, last 800 transitions repeat forever with 400 year cycle)
 * @param[in] timezone_name standard timezone name (for example, "US/Pacific")
 *
 * @return true if successful, false if failed to find/parse the timezone information
 **/
bool BuildTimezoneTransitionTable(std::vector<int64_t> &table, const std::string &timezone_name)
{
    using std::ios_base;
    std::string tz_filename("/usr/share/zoneinfo/");
    std::ifstream fin;
    std::vector<localtime_type_record_s> ttype;
    std::vector<int64_t> transition_times;
    std::vector<uint8_t> ttime_idx;
    std::vector<uint8_t> posix_tz_string;
    tzif_hdr_s tzh = {0};
    bool hdr64 = false;
    size_t earliest_std_idx;
    int64_t future_dstoff, future_stdoff, future_time;

    table.resize(0);
    if (timezone_name == "UTC" || !timezone_name.length())
    {
        // Return an empty table for UTC
        return true;
    }
    tz_filename += timezone_name;
#ifdef _MSC_VER
    for (size_t i = 0; i < tz_filename.length(); i++)
    {
        if (tz_filename[i] == '/')
        {
            tz_filename[i] = '\\';
        }
    }
#endif
    fin.open(tz_filename, ios_base::in | ios_base::binary | ios_base::ate);
    if (fin)
    {
        size_t file_size = fin.tellg(), file_pos;
        dst_transition_s dst_start = {0}, dst_end = {0};
        fin.seekg(0);
        fin.read(reinterpret_cast<char *>(&tzh), sizeof(tzh));
        if (fin.fail() || tzh.magic != TZIF_MAGIC)
        {
            return false;
        }
        // Convert fields to little endian
        tzh.isutccnt = bswap_32(tzh.isutccnt);
        tzh.isstdcnt = bswap_32(tzh.isstdcnt);
        tzh.leapcnt = bswap_32(tzh.leapcnt);
        tzh.timecnt = bswap_32(tzh.timecnt);
        tzh.typecnt = bswap_32(tzh.typecnt);
        tzh.charcnt = bswap_32(tzh.charcnt);
        // Check for 64-bit header
        if (tzh.version != 0)
        {
            size_t ofs64 = tzh.timecnt * 5 + tzh.typecnt * 6 + tzh.charcnt + tzh.leapcnt * 8 + tzh.isstdcnt + tzh.isutccnt;
            if (ofs64 + sizeof(tzh) < file_size)
            {
                fin.seekg(ofs64, ios_base::cur);
                hdr64 = true;
                if (fin.fail() || tzh.magic != TZIF_MAGIC)
                {
                    return false;
                }
                fin.read(reinterpret_cast<char *>(&tzh), sizeof(tzh));
                // Convert fields to little endian
                tzh.isutccnt = bswap_32(tzh.isutccnt);
                tzh.isstdcnt = bswap_32(tzh.isstdcnt);
                tzh.leapcnt = bswap_32(tzh.leapcnt);
                tzh.timecnt = bswap_32(tzh.timecnt);
                tzh.typecnt = bswap_32(tzh.typecnt);
                tzh.charcnt = bswap_32(tzh.charcnt);
            }
        }
        // Read transition times (convert from 32-bit to 64-bit if necessary)
        if (tzh.timecnt > 0)
        {
            if (tzh.timecnt > file_size)
            {
                return false;
            }
            transition_times.resize(tzh.timecnt);
            ttime_idx.resize(tzh.timecnt);
            if (hdr64)
            {
                int64_t *tt64 = transition_times.data();
                fin.read(reinterpret_cast<char *>(tt64), tzh.timecnt * sizeof(int64_t));
                for (uint32_t i = 0; i < tzh.timecnt; i++)
                {
                    tt64[i] = bswap_64(tt64[i]);
                }
            }
            else
            {
                int64_t *tt64 = transition_times.data();
                int32_t *tt32 = reinterpret_cast<int32_t *>(tt64);
                fin.read(reinterpret_cast<char *>(tt32), tzh.timecnt * sizeof(int32_t));
                for (uint32_t i = tzh.timecnt; i > 0; )
                {
                    --i;
                    tt64[i] = (int32_t)bswap_32(tt32[i]);
                }
            }
            fin.read(reinterpret_cast<char *>(ttime_idx.data()), tzh.timecnt * sizeof(uint8_t));
        }
        // Read time types
        if (tzh.typecnt <= 0 || tzh.typecnt > file_size / sizeof(localtime_type_record_s))
        {
            return false;
        }
        else
        {
            ttype.resize(tzh.typecnt);
            fin.read(reinterpret_cast<char *>(ttype.data()), tzh.typecnt * sizeof(localtime_type_record_s));
            if (fin.fail())
            {
                return false;
            }
            for (uint32_t i = 0; i < tzh.typecnt; i++)
            {
                ttype[i].utcoff = (int32_t)bswap_32(ttype[i].utcoff);
            }
        }
        // Read posix TZ string
        fin.seekg(tzh.charcnt + tzh.leapcnt * ((hdr64) ? 12 : 8) + tzh.isstdcnt + tzh.isutccnt, ios_base::cur);
        file_pos = fin.tellg();
        if (file_pos + 1 < file_size)
        {
            posix_tz_string.resize(file_size - file_pos);
            fin.read(reinterpret_cast<char *>(posix_tz_string.data()), file_size - file_pos);
        }
        fin.close();
        // Allocate transition table, add one entry for ancient rule, and 801 entries for future rules (2 transitions/year)
        table.resize((1 + (size_t)tzh.timecnt + 400 * 2 + 1) * 2);
        earliest_std_idx = 0;
        for (size_t t = 0; t < tzh.timecnt; t++)
        {
            int64_t ttime = transition_times[t];
            int64_t utcoff;
            uint32_t idx = ttime_idx[t];
            if (idx >= tzh.typecnt)
            {
                // Out-of-range type index
                return false;
            }
            utcoff = ttype[idx].utcoff;
            table[(1 + t) * 2 + 0] = ttime;
            table[(1 + t) * 2 + 1] = utcoff;
            if (!earliest_std_idx && !ttype[idx].isdst)
            {
                earliest_std_idx = 1 + t;
            }
        }
        if (!earliest_std_idx)
        {
            earliest_std_idx = 1;
        }
        table[0] = table[earliest_std_idx * 2 + 0];
        table[1] = table[earliest_std_idx * 2 + 1];
        // Generate entries for times after the last transition
        future_stdoff = table[(tzh.timecnt * 2) + 1];
        future_dstoff = future_stdoff;
        if (posix_tz_string.size() > 0)
        {
            const uint8_t *cur = posix_tz_string.data();
            const uint8_t *end = cur + posix_tz_string.size();
            cur = posix_parse_name(cur, end);
            cur = posix_parse_offset(cur, end, &future_stdoff);
            future_stdoff = -future_stdoff;
            if (cur + 1 < end)
            {
                // Parse Daylight Saving Time information
                cur = posix_parse_name(cur, end);
                if (cur < end && *cur != ',')
                {
                    cur = posix_parse_offset(cur, end, &future_dstoff);
                    future_dstoff = -future_dstoff;
                }
                else
                {
                    future_dstoff = future_stdoff + 60 * 60;
                }
                cur = posix_parse_transition(cur, end, &dst_start);
                cur = posix_parse_transition(cur, end, &dst_end);
            }
            else
            {
                future_dstoff = future_stdoff;
            }
        }
        // Add 2 entries per year for 400 years
        future_time = 0;
        for (size_t t = 0; t < 800; t += 2)
        {
            uint32_t year = 1970 + ((int)t >> 1);
            int64_t dst_start_time = GetTransitionTime(&dst_start, year);
            int64_t dst_end_time = GetTransitionTime(&dst_end, year);
            if (dst_start_time < dst_end_time)
            {
                table[(1 + tzh.timecnt + t) * 2 + 0] = future_time + dst_start_time - future_stdoff;
                table[(1 + tzh.timecnt + t) * 2 + 1] = future_dstoff;
                table[(1 + tzh.timecnt + t) * 2 + 2] = future_time + dst_end_time - future_dstoff;
                table[(1 + tzh.timecnt + t) * 2 + 3] = future_stdoff;
            }
            else
            {
                table[(1 + tzh.timecnt + t) * 2 + 0] = future_time + dst_end_time - future_dstoff;
                table[(1 + tzh.timecnt + t) * 2 + 1] = future_stdoff;
                table[(1 + tzh.timecnt + t) * 2 + 2] = future_time + dst_start_time - future_stdoff;
                table[(1 + tzh.timecnt + t) * 2 + 3] = future_dstoff;
            }
            future_time += (365 + IsLeapYear(year)) * 24 * 60 * 60;
        }
    }
    else
    {
        //printf("Failed to open \"%s\"\n", tz_filename.c_str());
        return false;
    }
    return true;
}

