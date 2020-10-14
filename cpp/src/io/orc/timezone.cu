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
#include "timezone.cuh"

#include <algorithm>
#include <fstream>

#include <cudf/utilities/error.hpp>

namespace cudf {
namespace io {
// NOTE: Assumes little-endian platform

#define TZIF_MAGIC (('T' << 0) | ('Z' << 8) | ('i' << 16) | ('f' << 24))

#define ORC_UTC_OFFSET 1420070400  // Seconds from Jan 1st, 1970 to Jan 1st, 2015

#pragma pack(push, 1)

struct localtime_type_record_s {
  int32_t utcoff;    // number of seconds to be added to UTC in order to determine local time
  uint8_t isdst;     // 0:standard time, 1:Daylight Savings Time (DST)
  uint8_t desigidx;  // index into the series of time zone designation characters
};

/**
 * @brief 32-bit TZif header
 */
struct tzif_header {
  uint32_t magic;          ///< "TZif"
  uint8_t version;         ///< 0:version1, '2':version2, '3':version3
  uint8_t reserved15[15];  ///< unused, reserved for future use
  uint32_t isutccnt;       ///< number of UTC/local indicators contained in the body
  uint32_t isstdcnt;       ///< number of standard/wall indicators contained in the body
  uint32_t leapcnt;        ///< number of leap second records contained in the body
  uint32_t timecnt;        ///< number of transition times contained in the body
  uint32_t typecnt;  ///< number of local time type Records contained in the body - MUST NOT be zero
  uint32_t charcnt;  ///< total number of octets used by the set of time zone designations contained
                     ///< in the body
};

struct tzif {
  tzif_header header;
  bool is_header_from_64bit = false;

  std::vector<int64_t> transition_times;
  std::vector<uint8_t> ttime_idx;
  std::vector<localtime_type_record_s> ttype;
  std::vector<uint8_t> posix_tz_string;

  auto timecnt() const { return header.timecnt; }
  auto typecnt() const { return header.typecnt; }

  void read_header(std::ifstream &input_file, size_t file_size)
  {
    input_file.read(reinterpret_cast<char *>(&header), sizeof(header));
    CUDF_EXPECTS(!input_file.fail() && header.magic == TZIF_MAGIC,
                 "Error reading time zones file header.");

    // Convert fields to little endian
    header.isutccnt = __builtin_bswap32(header.isutccnt);
    header.isstdcnt = __builtin_bswap32(header.isstdcnt);
    header.leapcnt  = __builtin_bswap32(header.leapcnt);
    header.timecnt  = __builtin_bswap32(header.timecnt);
    header.typecnt  = __builtin_bswap32(header.typecnt);
    header.charcnt  = __builtin_bswap32(header.charcnt);

    // Check for 64-bit header
    if (header.version != 0) {
      size_t const ofs64 = header.timecnt * 5 + header.typecnt * 6 + header.charcnt +
                           header.leapcnt * 8 + header.isstdcnt + header.isutccnt;
      if (ofs64 + sizeof(header) < file_size) {
        input_file.seekg(ofs64, std::ios_base::cur);
        is_header_from_64bit = true;
        input_file.read(reinterpret_cast<char *>(&header), sizeof(header));
        // Convert fields to little endian
        header.isutccnt = __builtin_bswap32(header.isutccnt);
        header.isstdcnt = __builtin_bswap32(header.isstdcnt);
        header.leapcnt  = __builtin_bswap32(header.leapcnt);
        header.timecnt  = __builtin_bswap32(header.timecnt);
        header.typecnt  = __builtin_bswap32(header.typecnt);
        header.charcnt  = __builtin_bswap32(header.charcnt);
      }
    }
    CUDF_EXPECTS(
      header.typecnt > 0 && header.typecnt <= file_size / sizeof(localtime_type_record_s),
      "Invalid number number of time types in timezone file.");
    CUDF_EXPECTS(header.timecnt <= file_size,
                 "Number of transition times is larger than the file size.");
  }

  tzif(std::string const &timezone_name)
  {
    using std::ios_base;

    // Open the input file
    std::string const tz_filename = "/usr/share/zoneinfo/" + timezone_name;
    std::ifstream fin;
    fin.open(tz_filename, ios_base::in | ios_base::binary | ios_base::ate);
    CUDF_EXPECTS(fin, "Failed to open the timezone file.");
    auto const file_size = fin.tellg();
    fin.seekg(0);

    read_header(fin, file_size);

    // Read transition times (convert from 32-bit to 64-bit if necessary)
    transition_times.resize(timecnt());
    if (is_header_from_64bit) {
      fin.read(reinterpret_cast<char *>(transition_times.data()),
               transition_times.size() * sizeof(int64_t));
      for (auto &tt : transition_times) { tt = __builtin_bswap64(tt); }
    } else {
      std::vector<int32_t> tt32(timecnt());
      fin.read(reinterpret_cast<char *>(tt32.data()), tt32.size() * sizeof(int32_t));
      std::transform(
        tt32.cbegin(), tt32.cend(), std::back_inserter(transition_times), [](auto &tt) {
          return __builtin_bswap32(tt);
        });
    }
    ttime_idx.resize(timecnt());
    fin.read(reinterpret_cast<char *>(ttime_idx.data()), timecnt() * sizeof(uint8_t));

    // Read time types
    ttype.resize(typecnt());
    fin.read(reinterpret_cast<char *>(ttype.data()), typecnt() * sizeof(localtime_type_record_s));
    CUDF_EXPECTS(!fin.fail(), "Failed to read time types from the time zone file.");
    for (uint32_t i = 0; i < typecnt(); i++) {
      ttype[i].utcoff = __builtin_bswap32(ttype[i].utcoff);
    }

    // Read posix TZ string
    fin.seekg(header.charcnt + header.leapcnt * ((is_header_from_64bit) ? 12 : 8) +
                header.isstdcnt + header.isutccnt,
              ios_base::cur);
    auto const file_pos = fin.tellg();
    if (file_size - file_pos > 1) {
      posix_tz_string.resize(file_size - file_pos);
      fin.read(reinterpret_cast<char *>(posix_tz_string.data()), file_size - file_pos);
    }
  }
};

struct dst_transition_s {
  int type;   // Transition type ('J','M' or day)
  int month;  // Month of transition
  int week;   // Week of transition
  int day;    // Day of transition
  int time;   // Time of day
};
#pragma pack(pop)

/**
 * @brief Parse a name from the posix TZ string
 *
 * @param[in] cur current position in TZ string
 * @param[in] end end of TZ string
 *
 * @return position after parsing the name
 **/
static const uint8_t *posix_parse_name(const uint8_t *cur, const uint8_t *end)
{
  if (cur < end) {
    int c = *cur;
    if (c == '<') {
      cur++;
      while (cur < end) {
        if (*cur++ == '>') { break; }
      }
    } else {
      while ((c < '0' || c > '9') && (c != '-') && (c != '+') && (c != ',')) {
        if (++cur >= end) { break; }
        c = *cur;
      }
    }
  }
  return cur;
}

/**
 * @brief Parse a number from the posix TZ string
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
  while (cur < end) {
    uint32_t c = *cur - '0';
    if (c > 9u) { break; }
    v = v * 10 + c;
    cur++;
  }
  *pval = v;
  return cur;
}

/**
 * @brief Parse a UTC offset from the posix TZ string
 *
 * @param[in] cur current position in TZ string
 * @param[in] end end of TZ string
 * @param[out] putcoff pointer to UTC offset
 *
 * @return position after parsing the UTC offset
 **/
static const uint8_t *posix_parse_offset(const uint8_t *cur, const uint8_t *end, int32_t *putcoff)
{
  int64_t v = 0;
  if (cur < end) {
    auto scale = 60 * 60;
    int sign   = *cur;
    cur += (sign == '-' || sign == '+');
    cur = posix_parse_number(cur, end, &v);
    v *= scale;
    while (cur < end && scale > 1 && *cur == ':') {
      int64_t v2;
      cur = posix_parse_number(cur + 1, end, &v2);
      scale /= 60;
      v += v2 * scale;
    }
    if (sign == '-') { v = -v; }
  }
  *putcoff = v;
  return cur;
}

/**
 * @brief Parse a DST transition time from the posix TZ string
 *
 * @param[in] cur current position in TZ string
 * @param[in] end end of TZ string
 * @param[out] ptrans pointer to resulting transition
 *
 * @return position after parsing the transition
 **/
static const uint8_t *posix_parse_transition(const uint8_t *cur,
                                             const uint8_t *end,
                                             dst_transition_s *ptrans)
{
  int type     = 0;
  int month    = 0;
  int week     = 0;
  int day      = 0;
  int32_t time = 2 * 60 * 60;
  if (cur + 2 <= end && *cur == ',') {
    int64_t v;
    type = cur[1];
    cur += (type == 'M' || type == 'J') ? 2 : 1;
    if (type == 'M') {
      cur   = posix_parse_number(cur, end, &v);
      month = (int)v;
      if (cur < end && *cur == '.') {
        cur  = posix_parse_number(cur + 1, end, &v);
        week = (int)v;
        if (cur < end && *cur == '.') {
          cur = posix_parse_number(cur + 1, end, &v);
          day = (int)v;
        }
      }
    } else {
      cur = posix_parse_number(cur, end, &v);
      day = (int)v;
    }
    if (cur < end && *cur == '/') { cur = posix_parse_offset(cur + 1, end, &time); }
  }
  ptrans->type  = type;
  ptrans->month = month;
  ptrans->week  = week;
  ptrans->day   = day;
  ptrans->time  = time;
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
  int day   = trans->day;

  if (trans->type == 'M') {
    int is_leap = IsLeapYear(year);
    int month   = std::min(std::max(trans->month, 1), 12);
    int week    = std::min(std::max(trans->week, 1), 52);
    // Compute day of week
    int adjustedMonth = (month + 9) % 12 + 1;
    int adjustedYear  = year - (month <= 2);
    int dayOfWeek     = ((26 * adjustedMonth - 2) / 10 + 1 + (adjustedYear % 100) +
                     (adjustedYear % 100) / 4 + (adjustedYear / 400) - 2 * (adjustedYear / 100)) %
                    7;
    if (dayOfWeek < 0) { dayOfWeek += 7; }
    day -= dayOfWeek;
    if (day < 0) { day += 7; }
    while (week > 1 && day + 7 < DaysInMonth(month, is_leap)) {
      week--;
      day += 7;
    }
    for (int m = 1; m < month; m++) { day += DaysInMonth(m, is_leap); }
  } else if (trans->type == 'J') {
    day += (day > 60 && IsLeapYear(year));
  }
  return t + day * 24 * 60 * 60;
}

/**
 * @brief Returns the gmt offset for a given date
 *
 * @param[in] TODO
 * @param[in] ts ORC timestamp
 *
 * @return gmt offset
 **/
static int32_t GetGmtOffset(int64_t const *ttimes, int32_t const *offsets, size_t count, int64_t ts)
{
  uint32_t dst_cycle   = 800;
  uint32_t num_entries = (uint32_t)(count - dst_cycle);
  uint32_t first = 0, last = 0;

  auto const first_transition = ttimes[0];
  auto const last_transition  = ttimes[num_entries - 1];
  if (ts <= first_transition) {
    return offsets[0];
  } else if (ts <= last_transition) {
    first = 0;
    last  = num_entries - 1;
  } else {
    // Apply 400-year cycle rule
    const int64_t k400Years = (365 * 400 + (100 - 3)) * 24 * 60 * 60ll;
    ts %= k400Years;
    if (ts < 0) { ts += k400Years; }
    first = num_entries;
    last  = num_entries + dst_cycle - 1;
    if (ts < ttimes[num_entries]) { return offsets[last]; }
  }
  // Binary search the table from first to last for ts
  do {
    uint32_t mid = first + ((last - first + 1) >> 1);
    int64_t tmid = ttimes[mid];
    if (tmid <= ts) {
      first = mid;
    } else {
      if (mid == last) { break; }
      last = mid;
    }
  } while (first < last);
  return offsets[first];
}

timezone_table BuildTimezoneTransitionTable(std::string const &timezone_name)
{
  if (timezone_name == "UTC" || !timezone_name.length()) {
    // Return an empty table for UTC
    return {};
  }

  tzif const tz(timezone_name);

  // Allocate transition table, add one entry for ancient rule, and 800 entries for future rules
  // (2 transitions/year)
  std::vector<int64_t> ttimes(1 + (size_t)tz.timecnt() + 400 * 2);
  std::vector<int32_t> offsets(1 + (size_t)tz.timecnt() + 400 * 2);
  size_t earliest_std_idx = 0;
  for (size_t t = 0; t < tz.timecnt(); t++) {
    auto const ttime = tz.transition_times[t];
    auto const idx   = tz.ttime_idx[t];
    CUDF_EXPECTS(idx < tz.typecnt(), "Out-of-range type index");
    auto const utcoff = tz.ttype[idx].utcoff;
    ttimes[1 + t]     = ttime;
    offsets[1 + t]    = utcoff;
    if (!earliest_std_idx && !tz.ttype[idx].isdst) { earliest_std_idx = 1 + t; }
  }
  if (!earliest_std_idx) { earliest_std_idx = 1; }
  ttimes[0]  = ttimes[earliest_std_idx];
  offsets[0] = offsets[earliest_std_idx];

  // Generate entries for times after the last transition
  auto future_stdoff = offsets[tz.timecnt()];
  auto future_dstoff = future_stdoff;

  dst_transition_s dst_start{};
  dst_transition_s dst_end{};
  if (tz.posix_tz_string.size() > 0) {
    const uint8_t *cur = tz.posix_tz_string.data();
    const uint8_t *end = cur + tz.posix_tz_string.size();
    cur                = posix_parse_name(cur, end);
    cur                = posix_parse_offset(cur, end, &future_stdoff);
    future_stdoff      = -future_stdoff;
    if (cur + 1 < end) {
      // Parse Daylight Saving Time information
      cur = posix_parse_name(cur, end);
      if (cur < end && *cur != ',') {
        cur           = posix_parse_offset(cur, end, &future_dstoff);
        future_dstoff = -future_dstoff;
      } else {
        future_dstoff = future_stdoff + 60 * 60;
      }
      cur = posix_parse_transition(cur, end, &dst_start);
      cur = posix_parse_transition(cur, end, &dst_end);
    } else {
      future_dstoff = future_stdoff;
    }
  }

  // Add 2 entries per year for 400 years
  int64_t future_time = 0;
  for (size_t t = 0; t < 800; t += 2) {
    uint32_t const year          = 1970 + ((int)t >> 1);
    int64_t const dst_start_time = GetTransitionTime(&dst_start, year);
    int64_t const dst_end_time   = GetTransitionTime(&dst_end, year);
    auto const dst_idx           = 1 + tz.timecnt() + t;

    ttimes[dst_idx]      = future_time + dst_end_time - future_dstoff;
    offsets[dst_idx]     = future_stdoff;
    ttimes[dst_idx + 1]  = future_time + dst_start_time - future_stdoff;
    offsets[dst_idx + 1] = future_dstoff;
    if (dst_start_time < dst_end_time) {
      std::swap(ttimes[dst_idx], ttimes[dst_idx + 1]);
      std::swap(offsets[dst_idx], offsets[dst_idx + 1]);
    }

    future_time += (365 + IsLeapYear(year)) * 24 * 60 * 60;
  }
  // Add gmt offset
  timezone_table tz_table;
  tz_table.gmt_offset = GetGmtOffset(ttimes.data(), offsets.data(), ttimes.size(), ORC_UTC_OFFSET);
  tz_table.ttimes     = ttimes;
  tz_table.offsets    = offsets;

  return tz_table;
}

}  // namespace io
}  // namespace cudf
