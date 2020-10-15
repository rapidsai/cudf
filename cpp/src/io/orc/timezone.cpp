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

namespace cudf {
namespace io {

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
struct timezone_file_header {
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

struct timezone_file {
  timezone_file_header header;
  bool is_header_from_64bit = false;

  std::vector<int64_t> transition_times;
  std::vector<uint8_t> ttime_idx;
  std::vector<localtime_type_record_s> ttype;
  std::vector<char> posix_tz_string;

  auto timecnt() const { return header.timecnt; }
  auto typecnt() const { return header.typecnt; }

  void header_to_little_endian()
  {
    header.isutccnt = __builtin_bswap32(header.isutccnt);
    header.isstdcnt = __builtin_bswap32(header.isstdcnt);
    header.leapcnt  = __builtin_bswap32(header.leapcnt);
    header.timecnt  = __builtin_bswap32(header.timecnt);
    header.typecnt  = __builtin_bswap32(header.typecnt);
    header.charcnt  = __builtin_bswap32(header.charcnt);
  }

  void read_header(std::ifstream &input_file, size_t file_size)
  {
    input_file.read(reinterpret_cast<char *>(&header), sizeof(header));
    CUDF_EXPECTS(!input_file.fail() && header.magic == TZIF_MAGIC,
                 "Error reading time zones file header.");
    header_to_little_endian();

    // Check for 64-bit header
    if (header.version != 0) {
      size_t const ofs64 = header.timecnt * 5 + header.typecnt * 6 + header.charcnt +
                           header.leapcnt * 8 + header.isstdcnt + header.isutccnt;
      if (ofs64 + sizeof(header) < file_size) {
        input_file.seekg(ofs64, std::ios_base::cur);
        is_header_from_64bit = true;
        input_file.read(reinterpret_cast<char *>(&header), sizeof(header));
        header_to_little_endian();
      }
    }
    CUDF_EXPECTS(
      header.typecnt > 0 && header.typecnt <= file_size / sizeof(localtime_type_record_s),
      "Invalid number number of time types in timezone file.");
    CUDF_EXPECTS(header.timecnt <= file_size,
                 "Number of transition times is larger than the file size.");
  }

  timezone_file(std::string const &timezone_name)
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
      fin.read(posix_tz_string.data(), file_size - file_pos);
    }
  }
};

struct dst_transition_s {
  char type;  // Transition type ('J','M' or day)
  int month;  // Month of transition
  int week;   // Week of transition
  int day;    // Day of transition
  int time;   // Time of day (seconds)
};
#pragma pack(pop)

/**
 * @brief Posix TZ parser
 */
template <class Container>
class posix_parser {
 public:
  posix_parser(Container const &tz_string) : cur{tz_string.begin()}, end{tz_string.end()} {}

  /**
   * @brief Advances the parser past a name from the posix TZ string.
   */
  void skip_name();

  /**
   * @brief Parses a number from the posix TZ string.
   *
   * @return Parsed number
   */
  int64_t parse_number();

  /**
   * @brief Parses a UTC offset from the posix TZ string.
   *
   * @return Parsed offset
   */
  int32_t parse_offset();

  /**
   * @brief Parses a DST transition time from the posix TZ string.
   *
   * @return Parsed transition time
   */
  dst_transition_s parse_transition();

  /**
   * @brief Returns the remaining number of characters in the input.
   */
  auto remaining_char_cnt() const { return end - cur; }

  /**
   * @brief Returns the next character in the input.
   */
  char next_character() const { return *cur; }

 private:
  typename Container::const_iterator cur;
  typename Container::const_iterator const end;
};

template <class Container>
void posix_parser<Container>::skip_name()
{
  if (cur < end) {
    auto c = *cur;
    if (c == '<') {
      ++cur;
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
}

template <class Container>
int64_t posix_parser<Container>::parse_number()
{
  int64_t v = 0;
  while (cur < end) {
    auto const c = *cur - '0';
    if (c > 9u) { break; }
    v = v * 10 + c;
    ++cur;
  }
  return v;
}

template <class Container>
int32_t posix_parser<Container>::parse_offset()
{
  if (cur < end) {
    auto scale      = 60 * 60;
    auto const sign = *cur;
    cur += (sign == '-' || sign == '+');
    auto v = parse_number();
    v *= scale;
    while (cur < end && scale > 1 && *cur == ':') {
      ++cur;
      auto const v2 = parse_number();
      scale /= 60;
      v += v2 * scale;
    }
    if (sign == '-') { v = -v; }
    return v;
  }
  return 0;
}

template <class Container>
dst_transition_s posix_parser<Container>::parse_transition()
{
  if (cur + 2 <= end && *cur == ',') {
    char const type = cur[1];
    int month       = 0;
    int week        = 0;
    int day         = 0;
    cur += (type == 'M' || type == 'J') ? 2 : 1;
    if (type == 'M') {
      month = parse_number();
      if (cur < end && *cur == '.') {
        ++cur;
        week = parse_number();
        if (cur < end && *cur == '.') {
          ++cur;
          day = parse_number();
        }
      }
    } else {
      day = parse_number();
    }
    int32_t time = 2 * 60 * 60;
    if (cur < end && *cur == '/') {
      ++cur;
      time = parse_offset();
    }
    return {type, month, week, day, time};
  }
  return {};
}

/**
 * @brief Checks if a given year is a leap year.
 */
static bool is_leap_year(uint32_t year)
{
  return ((year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0)));
}

/**
 * @brief Returns the number of days in a month.
 */
static int days_in_month(int month, bool is_leap_year)
{
  CUDF_EXPECTS(month > 0 && month <= 12, "Invalid month");

  if (month == 2) return 28 + is_leap_year;
  return 30 + ((0b1010110101010 >> month) & 1);
}

/**
 * @brief Converts a daylight saving transition time to a number of seconds.
 *
 * @param trans transition day information
 * @param year year of transition
 *
 * @return transition time in seconds from the beginning of the year
 */
static int64_t get_transition_time(dst_transition_s const &trans, int year)
{
  auto day = trans.day;

  if (trans.type == 'M') {
    auto const is_leap = is_leap_year(year);
    auto const month   = std::min(std::max(trans.month, 1), 12);
    auto week          = std::min(std::max(trans.week, 1), 52);

    // Compute day of week
    auto const adjusted_month = (month + 9) % 12 + 1;
    auto const adjusted_year  = year - (month <= 2);
    auto day_of_week =
      ((26 * adjusted_month - 2) / 10 + 1 + (adjusted_year % 100) + (adjusted_year % 100) / 4 +
       (adjusted_year / 400) - 2 * (adjusted_year / 100)) %
      7;
    if (day_of_week < 0) { day_of_week += 7; }
    day -= day_of_week;
    if (day < 0) { day += 7; }
    while (week > 1 && day + 7 < days_in_month(month, is_leap)) {
      week--;
      day += 7;
    }
    for (int m = 1; m < month; m++) { day += days_in_month(m, is_leap); }
  } else if (trans.type == 'J') {
    day += (day > 60 && is_leap_year(year));
  }

  return trans.time + day * 24 * 60 * 60;
}

timezone_table build_timezone_transition_table(std::string const &timezone_name)
{
  if (timezone_name == "UTC" || timezone_name.empty()) {
    // Return an empty table for UTC
    return {};
  }

  timezone_file const tzf(timezone_name);

  std::vector<int64_t> ttimes(1);
  std::vector<int32_t> offsets(1);
  // One ancient rule entry, one per TZ file entry, 2 entries per year in the future cycle
  ttimes.reserve(1 + tzf.timecnt() + cycle_entry_cnt);
  offsets.reserve(1 + tzf.timecnt() + cycle_entry_cnt);
  size_t earliest_std_idx = 0;
  for (size_t t = 0; t < tzf.timecnt(); t++) {
    auto const ttime = tzf.transition_times[t];
    auto const idx   = tzf.ttime_idx[t];
    CUDF_EXPECTS(idx < tzf.typecnt(), "Out-of-range type index");
    auto const utcoff = tzf.ttype[idx].utcoff;
    ttimes.push_back(ttime);
    offsets.push_back(utcoff);
    if (!earliest_std_idx && !tzf.ttype[idx].isdst) { earliest_std_idx = ttimes.size() - 1; }
  }
  if (!earliest_std_idx) { earliest_std_idx = 1; }
  ttimes[0]  = ttimes[earliest_std_idx];
  offsets[0] = offsets[earliest_std_idx];

  // Generate entries for times after the last transition
  auto future_std_offset = offsets[tzf.timecnt()];
  auto future_dst_offset = future_std_offset;
  dst_transition_s dst_start{};
  dst_transition_s dst_end{};
  if (!tzf.posix_tz_string.empty()) {
    posix_parser<decltype(tzf.posix_tz_string)> parser(tzf.posix_tz_string);
    parser.skip_name();
    future_std_offset = -parser.parse_offset();
    if (parser.remaining_char_cnt() > 1) {
      // Parse Daylight Saving Time information
      parser.skip_name();
      if (parser.remaining_char_cnt() > 0 && parser.next_character() != ',') {
        future_dst_offset = -parser.parse_offset();
      } else {
        future_dst_offset = future_std_offset + 60 * 60;
      }
      dst_start = parser.parse_transition();
      dst_end   = parser.parse_transition();
    } else {
      future_dst_offset = future_std_offset;
    }
  }

  // Add entries to fill the transition cycle
  int64_t year_timestamp = 0;
  for (uint32_t year = 1970; year < 1970 + cycle_entry_cnt; ++year) {
    int64_t const dst_start_time = get_transition_time(dst_start, year);
    int64_t const dst_end_time   = get_transition_time(dst_end, year);

    // Two entries per year, since there are two transitions
    ttimes.push_back(year_timestamp + dst_end_time - future_dst_offset);
    offsets.push_back(future_std_offset);
    ttimes.push_back(year_timestamp + dst_start_time - future_std_offset);
    offsets.push_back(future_dst_offset);

    // Swap the newly added transitions if in descending order
    if (ttimes.rbegin()[1] > ttimes.rbegin()[0]) {
      std::swap(ttimes.rbegin()[0], ttimes.rbegin()[1]);
      std::swap(offsets.rbegin()[0], offsets.rbegin()[1]);
    }

    year_timestamp += (365 + is_leap_year(year)) * 24 * 60 * 60;
  }

  return {get_gmt_offset(ttimes, offsets, ORC_UTC_OFFSET), ttimes, offsets};
}

}  // namespace io
}  // namespace cudf
