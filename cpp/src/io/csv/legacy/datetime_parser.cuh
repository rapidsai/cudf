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

/**
 * Formats
 * - Day and Months can be single or two digits
 * - Time separator can be either 'T' or a single space
 * -
 *
 * 												(start at 0)
	ID		Format					Size		First Sep Idx		Second Sep		'T' Sep		Time Sep 1			Time Sep 2
 	01		06/2018					6 - 7			1, 2			-				-					-					-
    02		06-2018					6 - 7			1, 2			-				-					-					-
    03		2018/06					6 - 7			4				-				-					-					-
    04		2018-06					6 - 7			4				-				-					-					-
    05		06/01/2018				8 - 10			1, 2			3, 4, 5			-					-					-
    06		06-01-2018				8 - 10			1, 2			3, 4, 5			-					-					-
    07		01/06/2018				8 - 10			1, 2			3, 4, 5			-					-					-
    08		01-06-2018				8 - 10			1, 2			3, 4, 5			-					-					-
    09		2018/06/01				8 - 10			4				6, 7			-					-					-
    10		2018-06-01				8 - 10			4				6, 7			-					-					-
    11		06/01/2018T10:16		12 - 16			1, 2			3, 4, 5			8, 9, 10		10, 11, 12, 13			-
    12		06-01-2018T10:16		12 - 16			1, 2			3, 4, 5			8. 9. 10		10, 11, 12, 13			-
    13		01/06/2018T10:16		12 - 16			1, 2			3, 4, 5			8, 9, 10		10, 11, 12, 13			-
    14		01-06-2018T10:16		12 - 16			1, 2			3, 4, 5			8, 9, 10		10, 11, 12, 13			-
    15		2018/06/01T10:16		12 - 16			4				6, 7			8, 9, 10		10, 11, 12, 13			-
    16		2018-06-01T10:16		12 - 16			4				6, 7			8, 9, 10		10, 11, 12, 13			-
    17		06/01/2018T10:16:12		14 - 19			1, 2			3, 4, 5			8, 9, 10		10, 11, 12, 13		11, 12, 13, 14, 15
    18		06-01-2018T10:16:12		14 - 19			1, 2			3, 4, 5			8, 9, 10		10, 11, 12, 13		11, 12, 13, 14, 15
    19		01/06/2018T10:16:12		14 - 19			1, 2			3, 4, 5			8, 9, 10		10, 11, 12, 13		11, 12, 13, 14, 15
    20		01-06-2018T10:16:12		14 - 19			1, 2			3, 4, 5			8, 9, 10		10, 11, 12, 13		11, 12, 13, 14, 15
    21		2018/06/01T10:16:12		14 - 19			4				6, 7			8, 9, 10		10, 11, 12, 13		11, 12, 13, 14, 15
    22		2018-06-01T10:16:12		14 - 19			4				6, 7			8, 9, 10		10, 11, 12, 13		11, 12, 13, 14, 15
    23		2018/06/01T10:16:12AM	14 - 19			4				6, 7			8, 9, 10		10, 11, 12, 13		11, 12, 13, 14, 15
    24		2018-06-01T10:16:12PM	14 - 19			4				6, 7			8, 9, 10		10, 11, 12, 13		11, 12, 13, 14, 15




 * What is not included here
 *  - Time zones and offsets
 *  - decimal fractions of seconds
 *
 *  Style:
 *  	date argument order is always year, month, day  and/or hour, minute, second
 *
 */


#pragma once

#include <cudf/cudf.h>

__inline__ __device__ bool extractDate(const char *data, long sIdx, long eIdx,
                            bool dayfirst, int *year, int *month, int *day);
__inline__ __device__ void extractTime(const char *data, long start, long end, int *hour,
                            int *minute, int *second, int *millisecond);

__inline__ __device__ constexpr int32_t daysSinceEpoch(int year, int month, int day);
__inline__ __device__ constexpr int64_t secondsSinceEpoch(int year, int month, int day,
                                               int hour, int minute,
                                               int second);

/**---------------------------------------------------------------------------*
 * @brief Simplified parsing function for use by date and time parsing
 *
 * This helper function is only intended to handle positive integers. The input
 * character string is expected to be well-formed.
 *
 * @param[in] data The character string for parse
 * @param[in] start The index within data to start parsing from
 * @param[in] end The end index within data to end parsing
 *
 * @return The parsed and converted value
 *---------------------------------------------------------------------------**/
template <typename T>
__inline__  __device__ T convertStrToInteger(const char *data, long start, long end) {
  T value = 0;

  long index = start;
  while (index <= end) {
    if (data[index] >= '0' && data[index] <= '9') {
      value *= 10;
      value += data[index] - '0';
    }
    ++index;
  }

  return value;
}

/**
 * @brief Returns location to the first occurrence of a character in a string
 *
 * This helper function takes a string and a search range to return the location
 * of the first instance of the specified character.
 *
 * @param[in] data 		Pointer to the data block
 * @param[in] start_idx Starting index within the data block
 * @param[in] end_idx 	Ending index within the data block
 * @param[in] c 		Character to find
 *
 * @return index into the string, or -1 if the character is not found
 */
__inline__ __device__ long findFirstOccurrence(const char *data, long start_idx,
                                    long end_idx, char c) {
  for (long i = start_idx; i <= end_idx; ++i) {
    if (data[i] == c) {
      return i;
    }
  }

  return -1;
}

/**
 * @brief Parse a Date string into a date32, days since epoch
 *
 * This function takes a string and produces a date32 representation
 * Acceptable formats are a combination of MM/YYYY and MM/DD/YYYY
 *
 * @param[in] data 		Pointer to the data block
 * @param[in] start_idx Starting index within the data block
 * @param[in] end_idx 	Ending index within the data block
 * @param[in] dayfirst 	Flag to indicate that day is the first field - DD/MM/YYYY
 *
 * @return returns the number of days since epoch
 */
__inline__ __device__
int32_t parseDateFormat(const char *data, long start_idx, long end_idx, bool dayfirst) {

	int day, month, year;
	int32_t e = -1;

	bool status = extractDate(data, start_idx, end_idx, dayfirst, &year, &month, &day);

	if ( status )
		e = daysSinceEpoch(year, month, day);

	return e;
}

/**
 * @brief Parses a datetime character stream and computes the number of
 * milliseconds since epoch. 
 * 
 * This function takes a string and produces a date32 representation
 * Acceptable formats are a combination of MM/YYYY and MM/DD/YYYY
 *
 * @param[in] data The character stream to parse
 * @param[in] start The start index of the character stream
 * @param[in] end The end index of the character stream
 * @param[in] dayfirst Flag to indicate day/month or month/day order
 * 
 * @return Milliseconds since epoch
 */
__inline__ __device__ int64_t parseDateTimeFormat(const char *data, long start,
                                          long end, bool dayfirst) {
  int day, month, year;
  int hour, minute, second, millisecond = 0;
  int64_t answer = -1;

  // Find end of the date portion
  // TODO: Refactor all the date/time parsing to remove multiple passes over
  // each character because of find() then convert(); that can also avoid the
  // ugliness below.
  auto sep_pos = findFirstOccurrence(data, start, end, 'T');
  if (sep_pos == -1) {
    // Attempt to locate the position between date and time, ignore premature
    // space separators around the day/month/year portions
    int count = 0;
    for (long i = start; i <= end; ++i) {
      if (count == 3 && data[i] == ' ') {
        sep_pos = i;
        break;
      } else if ((data[i] == '/' || data[i] == '-') ||
                 (count == 2 && data[i] != ' ')) {
        count++;
      }
    }
  }

  // There is only date if there's no separator, otherwise it's malformed
  if (sep_pos != -1) {
    if (extractDate(data, start, sep_pos - 1, dayfirst, &year, &month, &day)) {
      extractTime(data, sep_pos + 1, end, &hour, &minute, &second, &millisecond);
      answer = secondsSinceEpoch(year, month, day, hour, minute, second) * 1000 + millisecond;
    }
  } else {
    if (extractDate(data, start, end, dayfirst, &year, &month, &day)) {
      answer = secondsSinceEpoch(year, month, day, 0, 0, 0) * 1000;
    }
  }

  return answer;
}

/**
 * @brief Extract the Day, Month, and Year from a string
 *
 * @param[in] data	Pointer to data block
 * @param[in] sIdx	String index within data block
 * @param[in] eIdx	Ending index within data block
 * @param[in] dayfirst	Flag indicating that first field is the day
 *
 * @param[out] year
 * @param[out] month
 * @param[out] day
 *
 * @return T/F - false indicates that an error occurred
 */
__inline__ __device__
bool extractDate(const char *data, long sIdx, long eIdx, bool dayfirst, int *year, int *month, int *day) {

	char sep = '/';

	long sep_pos = findFirstOccurrence(data, sIdx, eIdx, sep);

	if ( sep_pos == -1 ) {
		sep = '-';
		sep_pos = findFirstOccurrence(data, sIdx, eIdx, sep);
	}

	if ( sep_pos == -1)
		return false;

	//--- is year the first filed?
	if ( (sep_pos - sIdx) == 4  ) {

		*year = convertStrToInteger<int>(data, sIdx, (sep_pos -1) );

		// Month
		long s2 = sep_pos +1;
		sep_pos = findFirstOccurrence(data, s2, eIdx, sep);

		if (sep_pos == -1 ) {

			//--- Data is just Year and Month - no day
			*month = convertStrToInteger<int>(data, s2, eIdx );
			*day = 1;

		} else {
			*month = convertStrToInteger<int>(data, s2, (sep_pos -1) );
			*day = convertStrToInteger<int>(data, (sep_pos + 1), eIdx);
		}

	} else {

		//--- if the dayfirst flag is set, then restricts the format options
		if ( dayfirst) {

			*day = convertStrToInteger<int>(data, sIdx, (sep_pos -1) );

			long s2 = sep_pos +1;
			sep_pos = findFirstOccurrence(data, s2, eIdx, sep);

			*month = convertStrToInteger<int>(data, s2, (sep_pos -1) );
			*year = convertStrToInteger<int>(data, (sep_pos + 1), eIdx);

		} else {

			*month = convertStrToInteger<int>(data, sIdx, (sep_pos -1) );

			long s2 = sep_pos +1;
			sep_pos = findFirstOccurrence(data, s2, eIdx, sep);

			if (sep_pos == -1 )
			{
				//--- Data is just Year and Month - no day
				*year = convertStrToInteger<int>(data, s2, eIdx );
				*day = 1;

			} else {
				*day = convertStrToInteger<int>(data, s2, (sep_pos -1) );
				*year = convertStrToInteger<int>(data, (sep_pos + 1), eIdx);
			}
		}
	}

	return true;
}

/**
 * @brief Parse a character stream to extract the hour, minute, second and
 * millisecond time field values.
 * 
 * Incoming format is expected to be HH:MM:SS.MS, with the latter second and
 * millisecond fields optional. Each time field can be a single, double,
 * or triple (in the case of milliseconds) digits. 12-hr and 24-hr time format
 * is detected via the absence or presence of AM/PM characters at the end.
 *
 * @param[in] data The character string time to parse
 * @param[in] start The start index of the character stream
 * @param[in] end The end index of the character stream
 * @param[out] hour The hour value
 * @param[out] minute The minute value
 * @param[out] second The second value (0 if not present)
 * @param[out] millisecond The millisecond (0 if not present)
 */
__inline__ __device__ void extractTime(const char *data, long start, long end, int *hour,
                            int *minute, int *second, int *millisecond) {
  constexpr char sep = ':';

  // Adjust for AM/PM and any whitespace before
  int hour_adjust = 0;
  if (data[end] == 'M' || data[end] == 'm') {
    if (data[end - 1] == 'P' || data[end - 1] == 'p') {
      hour_adjust = 12;
    }
    end = end - 2;
    while (data[end] == ' ') {
      --end;
    }
  }

  // Find hour-minute separator
  const auto hm_sep = findFirstOccurrence(data, start, end, sep);
  *hour = convertStrToInteger<int>(data, start, hm_sep - 1) + hour_adjust;

  // Find minute-second separator (if present)
  const auto ms_sep = findFirstOccurrence(data, hm_sep + 1, end, sep);
  if (ms_sep == -1) {
    *minute = convertStrToInteger<int>(data, hm_sep + 1, end);
    *second = 0;
    *millisecond = 0;
  } else {
    *minute = convertStrToInteger<int>(data, hm_sep + 1, ms_sep - 1);

    // Find second-millisecond separator (if present)
    const auto sms_sep = findFirstOccurrence(data, ms_sep + 1, end, '.');
    if (sms_sep == -1) {
      *second = convertStrToInteger<int>(data, ms_sep + 1, end);
      *millisecond = 0;
    } else {
      *second = convertStrToInteger<int>(data, ms_sep + 1, sms_sep - 1);
      *millisecond = convertStrToInteger<int>(data, sms_sep + 1, end);
    }
  }
}

// User-defined literals to clarify numbers and units for time calculation
__inline__ __device__
constexpr uint32_t operator "" _days(unsigned long long int days) {
  return days;
}
__inline__ __device__
constexpr uint32_t operator "" _erasInDays(unsigned long long int eras) {
  // multiply by number of days within an era (400 year span)
  return eras * 146097_days;
}
__inline__ __device__
constexpr uint32_t operator "" _years(unsigned long long int years) {
  return years;
}
__inline__ __device__
constexpr uint32_t operator "" _erasInYears(unsigned long long int eras) {
  return (eras * 1_erasInDays) / 365_days;
}

/**
 * @brief Compute number of days since "March 1, 0000", given a date
 *
 * This function takes year, month, and day and returns the number of days
 * since the baseline which is taken as 0000-03-01. This value is chosen as the
 * origin for ease of calculation (now February becomes the last month).
 *
 * @param[in] year
 * @param[in] month
 * @param[in] day
 *
 * @return days since March 1, 0000
 */
__inline__ __device__ constexpr int32_t daysSinceBaseline(int year, int month, int day) {
  // More details of this formula are located in cuDF datetime_ops
  // In brief, the calculation is split over several components:
  //     era: a 400 year range, where the date cycle repeats exactly
  //     yoe: year within the 400 range of an era
  //     doy: day within the 364 range of a year
  //     doe: exact day within the whole era
  // The months are shifted so that March is the starting month and February
  // (possible leap day in it) is the last month for the linear calculation
  year -= (month <= 2) ? 1 : 0;

  const int32_t era = (year >= 0 ? year : year - 399_years) / 1_erasInYears;
  const int32_t yoe = year - era * 1_erasInYears;
  const int32_t doy = (153_days * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
  const int32_t doe = (yoe * 365_days) + (yoe / 4_years) - (yoe / 100_years) + doy;

  return (era * 1_erasInDays) + doe;
}


/**
 * @brief Compute number of days since epoch, given a date
 *
 * This function takes year, month, and day and returns the number of days
 * since epoch (1970-01-01).
 *
 * @param[in] year
 * @param[in] month
 * @param[in] day
 *
 * @return days since epoch
 */
__inline__ __device__ constexpr int32_t daysSinceEpoch(int year, int month, int day) {
  // Shift the start date to epoch to match unix time
  static_assert(daysSinceBaseline(1970, 1, 1) == 719468_days,
                "Baseline to epoch returns incorrect number of days");

  return daysSinceBaseline(year, month, day) - daysSinceBaseline(1970, 1, 1);
}


/**
 * @brief Compute the number of seconds since epoch, given a date and time
 *
 * This function takes year, month, day, hour, minute and second and returns
 * the number of seconds since epoch (1970-01-01)
 *
 * @param[in] year
 * @param[in] month
 * @param[in] day
 * @param[in] hour
 * @param[in] minute
 * @param[in] second
 *
 * @return seconds since epoch
 */
__inline__ __device__ constexpr int64_t secondsSinceEpoch(int year, int month, int day,
                                               int hour, int minute,
                                               int second) {
  // Leverage the function to find the days since epoch
  const auto days = daysSinceEpoch(year, month, day);

  // Return sum total seconds from each time portion
  return (days * 24 * 60 * 60) + (hour * 60 * 60) + (minute * 60) + second;
}
