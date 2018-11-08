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

#include <gdf/gdf.h>
#include "type_conversion.cuh"

__host__ __device__ gdf_date32 parseDateFormat(char *data, long start_idx, long end_idx, bool dayfirst);
__host__ __device__ gdf_date64 parseDateTimeFormat(char *data, long start_idx, long end_idx, bool dayfirst);

__host__ __device__ bool extractDate(char *data, long sIdx, long eIdx, bool dayfirst, int *year_out, int *month_out, int *day_out);
__host__ __device__ bool extractTime(char *data, int sIdx, int eIdx, int *hour_out, int *minute_out, int *second_out);

__host__ __device__ int32_t daysSinceEpoch(int year, int month, int day);
__host__ __device__ int64_t secondsFromEpoch(int year, int month, int day, int hour, int minute, int second);


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
__host__ __device__
gdf_date32 parseDateFormat(char *data, long start_idx, long end_idx, bool dayfirst) {

	int day, month, year;
	gdf_date32 e = -1;

	bool status = extractDate(data, start_idx, end_idx, dayfirst, &year, &month, &day);

	if ( status )
		e = daysSinceEpoch(year, month, day);

	return e;
}

/**
 * @brief Parse a Date string into a date64, milliseconds since epoch
 *
 * This function takes a string and produces a date32 representation
 * Acceptable formats are a combination of MM/YYYY and MM/DD/YYYY
 *
 * @param[in] data 		Pointer to the data block
 * @param[in] start_idx Starting index within the data block
 * @param[in] end_idx 	Ending index within the data block
 * @param[in] dayfirst 	Flag to indicate that day is the first field - DD/MM/YYYY
 *
 * @return milliseconds since epoch
 */
__host__ __device__
gdf_date64 parseDateTimeFormat(char *data, long start_idx, long end_idx, bool dayfirst) {

	int 		day, month, year;
	int 		hour, minute, second;
	gdf_date64 	answer = -1;

	// find the time separator between date and time
	long t_pos = firstOcurance(data, start_idx, end_idx, 'T');

	if ( t_pos == -1) {
		t_pos = firstOcurance(data, start_idx, end_idx, ' ');
		if ( t_pos < 8 || t_pos > 10)
			t_pos = -1;
	}

	// if the position was not found then we are in trouble, return -1 (error code)
	if ( t_pos != -1 ) {

		if ( extractDate(data, start_idx, (t_pos - 1), dayfirst, &year, &month, &day) )
		{

			if ( extractTime(data, (t_pos + 1), end_idx, &hour, &minute, &second) )
				answer = secondsFromEpoch(year, month, day, hour, minute, second);
		}
	} else {

		if ( (end_idx - start_idx) < 11 ) {
			// only have a date portion, no time
			extractDate(data, start_idx, end_idx, dayfirst, &year, &month, &day);
			answer = secondsFromEpoch(year, month, day, 0, 0, 0);
		} else {
			answer = -1;
		}
	}

	// convert to milliseconds
	answer *= 1000;

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
__host__ __device__
bool extractDate(char *data, long sIdx, long eIdx, bool dayfirst, int *year, int *month, int *day) {

	char sep = '/';

	long sep_pos = firstOcurance(data, sIdx, eIdx, sep);

	if ( sep_pos == -1 ) {
		sep = '-';
		sep_pos = firstOcurance(data, sIdx, eIdx, sep);
	}

	if ( sep_pos == -1)
		return false;

	//--- is year the first filed?
	if ( (sep_pos - sIdx) == 4  ) {

		*year = convertStrtoInt<int>(data, sIdx, (sep_pos -1) );

		// Month
		long s2 = sep_pos +1;
		sep_pos = firstOcurance(data, s2, eIdx, sep);

		if (sep_pos == -1 ) {

			//--- Data is just Year and Month - no day
			*month = convertStrtoInt<int>(data, s2, eIdx );
			*day = 1;

		} else {
			*month = convertStrtoInt<int>(data, s2, (sep_pos -1) );
			*day = convertStrtoInt<int>(data, (sep_pos + 1), eIdx);
		}

	} else {

		//--- if the dayfirst flag is set, then restricts the format options
		if ( dayfirst) {

			*day = convertStrtoInt<int>(data, sIdx, (sep_pos -1) );

			long s2 = sep_pos +1;
			sep_pos = firstOcurance(data, s2, eIdx, sep);

			*month = convertStrtoInt<int>(data, s2, (sep_pos -1) );
			*year = convertStrtoInt<int>(data, (sep_pos + 1), eIdx);

		} else {

			*month = convertStrtoInt<int>(data, sIdx, (sep_pos -1) );

			long s2 = sep_pos +1;
			sep_pos = firstOcurance(data, s2, eIdx, sep);

			if (sep_pos == -1 )
			{
				//--- Data is just Year and Month - no day
				*year = convertStrtoInt<int>(data, s2, eIdx );
				*day = 1;

			} else {
				*day = convertStrtoInt<int>(data, s2, (sep_pos -1) );
				*year = convertStrtoInt<int>(data, (sep_pos + 1), eIdx);
			}
		}
	}

	return true;
}



/**
 * @brief Extract the Hour, Minute, Second from a string
 *
 *  Formats are H:M to HH:MM, then with seconds H:M:S to HH:MM:SS
 *  Also handles 24 and 12-hours with AM/PM
 *
 * @param[in] data	Pointer to data block
 * @param[in] sIdx	String index within data block
 * @param[in] eIdx	Ending index within data block
 *
 * @param[out] hour
 * @param[out] minute
 * @param[out] second
 *
 * @return T/F - false indicates that an error occurred
 */
__host__ __device__
bool extractTime(char *data, int sIdx, int eIdx, int *hour, int *minute, int *second) {

	char sep = ':';

	// is there AM/PM
	int hour_adjust = 0;

	if ( data[eIdx] == 'M' || data[eIdx] == 'm')
	{
		if ( data[eIdx - 1] == 'P' || data[eIdx -1] == 'p')
			hour_adjust = 12;

		eIdx = eIdx - 2;

		while (data[eIdx] == ' ')
			--eIdx;
	}


	// Hour to Minute Separator
	int hm_sep = firstOcurance(data, sIdx, eIdx, sep);

	*hour = convertStrtoInt<int>(data, sIdx, (hm_sep -1) );

	*hour += hour_adjust;

	// now minute
	long ms_sep = firstOcurance(data, (hm_sep + 1), eIdx, sep);

	if (ms_sep == -1 ) {
		//--- Data is just Hour and Minutes, no seconds
		*minute = convertStrtoInt<int>(data, (hm_sep + 1), eIdx );
		*second = 0;

	} else {
		*minute = convertStrtoInt<int>(data, (hm_sep + 1), (ms_sep -1) );
		*second = convertStrtoInt<int>(data, (ms_sep + 1), eIdx);
	}

	return true;
}


/**
 * @brief compute number of days since epoch
 *
 * This function takes year, month, and day and return the
 * number of days since epcoh as a int32
 *
 * @param[in] year
 * @param[in] month
 * @param[in] day
 *
 * @return days since epoch
 *
 */
__host__ __device__
gdf_date32 daysSinceEpoch(int year, int month, int day)  {

	static unsigned short days[12] = {0,  31,  59,  90, 120, 151, 181, 212, 243, 273, 304, 334};

	// years since epoch
	int ye = year - 1970;

	// 1972 was a leap year, so how many have occurred between date and 1972? Do not include this year in count
	int lpy = ((year - 1972 - 1) / 4);

	// compute days since epoch
	gdf_date32 days_e = (ye * 365) + lpy;

	// is this a leap year?
	if ( year % 4 == 0 && month > 2)
		days_e++;

	// months since January
	int me = month - 01;

	// days up to start of month
	days_e += days[me];

	// now just add days, but not current full days since this one is not over
	days_e +=  day;

	return days_e;
}




/**
 * @brief Given year,month,day,hour,minute,second and return the epcoh
 *
 * Given year, month, day, hour, minute, second compute the number of
 * seconds since epoch
 *
 * @param[in] year
 * @param[in] month
 * @param[in] day
 * @param[in] hour
 * @param[in] minute
 * @param[in] second
 *
 * @return seconds since epoch
 *
 */
__host__ __device__
int64_t secondsFromEpoch(int year, int month, int day, int hour, int minute, int second)  {

	// leverage the epoch days function
	gdf_date32 days_e = daysSinceEpoch(year, month, day);

	// now convert to seconds
	int64_t t = (days_e * 24) * 60 * 60;

	t += hour * 60 * 60;
	t += minute * 60;
	t += second;

	return t;
}


