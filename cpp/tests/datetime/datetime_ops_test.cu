/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 William Scott Malpica <william@blazingdb.com>
 *     Copyright 2018 Rommel Quintanilla <rommel@blazingdb.com>
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

#include <tests/utilities/cudf_test_fixtures.h>

#include <cudf/cudf.h>
#include <cudf/functions.h>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/device_vector.h>

#include <iostream>
#include <vector>

#include <cstdlib>


struct gdf_extract_from_datetime_example_test : public GdfTest {};

TEST_F(gdf_extract_from_datetime_example_test, usage_example) {

	// gdf_column input examples for date32, date64 and timestamp (in seconds)

	std::vector<int32_t> inputDate32Data = {
		-1528, // '1965-10-26'
		17716, // '2018-07-04'
		19382 // '2023-01-25'
	};

	std::vector<int64_t> inputDate64Data = {
		-131968727238, // '1965-10-26 14:01:12.762'
		1530705600000, // '2018-07-04 12:00:00.000'
		1674631932929 // '2023-01-25 07:32:12.929'
	};

	std::vector<int64_t> inputTimestampSecsData = {
		-131968728, // '1965-10-26 14:01:12'
		1530705600, // '2018-07-04 12:00:00'
		1674631932 // '2023-01-25 07:32:12'
	};

	int colSize = 3;

	// Input column for date32
	rmm::device_vector<int32_t> intputDate32DataDev(inputDate32Data);
	rmm::device_vector<gdf_valid_type> inputDate32ValidDev(1,0);

	gdf_column inputDate32Col{};
	inputDate32Col.dtype = GDF_DATE32;
	inputDate32Col.size = colSize;

	inputDate32Col.data = thrust::raw_pointer_cast(intputDate32DataDev.data());
	inputDate32Col.valid = thrust::raw_pointer_cast(inputDate32ValidDev.data());

	// Input column for date64
	rmm::device_vector<int64_t> intputDate64DataDev(inputDate64Data);
	rmm::device_vector<gdf_valid_type> inputDate64ValidDev(1,0);

	gdf_column inputDate64Col{};
	inputDate64Col.dtype = GDF_DATE64;
	inputDate64Col.size = colSize;

	inputDate64Col.data = thrust::raw_pointer_cast(intputDate64DataDev.data());
	inputDate64Col.valid = thrust::raw_pointer_cast(inputDate64ValidDev.data());

	// Input column for timestamp in seconds
	rmm::device_vector<int64_t> intputTimestampSecsDataDev(inputTimestampSecsData);
	rmm::device_vector<gdf_valid_type> inputTimestampSecsValidDev(1,0);

	gdf_column inputTimestampSecsCol{};
	inputTimestampSecsCol.dtype = GDF_TIMESTAMP;
	inputTimestampSecsCol.size = colSize;
	inputTimestampSecsCol.dtype_info.time_unit = TIME_UNIT_s;

	inputTimestampSecsCol.data = thrust::raw_pointer_cast(intputTimestampSecsDataDev.data());
	inputTimestampSecsCol.valid = thrust::raw_pointer_cast(inputTimestampSecsValidDev.data());

	// Output column
	rmm::device_vector<int16_t> outDataDev(colSize);
	rmm::device_vector<gdf_valid_type> outValidDev(1,0);

	gdf_column outputInt16Col{};
	outputInt16Col.dtype = GDF_INT16;
	outputInt16Col.size = colSize;

	outputInt16Col.data = thrust::raw_pointer_cast(outDataDev.data());
	outputInt16Col.valid = thrust::raw_pointer_cast(outValidDev.data());

	std::vector<int16_t> results(colSize);
	gdf_error gdfError;

	// example for gdf_error gdf_extract_datetime_year(gdf_column *input, gdf_column *output)
	{
		// from date32
		gdfError = gdf_extract_datetime_year(&inputDate32Col, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 1965 );
		EXPECT_TRUE( results[1] == 2018 );
		EXPECT_TRUE( results[2] == 2023 );

		// from date64
		gdfError = gdf_extract_datetime_year(&inputDate64Col, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 1965 );
		EXPECT_TRUE( results[1] == 2018 );
		EXPECT_TRUE( results[2] == 2023 );

		// from timestamp in seconds
		gdfError = gdf_extract_datetime_year(&inputTimestampSecsCol, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 1965 );
		EXPECT_TRUE( results[1] == 2018 );
		EXPECT_TRUE( results[2] == 2023 );
	}

	// example for gdf_error gdf_extract_datetime_month(gdf_column *input, gdf_column *output)
	{
		// from date32
		gdfError = gdf_extract_datetime_month(&inputDate32Col, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 10 );
		EXPECT_TRUE( results[1] == 7 );
		EXPECT_TRUE( results[2] == 1 );

		// from date64
		gdfError = gdf_extract_datetime_month(&inputDate64Col, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 10 );
		EXPECT_TRUE( results[1] == 7 );
		EXPECT_TRUE( results[2] == 1 );

		// from timestamp in seconds
		gdfError = gdf_extract_datetime_month(&inputTimestampSecsCol, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 10 );
		EXPECT_TRUE( results[1] == 7 );
		EXPECT_TRUE( results[2] == 1 );
	}

	// example for gdf_error gdf_extract_datetime_day(gdf_column *input, gdf_column *output)
	{
		// from date32
		gdfError = gdf_extract_datetime_day(&inputDate32Col, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 26 );
		EXPECT_TRUE( results[1] == 4 );
		EXPECT_TRUE( results[2] == 25 );

		// from date64
		gdfError = gdf_extract_datetime_day(&inputDate64Col, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 26 );
		EXPECT_TRUE( results[1] == 4 );
		EXPECT_TRUE( results[2] == 25 );

		// from timestamp in seconds
		gdfError = gdf_extract_datetime_day(&inputTimestampSecsCol, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 26 );
		EXPECT_TRUE( results[1] == 4 );
		EXPECT_TRUE( results[2] == 25 );
	}

	// example for gdf_error gdf_extract_datetime_hour(gdf_column *input, gdf_column *output)
	{
		// from date64
		gdfError = gdf_extract_datetime_hour(&inputDate64Col, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 14 );
		EXPECT_TRUE( results[1] == 12 );
		EXPECT_TRUE( results[2] == 7 );

		// from timestamp in seconds
		gdfError = gdf_extract_datetime_hour(&inputTimestampSecsCol, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 14 );
		EXPECT_TRUE( results[1] == 12 );
		EXPECT_TRUE( results[2] == 7 );
	}

	// example for gdf_error gdf_extract_datetime_minute(gdf_column *input, gdf_column *output)
	{
		// from date64
		gdfError = gdf_extract_datetime_minute(&inputDate64Col, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 1 );
		EXPECT_TRUE( results[1] == 0 );
		EXPECT_TRUE( results[2] == 32 );

		// from timestamp in seconds
		gdfError = gdf_extract_datetime_minute(&inputTimestampSecsCol, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 1 );
		EXPECT_TRUE( results[1] == 0 );
		EXPECT_TRUE( results[2] == 32 );
	}

	// example for gdf_error gdf_extract_datetime_second(gdf_column *input, gdf_column *output)
	{
		// from date64
		gdfError = gdf_extract_datetime_second(&inputDate64Col, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 12 );
		EXPECT_TRUE( results[1] == 0 );
		EXPECT_TRUE( results[2] == 12 );

		// from timestamp in seconds
		gdfError = gdf_extract_datetime_second(&inputTimestampSecsCol, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 12 );
		EXPECT_TRUE( results[1] == 0 );
		EXPECT_TRUE( results[2] == 12 );
	}

	// example for gdf_error gdf_extract_datetime_day(gdf_column *input, gdf_column *output)
	{
		// from date32
		gdfError = gdf_extract_datetime_weekday(&inputDate32Col, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 1 );
		EXPECT_TRUE( results[1] == 2 );
		EXPECT_TRUE( results[2] == 2 );

		// from date64
		gdfError = gdf_extract_datetime_weekday(&inputDate64Col, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 1 );
		EXPECT_TRUE( results[1] == 2 );
		EXPECT_TRUE( results[2] == 2 );

		// from timestamp in seconds
		gdfError = gdf_extract_datetime_weekday(&inputTimestampSecsCol, &outputInt16Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 1 );
		EXPECT_TRUE( results[1] == 2 );
		EXPECT_TRUE( results[2] == 2 );
	}


}


struct gdf_extract_from_datetime_test : public GdfTest {

	void SetUp() {

		inputYears = {
				2018, 2018, 1919, 2020, 1970, 2043, 1998, 1815, 1969, 1969, 1969,
				1906, 1908,	1917, 1932,	1981, 1989,	1991, 1993,	2009, 2018,	2024,
				1915, 1939, 1944, 1969,	1976, 1998,	2010, 2031
		};
		inputMonths = {
					6,
					6,
					12,
					2,
					1,
					3,
					4,
					8,
					12,
					12,
					12,
					11,
					1,
					2,
					6,
					6,
					12,
					12,
					4,
					1,
					3,
					9,
					5,
					12,
					4,
					5,
					1,
					6,
					1,
					1
			};
			inputDays = {
					14,
					14,
					31,
					29,
					1,
					11,
					20,
					28,
					31,
					30,
					31,
					28,
					28,
					3,
					19,
					22,
					14,
					7,
					13,
					3,
					25,
					14,
					29,
					15,
					9,
					4,
					23,
					1,
					7,
					15
			};
			inputHours = {
					0,
					0,
					23,
					0,
					0,
					2,
					12,
					16,
					23,
					1,
					23,
					6,
					0,
					18,
					21,
					17,
					14,
					2,
					20,
					16,
					14,
					23,
					6,
					1,
					16,
					15,
					0,
					21,
					22,
					0
			};
			inputMinutes = {
					19,
					19,
					59,
					0,
					0,
					22,
					30,
					49,
					59,
					1,
					59,
					44,
					9,
					51,
					8,
					39,
					47,
					47,
					59,
					8,
					59,
					8,
					12,
					8,
					43,
					45,
					58,
					18,
					24,
					32
			};
			inputSeconds = {
					50,
					59,
					59,
					1,
					0,
					22,
					30,
					1,
					55,
					1,
					54,
					8,
					21,
					45,
					44,
					35,
					20,
					40,
					24,
					14,
					51,
					14,
					29,
					27,
					58,
					35,
					24,
					32,
					53,
					48
			};

			colSize = inputYears.size();

			outDataDev.resize(colSize);
			outputValidDev.resize(4,0);

			outputCol.dtype = GDF_INT16;
			outputCol.size = colSize;
			outputCol.data = thrust::raw_pointer_cast(outDataDev.data());
			outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

	}

	void TearDown() {
	}

	void validate_output(gdf_error gdfError, std::vector<int16_t> & correct_result) {

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == correct_result[i]);
		}
	}

	void test_all_extract_functions(gdf_column & inputCol){
		gdf_error gdfError = gdf_extract_datetime_year(&inputCol, &outputCol);
		validate_output(gdfError, inputYears);

		gdfError = gdf_extract_datetime_month(&inputCol, &outputCol);
		validate_output(gdfError, inputMonths);

		gdfError = gdf_extract_datetime_day(&inputCol, &outputCol);
		validate_output(gdfError, inputDays);

		gdfError = gdf_extract_datetime_hour(&inputCol, &outputCol);
		validate_output(gdfError, inputHours);

		gdfError = gdf_extract_datetime_minute(&inputCol, &outputCol);
		validate_output(gdfError, inputMinutes);

		gdfError = gdf_extract_datetime_second(&inputCol, &outputCol);
		validate_output(gdfError, inputSeconds);
	}


	std::vector<int16_t> inputYears;
	std::vector<int16_t> inputMonths;
	std::vector<int16_t> inputDays;
	std::vector<int16_t> inputHours;
	std::vector<int16_t> inputMinutes;
	std::vector<int16_t> inputSeconds;

	int colSize;

	gdf_column outputCol;
	rmm::device_vector<int16_t> outDataDev;
	rmm::device_vector<gdf_valid_type> outputValidDev;

};



TEST_F(gdf_extract_from_datetime_test, date64Tests) {

	// extract from milliseconds
	{
		std::vector<int64_t> inputData = {
				1528935590000, // '2018-06-14 00:19:50.000'
				1528935599999, // '2018-06-14 00:19:59.999'
				-1577923201000, // '1919-12-31 23:59:59.000'
				1582934401123, // '2020-02-29 00:00:01.123'
				0,             // '1970-01-01 00:00:00.000'
				2309653342222, // '2043-03-11 02:22:22.222'
				893075430345, // '1998-04-20 12:30:30.345'
				-4870653058987,  // '1815-08-28 16:49:01.013
				-4500,            // '1969-12-31 23:59:55.500'
				-169138999,    // '1969-12-30 01:01:01.001
				-5999,        // '1969-12-31 23:59:54.001'
				-1991063752000, //	1906-11-28 06:44:08
				-1954281039000, //	1908-01-28 00:09:21
				-1669612095000, //	1917-02-03 18:51:45
				-1184467876000, //	1932-06-19 21:08:44
				362079575000, //	1981-06-22 17:39:35
				629650040000, //	1989-12-14 14:47:20
				692074060000, //	1991-12-07 02:47:40
				734734764000, //	1993-04-13 20:59:24
				1230998894000, //	2009-01-03 16:08:14
				1521989991000, //	2018-03-25 14:59:51
				1726355294000, //	2024-09-14 23:08:14
				-1722880051000, //	1915-05-29 06:12:29
				-948235893000, //	1939-12-15 01:08:27
				-811926962000, //	1944-04-09 16:43:58
				-20852065000, //	1969-05-04 15:45:35
				191206704000, //	1976-01-23 00:58:24
				896735912000, //	1998-06-01 21:18:32
				1262903093000, //	2010-01-07 22:24:53
				1926203568000 //	2031-01-15 00:32:48
		};

		rmm::device_vector<int64_t> intputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4,0);

		gdf_column inputCol{};
		inputCol.dtype = GDF_DATE64;
		inputCol.size = colSize;
		inputCol.data = thrust::raw_pointer_cast(intputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());

		test_all_extract_functions(inputCol);

		inputCol.valid = NULL;
        test_all_extract_functions(inputCol);
        inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());

		// TIMESTAMP in ms should be the same as DATE64
		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.dtype_info.time_unit = TIME_UNIT_ms;

		test_all_extract_functions(inputCol);

        inputCol.valid = NULL;
        test_all_extract_functions(inputCol);

	}

	// extract from seconds
	{
		std::vector<int64_t> inputData = {
				1528935590, // '2018-06-14 00:19:50'
				1528935599, // '2018-06-14 00:19:59'
				-1577923201, // '1919-12-31 23:59:59'
				1582934401, // '2020-02-29 00:00:01'
				0,             // '1970-01-01 00:00:00'
				2309653342, // '2043-03-11 02:22:22'
				893075430, // '1998-04-20 12:30:30'
				-4870653059,  // '1815-08-28 16:49:01
				-5,            // '1969-12-31 23:59:55'
				-169139,    // '1969-12-30 01:01:01
				-6,        // '1969-12-31 23:59:54'
				-1991063752, //	1906-11-28 06:44:08
				-1954281039, //	1908-01-28 00:09:21
				-1669612095, //	1917-02-03 18:51:45
				-1184467876, //	1932-06-19 21:08:44
				362079575, //	1981-06-22 17:39:35
				629650040, //	1989-12-14 14:47:20
				692074060, //	1991-12-07 02:47:40
				734734764, //	1993-04-13 20:59:24
				1230998894, //	2009-01-03 16:08:14
				1521989991, //	2018-03-25 14:59:51
				1726355294, //	2024-09-14 23:08:14
				-1722880051, //	1915-05-29 06:12:29
				-948235893, //	1939-12-15 01:08:27
				-811926962, //	1944-04-09 16:43:58
				-20852065, //	1969-05-04 15:45:35
				191206704, //	1976-01-23 00:58:24
				896735912, //	1998-06-01 21:18:32
				1262903093, //	2010-01-07 22:24:53
				1926203568 //	2031-01-15 00:32:48
		};

		rmm::device_vector<int64_t> intputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4,0);

		gdf_column inputCol{};
		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.dtype_info.time_unit = TIME_UNIT_s;
		inputCol.size = colSize;
		inputCol.data = thrust::raw_pointer_cast(intputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());

		test_all_extract_functions(inputCol);

        inputCol.valid = NULL;
        test_all_extract_functions(inputCol);
	}

	// extract from microseconds
	{
		std::vector<int64_t> inputData = {
				1528935590000000, // '2018-06-14 00:19:50.000000'
				1528935599999999, // '2018-06-14 00:19:59.999999'
				-1577923201000000, // '1919-12-31 23:59:59.000000'
				1582934401123123, // '2020-02-29 00:00:01.123123'
				0,             // '1970-01-01 00:00:00.000000'
				2309653342222222, // '2043-03-11 02:22:22.222222'
				893075430345543, // '1998-04-20 12:30:30.345543'
				-4870653058987789,  // '1815-08-28 16:49:01.012211
				-4500005,            // '1969-12-31 23:59:55.499995'
				-169138999999,    // '1969-12-30 01:01:01.000001'
				-5999999,        // '1969-12-31 23:59:54.000001'
				-1991063752000000, //	1906-11-28 06:44:08
				-1954281039000000, //	1908-01-28 00:09:21
				-1669612095000000, //	1917-02-03 18:51:45
				-1184467876000000, //	1932-06-19 21:08:44
				362079575000000, //	1981-06-22 17:39:35
				629650040000000, //	1989-12-14 14:47:20
				692074060000000, //	1991-12-07 02:47:40
				734734764000000, //	1993-04-13 20:59:24
				1230998894000000, //	2009-01-03 16:08:14
				1521989991000000, //	2018-03-25 14:59:51
				1726355294000000, //	2024-09-14 23:08:14
				-1722880051000000, //	1915-05-29 06:12:29
				-948235893000000, //	1939-12-15 01:08:27
				-811926962000000, //	1944-04-09 16:43:58
				-20852065000000, //	1969-05-04 15:45:35
				191206704000000, //	1976-01-23 00:58:24
				896735912000000, //	1998-06-01 21:18:32
				1262903093000000, //	2010-01-07 22:24:53
				1926203568000000 //	2031-01-15 00:32:48
		};

		rmm::device_vector<int64_t> intputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4,0);

		gdf_column inputCol{};
		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.dtype_info.time_unit = TIME_UNIT_us;
		inputCol.size = colSize;
		inputCol.data = thrust::raw_pointer_cast(intputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());

		test_all_extract_functions(inputCol);

        inputCol.valid = NULL;
        test_all_extract_functions(inputCol);
	}

	// extract from nanoseconds
	{
		std::vector<int64_t> inputData = {
				1528935590000000000, // '2018-06-14 00:19:50.000000000'
				1528935599999999999, // '2018-06-14 00:19:59.999999999'
				-1577923201000000000, // '1919-12-31 23:59:59.000000000'
				1582934401123123123, // '2020-02-29 00:00:01.123123123'
				0,             // '1970-01-01 00:00:00.000000'
				2309653342222222222, // '2043-03-11 02:22:22.222222222'
				893075430345543345, // '1998-04-20 12:30:30.345543345'
				-4870653058987789987,  // '1815-08-28 16:49:01.012210013
				-4500000005,            // '1969-12-31 23:59:55.499999995'
				-169138999999999,    // '1969-12-30 01:01:01.000000001'
				-5999999999,        // '1969-12-31 23:59:54.000000001'
				-1991063752000000000, //	1906-11-28 06:44:08
				-1954281039000000000, //	1908-01-28 00:09:21
				-1669612095000000000, //	1917-02-03 18:51:45
				-1184467876000000000, //	1932-06-19 21:08:44
				362079575000000000, //	1981-06-22 17:39:35
				629650040000000000, //	1989-12-14 14:47:20
				692074060000000000, //	1991-12-07 02:47:40
				734734764000000000, //	1993-04-13 20:59:24
				1230998894000000000, //	2009-01-03 16:08:14
				1521989991000000000, //	2018-03-25 14:59:51
				1726355294000000000, //	2024-09-14 23:08:14
				-1722880051000000000, //	1915-05-29 06:12:29
				-948235893000000000, //	1939-12-15 01:08:27
				-811926962000000000, //	1944-04-09 16:43:58
				-20852065000000000, //	1969-05-04 15:45:35
				191206704000000000, //	1976-01-23 00:58:24
				896735912000000000, //	1998-06-01 21:18:32
				1262903093000000000, //	2010-01-07 22:24:53
				1926203568000000000 //	2031-01-15 00:32:48
		};

		rmm::device_vector<int64_t> intputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4,0);

		gdf_column inputCol{};
		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.dtype_info.time_unit = TIME_UNIT_ns;
		inputCol.size = colSize;
		inputCol.data = thrust::raw_pointer_cast(intputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());

		test_all_extract_functions(inputCol);

        inputCol.valid = NULL;
        test_all_extract_functions(inputCol);
	}
}

struct gdf_extract_datetime_TEST : public GdfTest {};

TEST_F(gdf_extract_datetime_TEST, date32Tests) {

	int colSize = 8;

	gdf_column inputCol{};
	gdf_column outputCol{};

	inputCol.dtype = GDF_DATE32;
	inputCol.size = colSize;
	outputCol.dtype = GDF_INT16;
	outputCol.size = colSize;

	std::vector<int32_t> inputData(colSize);
	inputData[0] = 17696;	// '2018-06-14'
	inputData[1] = 17697;	// '2018-06-15'
	inputData[2] = -18264;	// '1919-12-31'
	inputData[3] = 18321;   // '2020-02-29'
	inputData[4] = 0;               // '1970-01-01'
	inputData[5] = 26732;   // '2043-03-11'
	inputData[6] = 10336;    // '1998-04-20'
	inputData[7] = -56374;  // '1815-08-28'

	rmm::device_vector<int32_t> intputDataDev(inputData);
	rmm::device_vector<gdf_valid_type> inputValidDev(1,0);
	rmm::device_vector<int16_t> outDataDev(colSize);
	rmm::device_vector<gdf_valid_type> outputValidDev(1,0);

	inputCol.data = thrust::raw_pointer_cast(intputDataDev.data());
	inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
	outputCol.data = thrust::raw_pointer_cast(outDataDev.data());
	outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

	{
		gdf_error gdfError = gdf_extract_datetime_year(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 2018);
		EXPECT_TRUE( results[1] == 2018);
		EXPECT_TRUE( results[2] == 1919);
		EXPECT_TRUE( results[3] == 2020);
		EXPECT_TRUE( results[4] == 1970);
		EXPECT_TRUE( results[5] == 2043);
		EXPECT_TRUE( results[6] == 1998);
		EXPECT_TRUE( results[7] == 1815);
	}
	{
		gdf_error gdfError = gdf_extract_datetime_month(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 6);
		EXPECT_TRUE( results[1] == 6);
		EXPECT_TRUE( results[2] == 12);
		EXPECT_TRUE( results[3] == 2);
		EXPECT_TRUE( results[4] == 1);
		EXPECT_TRUE( results[5] == 3);
		EXPECT_TRUE( results[6] == 4);
		EXPECT_TRUE( results[7] == 8);
	}
	{
		gdf_error gdfError = gdf_extract_datetime_day(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 14);
		EXPECT_TRUE( results[1] == 15);
		EXPECT_TRUE( results[2] == 31);
		EXPECT_TRUE( results[3] == 29);
		EXPECT_TRUE( results[4] == 1);
		EXPECT_TRUE( results[5] == 11);
		EXPECT_TRUE( results[6] == 20);
		EXPECT_TRUE( results[7] == 28);
	}
}

TEST_F(gdf_extract_datetime_TEST, testErrors) {

	// WRONG SIZE OF OUTPUT
	{
		int colSize = 8;

		gdf_column inputCol{};
		gdf_column outputCol{};

		inputCol.dtype = GDF_DATE32;
		inputCol.size = colSize;
		outputCol.dtype = GDF_INT32;
		outputCol.size = colSize;

		std::vector<int32_t> inputData(colSize);
		inputData[0] = 17696;	// '2018-06-14'
		inputData[1] = 17697;	// '2018-06-15'
		inputData[2] = -18264;	// '1919-12-31'
		inputData[3] = 18321;   // '2020-02-29'
		inputData[4] = 0;               // '1970-01-01'
		inputData[5] = 26732;   // '2043-03-11'
		inputData[6] = 10336;    // '1998-04-20'
		inputData[7] = -56374;  // '1815-08-287


		rmm::device_vector<int32_t> intputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,0);
		rmm::device_vector<int16_t> outDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,0);

		inputCol.data = thrust::raw_pointer_cast(intputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());
		gdf_error gdfError = gdf_extract_datetime_year(&inputCol, &outputCol);
		EXPECT_TRUE( gdfError == GDF_UNSUPPORTED_DTYPE );
	}
	// MISMATCHED SIZE
	{
		int colSize = 8;

		gdf_column inputCol{};
		gdf_column outputCol{};

		inputCol.dtype = GDF_DATE32;
		inputCol.size = colSize;
		outputCol.dtype = GDF_INT16;
		outputCol.size = colSize + 10;

		std::vector<int32_t> inputData(colSize);
		inputData[0] = 17696;	// '2018-06-14'
		inputData[1] = 17697;	// '2018-06-15'
		inputData[2] = -18264;	// '1919-12-31'
		inputData[3] = 18321;   // '2020-02-29'
		inputData[4] = 0;               // '1970-01-01'
		inputData[5] = 26732;   // '2043-03-11'
		inputData[6] = 10336;    // '1998-04-20'
		inputData[7] = -56374;  // '1815-08-287


		rmm::device_vector<int32_t> intputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,0);
		rmm::device_vector<int16_t> outDataDev(colSize + 10);
		rmm::device_vector<gdf_valid_type> outputValidDev(3,0);

		inputCol.data = thrust::raw_pointer_cast(intputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_extract_datetime_year(&inputCol, &outputCol);
		EXPECT_TRUE( gdfError == GDF_COLUMN_SIZE_MISMATCH );
	}

}
