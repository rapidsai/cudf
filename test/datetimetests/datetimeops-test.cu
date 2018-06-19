#include <cstdlib>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>



TEST(gdf_extract_datetime_TEST, date64Tests) {

	std::vector<int64_t> inputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058987,  // '1815-08-28 16:49:01.987
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

	std::vector<int16_t> inputYears = {
			2018, 2018, 1919, 2020, 1970, 2043, 1998, 1815, 1969, 1969, 1969,
			1906, 1908,	1917, 1932,	1981, 1989,	1991, 1993,	2009, 2018,	2024,
			1915, 1939, 1944, 1969,	1976, 1998,	2010, 2031
	};
	std::vector<int16_t> inputMonths = {
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
	std::vector<int16_t> inputDays = {
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
	std::vector<int16_t> inputHours = {
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
	std::vector<int16_t> inputMinutes = {
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
	std::vector<int16_t> inputSeconds = {
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

	int colSize = inputData.size();

	gdf_column inputCol;
	gdf_column outputCol;

	inputCol.dtype = GDF_DATE64;
	inputCol.size = colSize;
	outputCol.dtype = GDF_INT16;
	outputCol.size = colSize;

	thrust::device_vector<int64_t> intputDataDev(inputData);
	thrust::device_vector<gdf_valid_type> inputValidDev(4,0);
	thrust::device_vector<int16_t> outDataDev(colSize);
	thrust::device_vector<gdf_valid_type> outputValidDev(4,0);

	inputCol.data = thrust::raw_pointer_cast(intputDataDev.data());
	inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
	outputCol.data = thrust::raw_pointer_cast(outDataDev.data());
	outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

	{
		gdf_error gdfError = gdf_extract_datetime_year(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == inputYears[i]);
		}
	}
	{
		gdf_error gdfError = gdf_extract_datetime_month(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == inputMonths[i]);
		}
	}
	{
		gdf_error gdfError = gdf_extract_datetime_day(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == inputDays[i]);
		}
	}
	{
		gdf_error gdfError = gdf_extract_datetime_hour(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == inputHours[i]);
		}
	}
	{
		gdf_error gdfError = gdf_extract_datetime_minute(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == inputMinutes[i]);
		}
	}
	{
		gdf_error gdfError = gdf_extract_datetime_second(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == inputSeconds[i]);
		}
	}
}

TEST(gdf_extract_datetime_TEST, date32Tests) {

	int colSize = 8;

	gdf_column inputCol;
	gdf_column outputCol;

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
	inputData[7] = -56374;  // '1815-08-287


	thrust::device_vector<int32_t> intputDataDev(inputData);
	thrust::device_vector<gdf_valid_type> inputValidDev(1,0);
	thrust::device_vector<int16_t> outDataDev(colSize);
	thrust::device_vector<gdf_valid_type> outputValidDev(1,0);

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

TEST(gdf_extract_datetime_TEST, testErrors) {

	// WRONG SIZE OF OUTPUT
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

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


		thrust::device_vector<int32_t> intputDataDev(inputData);
		thrust::device_vector<gdf_valid_type> inputValidDev(1,0);
		thrust::device_vector<int16_t> outDataDev(colSize);
		thrust::device_vector<gdf_valid_type> outputValidDev(1,0);

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

		gdf_column inputCol;
		gdf_column outputCol;

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


		thrust::device_vector<int32_t> intputDataDev(inputData);
		thrust::device_vector<gdf_valid_type> inputValidDev(1,0);
		thrust::device_vector<int16_t> outDataDev(colSize + 10);
		thrust::device_vector<gdf_valid_type> outputValidDev(3,0);

		inputCol.data = thrust::raw_pointer_cast(intputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_extract_datetime_year(&inputCol, &outputCol);
		EXPECT_TRUE( gdfError == GDF_COLUMN_SIZE_MISMATCH );
	}

}
