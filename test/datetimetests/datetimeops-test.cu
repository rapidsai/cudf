#include <cstdlib>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include "../include/gdf/cffi/functions.h"



TEST(gdf_extract_datetime_TEST, date64Tests) {

	int colSize = 5;

	gdf_column inputCol;
	gdf_column outputCol;

	inputCol.dtype = GDF_DATE64;
	inputCol.size = colSize;
	outputCol.dtype = GDF_INT16;
	outputCol.size = colSize;

	int64_t inputData(colSize);
	inputData[0] = 1528935590000;	// '2018-06-14 00:19:50.000'
	inputData[1] = 1528935599999;	// '2018-06-14 00:19:59.999'
	inputData[2] = -1577923201000;	// '1919-12-31 23:59:59.000'
	inputData[3] = 1582934401123;   // '2020-02-29 00:00:01.123'
	inputData[4] = 0;               // '1970-01-01 00:00:00.000'


	thrust::device_vector<int64_t> intputDataDev(inputData);
	thrust::device_vector<char> inputValidDev(1,0);
	thrust::device_vector<int64_t> outDataDev(colSize);
	thrust::device_vector<char> outputValidDev(1,0);

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
	}
	{
		gdf_error gdfError = gdf_extract_datetime_day(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 14);
		EXPECT_TRUE( results[1] == 14);
		EXPECT_TRUE( results[2] == 31);
		EXPECT_TRUE( results[3] == 29);
		EXPECT_TRUE( results[4] == 1);
	}
	{
		gdf_error gdfError = gdf_extract_datetime_hour(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 0);
		EXPECT_TRUE( results[1] == 0);
		EXPECT_TRUE( results[2] == 23);
		EXPECT_TRUE( results[3] == 0);
		EXPECT_TRUE( results[4] == 0);
	}
	{
		gdf_error gdfError = gdf_extract_datetime_minute(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 19);
		EXPECT_TRUE( results[1] == 19);
		EXPECT_TRUE( results[2] == 59);
		EXPECT_TRUE( results[3] == 0);
		EXPECT_TRUE( results[4] == 0);
	}
	{
		gdf_error gdfError = gdf_extract_datetime_second(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		std::vector<int16_t> results(colSize);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 50);
		EXPECT_TRUE( results[1] == 59);
		EXPECT_TRUE( results[2] == 59);
		EXPECT_TRUE( results[3] == 1);
		EXPECT_TRUE( results[4] == 0);
	}
}
