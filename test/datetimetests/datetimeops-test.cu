#include <cstdlib>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>



TEST(gdf_extract_datetime_TEST, date64Tests) {

	int colSize = 8;

	gdf_column inputCol;
	gdf_column outputCol;

	inputCol.dtype = GDF_DATE64;
	inputCol.size = colSize;
	outputCol.dtype = GDF_INT16;
	outputCol.size = colSize;

	std::vector<int64_t> inputData(colSize);
	inputData[0] = 1528935590000;	// '2018-06-14 00:19:50.000'
	inputData[1] = 1528935599999;	// '2018-06-14 00:19:59.999'
	inputData[2] = -1577923201000;	// '1919-12-31 23:59:59.000'
	inputData[3] = 1582934401123;   // '2020-02-29 00:00:01.123'
	inputData[4] = 0;               // '1970-01-01 00:00:00.000'
	inputData[5] = 2309653342222;   // '2043-03-11 02:22:22.222'
	inputData[6] = 893032230345;    // '1998-04-20 12:30:30.345'
	inputData[7] = -4870653059987;  // '1815-08-28 16:49:01.987


	thrust::device_vector<int64_t> intputDataDev(inputData);
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
		EXPECT_TRUE( results[1] == 14);
		EXPECT_TRUE( results[2] == 31);
		EXPECT_TRUE( results[3] == 29);
		EXPECT_TRUE( results[4] == 1);
		EXPECT_TRUE( results[5] == 11);
		EXPECT_TRUE( results[6] == 20);
		EXPECT_TRUE( results[7] == 28);
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
		EXPECT_TRUE( results[5] == 2);
		EXPECT_TRUE( results[6] == 12);
		EXPECT_TRUE( results[7] == 16);
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
		EXPECT_TRUE( results[5] == 22);
		EXPECT_TRUE( results[6] == 30);
		EXPECT_TRUE( results[7] == 1);
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
		EXPECT_TRUE( results[5] == 222);
		EXPECT_TRUE( results[6] == 345);
		EXPECT_TRUE( results[7] == 987);
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
	inputData[2] = -18364;	// '1919-12-31'
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
