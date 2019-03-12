/*
 * Copyright 2018 BlazingDB, Inc.
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

#include <cstdlib>
#include <iostream>
#include <vector>
#include <numeric>
#include <limits>
#include <random>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include "gtest/gtest.h"
#include "tests/utilities/cudf_test_fixtures.h"
#include <bitmask/legacy_bitmask.hpp>

#include <cudf.h>
#include <cudf/functions.h>
#include <utilities/cudf_utils.h>
#include <rmm/thrust_rmm_allocator.h>


struct gdf_cast_test : public GdfTest {};

TEST_F(gdf_cast_test, usage_example) {

	// gdf_column input examples for int32, int64, float32, float64, date32, date64 and timestamp (in milliseconds)

	std::vector<int32_t> inputInt32Data = {
		-1528,
		1,
		19382
	};

	std::vector<int64_t> inputInt64Data = {
		-1528,
		1,
		19382
	};

	std::vector<float> inputFloat32Data = {
		-1528.6,
		1.3,
		19382.9
	};

	std::vector<double> inputFloat64Data = {
		-1528,
		17716,
		19382
	};

	std::vector<int32_t> inputDate32Data = {
		-1528, // '1965-10-26'
		17716, // '2018-07-04'
		19382 // '2023-01-25'
	};

	std::vector<int64_t> inputDate64Data = {
		1528935590000, // '2018-06-14 00:19:50.000'
		1528935599999, // '2018-06-14 00:19:59.999'
		-1577923201000, // '1919-12-31 23:59:59.000'
	};

	std::vector<int64_t> inputTimestampMilliData = {
		1528935590000, // '2018-06-14 00:19:50.000'
		1528935599999, // '2018-06-14 00:19:59.999'
		-1577923201000, // '1919-12-31 23:59:59.000'
	};

	int colSize = 3;

	// Input column for int32
	rmm::device_vector<int32_t> inputInt32DataDev(inputInt32Data);
	rmm::device_vector<gdf_valid_type> inputInt32ValidDev(1,255);

	gdf_column inputInt32Col;
	inputInt32Col.dtype = GDF_INT32;
	inputInt32Col.size = colSize;

	inputInt32Col.data = thrust::raw_pointer_cast(inputInt32DataDev.data());
	inputInt32Col.valid = thrust::raw_pointer_cast(inputInt32ValidDev.data());

	// Input column for float32
	rmm::device_vector<float> inputFloat32DataDev(inputFloat32Data);
	rmm::device_vector<gdf_valid_type> inputFloat32ValidDev(1,255);

	gdf_column inputFloat32Col;
	inputFloat32Col.dtype = GDF_FLOAT32;
	inputFloat32Col.size = colSize;

	inputFloat32Col.data = thrust::raw_pointer_cast(inputFloat32DataDev.data());
	inputFloat32Col.valid = thrust::raw_pointer_cast(inputFloat32ValidDev.data());

	// Input column for int64
	rmm::device_vector<int64_t> inputInt64DataDev(inputInt64Data);
	rmm::device_vector<gdf_valid_type> inputInt64ValidDev(1,255);

	gdf_column inputInt64Col;
	inputInt64Col.dtype = GDF_INT64;
	inputInt64Col.size = colSize;

	inputInt64Col.data = thrust::raw_pointer_cast(inputInt64DataDev.data());
	inputInt64Col.valid = thrust::raw_pointer_cast(inputInt64ValidDev.data());

	// Input column for date32
	rmm::device_vector<int32_t> inputDate32DataDev(inputDate32Data);
	rmm::device_vector<gdf_valid_type> inputDate32ValidDev(1,255);

	gdf_column inputDate32Col;
	inputDate32Col.dtype = GDF_DATE32;
	inputDate32Col.size = colSize;

	inputDate32Col.data = thrust::raw_pointer_cast(inputDate32DataDev.data());
	inputDate32Col.valid = thrust::raw_pointer_cast(inputDate32ValidDev.data());

	// Input column for date64
	rmm::device_vector<int64_t> inputDate64DataDev(inputDate64Data);
	rmm::device_vector<gdf_valid_type> inputDate64ValidDev(1,255);

	gdf_column inputDate64Col;
	inputDate64Col.dtype = GDF_DATE64;
	inputDate64Col.size = colSize;

	inputDate64Col.data = thrust::raw_pointer_cast(inputDate64DataDev.data());
	inputDate64Col.valid = thrust::raw_pointer_cast(inputDate64ValidDev.data());

	// Input column for timestamp in ms
	rmm::device_vector<int64_t> inputTimestampMilliDataDev(inputTimestampMilliData);
	rmm::device_vector<gdf_valid_type> inputTimestampMilliValidDev(1,255);

	gdf_column inputTimestampMilliCol;
	inputTimestampMilliCol.dtype = GDF_TIMESTAMP;
	inputTimestampMilliCol.size = colSize;
	inputTimestampMilliCol.dtype_info.time_unit = TIME_UNIT_ms;

	inputTimestampMilliCol.data = thrust::raw_pointer_cast(inputTimestampMilliDataDev.data());
	inputTimestampMilliCol.valid = thrust::raw_pointer_cast(inputTimestampMilliValidDev.data());

	gdf_error gdfError;

	// example for gdf_error gdf_cast_generic_to_f32(gdf_column *input, gdf_column *output)
	{
		// Output column
		rmm::device_vector<float> outDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outValidDev(1,0);

		gdf_column outputFloat32Col;
		outputFloat32Col.dtype = GDF_FLOAT32;
		outputFloat32Col.size = colSize;

		outputFloat32Col.data = thrust::raw_pointer_cast(outDataDev.data());
		outputFloat32Col.valid = thrust::raw_pointer_cast(outValidDev.data());

		std::vector<float> results(colSize);

		// from int32
		gdfError = gdf_cast_generic_to_f32(&inputInt32Col, &outputFloat32Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == -1528.0 );
		EXPECT_TRUE( results[1] == 1.0 );
		EXPECT_TRUE( results[2] == 19382.0 );
	}

	// example for gdf_error gdf_cast_generic_to_i32(gdf_column *input, gdf_column *output);
	{
		// Output column
		rmm::device_vector<int32_t> outDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outValidDev(1,0);

		gdf_column outputInt32Col;
		outputInt32Col.dtype = GDF_INT32;
		outputInt32Col.size = colSize;

		outputInt32Col.data = thrust::raw_pointer_cast(outDataDev.data());
		outputInt32Col.valid = thrust::raw_pointer_cast(outValidDev.data());

		std::vector<int32_t> results(colSize);

		// from float32
		gdfError = gdf_cast_generic_to_i32(&inputFloat32Col, &outputInt32Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == -1528 );
		EXPECT_TRUE( results[1] == 1 );
		EXPECT_TRUE( results[2] == 19382 );
	}

	// example for gdf_error gdf_cast_generic_to_i64(gdf_column *input, gdf_column *output) - upcast
	{
		// Output column
		rmm::device_vector<int64_t> outDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outValidDev(1,0);

		gdf_column outputInt64Col;
		outputInt64Col.dtype = GDF_INT64;
		outputInt64Col.size = colSize;

		outputInt64Col.data = thrust::raw_pointer_cast(outDataDev.data());
		outputInt64Col.valid = thrust::raw_pointer_cast(outValidDev.data());

		std::vector<int64_t> results(colSize);

		// from int32
		gdfError = gdf_cast_generic_to_i64(&inputInt32Col, &outputInt64Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == -1528 );
		EXPECT_TRUE( results[1] == 1 );
		EXPECT_TRUE( results[2] == 19382 );
	}

	// example for gdf_error gdf_cast_generic_to_i32(gdf_column *input, gdf_column *output) - downcast
	{
		// Output column
		rmm::device_vector<int32_t> outDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outValidDev(1,0);

		gdf_column outputInt32Col;
		outputInt32Col.dtype = GDF_INT32;
		outputInt32Col.size = colSize;

		outputInt32Col.data = thrust::raw_pointer_cast(outDataDev.data());
		outputInt32Col.valid = thrust::raw_pointer_cast(outValidDev.data());

		std::vector<int32_t> results(colSize);

		// from int64
		gdfError = gdf_cast_generic_to_i32(&inputInt64Col, &outputInt32Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == -1528 );
		EXPECT_TRUE( results[1] == 1 );
		EXPECT_TRUE( results[2] == 19382 );
	}

	// example for gdf_error gdf_cast_generic_to_i32(gdf_column *input, gdf_column *output)
	{
		// Output column
		rmm::device_vector<int32_t> outDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outValidDev(1,0);

		gdf_column outputInt32Col;
		outputInt32Col.dtype = GDF_INT32;
		outputInt32Col.size = colSize;

		outputInt32Col.data = thrust::raw_pointer_cast(outDataDev.data());
		outputInt32Col.valid = thrust::raw_pointer_cast(outValidDev.data());

		std::vector<int32_t> results(colSize);

		// from date32
		gdfError = gdf_cast_generic_to_i32(&inputDate32Col, &outputInt32Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == -1528 );
		EXPECT_TRUE( results[1] == 17716 );
		EXPECT_TRUE( results[2] == 19382 );
	}

	// example for gdf_error gdf_cast_generic_to_date32(gdf_column *input, gdf_column *output)
	{
		// Output column
		rmm::device_vector<int32_t> outDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outValidDev(1,0);

		gdf_column outputDate32Col;
		outputDate32Col.dtype = GDF_DATE32;
		outputDate32Col.size = colSize;

		outputDate32Col.data = thrust::raw_pointer_cast(outDataDev.data());
		outputDate32Col.valid = thrust::raw_pointer_cast(outValidDev.data());

		std::vector<int32_t> results(colSize);

		// from int32
		gdfError = gdf_cast_generic_to_date32(&inputInt32Col, &outputDate32Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == -1528 );
		EXPECT_TRUE( results[1] == 1 );
		EXPECT_TRUE( results[2] == 19382 );
	}

	// example for gdf_error gdf_cast_generic_to_timestamp(gdf_column *input, gdf_column *output, gdf_time_unit time_unit)
	{
		// Output column
		rmm::device_vector<int64_t> outDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outValidDev(1,0);

		gdf_column outputTimestampMicroCol;
		outputTimestampMicroCol.dtype = GDF_TIMESTAMP;
		outputTimestampMicroCol.size = colSize;

		outputTimestampMicroCol.data = thrust::raw_pointer_cast(outDataDev.data());
		outputTimestampMicroCol.valid = thrust::raw_pointer_cast(outValidDev.data());

		std::vector<int64_t> results(colSize);

		// from date64
		gdfError = gdf_cast_generic_to_timestamp(&inputDate64Col, &outputTimestampMicroCol, TIME_UNIT_us);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 1528935590000000 ); // '2018-06-14 00:19:50.000000'
		EXPECT_TRUE( results[1] == 1528935599999000 ); // '2018-06-14 00:19:59.999000'
		EXPECT_TRUE( results[2] == -1577923201000000 ); // '1919-12-31 23:59:59.000000'
	}

	// example for gdf_error gdf_cast_generic_to_date32(gdf_column *input, gdf_column *output)
	{
		// Output column
		rmm::device_vector<int32_t> outDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outValidDev(1,0);

		gdf_column outputDate32Col;
		outputDate32Col.dtype = GDF_DATE32;
		outputDate32Col.size = colSize;

		outputDate32Col.data = thrust::raw_pointer_cast(outDataDev.data());
		outputDate32Col.valid = thrust::raw_pointer_cast(outValidDev.data());

		std::vector<int32_t> results(colSize);

		// from timestamp in ms
		gdfError = gdf_cast_generic_to_date32(&inputTimestampMilliCol, &outputDate32Col);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );

		results.clear();
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == 17696 ); // '2018-06-14'
		EXPECT_TRUE( results[1] == 17696 ); // '2018-06-14'
		EXPECT_TRUE( results[2] == -18264 ); // '1919-12-31'
	}
}

// Use partial template specialization to choose the uniform_real_distribution type without 
// warnings for unused code paths.
// Generic case for all combinations where one is not a double
template <typename TOUT, typename TFROM> struct DisType { typedef float value_type; };

// Specialization for both doubles
template <> struct DisType<double, double> { typedef double value_type; };
// Specializations for one type is double
template <typename TOUT> struct DisType<TOUT, double> { typedef double value_type; };
template <typename TFROM> struct DisType<double, TFROM> { typedef double value_type; };

// Generates random values between 0 and the maximum possible value of the data type with the minimum max() value
template<typename TOUT, typename TFROM>
void fill_with_random_values(std::vector<TFROM>& input, size_t size)
{
	std::random_device rd;
	std::default_random_engine eng(rd());

	using T = typename DisType<TOUT, TFROM>::value_type;
	std::uniform_real_distribution<T> floating_dis;
	if( std::numeric_limits<TFROM>::max() < std::numeric_limits<TOUT>::max() )
		floating_dis = std::uniform_real_distribution<T>(std::numeric_limits<TFROM>::min(), std::numeric_limits<TFROM>::max());
	else
		floating_dis = std::uniform_real_distribution<T>(std::numeric_limits<TOUT>::min(), std::numeric_limits<TOUT>::max());

	std::generate(input.begin(), input.end(), [floating_dis, eng]() mutable {
		return static_cast<TFROM>(floating_dis(eng));
	});
}

// Generates random bitmaps
void fill_random_bitmap(std::vector<gdf_valid_type>& valid_input, size_t size)
{
	std::random_device rd;
	std::default_random_engine eng(rd());

	std::uniform_int_distribution<gdf_valid_type> int_dis;
	int_dis = std::uniform_int_distribution<gdf_valid_type>(std::numeric_limits<gdf_valid_type>::min(), std::numeric_limits<gdf_valid_type>::max());

	std::generate(valid_input.begin(), valid_input.end(), [int_dis, eng]() mutable {
		return int_dis(eng);
	});
}

// CPU casting

struct gdf_cast_CPU_VS_GPU_TEST : public GdfTest {};

template<typename T, typename Tout>
struct HostUnaryOp {
    static
    gdf_error launch(gdf_column *input, gdf_column *output) {
        /* check for size of the columns */
        if (input->size != output->size) {
            return GDF_COLUMN_SIZE_MISMATCH;
        }

        std::transform((const T*)input->data, (const T*)input->data + input->size, (Tout*)output->data, [](T data) -> Tout { return (Tout)data; });
        return GDF_SUCCESS;
    }
};

#define DEF_HOST_CAST_IMPL(VFROM, VTO, TFROM, TTO)                                 \
gdf_error gdf_host_cast_##VFROM##_to_##VTO(gdf_column *input, gdf_column *output)  \
{ return HostUnaryOp<TFROM, TTO>::launch(input, output); }

// Comparing CPU and GPU casting results
#define DEF_CAST_IMPL_TEST(VFROM, VTO, VVFROM, VVTO, TFROM, TTO)				\
	TEST_F(gdf_cast_CPU_VS_GPU_TEST, VFROM##_to_##VTO) {							\
	{																			\
		int colSize = 1024;														\
		gdf_column inputCol;													\
		gdf_column outputCol;													\
																				\
		inputCol.dtype = VVFROM;												\
		inputCol.size = colSize;												\
		outputCol.dtype = VVTO;													\
		outputCol.size = colSize;												\
																				\
		std::vector<TFROM> inputData(colSize);									\
		fill_with_random_values<TTO, TFROM>(inputData, colSize);				\
																				\
		rmm::device_vector<TFROM> inputDataDev(inputData);					\
		rmm::device_vector<TTO> outputDataDev(colSize);						\
																				\
		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());			\
		inputCol.valid = nullptr;												\
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());		\
		outputCol.valid = nullptr;												\
																				\
		gdf_error gdfError = gdf_cast_##VFROM##_to_##VTO(&inputCol, &outputCol);\
		EXPECT_TRUE( gdfError == GDF_SUCCESS );									\
																				\
		std::vector<TTO> results(colSize);										\
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());	\
																				\
		std::vector<TTO> outputData(colSize);									\
		inputCol.data = inputData.data();										\
																				\
		gdf_column outputColHost;												\
		outputColHost.dtype = VVTO;												\
		outputColHost.size = colSize;											\
		outputColHost.data = static_cast<void*>(outputData.data());				\
																				\
		gdfError = gdf_host_cast_##VFROM##_to_##VTO(&inputCol, &outputColHost);	\
		EXPECT_TRUE( gdfError == GDF_SUCCESS );									\
																				\
		for (int i = 0; i < colSize; i++){										\
			EXPECT_TRUE( results[i] == outputData[i] );							\
		}																		\
	}																			\
}

#define DEF_HOST_CAST_IMPL_TEMPLATE(ABREV, PHYSICAL_TYPE)  	\
DEF_HOST_CAST_IMPL(i8,        ABREV,  int8_t, PHYSICAL_TYPE)\
DEF_HOST_CAST_IMPL(i32,       ABREV, int32_t, PHYSICAL_TYPE)\
DEF_HOST_CAST_IMPL(i64,       ABREV, int64_t, PHYSICAL_TYPE)\
DEF_HOST_CAST_IMPL(f32,       ABREV,   float, PHYSICAL_TYPE)\
DEF_HOST_CAST_IMPL(f64,       ABREV,  double, PHYSICAL_TYPE)\
DEF_HOST_CAST_IMPL(date32,    ABREV, int32_t, PHYSICAL_TYPE)\
DEF_HOST_CAST_IMPL(date64,    ABREV, int64_t, PHYSICAL_TYPE)\
DEF_HOST_CAST_IMPL(timestamp, ABREV, int64_t, PHYSICAL_TYPE)

#define DEF_CAST_TYPE_TEST(ABREV, LOGICAL_TYPE, PHYSICAL_TYPE) 							\
DEF_HOST_CAST_IMPL_TEMPLATE(ABREV, PHYSICAL_TYPE) 										\
DEF_CAST_IMPL_TEST(i8, ABREV, GDF_INT8, LOGICAL_TYPE,  int8_t, PHYSICAL_TYPE) 			\
DEF_CAST_IMPL_TEST(i32, ABREV, GDF_INT32, LOGICAL_TYPE, int32_t, PHYSICAL_TYPE) 		\
DEF_CAST_IMPL_TEST(i64, ABREV, GDF_INT64, LOGICAL_TYPE,  int64_t, PHYSICAL_TYPE) 		\
DEF_CAST_IMPL_TEST(f32, ABREV, GDF_FLOAT32, LOGICAL_TYPE, float, PHYSICAL_TYPE) 		\
DEF_CAST_IMPL_TEST(f64, ABREV, GDF_FLOAT64, LOGICAL_TYPE, double, PHYSICAL_TYPE) 		\
DEF_CAST_IMPL_TEST(date32, ABREV, GDF_DATE32, LOGICAL_TYPE, int32_t, PHYSICAL_TYPE) 	\
DEF_CAST_IMPL_TEST(date64, ABREV, GDF_DATE64, LOGICAL_TYPE, int64_t, PHYSICAL_TYPE) 	\
DEF_CAST_IMPL_TEST(timestamp, ABREV, GDF_TIMESTAMP, LOGICAL_TYPE, int64_t, PHYSICAL_TYPE)

DEF_CAST_TYPE_TEST(i8, GDF_INT8, int8_t)
DEF_CAST_TYPE_TEST(i32, GDF_INT32, int32_t)
DEF_CAST_TYPE_TEST(i64, GDF_INT64, int64_t)
DEF_CAST_TYPE_TEST(f32, GDF_FLOAT32, float)
DEF_CAST_TYPE_TEST(f64, GDF_FLOAT64, double)

struct gdf_cast_swap_TEST : public GdfTest {};

// Casting from T1 to T2, and then casting from T2 to T1 results in the same value 
#define DEF_CAST_SWAP_TEST(VFROM, VTO, VVFROM, VVTO, TFROM, TTO)				\
	TEST_F(gdf_cast_swap_TEST, VFROM##_to_##VTO) {								\
	{																			\
		int colSize = 1024;														\
		gdf_column inputCol;													\
		gdf_column outputCol;													\
		gdf_column originalOutputCol;											\
																				\
		inputCol.dtype = VVFROM;												\
		inputCol.size = colSize;												\
		outputCol.dtype = VVTO;													\
		outputCol.size = colSize;												\
		originalOutputCol.dtype = VVFROM;										\
		originalOutputCol.size = colSize;										\
																				\
		std::vector<TFROM> inputData(colSize);									\
		fill_with_random_values<TTO, TFROM>(inputData, colSize);				\
																				\
		rmm::device_vector<TFROM> inputDataDev(inputData);					\
		rmm::device_vector<TTO> outputDataDev(colSize);						\
		rmm::device_vector<TFROM> origOutputDataDev(colSize);				\
																				\
		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());			\
		inputCol.valid = nullptr;												\
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());		\
		outputCol.valid = nullptr;												\
		originalOutputCol.data = thrust::raw_pointer_cast(origOutputDataDev.data());\
		originalOutputCol.valid = nullptr;										\
																				\
		gdf_error gdfError = gdf_cast_##VFROM##_to_##VTO(&inputCol, &outputCol);\
		EXPECT_TRUE( gdfError == GDF_SUCCESS );									\
		gdfError = gdf_cast_##VTO##_to_##VFROM(&outputCol, &originalOutputCol);	\
		EXPECT_TRUE( gdfError == GDF_SUCCESS );									\
																				\
		std::vector<TFROM> results(colSize);									\
		thrust::copy(origOutputDataDev.begin(), origOutputDataDev.end(), results.begin());\
																				\
		for (int i = 0; i < colSize; i++){										\
			EXPECT_TRUE( results[i] == inputData[i] );							\
		}																		\
																				\
		EXPECT_TRUE( gdfError == GDF_SUCCESS );									\
	}																			\
}

// Casting from T1 to T2, and then casting from T2 to T1 results in the same value
#define DEF_CAST_SWAP_TEST_TO_TIMESTAMP(VFROM, VVFROM, TFROM)				\
	TEST_F(gdf_cast_swap_TEST, VFROM##_to_timestamp) {								\
	{																			\
		int colSize = 1024;														\
		gdf_column inputCol;													\
		gdf_column outputCol;													\
		gdf_column originalOutputCol;											\
																				\
		inputCol.dtype = VVFROM;												\
		inputCol.size = colSize;												\
		outputCol.dtype = GDF_TIMESTAMP;													\
		outputCol.size = colSize;												\
		originalOutputCol.dtype = VVFROM;										\
		originalOutputCol.size = colSize;										\
																				\
		std::vector<TFROM> inputData(colSize);									\
		fill_with_random_values<int64_t, TFROM>(inputData, colSize);				\
																				\
		rmm::device_vector<TFROM> inputDataDev(inputData);					\
		rmm::device_vector<int64_t> outputDataDev(colSize);						\
		rmm::device_vector<TFROM> origOutputDataDev(colSize);				\
																				\
		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());			\
		inputCol.valid = nullptr;												\
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());		\
		outputCol.valid = nullptr;												\
		originalOutputCol.data = thrust::raw_pointer_cast(origOutputDataDev.data());\
		originalOutputCol.valid = nullptr;										\
																				\
		gdf_error gdfError = gdf_cast_##VFROM##_to_timestamp(&inputCol, &outputCol, TIME_UNIT_ms);\
		EXPECT_TRUE( gdfError == GDF_SUCCESS );									\
		gdfError = gdf_cast_timestamp_to_##VFROM(&outputCol, &originalOutputCol);	\
		EXPECT_TRUE( gdfError == GDF_SUCCESS );									\
																				\
		std::vector<TFROM> results(colSize);									\
		thrust::copy(origOutputDataDev.begin(), origOutputDataDev.end(), results.begin());\
																				\
		for (int i = 0; i < colSize; i++){										\
			EXPECT_TRUE( results[i] == inputData[i] );							\
		}																		\
																				\
		EXPECT_TRUE( gdfError == GDF_SUCCESS );									\
	}																			\
}

DEF_CAST_SWAP_TEST(i8, i32, GDF_INT8, GDF_INT32,  int8_t, int32_t)
DEF_CAST_SWAP_TEST(i8, i64, GDF_INT8, GDF_INT64,  int8_t, int64_t)
DEF_CAST_SWAP_TEST(i8, f32, GDF_INT8, GDF_FLOAT32,  int8_t, float)
DEF_CAST_SWAP_TEST(i8, f64, GDF_INT8, GDF_FLOAT64,  int8_t, double)
DEF_CAST_SWAP_TEST(i8, date32, GDF_INT8, GDF_DATE32,  int8_t, int32_t)
DEF_CAST_SWAP_TEST(i8, date64, GDF_INT8, GDF_DATE64,  int8_t, int64_t)
DEF_CAST_SWAP_TEST(i32, i64, GDF_INT32, GDF_INT64,  int32_t, int64_t)
DEF_CAST_SWAP_TEST(i32, f64, GDF_INT32, GDF_FLOAT64,  int32_t, double)
DEF_CAST_SWAP_TEST(i32, f32, GDF_INT32, GDF_FLOAT32,  int32_t, float)
DEF_CAST_SWAP_TEST(f32, f64, GDF_FLOAT32, GDF_FLOAT64,  float, double)
DEF_CAST_SWAP_TEST(date32, date64, GDF_DATE32, GDF_DATE64,  int32_t, int64_t)

DEF_CAST_SWAP_TEST_TO_TIMESTAMP(i8, GDF_INT8, int8_t)
DEF_CAST_SWAP_TEST_TO_TIMESTAMP(date32, GDF_DATE32, int32_t)
DEF_CAST_SWAP_TEST_TO_TIMESTAMP(date64, GDF_DATE64, int64_t)

struct generateValidRandom
{
    __device__
    gdf_valid_type operator () (int idx)
    {
        thrust::default_random_engine eng;
        thrust::uniform_int_distribution<gdf_valid_type> int_dis;
        eng.discard(idx);
        return int_dis(eng);
    }
};

struct gdf_unaryops_output_valid_TEST : public GdfTest {};

TEST_F(gdf_unaryops_output_valid_TEST, checkingValidAndDtype) {

	//The output datatype is set by the casting function
	{
		const int colSize = 1024;
		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_FLOAT32;
		inputCol.size = colSize;
		outputCol.size = colSize;

		std::vector<float> inputData(colSize);
		fill_with_random_values<double, float>(inputData, colSize);

		rmm::device_vector<float> inputDataDev(inputData);
		rmm::device_vector<double> outputDataDev(colSize);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = nullptr;
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = nullptr;

		gdf_error gdfError = gdf_cast_f32_to_f64(&inputCol, &outputCol);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_FLOAT64 );
	}

	//The input and output valid bitmaps are equal
	{
		const int colSize = 1024;
		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_FLOAT32;
		inputCol.size = colSize;
		outputCol.dtype = GDF_FLOAT32;
		outputCol.size = colSize;

		std::vector<float> inputData(colSize);
		fill_with_random_values<float, float>(inputData, colSize);

		rmm::device_vector<float> inputDataDev(inputData);
		rmm::device_vector<float> outputDataDev(colSize);

		rmm::device_vector<gdf_valid_type> inputValidDev(gdf_valid_allocation_size(inputCol.size));
		rmm::device_vector<gdf_valid_type> outputValidDev(gdf_valid_allocation_size(inputCol.size));

        thrust::transform(thrust::make_counting_iterator(static_cast<gdf_size_type>(0)),
                          thrust::make_counting_iterator( gdf_num_bitmask_elements(inputCol.size)),
                          inputValidDev.begin(), generateValidRandom());

        inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_f32_to_f32(&inputCol, &outputCol);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_FLOAT32 );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.begin() + gdf_num_bitmask_elements(inputCol.size), outputValidDev.begin());

		EXPECT_TRUE( result == true );
	}

	//Testing with a colSize not divisible by 8
	{
		const int colSize = 1000;
		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_FLOAT32;
		inputCol.size = colSize;
		outputCol.dtype = GDF_FLOAT32;
		outputCol.size = colSize;

		std::vector<float> inputData(colSize);
		fill_with_random_values<float, float>(inputData, colSize);

		rmm::device_vector<float> inputDataDev(inputData);
		rmm::device_vector<float> outputDataDev(colSize);

		rmm::device_vector<gdf_valid_type> inputValidDev(gdf_valid_allocation_size(inputCol.size));
		rmm::device_vector<gdf_valid_type> outputValidDev(gdf_valid_allocation_size(inputCol.size));

        thrust::transform(thrust::make_counting_iterator( static_cast<gdf_size_type>(0)),
                        thrust::make_counting_iterator( gdf_num_bitmask_elements(inputCol.size)),
                        inputValidDev.begin(), generateValidRandom());

        inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_f32_to_f32(&inputCol, &outputCol);
		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_FLOAT32 );

        bool result =
            thrust::equal(inputValidDev.begin(),
                            inputValidDev.begin() +
                                gdf_num_bitmask_elements(inputCol.size),
                            outputValidDev.begin());

        EXPECT_TRUE( result );

		std::vector<float> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputDataDev[i] );
		}
	}
}

struct gdf_date_casting_TEST : public GdfTest {};

TEST_F(gdf_date_casting_TEST, date32_to_date64) {

	//date32 to date64
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_DATE32;
		inputCol.size = colSize;
		outputCol.dtype = GDF_DATE64;
		outputCol.size = colSize;

		std::vector<int32_t> inputData = {
			17696,	// '2018-06-14'
			17697,	// '2018-06-15'
			-18264,	// '1919-12-31'
			18321,   // '2020-02-29'
			0,       // '1970-01-01'
			26732,   // '2043-03-11'
			10336,    // '1998-04-20'
			-56374  // '1815-08-28
		};

		std::vector<int64_t> outputData = {
			1528934400000,	// '2018-06-14 00:00:00.000'
			1529020800000,	// '2018-06-15 00:00:00.000'
			-1578009600000,	// '1919-12-31 00:00:00.000'
			1582934400000,   // '2020-02-29 00:00:00.000'
			0,            // '1970-01-01 00:00:00.000'
			2309644800000,   // '2043-03-11 00:00:00.000'
			893030400000,    // '1998-04-20 00:00:00.000'
			-4870713600000  // '1815-08-28 00:00:00.000'
		};

		rmm::device_vector<int32_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_date32_to_date64(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_DATE64 );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	// date64 to date32
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_DATE64;
		inputCol.size = colSize;
		outputCol.dtype = GDF_DATE32;
		outputCol.size = colSize;

		// timestamps with milliseconds
		std::vector<int64_t> inputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058987,  // '1815-08-28 16:49:01.013'
			-4500,            // '1969-12-31 23:59:55.500'
			-169138999,    // '1969-12-30 01:01:01.001'
			-5999,        // '1969-12-31 23:59:54.001'
			-1991063752000, //	'1906-11-28 06:44:08'
			-1954281039000, //	'1908-01-28 00:09:21'
			-1669612095000, //	'1917-02-03 18:51:45'
			-1184467876000, //	'1932-06-19 21:08:44'
			362079575000, //	'1981-06-22 17:39:35'
			629650040000, //	'1989-12-14 14:47:20'
			692074060000, //	'1991-12-07 02:47:40'
			734734764000, //	'1993-04-13 20:59:24'
			1230998894000, //	'2009-01-03 16:08:14'
			1521989991000, //	'2018-03-25 14:59:51'
			1726355294000, //	'2024-09-14 23:08:14'
			-1722880051000, //	'1915-05-29 06:12:29'
			-948235893000, //	'1939-12-15 01:08:27'
			-811926962000, //	'1944-04-09 16:43:58'
			-20852065000, //	'1969-05-04 15:45:35'
			191206704000, //	'1976-01-23 00:58:24'
			896735912000, //	'1998-06-01 21:18:32'
			1262903093000, //	'2010-01-07 22:24:53'
			1926203568000 //	'2031-01-15 00:32:48'
		};

		std::vector<int32_t> outputData = {
			17696,	// '2018-06-14'
			17696,	// '2018-06-14'
			-18264,	// '1919-12-31'
			18321,  // '2020-02-29'
			0,      // '1970-01-01'
			26732,  // '2043-03-11'
			10336,  // '1998-04-20'
			-56374, // '1815-08-28'
			-1,		// '1969-12-31'
			-2,		// '1969-12-30'
			-1,		// '1969-12-31'
			-23045,	// '1906-11-28'
			-22619,	// '1908-01-28'
			-19325,	// '1917-02-03'
			-13710,	// '1932-06-19'
			4190,	// '1981-06-22'
			7287,	// '1989-12-14'
			8010,	// '1991-12-07'
			8503,	// '1993-04-13'
			14247,	// '2009-01-03'
			17615,	// '2018-03-25'
			19980,	// '2024-09-14'
			-19941,	// '1915-05-29'
			-10975,	// '1939-12-15'
			-9398,	// '1944-04-09'
			-242,	// '1969-05-04'
			2213,	// '1976-01-23'
			10378,	// '1998-06-01'
			14616,	// '2010-01-07'
			22294	// '2031-01-15'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int32_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_date64_to_date32(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_DATE32 );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int32_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}
}

TEST_F(gdf_date_casting_TEST, date32_to_date64_over_valid_bitmask) {

	//date32 to date64 over valid bitmask
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_DATE32;
		inputCol.size = colSize;
		outputCol.dtype = GDF_DATE64;
		outputCol.size = colSize;

		std::vector<int32_t> inputData = {
			17696,	// '2018-06-14'
			17697,	// '2018-06-15'
			-18264,	// '1919-12-31'
			18321,   // '2020-02-29'
			0,       // '1970-01-01'
			26732,   // '2043-03-11'
			10336,    // '1998-04-20'
			-56374  // '1815-08-28
		};

		std::vector<int64_t> outputData = {
			1528934400000,	// '2018-06-14 00:00:00.000'
			0, // no operation
			-1578009600000,	// '1919-12-31 00:00:00.000'
			0, // no operation
			0,            // '1970-01-01 00:00:00.000'
			0, // no operation
			893030400000,    // '1998-04-20 00:00:00.000'
			0 // no operation
		};

		rmm::device_vector<int32_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,85); //01010101
		rmm::device_vector<int64_t> outputDataDev(colSize, 0);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_date32_to_date64(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_DATE64 );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}
}

TEST_F(gdf_date_casting_TEST, date32_to_timestamp) {

	//date32 to timestamp s
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_DATE32;
		inputCol.size = colSize;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		std::vector<int32_t> inputData = {
			17696,	// '2018-06-14'
			17697,	// '2018-06-15'
			-18264,	// '1919-12-31'
			18321,   // '2020-02-29'
			0,       // '1970-01-01'
			26732,   // '2043-03-11'
			10336,    // '1998-04-20'
			-56374  // '1815-08-28
		};

		std::vector<int64_t> outputData = {
			1528934400,	// '2018-06-14 00:00:00'
			1529020800,	// '2018-06-15 00:00:00'
			-1578009600,	// '1919-12-31 00:00:00'
			1582934400,   // '2020-02-29 00:00:00'
			0,            // '1970-01-01 00:00:00'
			2309644800,   // '2043-03-11 00:00:00'
			893030400,    // '1998-04-20 00:00:00'
			-4870713600  // '1815-08-28 00:00:00'
		};

		rmm::device_vector<int32_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_date32_to_timestamp(&inputCol, &outputCol, TIME_UNIT_s);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_s );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp s to date32
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_s;
		outputCol.dtype = GDF_DATE32;
		outputCol.size = colSize;

		std::vector<int64_t> inputData = {
			1528934400,	// '2018-06-14 00:00:00'
			1529020800,	// '2018-06-15 00:00:00'
			-1578009600,	// '1919-12-31 00:00:00'
			1582934400,   // '2020-02-29 00:00:00'
			0,            // '1970-01-01 00:00:00'
			2309644800,   // '2043-03-11 00:00:00'
			893030400,    // '1998-04-20 00:00:00'
			-4870713600  // '1815-08-28 00:00:00'
		};

		std::vector<int32_t> outputData = {
			17696,	// '2018-06-14'
			17697,	// '2018-06-15'
			-18264,	// '1919-12-31'
			18321,   // '2020-02-29'
			0,       // '1970-01-01'
			26732,   // '2043-03-11'
			10336,    // '1998-04-20'
			-56374  // '1815-08-28
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int32_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_date32(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_DATE32 );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int32_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}
	
	//date32 to timestamp ms
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_DATE32;
		inputCol.size = colSize;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		std::vector<int32_t> inputData = {
			17696,	// '2018-06-14'
			17697,	// '2018-06-15'
			-18264,	// '1919-12-31'
			18321,   // '2020-02-29'
			0,       // '1970-01-01'
			26732,   // '2043-03-11'
			10336,    // '1998-04-20'
			-56374  // '1815-08-28
		};

		std::vector<int64_t> outputData = {
			1528934400000,	// '2018-06-14 00:00:00.000'
			1529020800000,	// '2018-06-15 00:00:00.000'
			-1578009600000,	// '1919-12-31 00:00:00.000'
			1582934400000,   // '2020-02-29 00:00:00.000'
			0,            // '1970-01-01 00:00:00.000'
			2309644800000,   // '2043-03-11 00:00:00.000'
			893030400000,    // '1998-04-20 00:00:00.000'
			-4870713600000  // '1815-08-28 00:00:00.000'
		};

		rmm::device_vector<int32_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_date32_to_timestamp(&inputCol, &outputCol, TIME_UNIT_ms);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_ms );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp ms to date32
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_ms;
		outputCol.dtype = GDF_DATE32;
		outputCol.size = colSize;

		std::vector<int64_t> inputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058987,  // '1815-08-28 16:49:01.013'
		};

		std::vector<int32_t> outputData = {
			17696,	// '2018-06-14'
			17696,	// '2018-06-14'
			-18264,	// '1919-12-31'
			18321,   // '2020-02-29'
			0,       // '1970-01-01'
			26732,   // '2043-03-11'
			10336,    // '1998-04-20'
			-56374  // '1815-08-28
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int32_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_date32(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_DATE32 );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int32_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//date32 to timestamp ns
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_DATE32;
		inputCol.size = colSize;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		std::vector<int32_t> inputData = {
			17696,	// '2018-06-14'
			17697,	// '2018-06-15'
			-18264,	// '1919-12-31'
			18321,   // '2020-02-29'
			0,       // '1970-01-01'
			26732,   // '2043-03-11'
			10336,    // '1998-04-20'
			-56374  // '1815-08-28
		};

		std::vector<int64_t> outputData = {
			1528934400000000000,	// '2018-06-14 00:00:00.000000000'
			1529020800000000000,	// '2018-06-15 00:00:00.000000000'
			-1578009600000000000,	// '1919-12-31 00:00:00.000000000'
			1582934400000000000,	// '2020-02-29 00:00:00.000000000'
			0,						// '1970-01-01 00:00:00.000000000'
			2309644800000000000,	// '2043-03-11 00:00:00.000000000'
			893030400000000000,		// '1998-04-20 00:00:00.000000000'
			-4870713600000000000	// '1815-08-28 00:00:00.000000000'
		};

		rmm::device_vector<int32_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_date32_to_timestamp(&inputCol, &outputCol, TIME_UNIT_ns);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_ns );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp ns to date32
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_ns;
		outputCol.dtype = GDF_DATE32;
		outputCol.size = colSize;

		std::vector<int64_t> inputData = {
			1528935590000000000, // '2018-06-14 00:19:50.000000000'
			1528935599999999999, // '2018-06-14 00:19:59.999999999'
			-1577923201000000000, // '1919-12-31 23:59:59.000000000'
			1582934401123123123, // '2020-02-29 00:00:01.123123123'
			0,             // '1970-01-01 00:00:00.000000000'
			2309653342222222222, // '2043-03-11 02:22:22.222222222'
			893075430345543345, // '1998-04-20 12:30:30.345543345'
			-4870653058987789987,  // '1815-08-28 16:49:01.012210013'
		};

		std::vector<int32_t> outputData = {
			17696,	// '2018-06-14'
			17696,	// '2018-06-14'
			-18264,	// '1919-12-31'
			18321,   // '2020-02-29'
			0,       // '1970-01-01'
			26732,   // '2043-03-11'
			10336,    // '1998-04-20'
			-56374  // '1815-08-28
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int32_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_date32(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_DATE32 );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int32_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//date32 to timestamp us
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_DATE32;
		inputCol.size = colSize;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		std::vector<int32_t> inputData = {
			17696,	// '2018-06-14'
			17697,	// '2018-06-15'
			-18264,	// '1919-12-31'
			18321,   // '2020-02-29'
			0,       // '1970-01-01'
			26732,   // '2043-03-11'
			10336,    // '1998-04-20'
			-56374  // '1815-08-28
		};

		std::vector<int64_t> outputData = {
			1528934400000000,	// '2018-06-14 00:00:00.000000'
			1529020800000000,	// '2018-06-15 00:00:00.000000'
			-1578009600000000,	// '1919-12-31 00:00:00.000000'
			1582934400000000,   // '2020-02-29 00:00:00.000000'
			0,            // '1970-01-01 00:00:00.000000'
			2309644800000000,   // '2043-03-11 00:00:00.000000'
			893030400000000,    // '1998-04-20 00:00:00.000000'
			-4870713600000000  // '1815-08-28 00:00:00.000000'
		};

		rmm::device_vector<int32_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_date32_to_timestamp(&inputCol, &outputCol, TIME_UNIT_us);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_us );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp us to date32
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_us;
		outputCol.dtype = GDF_DATE32;
		outputCol.size = colSize;

		std::vector<int64_t> inputData = {
			1528935590000000, // '2018-06-14 00:19:50.000000'
			1528935599000000, // '2018-06-14 00:19:59.000000'
			-1577923201000000, // '1919-12-31 23:59:59.000000'
			1582934401000000, // '2020-02-29 00:00:01.000000'
			0,             // '1970-01-01 00:00:00.000000'
			2309653342000000, // '2043-03-11 02:22:22.000000'
			893075430000000, // '1998-04-20 12:30:30.000000'
			-4870653059000000,  // '1815-08-28 16:49:01.000000'
		};

		std::vector<int32_t> outputData = {
			17696,	// '2018-06-14'
			17696,	// '2018-06-14'
			-18264,	// '1919-12-31'
			18321,   // '2020-02-29'
			0,       // '1970-01-01'
			26732,   // '2043-03-11'
			10336,    // '1998-04-20'
			-56374  // '1815-08-28
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int32_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_date32(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_DATE32 );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int32_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}
}

TEST_F(gdf_date_casting_TEST, date64_to_timestamp) {

	// date64 to timestamp ms, internally the output is equal to the input
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_DATE64;
		inputCol.size = colSize;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		// timestamps with milliseconds
		std::vector<int64_t> inputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058987,  // '1815-08-28 16:49:01.013'
			-4500,            // '1969-12-31 23:59:55.500'
			-169138999,    // '1969-12-30 01:01:01.001'
			-5999,        // '1969-12-31 23:59:54.001'
			-1991063752000, //	'1906-11-28 06:44:08'
			-1954281039000, //	'1908-01-28 00:09:21'
			-1669612095000, //	'1917-02-03 18:51:45'
			-1184467876000, //	'1932-06-19 21:08:44'
			362079575000, //	'1981-06-22 17:39:35'
			629650040000, //	'1989-12-14 14:47:20'
			692074060000, //	'1991-12-07 02:47:40'
			734734764000, //	'1993-04-13 20:59:24'
			1230998894000, //	'2009-01-03 16:08:14'
			1521989991000, //	'2018-03-25 14:59:51'
			1726355294000, //	'2024-09-14 23:08:14'
			-1722880051000, //	'1915-05-29 06:12:29'
			-948235893000, //	'1939-12-15 01:08:27'
			-811926962000, //	'1944-04-09 16:43:58'
			-20852065000, //	'1969-05-04 15:45:35'
			191206704000, //	'1976-01-23 00:58:24'
			896735912000, //	'1998-06-01 21:18:32'
			1262903093000, //	'2010-01-07 22:24:53'
			1926203568000 //	'2031-01-15 00:32:48'
		};

		std::vector<int64_t> outputData(colSize);

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_date64_to_timestamp(&inputCol, &outputCol, TIME_UNIT_ms);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit = TIME_UNIT_ms );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == inputData[i] );
		}
	}

	// timestamp ms to date64, internally the output is equal to the input
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_ms;
		outputCol.dtype = GDF_DATE64;
		outputCol.size = colSize;

		// timestamps with milliseconds
		std::vector<int64_t> inputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058987,  // '1815-08-28 16:49:01.013'
			-4500,            // '1969-12-31 23:59:55.500'
			-169138999,    // '1969-12-30 01:01:01.001'
			-5999,        // '1969-12-31 23:59:54.001'
			-1991063752000, //	'1906-11-28 06:44:08'
			-1954281039000, //	'1908-01-28 00:09:21'
			-1669612095000, //	'1917-02-03 18:51:45'
			-1184467876000, //	'1932-06-19 21:08:44'
			362079575000, //	'1981-06-22 17:39:35'
			629650040000, //	'1989-12-14 14:47:20'
			692074060000, //	'1991-12-07 02:47:40'
			734734764000, //	'1993-04-13 20:59:24'
			1230998894000, //	'2009-01-03 16:08:14'
			1521989991000, //	'2018-03-25 14:59:51'
			1726355294000, //	'2024-09-14 23:08:14'
			-1722880051000, //	'1915-05-29 06:12:29'
			-948235893000, //	'1939-12-15 01:08:27'
			-811926962000, //	'1944-04-09 16:43:58'
			-20852065000, //	'1969-05-04 15:45:35'
			191206704000, //	'1976-01-23 00:58:24'
			896735912000, //	'1998-06-01 21:18:32'
			1262903093000, //	'2010-01-07 22:24:53'
			1926203568000 //	'2031-01-15 00:32:48'
		};

		std::vector<int64_t> outputData(colSize);

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_date64(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_DATE64 );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == inputData[i] );
		}
	}

	//date64 to timestamp s
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_DATE64;
		inputCol.size = colSize;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		std::vector<int64_t> inputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058987,  // '1815-08-28 16:49:01.013'
		};

		std::vector<int64_t> outputData = {
			1528935590, // '2018-06-14 00:19:50'
			1528935599, // '2018-06-14 00:19:59'
			-1577923201, // '1919-12-31 23:59:59'
			1582934401, // '2020-02-29 00:00:01'
			0,             // '1970-01-01 00:00:00'
			2309653342, // '2043-03-11 02:22:22'
			893075430, // '1998-04-20 12:30:30'
			-4870653059,  // '1815-08-28 16:49:01'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_date64_to_timestamp(&inputCol, &outputCol, TIME_UNIT_s);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_s );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp s to date64
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_s;
		outputCol.dtype = GDF_DATE64;
		outputCol.size = colSize;

		std::vector<int64_t> inputData = {
			1528935590, // '2018-06-14 00:19:50'
			1528935599, // '2018-06-14 00:19:59'
			-1577923201, // '1919-12-31 23:59:59'
			1582934401, // '2020-02-29 00:00:01'
			0,             // '1970-01-01 00:00:00'
			2309653342, // '2043-03-11 02:22:22'
			893075430, // '1998-04-20 12:30:30'
			-4870653059,  // '1815-08-28 16:49:01'
		};

		std::vector<int64_t> outputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599000, // '2018-06-14 00:19:59.000'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401000, // '2020-02-29 00:00:01.000'
			0,             // '1970-01-01 00:00:00.000'
			2309653342000, // '2043-03-11 02:22:22.000'
			893075430000, // '1998-04-20 12:30:30.000'
			-4870653059000,  // '1815-08-28 16:49:01.000
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_date64(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_DATE64 );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//date64 to timestamp us
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_DATE64;
		inputCol.size = colSize;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		std::vector<int64_t> inputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058987,  // '1815-08-28 16:49:01.013'
		};

		std::vector<int64_t> outputData = {
			1528935590000000, // '2018-06-14 00:19:50.000000'
			1528935599999000, // '2018-06-14 00:19:59.999000'
			-1577923201000000, // '1919-12-31 23:59:59.000000'
			1582934401123000, // '2020-02-29 00:00:01.123000'
			0,             // '1970-01-01 00:00:00.000000'
			2309653342222000, // '2043-03-11 02:22:22.222000'
			893075430345000, // '1998-04-20 12:30:30.345000'
			-4870653058987000,  // '1815-08-28 16:49:01.013000'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_date64_to_timestamp(&inputCol, &outputCol, TIME_UNIT_us);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_us );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp us to date64
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_us;
		outputCol.dtype = GDF_DATE64;
		outputCol.size = colSize;

		std::vector<int64_t> inputData = {
			1528935590000000, // '2018-06-14 00:19:50.000000'
			1528935599999999, // '2018-06-14 00:19:59.999999'
			-1577923201000000, // '1919-12-31 23:59:59.000000'
			1582934401123123, // '2020-02-29 00:00:01.123123'
			0,             // '1970-01-01 00:00:00.000000'
			2309653342222222, // '2043-03-11 02:22:22.222222'
			893075430345543, // '1998-04-20 12:30:30.345543'
			-4870653058987789,  // '1815-08-28 16:49:01.012211'
		};

		std::vector<int64_t> outputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058988,  // '1815-08-28 16:49:01.012'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_date64(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_DATE64 );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//date64 to timestamp ns
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_DATE64;
		inputCol.size = colSize;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		std::vector<int64_t> inputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058987,  // '1815-08-28 16:49:01.013'
		};

		std::vector<int64_t> outputData = {
			1528935590000000000, // '2018-06-14 00:19:50.000000000'
			1528935599999000000, // '2018-06-14 00:19:59.999000000'
			-1577923201000000000, // '1919-12-31 23:59:59.000000000'
			1582934401123000000, // '2020-02-29 00:00:01.123000000'
			0,             // '1970-01-01 00:00:00.000000000'
			2309653342222000000, // '2043-03-11 02:22:22.222000000'
			893075430345000000, // '1998-04-20 12:30:30.345000000'
			-4870653058987000000,  // '1815-08-28 16:49:01.013000000'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_date64_to_timestamp(&inputCol, &outputCol, TIME_UNIT_ns);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_ns );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp ns to date64
	{
		int colSize = 8;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_ns;
		outputCol.dtype = GDF_DATE64;
		outputCol.size = colSize;

		std::vector<int64_t> inputData = {
			1528935590000000000, // '2018-06-14 00:19:50.000000000'
			1528935599999999999, // '2018-06-14 00:19:59.999999999'
			-1577923201000000000, // '1919-12-31 23:59:59.000000000'
			1582934401123123123, // '2020-02-29 00:00:01.123123123'
			0,             // '1970-01-01 00:00:00.000000000'
			2309653342222222222, // '2043-03-11 02:22:22.222222222'
			893075430345543345, // '1998-04-20 12:30:30.345543345'
			-4870653058987789987,  // '1815-08-28 16:49:01.012210013'
		};

		std::vector<int64_t> outputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058988,  // '1815-08-28 16:49:01.012'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(1,255);
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(1,255);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_date64(&inputCol, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_DATE64 );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}
}

struct gdf_timestamp_casting_TEST : public GdfTest {};

TEST_F(gdf_timestamp_casting_TEST, timestamp_to_timestamp) {

	//timestamp to timestamp from s to ms
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_s;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		// timestamps with seconds
		std::vector<int64_t> inputData = {
			1528935590, // '2018-06-14 00:19:50'
			1528935599, // '2018-06-14 00:19:59'
			-1577923201, // '1919-12-31 23:59:59'
			1582934401, // '2020-02-29 00:00:01'
			0,             // '1970-01-01 00:00:00'
			2309653342, // '2043-03-11 02:22:22'
			893075430, // '1998-04-20 12:30:30'
			-4870653059,  // '1815-08-28 16:49:01'
			-5,            // '1969-12-31 23:59:55'
			-169139,    // '1969-12-30 01:01:01'
			-6,        // '1969-12-31 23:59:54'
			-1991063752, //	'1906-11-28 06:44:08'
			-1954281039, //	'1908-01-28 00:09:21'
			-1669612095, //	'1917-02-03 18:51:45'
			-1184467876, //	'1932-06-19 21:08:44'
			362079575, //	'1981-06-22 17:39:35'
			629650040, //	'1989-12-14 14:47:20'
			692074060, //	'1991-12-07 02:47:40'
			734734764, //	'1993-04-13 20:59:24'
			1230998894, //	'2009-01-03 16:08:14'
			1521989991, //	'2018-03-25 14:59:51'
			1726355294, //	'2024-09-14 23:08:14'
			-1722880051, //	'1915-05-29 06:12:29'
			-948235893, //	'1939-12-15 01:08:27'
			-811926962, //	'1944-04-09 16:43:58'
			-20852065, //	'1969-05-04 15:45:35'
			191206704, //	'1976-01-23 00:58:24'
			896735912, //	'1998-06-01 21:18:32'
			1262903093, //	'2010-01-07 22:24:53'
			1926203568 //	'2031-01-15 00:32:48'
		};

		// timestamps with milliseconds
		std::vector<int64_t> outputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599000, // '2018-06-14 00:19:59.000'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401000, // '2020-02-29 00:00:01.000'
			0,             // '1970-01-01 00:00:00.000'
			2309653342000, // '2043-03-11 02:22:22.000'
			893075430000, // '1998-04-20 12:30:30.000'
			-4870653059000,  // '1815-08-28 16:49:01.000
			-5000,            // '1969-12-31 23:59:55.000'
			-169139000,    // '1969-12-30 01:01:01.000
			-6000,        // '1969-12-31 23:59:54.000'
			-1991063752000, //	1906-11-28 06:44:08.000
			-1954281039000, //	1908-01-28 00:09:21.000
			-1669612095000, //	1917-02-03 18:51:45.000
			-1184467876000, //	1932-06-19 21:08:44.000
			362079575000, //	1981-06-22 17:39:35.000
			629650040000, //	1989-12-14 14:47:20.000
			692074060000, //	1991-12-07 02:47:40.000
			734734764000, //	1993-04-13 20:59:24.000
			1230998894000, //	2009-01-03 16:08:14.000
			1521989991000, //	2018-03-25 14:59:51.000
			1726355294000, //	2024-09-14 23:08:14.000
			-1722880051000, //	1915-05-29 06:12:29.000
			-948235893000, //	1939-12-15 01:08:27.000
			-811926962000, //	1944-04-09 16:43:58.000
			-20852065000, //	1969-05-04 15:45:35.000
			191206704000, //	1976-01-23 00:58:24.000
			896735912000, //	1998-06-01 21:18:32.000
			1262903093000, //	2010-01-07 22:24:53.000
			1926203568000 //	2031-01-15 00:32:48.000
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_timestamp(&inputCol, &outputCol, TIME_UNIT_ms);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_ms );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp to timestamp from ms to s
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_ms;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		// timestamps with milliseconds
		std::vector<int64_t> inputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058987,  // '1815-08-28 16:49:01.013'
			-4500,            // '1969-12-31 23:59:55.500'
			-169138999,    // '1969-12-30 01:01:01.001'
			-5999,        // '1969-12-31 23:59:54.001'
			-1991063752000, //	'1906-11-28 06:44:08.000'
			-1954281039000, //	'1908-01-28 00:09:21.000'
			-1669612095000, //	'1917-02-03 18:51:45.000'
			-1184467876000, //	'1932-06-19 21:08:44.000'
			362079575000, //	'1981-06-22 17:39:35.000'
			629650040000, //	'1989-12-14 14:47:20.000'
			692074060000, //	'1991-12-07 02:47:40.000'
			734734764000, //	'1993-04-13 20:59:24.000'
			1230998894000, //	'2009-01-03 16:08:14.000'
			1521989991000, //	'2018-03-25 14:59:51.000'
			1726355294000, //	'2024-09-14 23:08:14.000'
			-1722880051000, //	'1915-05-29 06:12:29.000'
			-948235893000, //	'1939-12-15 01:08:27.000'
			-811926962000, //	'1944-04-09 16:43:58.000'
			-20852065000, //	'1969-05-04 15:45:35.000'
			191206704000, //	'1976-01-23 00:58:24.000'
			896735912000, //	'1998-06-01 21:18:32.000'
			1262903093000, //	'2010-01-07 22:24:53.000'
			1926203568000 //	'2031-01-15 00:32:48.000'
		};

		// timestamps with seconds
		std::vector<int64_t> outputData = {
			1528935590, // '2018-06-14 00:19:50'
			1528935599, // '2018-06-14 00:19:59'
			-1577923201, // '1919-12-31 23:59:59'
			1582934401, // '2020-02-29 00:00:01'
			0,             // '1970-01-01 00:00:00'
			2309653342, // '2043-03-11 02:22:22'
			893075430, // '1998-04-20 12:30:30'
			-4870653059,  // '1815-08-28 16:49:01'
			-5,            // '1969-12-31 23:59:55'
			-169139,    // '1969-12-30 01:01:01'
			-6,        // '1969-12-31 23:59:54'
			-1991063752, //	'1906-11-28 06:44:08'
			-1954281039, //	'1908-01-28 00:09:21'
			-1669612095, //	'1917-02-03 18:51:45'
			-1184467876, //	'1932-06-19 21:08:44'
			362079575, //	'1981-06-22 17:39:35'
			629650040, //	'1989-12-14 14:47:20'
			692074060, //	'1991-12-07 02:47:40'
			734734764, //	'1993-04-13 20:59:24'
			1230998894, //	'2009-01-03 16:08:14'
			1521989991, //	'2018-03-25 14:59:51'
			1726355294, //	'2024-09-14 23:08:14'
			-1722880051, //	'1915-05-29 06:12:29'
			-948235893, //	'1939-12-15 01:08:27'
			-811926962, //	'1944-04-09 16:43:58'
			-20852065, //	'1969-05-04 15:45:35'
			191206704, //	'1976-01-23 00:58:24'
			896735912, //	'1998-06-01 21:18:32'
			1262903093, //	'2010-01-07 22:24:53'
			1926203568 //	'2031-01-15 00:32:48'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_timestamp(&inputCol, &outputCol, TIME_UNIT_s);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_s );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp to timestamp from s to us
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_s;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		// timestamps with seconds
		std::vector<int64_t> inputData = {
			1528935590, // '2018-06-14 00:19:50'
			1528935599, // '2018-06-14 00:19:59'
			-1577923201, // '1919-12-31 23:59:59'
			1582934401, // '2020-02-29 00:00:01'
			0,             // '1970-01-01 00:00:00'
			2309653342, // '2043-03-11 02:22:22'
			893075430, // '1998-04-20 12:30:30'
			-4870653059,  // '1815-08-28 16:49:01'
			-5,            // '1969-12-31 23:59:55'
			-169139,    // '1969-12-30 01:01:01'
			-6,        // '1969-12-31 23:59:54'
			-1991063752, //	'1906-11-28 06:44:08'
			-1954281039, //	'1908-01-28 00:09:21'
			-1669612095, //	'1917-02-03 18:51:45'
			-1184467876, //	'1932-06-19 21:08:44'
			362079575, //	'1981-06-22 17:39:35'
			629650040, //	'1989-12-14 14:47:20'
			692074060, //	'1991-12-07 02:47:40'
			734734764, //	'1993-04-13 20:59:24'
			1230998894, //	'2009-01-03 16:08:14'
			1521989991, //	'2018-03-25 14:59:51'
			1726355294, //	'2024-09-14 23:08:14'
			-1722880051, //	'1915-05-29 06:12:29'
			-948235893, //	'1939-12-15 01:08:27'
			-811926962, //	'1944-04-09 16:43:58'
			-20852065, //	'1969-05-04 15:45:35'
			191206704, //	'1976-01-23 00:58:24'
			896735912, //	'1998-06-01 21:18:32'
			1262903093, //	'2010-01-07 22:24:53'
			1926203568 //	'2031-01-15 00:32:48'
		};

		// timestamps with microseconds
		std::vector<int64_t> outputData = {
			1528935590000000, // '2018-06-14 00:19:50.000000'
			1528935599000000, // '2018-06-14 00:19:59.000000'
			-1577923201000000, // '1919-12-31 23:59:59.000000'
			1582934401000000, // '2020-02-29 00:00:01.000000'
			0,             // '1970-01-01 00:00:00.000000'
			2309653342000000, // '2043-03-11 02:22:22.000000'
			893075430000000, // '1998-04-20 12:30:30.000000'
			-4870653059000000,  // '1815-08-28 16:49:01.000000'
			-5000000,            // '1969-12-31 23:59:55.000000'
			-169139000000,    // '1969-12-30 01:01:01.000000'
			-6000000,        // '1969-12-31 23:59:54.000000'
			-1991063752000000, //	'1906-11-28 06:44:08.000000'
			-1954281039000000, //	'1908-01-28 00:09:21.000000'
			-1669612095000000, //	'1917-02-03 18:51:45.000000'
			-1184467876000000, //	'1932-06-19 21:08:44.000000'
			362079575000000, //	'1981-06-22 17:39:35.000000'
			629650040000000, //	'1989-12-14 14:47:20.000000'
			692074060000000, //	'1991-12-07 02:47:40.000000'
			734734764000000, //	'1993-04-13 20:59:24.000000'
			1230998894000000, //	'2009-01-03 16:08:14.000000'
			1521989991000000, //	'2018-03-25 14:59:51.000000'
			1726355294000000, //	'2024-09-14 23:08:14.000000'
			-1722880051000000, //	'1915-05-29 06:12:29.000000'
			-948235893000000, //	'1939-12-15 01:08:27.000000'
			-811926962000000, //	'1944-04-09 16:43:58.000000'
			-20852065000000, //	'1969-05-04 15:45:35.000000'
			191206704000000, //	'1976-01-23 00:58:24.000000'
			896735912000000, //	'1998-06-01 21:18:32.000000'
			1262903093000000, //	'2010-01-07 22:24:53.000000'
			1926203568000000 //	'2031-01-15 00:32:48.000000'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_timestamp(&inputCol, &outputCol, TIME_UNIT_us);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_us );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp to timestamp from us to s
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_us;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		// timestamps with microseconds
		std::vector<int64_t> inputData = {
			1528935590000000, // '2018-06-14 00:19:50.000000'
			1528935599999999, // '2018-06-14 00:19:59.999999'
			-1577923201000000, // '1919-12-31 23:59:59.000000'
			1582934401123123, // '2020-02-29 00:00:01.123123'
			0,             // '1970-01-01 00:00:00.000000'
			2309653342222222, // '2043-03-11 02:22:22.222222'
			893075430345543, // '1998-04-20 12:30:30.345543'
			-4870653058987789,  // '1815-08-28 16:49:01.012211'
			-4500005,            // '1969-12-31 23:59:55.499995'
			-169138999999,    // '1969-12-30 01:01:01.000001'
			-5999999,        // '1969-12-31 23:59:54.000001'
			-1991063752000000, //	'1906-11-28 06:44:08.000000'
			-1954281039000000, //	'1908-01-28 00:09:21.000000'
			-1669612095000000, //	'1917-02-03 18:51:45.000000'
			-1184467876000000, //	'1932-06-19 21:08:44.000000'
			362079575000000, //	'1981-06-22 17:39:35.000000'
			629650040000000, //	'1989-12-14 14:47:20.000000'
			692074060000000, //	'1991-12-07 02:47:40.000000'
			734734764000000, //	'1993-04-13 20:59:24.000000'
			1230998894000000, //	'2009-01-03 16:08:14.000000'
			1521989991000000, //	'2018-03-25 14:59:51.000000'
			1726355294000000, //	'2024-09-14 23:08:14.000000'
			-1722880051000000, //	'1915-05-29 06:12:29.000000'
			-948235893000000, //	'1939-12-15 01:08:27.000000'
			-811926962000000, //	'1944-04-09 16:43:58.000000'
			-20852065000000, //	'1969-05-04 15:45:35.000000'
			191206704000000, //	'1976-01-23 00:58:24.000000'
			896735912000000, //	'1998-06-01 21:18:32.000000'
			1262903093000000, //	'2010-01-07 22:24:53.000000'
			1926203568000000 //	'2031-01-15 00:32:48.000000'
		};

		// timestamps with seconds
		std::vector<int64_t> outputData = {
			1528935590, // '2018-06-14 00:19:50'
			1528935599, // '2018-06-14 00:19:59'
			-1577923201, // '1919-12-31 23:59:59'
			1582934401, // '2020-02-29 00:00:01'
			0,             // '1970-01-01 00:00:00'
			2309653342, // '2043-03-11 02:22:22'
			893075430, // '1998-04-20 12:30:30'
			-4870653059,  // '1815-08-28 16:49:01'
			-5,            // '1969-12-31 23:59:55'
			-169139,    // '1969-12-30 01:01:01'
			-6,        // '1969-12-31 23:59:54'
			-1991063752, //	'1906-11-28 06:44:08'
			-1954281039, //	'1908-01-28 00:09:21'
			-1669612095, //	'1917-02-03 18:51:45'
			-1184467876, //	'1932-06-19 21:08:44'
			362079575, //	'1981-06-22 17:39:35'
			629650040, //	'1989-12-14 14:47:20'
			692074060, //	'1991-12-07 02:47:40'
			734734764, //	'1993-04-13 20:59:24'
			1230998894, //	'2009-01-03 16:08:14'
			1521989991, //	'2018-03-25 14:59:51'
			1726355294, //	'2024-09-14 23:08:14'
			-1722880051, //	'1915-05-29 06:12:29'
			-948235893, //	'1939-12-15 01:08:27'
			-811926962, //	'1944-04-09 16:43:58'
			-20852065, //	'1969-05-04 15:45:35'
			191206704, //	'1976-01-23 00:58:24'
			896735912, //	'1998-06-01 21:18:32'
			1262903093, //	'2010-01-07 22:24:53'
			1926203568 //	'2031-01-15 00:32:48'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_timestamp(&inputCol, &outputCol, TIME_UNIT_s);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_s );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp to timestamp from s to ns
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_s;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		// timestamps with seconds
		std::vector<int64_t> inputData = {
			1528935590, // '2018-06-14 00:19:50'
			1528935599, // '2018-06-14 00:19:59'
			-1577923201, // '1919-12-31 23:59:59'
			1582934401, // '2020-02-29 00:00:01'
			0,             // '1970-01-01 00:00:00'
			2309653342, // '2043-03-11 02:22:22'
			893075430, // '1998-04-20 12:30:30'
			-4870653059,  // '1815-08-28 16:49:01'
			-5,            // '1969-12-31 23:59:55'
			-169139,    // '1969-12-30 01:01:01'
			-6,        // '1969-12-31 23:59:54'
			-1991063752, //	'1906-11-28 06:44:08'
			-1954281039, //	'1908-01-28 00:09:21'
			-1669612095, //	'1917-02-03 18:51:45'
			-1184467876, //	'1932-06-19 21:08:44'
			362079575, //	'1981-06-22 17:39:35'
			629650040, //	'1989-12-14 14:47:20'
			692074060, //	'1991-12-07 02:47:40'
			734734764, //	'1993-04-13 20:59:24'
			1230998894, //	'2009-01-03 16:08:14'
			1521989991, //	'2018-03-25 14:59:51'
			1726355294, //	'2024-09-14 23:08:14'
			-1722880051, //	'1915-05-29 06:12:29'
			-948235893, //	'1939-12-15 01:08:27'
			-811926962, //	'1944-04-09 16:43:58'
			-20852065, //	'1969-05-04 15:45:35'
			191206704, //	'1976-01-23 00:58:24'
			896735912, //	'1998-06-01 21:18:32'
			1262903093, //	'2010-01-07 22:24:53'
			1926203568 //	'2031-01-15 00:32:48'
		};

		// timestamps with nanoseconds
		std::vector<int64_t> outputData = {
			1528935590000000000, // '2018-06-14 00:19:50.000000000'
			1528935599000000000, // '2018-06-14 00:19:59.000000000'
			-1577923201000000000, // '1919-12-31 23:59:59.000000000'
			1582934401000000000, // '2020-02-29 00:00:01.000000000'
			0,             // '1970-01-01 00:00:00.000000000'
			2309653342000000000, // '2043-03-11 02:22:22.000000000'
			893075430000000000, // '1998-04-20 12:30:30.000000000'
			-4870653059000000000,  // '1815-08-28 16:49:01.000000000'
			-5000000000,            // '1969-12-31 23:59:55.000000000'
			-169139000000000,    // '1969-12-30 01:01:01.000000000'
			-6000000000,        // '1969-12-31 23:59:54.000000000'
			-1991063752000000000, //	'1906-11-28 06:44:08.000000000'
			-1954281039000000000, //	'1908-01-28 00:09:21.000000000'
			-1669612095000000000, //	'1917-02-03 18:51:45.000000000'
			-1184467876000000000, //	'1932-06-19 21:08:44.000000000'
			362079575000000000, //	'1981-06-22 17:39:35.000000000'
			629650040000000000, //	'1989-12-14 14:47:20.000000000'
			692074060000000000, //	'1991-12-07 02:47:40.000000000'
			734734764000000000, //	'1993-04-13 20:59:24.000000000'
			1230998894000000000, //	'2009-01-03 16:08:14.000000000'
			1521989991000000000, //	'2018-03-25 14:59:51.000000000'
			1726355294000000000, //	'2024-09-14 23:08:14.000000000'
			-1722880051000000000, //	'1915-05-29 06:12:29.000000000'
			-948235893000000000, //	'1939-12-15 01:08:27.000000000'
			-811926962000000000, //	'1944-04-09 16:43:58.000000000'
			-20852065000000000, //	'1969-05-04 15:45:35.000000000'
			191206704000000000, //	'1976-01-23 00:58:24.000000000'
			896735912000000000, //	'1998-06-01 21:18:32.000000000'
			1262903093000000000, //	'2010-01-07 22:24:53.000000000'
			1926203568000000000 //	'2031-01-15 00:32:48.000000000'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_timestamp(&inputCol, &outputCol, TIME_UNIT_ns);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_ns );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp to timestamp from ns to s
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_ns;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		// timestamps with nanoseconds
		std::vector<int64_t> inputData = {
			1528935590000000000, // '2018-06-14 00:19:50.000000000'
			1528935599999999999, // '2018-06-14 00:19:59.999999999'
			-1577923201000000000, // '1919-12-31 23:59:59.000000000'
			1582934401123123123, // '2020-02-29 00:00:01.123123123'
			0,             // '1970-01-01 00:00:00.000000000'
			2309653342222222222, // '2043-03-11 02:22:22.222222222'
			893075430345543345, // '1998-04-20 12:30:30.345543345'
			-4870653058987789987,  // '1815-08-28 16:49:01.012210013'
			-4500000005,            // '1969-12-31 23:59:55.499999995'
			-169138999999999,    // '1969-12-30 01:01:01.000000001'
			-5999999999,        // '1969-12-31 23:59:54.000000001'
			-1991063752000000000, //	'1906-11-28 06:44:08.000000000'
			-1954281039000000000, //	'1908-01-28 00:09:21.000000000'
			-1669612095000000000, //	'1917-02-03 18:51:45.000000000'
			-1184467876000000000, //	'1932-06-19 21:08:44.000000000'
			362079575000000000, //	'1981-06-22 17:39:35.000000000'
			629650040000000000, //	'1989-12-14 14:47:20.000000000'
			692074060000000000, //	'1991-12-07 02:47:40.000000000'
			734734764000000000, //	'1993-04-13 20:59:24.000000000'
			1230998894000000000, //	'2009-01-03 16:08:14.000000000'
			1521989991000000000, //	'2018-03-25 14:59:51.000000000'
			1726355294000000000, //	'2024-09-14 23:08:14.000000000'
			-1722880051000000000, //	'1915-05-29 06:12:29.000000000'
			-948235893000000000, //	'1939-12-15 01:08:27.000000000'
			-811926962000000000, //	'1944-04-09 16:43:58.000000000'
			-20852065000000000, //	'1969-05-04 15:45:35.000000000'
			191206704000000000, //	'1976-01-23 00:58:24.000000000'
			896735912000000000, //	'1998-06-01 21:18:32.000000000'
			1262903093000000000, //	'2010-01-07 22:24:53.000000000'
			1926203568000000000 //	'2031-01-15 00:32:48.000000000'
		};

		// timestamps with seconds
		std::vector<int64_t> outputData = {
			1528935590, // '2018-06-14 00:19:50'
			1528935599, // '2018-06-14 00:19:59'
			-1577923201, // '1919-12-31 23:59:59'
			1582934401, // '2020-02-29 00:00:01'
			0,             // '1970-01-01 00:00:00'
			2309653342, // '2043-03-11 02:22:22'
			893075430, // '1998-04-20 12:30:30'
			-4870653059,  // '1815-08-28 16:49:01'
			-5,            // '1969-12-31 23:59:55'
			-169139,    // '1969-12-30 01:01:01'
			-6,        // '1969-12-31 23:59:54'
			-1991063752, //	'1906-11-28 06:44:08'
			-1954281039, //	'1908-01-28 00:09:21'
			-1669612095, //	'1917-02-03 18:51:45'
			-1184467876, //	'1932-06-19 21:08:44'
			362079575, //	'1981-06-22 17:39:35'
			629650040, //	'1989-12-14 14:47:20'
			692074060, //	'1991-12-07 02:47:40'
			734734764, //	'1993-04-13 20:59:24'
			1230998894, //	'2009-01-03 16:08:14'
			1521989991, //	'2018-03-25 14:59:51'
			1726355294, //	'2024-09-14 23:08:14'
			-1722880051, //	'1915-05-29 06:12:29'
			-948235893, //	'1939-12-15 01:08:27'
			-811926962, //	'1944-04-09 16:43:58'
			-20852065, //	'1969-05-04 15:45:35'
			191206704, //	'1976-01-23 00:58:24'
			896735912, //	'1998-06-01 21:18:32'
			1262903093, //	'2010-01-07 22:24:53'
			1926203568 //	'2031-01-15 00:32:48'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_timestamp(&inputCol, &outputCol, TIME_UNIT_s);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_s );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp to timestamp from us to ns
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_us;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		// timestamps with microseconds
		std::vector<int64_t> inputData = {
			1528935590000000, // '2018-06-14 00:19:50.000000'
			1528935599999999, // '2018-06-14 00:19:59.999999'
			-1577923201000000, // '1919-12-31 23:59:59.000000'
			1582934401123123, // '2020-02-29 00:00:01.123123'
			0,             // '1970-01-01 00:00:00.000000'
			2309653342222222, // '2043-03-11 02:22:22.222222'
			893075430345543, // '1998-04-20 12:30:30.345543'
			-4870653058987789,  // '1815-08-28 16:49:01.012211'
			-4500005,            // '1969-12-31 23:59:55.499995'
			-169138999999,    // '1969-12-30 01:01:01.000001'
			-5999999,        // '1969-12-31 23:59:54.000001'
			-1991063752000000, //	'1906-11-28 06:44:08.000000'
			-1954281039000000, //	'1908-01-28 00:09:21.000000'
			-1669612095000000, //	'1917-02-03 18:51:45.000000'
			-1184467876000000, //	'1932-06-19 21:08:44.000000'
			362079575000000, //	'1981-06-22 17:39:35.000000'
			629650040000000, //	'1989-12-14 14:47:20.000000'
			692074060000000, //	'1991-12-07 02:47:40.000000'
			734734764000000, //	'1993-04-13 20:59:24.000000'
			1230998894000000, //	'2009-01-03 16:08:14.000000'
			1521989991000000, //	'2018-03-25 14:59:51.000000'
			1726355294000000, //	'2024-09-14 23:08:14.000000'
			-1722880051000000, //	'1915-05-29 06:12:29.000000'
			-948235893000000, //	'1939-12-15 01:08:27.000000'
			-811926962000000, //	'1944-04-09 16:43:58.000000'
			-20852065000000, //	'1969-05-04 15:45:35.000000'
			191206704000000, //	'1976-01-23 00:58:24.000000'
			896735912000000, //	'1998-06-01 21:18:32.000000'
			1262903093000000, //	'2010-01-07 22:24:53.000000'
			1926203568000000 //	'2031-01-15 00:32:48.000000'
		};

		// timestamps with nanoseconds
		std::vector<int64_t> outputData = {
			1528935590000000000, // '2018-06-14 00:19:50.000000000'
			1528935599999999000, // '2018-06-14 00:19:59.999999000'
			-1577923201000000000, // '1919-12-31 23:59:59.000000000'
			1582934401123123000, // '2020-02-29 00:00:01.123123000'
			0,             // '1970-01-01 00:00:00.000000000'
			2309653342222222000, // '2043-03-11 02:22:22.222222000'
			893075430345543000, // '1998-04-20 12:30:30.345543000'
			-4870653058987789000,  // '1815-08-28 16:49:01.012211000'
			-4500005000,            // '1969-12-31 23:59:55.499995000'
			-169138999999000,    // '1969-12-30 01:01:01.000001000'
			-5999999000,        // '1969-12-31 23:59:54.000001000'
			-1991063752000000000, //	'1906-11-28 06:44:08.000000000'
			-1954281039000000000, //	'1908-01-28 00:09:21.000000000'
			-1669612095000000000, //	'1917-02-03 18:51:45.000000000'
			-1184467876000000000, //	'1932-06-19 21:08:44.000000000'
			362079575000000000, //	'1981-06-22 17:39:35.000000000'
			629650040000000000, //	'1989-12-14 14:47:20.000000000'
			692074060000000000, //	'1991-12-07 02:47:40.000000000'
			734734764000000000, //	'1993-04-13 20:59:24.000000000'
			1230998894000000000, //	'2009-01-03 16:08:14.000000000'
			1521989991000000000, //	'2018-03-25 14:59:51.000000000'
			1726355294000000000, //	'2024-09-14 23:08:14.000000000'
			-1722880051000000000, //	'1915-05-29 06:12:29.000000000'
			-948235893000000000, //	'1939-12-15 01:08:27.000000000'
			-811926962000000000, //	'1944-04-09 16:43:58.000000000'
			-20852065000000000, //	'1969-05-04 15:45:35.000000000'
			191206704000000000, //	'1976-01-23 00:58:24.000000000'
			896735912000000000, //	'1998-06-01 21:18:32.000000000'
			1262903093000000000, //	'2010-01-07 22:24:53.000000000'
			1926203568000000000 //	'2031-01-15 00:32:48.000000000'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_timestamp(&inputCol, &outputCol, TIME_UNIT_ns);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_ns );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp to timestamp from ns to us
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_ns;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		// timestamps with nanoseconds
		std::vector<int64_t> inputData = {
			1528935590000000000, // '2018-06-14 00:19:50.000000000'
			1528935599999999999, // '2018-06-14 00:19:59.999999999'
			-1577923201000000000, // '1919-12-31 23:59:59.000000000'
			1582934401123123123, // '2020-02-29 00:00:01.123123123'
			0,             // '1970-01-01 00:00:00.000000000'
			2309653342222222222, // '2043-03-11 02:22:22.222222222'
			893075430345543345, // '1998-04-20 12:30:30.345543345'
			-4870653058987789987,  // '1815-08-28 16:49:01.012210013'
			-4500000005,            // '1969-12-31 23:59:55.499999995'
			-169138999999999,    // '1969-12-30 01:01:01.000000001'
			-5999999999,        // '1969-12-31 23:59:54.000000001'
			-1991063752000000000, //	'1906-11-28 06:44:08.000000000'
			-1954281039000000000, //	'1908-01-28 00:09:21.000000000'
			-1669612095000000000, //	'1917-02-03 18:51:45.000000000'
			-1184467876000000000, //	'1932-06-19 21:08:44.000000000'
			362079575000000000, //	'1981-06-22 17:39:35.000000000'
			629650040000000000, //	'1989-12-14 14:47:20.000000000'
			692074060000000000, //	'1991-12-07 02:47:40.000000000'
			734734764000000000, //	'1993-04-13 20:59:24.000000000'
			1230998894000000000, //	'2009-01-03 16:08:14.000000000'
			1521989991000000000, //	'2018-03-25 14:59:51.000000000'
			1726355294000000000, //	'2024-09-14 23:08:14.000000000'
			-1722880051000000000, //	'1915-05-29 06:12:29.000000000'
			-948235893000000000, //	'1939-12-15 01:08:27.000000000'
			-811926962000000000, //	'1944-04-09 16:43:58.000000000'
			-20852065000000000, //	'1969-05-04 15:45:35.000000000'
			191206704000000000, //	'1976-01-23 00:58:24.000000000'
			896735912000000000, //	'1998-06-01 21:18:32.000000000'
			1262903093000000000, //	'2010-01-07 22:24:53.000000000'
			1926203568000000000 //	'2031-01-15 00:32:48.000000000'
		};

		// timestamps with microseconds
		std::vector<int64_t> outputData = {
			1528935590000000, // '2018-06-14 00:19:50.000000'
			1528935599999999, // '2018-06-14 00:19:59.999999'
			-1577923201000000, // '1919-12-31 23:59:59.000000'
			1582934401123123, // '2020-02-29 00:00:01.123123'
			0,             // '1970-01-01 00:00:00.000000'
			2309653342222222, // '2043-03-11 02:22:22.222222'
			893075430345543, // '1998-04-20 12:30:30.345543'
			-4870653058987790,  // '1815-08-28 16:49:01.012210'
			-4500001,            // '1969-12-31 23:59:55.499999'
			-169139000000,    // '1969-12-30 01:01:01.000000'
			-6000000,        // '1969-12-31 23:59:54.000000'
			-1991063752000000, //	'1906-11-28 06:44:08.000000'
			-1954281039000000, //	'1908-01-28 00:09:21.000000'
			-1669612095000000, //	'1917-02-03 18:51:45.000000'
			-1184467876000000, //	'1932-06-19 21:08:44.000000'
			362079575000000, //	'1981-06-22 17:39:35.000000'
			629650040000000, //	'1989-12-14 14:47:20.000000'
			692074060000000, //	'1991-12-07 02:47:40.000000'
			734734764000000, //	'1993-04-13 20:59:24.000000'
			1230998894000000, //	'2009-01-03 16:08:14.000000'
			1521989991000000, //	'2018-03-25 14:59:51.000000'
			1726355294000000, //	'2024-09-14 23:08:14.000000'
			-1722880051000000, //	'1915-05-29 06:12:29.000000'
			-948235893000000, //	'1939-12-15 01:08:27.000000'
			-811926962000000, //	'1944-04-09 16:43:58.000000'
			-20852065000000, //	'1969-05-04 15:45:35.000000'
			191206704000000, //	'1976-01-23 00:58:24.000000'
			896735912000000, //	'1998-06-01 21:18:32.000000'
			1262903093000000, //	'2010-01-07 22:24:53.000000'
			1926203568000000 //	'2031-01-15 00:32:48.000000'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_timestamp(&inputCol, &outputCol, TIME_UNIT_us);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_us );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp to timestamp from ms to ns
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_ms;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		// timestamps with milliseconds
		std::vector<int64_t> inputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058987,  // '1815-08-28 16:49:01.013'
			-4500,            // '1969-12-31 23:59:55.500'
			-169138999,    // '1969-12-30 01:01:01.001'
			-5999,        // '1969-12-31 23:59:54.001'
			-1991063752000, //	'1906-11-28 06:44:08.000'
			-1954281039000, //	'1908-01-28 00:09:21.000'
			-1669612095000, //	'1917-02-03 18:51:45.000'
			-1184467876000, //	'1932-06-19 21:08:44.000'
			362079575000, //	'1981-06-22 17:39:35.000'
			629650040000, //	'1989-12-14 14:47:20.000'
			692074060000, //	'1991-12-07 02:47:40.000'
			734734764000, //	'1993-04-13 20:59:24.000'
			1230998894000, //	'2009-01-03 16:08:14.000'
			1521989991000, //	'2018-03-25 14:59:51.000'
			1726355294000, //	'2024-09-14 23:08:14.000'
			-1722880051000, //	'1915-05-29 06:12:29.000'
			-948235893000, //	'1939-12-15 01:08:27.000'
			-811926962000, //	'1944-04-09 16:43:58.000'
			-20852065000, //	'1969-05-04 15:45:35.000'
			191206704000, //	'1976-01-23 00:58:24.000'
			896735912000, //	'1998-06-01 21:18:32.000'
			1262903093000, //	'2010-01-07 22:24:53.000'
			1926203568000 //	'2031-01-15 00:32:48.000'
		};

		// timestamps with nanoseconds
		std::vector<int64_t> outputData = {
			1528935590000000000, // '2018-06-14 00:19:50.000000000'
			1528935599999000000, // '2018-06-14 00:19:59.999000000'
			-1577923201000000000, // '1919-12-31 23:59:59.000000000'
			1582934401123000000, // '2020-02-29 00:00:01.123000000'
			0,             // '1970-01-01 00:00:00.000000000'
			2309653342222000000, // '2043-03-11 02:22:22.222000000'
			893075430345000000, // '1998-04-20 12:30:30.345000000'
			-4870653058987000000,  // '1815-08-28 16:49:01.013000000'
			-4500000000,            // '1969-12-31 23:59:55.500000000'
			-169138999000000,    // '1969-12-30 01:01:01.001000000'
			-5999000000,        // '1969-12-31 23:59:54.001000000'
			-1991063752000000000, //	'1906-11-28 06:44:08.000000000'
			-1954281039000000000, //	'1908-01-28 00:09:21.000000000'
			-1669612095000000000, //	'1917-02-03 18:51:45.000000000'
			-1184467876000000000, //	'1932-06-19 21:08:44.000000000'
			362079575000000000, //	'1981-06-22 17:39:35.000000000'
			629650040000000000, //	'1989-12-14 14:47:20.000000000'
			692074060000000000, //	'1991-12-07 02:47:40.000000000'
			734734764000000000, //	'1993-04-13 20:59:24.000000000'
			1230998894000000000, //	'2009-01-03 16:08:14.000000000'
			1521989991000000000, //	'2018-03-25 14:59:51.000000000'
			1726355294000000000, //	'2024-09-14 23:08:14.000000000'
			-1722880051000000000, //	'1915-05-29 06:12:29.000000000'
			-948235893000000000, //	'1939-12-15 01:08:27.000000000'
			-811926962000000000, //	'1944-04-09 16:43:58.000000000'
			-20852065000000000, //	'1969-05-04 15:45:35.000000000'
			191206704000000000, //	'1976-01-23 00:58:24.000000000'
			896735912000000000, //	'1998-06-01 21:18:32.000000000'
			1262903093000000000, //	'2010-01-07 22:24:53.000000000'
			1926203568000000000 //	'2031-01-15 00:32:48.000000000'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_timestamp(&inputCol, &outputCol, TIME_UNIT_ns);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_ns );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp to timestamp from ns to ms
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_ns;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		// timestamps with nanoseconds
		std::vector<int64_t> inputData = {
			1528935590000000000, // '2018-06-14 00:19:50.000000000'
			1528935599999999999, // '2018-06-14 00:19:59.999999999'
			-1577923201000000000, // '1919-12-31 23:59:59.000000000'
			1582934401123123123, // '2020-02-29 00:00:01.123123123'
			0,             // '1970-01-01 00:00:00.000000000'
			2309653342222222222, // '2043-03-11 02:22:22.222222222'
			893075430345543345, // '1998-04-20 12:30:30.345543345'
			-4870653058987789987,  // '1815-08-28 16:49:01.012210013'
			-4500000005,            // '1969-12-31 23:59:55.499999995'
			-169138999999999,    // '1969-12-30 01:01:01.000000001'
			-5999999999,        // '1969-12-31 23:59:54.000000001'
			-1991063752000000000, //	'1906-11-28 06:44:08.000000000'
			-1954281039000000000, //	'1908-01-28 00:09:21.000000000'
			-1669612095000000000, //	'1917-02-03 18:51:45.000000000'
			-1184467876000000000, //	'1932-06-19 21:08:44.000000000'
			362079575000000000, //	'1981-06-22 17:39:35.000000000'
			629650040000000000, //	'1989-12-14 14:47:20.000000000'
			692074060000000000, //	'1991-12-07 02:47:40.000000000'
			734734764000000000, //	'1993-04-13 20:59:24.000000000'
			1230998894000000000, //	'2009-01-03 16:08:14.000000000'
			1521989991000000000, //	'2018-03-25 14:59:51.000000000'
			1726355294000000000, //	'2024-09-14 23:08:14.000000000'
			-1722880051000000000, //	'1915-05-29 06:12:29.000000000'
			-948235893000000000, //	'1939-12-15 01:08:27.000000000'
			-811926962000000000, //	'1944-04-09 16:43:58.000000000'
			-20852065000000000, //	'1969-05-04 15:45:35.000000000'
			191206704000000000, //	'1976-01-23 00:58:24.000000000'
			896735912000000000, //	'1998-06-01 21:18:32.000000000'
			1262903093000000000, //	'2010-01-07 22:24:53.000000000'
			1926203568000000000 //	'2031-01-15 00:32:48.000000000'
		};

		// timestamps with milliseconds
		std::vector<int64_t> outputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058988,  // '1815-08-28 16:49:01.012'
			-4501,            // '1969-12-31 23:59:55.499'
			-169139000,    // '1969-12-30 01:01:01.000'
			-6000,        // '1969-12-31 23:59:54.000'
			-1991063752000, //	'1906-11-28 06:44:08.000'
			-1954281039000, //	'1908-01-28 00:09:21.000'
			-1669612095000, //	'1917-02-03 18:51:45.000'
			-1184467876000, //	'1932-06-19 21:08:44.000'
			362079575000, //	'1981-06-22 17:39:35.000'
			629650040000, //	'1989-12-14 14:47:20.000'
			692074060000, //	'1991-12-07 02:47:40.000'
			734734764000, //	'1993-04-13 20:59:24.000'
			1230998894000, //	'2009-01-03 16:08:14.000'
			1521989991000, //	'2018-03-25 14:59:51.000'
			1726355294000, //	'2024-09-14 23:08:14.000'
			-1722880051000, //	'1915-05-29 06:12:29.000'
			-948235893000, //	'1939-12-15 01:08:27.000'
			-811926962000, //	'1944-04-09 16:43:58.000'
			-20852065000, //	'1969-05-04 15:45:35.000'
			191206704000, //	'1976-01-23 00:58:24.000'
			896735912000, //	'1998-06-01 21:18:32.000'
			1262903093000, //	'2010-01-07 22:24:53.000'
			1926203568000 //	'2031-01-15 00:32:48.000'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_timestamp(&inputCol, &outputCol, TIME_UNIT_ms);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_ms );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp to timestamp from us to ms
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_us;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		// timestamps with microseconds
		std::vector<int64_t> inputData = {
			1528935590000000, // '2018-06-14 00:19:50.000000'
			1528935599999999, // '2018-06-14 00:19:59.999999'
			-1577923201000000, // '1919-12-31 23:59:59.000000'
			1582934401123123, // '2020-02-29 00:00:01.123123'
			0,             // '1970-01-01 00:00:00.000000'
			2309653342222222, // '2043-03-11 02:22:22.222222'
			893075430345543, // '1998-04-20 12:30:30.345543'
			-4870653058987789,  // '1815-08-28 16:49:01.012211'
			-4500005,            // '1969-12-31 23:59:55.499995'
			-169138999999,    // '1969-12-30 01:01:01.000001'
			-5999999,        // '1969-12-31 23:59:54.000001'
			-1991063752000000, //	'1906-11-28 06:44:08.000000'
			-1954281039000000, //	'1908-01-28 00:09:21.000000'
			-1669612095000000, //	'1917-02-03 18:51:45.000000'
			-1184467876000000, //	'1932-06-19 21:08:44.000000'
			362079575000000, //	'1981-06-22 17:39:35.000000'
			629650040000000, //	'1989-12-14 14:47:20.000000'
			692074060000000, //	'1991-12-07 02:47:40.000000'
			734734764000000, //	'1993-04-13 20:59:24.000000'
			1230998894000000, //	'2009-01-03 16:08:14.000000'
			1521989991000000, //	'2018-03-25 14:59:51.000000'
			1726355294000000, //	'2024-09-14 23:08:14.000000'
			-1722880051000000, //	'1915-05-29 06:12:29.000000'
			-948235893000000, //	'1939-12-15 01:08:27.000000'
			-811926962000000, //	'1944-04-09 16:43:58.000000'
			-20852065000000, //	'1969-05-04 15:45:35.000000'
			191206704000000, //	'1976-01-23 00:58:24.000000'
			896735912000000, //	'1998-06-01 21:18:32.000000'
			1262903093000000, //	'2010-01-07 22:24:53.000000'
			1926203568000000 //	'2031-01-15 00:32:48.000000'
		};

		// timestamps with milliseconds
		std::vector<int64_t> outputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058988,  // '1815-08-28 16:49:01.012'
			-4501,            // '1969-12-31 23:59:55.499'
			-169139000,    // '1969-12-30 01:01:01.000'
			-6000,        // '1969-12-31 23:59:54.000'
			-1991063752000, //	'1906-11-28 06:44:08.000'
			-1954281039000, //	'1908-01-28 00:09:21.000'
			-1669612095000, //	'1917-02-03 18:51:45.000'
			-1184467876000, //	'1932-06-19 21:08:44.000'
			362079575000, //	'1981-06-22 17:39:35.000'
			629650040000, //	'1989-12-14 14:47:20.000'
			692074060000, //	'1991-12-07 02:47:40.000'
			734734764000, //	'1993-04-13 20:59:24.000'
			1230998894000, //	'2009-01-03 16:08:14.000'
			1521989991000, //	'2018-03-25 14:59:51.000'
			1726355294000, //	'2024-09-14 23:08:14.000'
			-1722880051000, //	'1915-05-29 06:12:29.000'
			-948235893000, //	'1939-12-15 01:08:27.000'
			-811926962000, //	'1944-04-09 16:43:58.000'
			-20852065000, //	'1969-05-04 15:45:35.000'
			191206704000, //	'1976-01-23 00:58:24.000'
			896735912000, //	'1998-06-01 21:18:32.000'
			1262903093000, //	'2010-01-07 22:24:53.000'
			1926203568000 //	'2031-01-15 00:32:48.000'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_timestamp(&inputCol, &outputCol, TIME_UNIT_ms);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_ms );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}

	//timestamp to timestamp from ms to us
	{
		int colSize = 30;

		gdf_column inputCol;
		gdf_column outputCol;

		inputCol.dtype = GDF_TIMESTAMP;
		inputCol.size = colSize;
		inputCol.dtype_info.time_unit = TIME_UNIT_ms;
		outputCol.dtype = GDF_TIMESTAMP;
		outputCol.size = colSize;

		// timestamps with milliseconds
		std::vector<int64_t> inputData = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			0,             // '1970-01-01 00:00:00.000'
			2309653342222, // '2043-03-11 02:22:22.222'
			893075430345, // '1998-04-20 12:30:30.345'
			-4870653058987,  // '1815-08-28 16:49:01.013'
			-4500,            // '1969-12-31 23:59:55.500'
			-169138999,    // '1969-12-30 01:01:01.001'
			-5999,        // '1969-12-31 23:59:54.001'
			-1991063752000, //	'1906-11-28 06:44:08.000'
			-1954281039000, //	'1908-01-28 00:09:21.000'
			-1669612095000, //	'1917-02-03 18:51:45.000'
			-1184467876000, //	'1932-06-19 21:08:44.000'
			362079575000, //	'1981-06-22 17:39:35.000'
			629650040000, //	'1989-12-14 14:47:20.000'
			692074060000, //	'1991-12-07 02:47:40.000'
			734734764000, //	'1993-04-13 20:59:24.000'
			1230998894000, //	'2009-01-03 16:08:14.000'
			1521989991000, //	'2018-03-25 14:59:51.000'
			1726355294000, //	'2024-09-14 23:08:14.000'
			-1722880051000, //	'1915-05-29 06:12:29.000'
			-948235893000, //	'1939-12-15 01:08:27.000'
			-811926962000, //	'1944-04-09 16:43:58.000'
			-20852065000, //	'1969-05-04 15:45:35.000'
			191206704000, //	'1976-01-23 00:58:24.000'
			896735912000, //	'1998-06-01 21:18:32.000'
			1262903093000, //	'2010-01-07 22:24:53.000'
			1926203568000 //	'2031-01-15 00:32:48.000'
		};

		// timestamps with microseconds
		std::vector<int64_t> outputData = {
			1528935590000000, // '2018-06-14 00:19:50.000000'
			1528935599999000, // '2018-06-14 00:19:59.999000'
			-1577923201000000, // '1919-12-31 23:59:59.000000'
			1582934401123000, // '2020-02-29 00:00:01.123000'
			0,             // '1970-01-01 00:00:00.000000'
			2309653342222000, // '2043-03-11 02:22:22.222000'
			893075430345000, // '1998-04-20 12:30:30.345000'
			-4870653058987000,  // '1815-08-28 16:49:01.013000'
			-4500000,            // '1969-12-31 23:59:55.500000'
			-169138999000,    // '1969-12-30 01:01:01.001000'
			-5999000,        // '1969-12-31 23:59:54.001000'
			-1991063752000000, //	'1906-11-28 06:44:08.000000'
			-1954281039000000, //	'1908-01-28 00:09:21.000000'
			-1669612095000000, //	'1917-02-03 18:51:45.000000'
			-1184467876000000, //	'1932-06-19 21:08:44.000000'
			362079575000000, //	'1981-06-22 17:39:35.000000'
			629650040000000, //	'1989-12-14 14:47:20.000000'
			692074060000000, //	'1991-12-07 02:47:40.000000'
			734734764000000, //	'1993-04-13 20:59:24.000000'
			1230998894000000, //	'2009-01-03 16:08:14.000000'
			1521989991000000, //	'2018-03-25 14:59:51.000000'
			1726355294000000, //	'2024-09-14 23:08:14.000000'
			-1722880051000000, //	'1915-05-29 06:12:29.000000'
			-948235893000000, //	'1939-12-15 01:08:27.000000'
			-811926962000000, //	'1944-04-09 16:43:58.000000'
			-20852065000000, //	'1969-05-04 15:45:35.000000'
			191206704000000, //	'1976-01-23 00:58:24.000000'
			896735912000000, //	'1998-06-01 21:18:32.000000'
			1262903093000000, //	'2010-01-07 22:24:53.000000'
			1926203568000000 //	'2031-01-15 00:32:48.000000'
		};

		rmm::device_vector<int64_t> inputDataDev(inputData);
		rmm::device_vector<gdf_valid_type> inputValidDev(4);
		inputValidDev[0] = 255;
		inputValidDev[1] = 255;
		inputValidDev[2] = 255;
		inputValidDev[3] = 63;
		rmm::device_vector<int64_t> outputDataDev(colSize);
		rmm::device_vector<gdf_valid_type> outputValidDev(4);

		inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
		inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
		outputCol.data = thrust::raw_pointer_cast(outputDataDev.data());
		outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

		gdf_error gdfError = gdf_cast_timestamp_to_timestamp(&inputCol, &outputCol, TIME_UNIT_us);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_TRUE( outputCol.dtype == GDF_TIMESTAMP );
		EXPECT_TRUE( outputCol.dtype_info.time_unit == TIME_UNIT_us );

		bool result = thrust::equal(inputValidDev.begin(), inputValidDev.end(), outputValidDev.begin());
		EXPECT_TRUE( result );

		std::vector<int64_t> results(colSize);
		thrust::copy(outputDataDev.begin(), outputDataDev.end(), results.begin());

		for (int i = 0; i < colSize; i++){
			EXPECT_TRUE( results[i] == outputData[i] );
		}
	}
}
