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

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/column_wrapper.cuh>
#include <utilities/cudf_utils.h>
#include <cudf/utilities/legacy/wrapper_types.hpp>
#include <cudf/cudf.h>
#include <cudf/unary.hpp>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include <rmm/thrust_rmm_allocator.h>

#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#include <numeric>
#include <limits>
#include <random>
#include <algorithm>
#include <cstdlib>

struct col_cast_test : public GdfTest {};

TEST_F(col_cast_test, usage_example) {

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

	std::vector<cudf::date32> inputDate32Data = {
		cudf::date32{-1528}, // '1965-10-26'
		cudf::date32{17716}, // '2018-07-04'
		cudf::date32{19382} // '2023-01-25'
	};

	std::vector<cudf::date64> inputDate64Data = {
		cudf::date64{1528935590000}, // '2018-06-14 00:19:50.000'
		cudf::date64{1528935599999}, // '2018-06-14 00:19:59.999'
		cudf::date64{-1577923201000}, // '1919-12-31 23:59:59.000'
	};

	std::vector<cudf::timestamp> inputTimestampMilliData = {
		cudf::timestamp{1528935590000}, // '2018-06-14 00:19:50.000'
		cudf::timestamp{1528935599999}, // '2018-06-14 00:19:59.999'
		cudf::timestamp{-1577923201000}, // '1919-12-31 23:59:59.000'
	};

	// Input column for int32
	auto inputInt32Col = cudf::test::column_wrapper<int32_t>(inputInt32Data);

	// Input column for float32
	auto inputFloat32Col = cudf::test::column_wrapper<float>(inputFloat32Data);

	// Input column for int64
	auto inputInt64Col = cudf::test::column_wrapper<int64_t>(inputInt64Data);

	// Input column for date32
	auto inputDate32Col = cudf::test::column_wrapper<cudf::date32>(inputDate32Data);

	// Input column for date64
	auto inputDate64Col = cudf::test::column_wrapper<cudf::date64>(inputDate64Data);

	// Input column for timestamp in ms
	auto inputTimestampMilliCol = cudf::test::column_wrapper<cudf::timestamp>(inputTimestampMilliData);
	inputTimestampMilliCol.get()->dtype_info.time_unit = TIME_UNIT_ms;

	// example for cudf::cast generic to f32
	{
		auto results = cudf::test::column_wrapper<float>(std::vector<float>{
			-1528.0,
			1.0,
			19382.0,});

		// from int32
		gdf_column output;
		EXPECT_NO_THROW( output = cudf::cast(inputInt32Col, GDF_FLOAT32) );

		// Output column
		auto outputFloat32Col = cudf::test::column_wrapper<float>(output);
		EXPECT_TRUE( results == outputFloat32Col );
	}

	// example for cudf::cast generic to i32
	{
		auto results = cudf::test::column_wrapper<int32_t>(std::vector<int32_t>{
			-1528,
			1,
			19382,});

		// from float32
		gdf_column output;
		EXPECT_NO_THROW( output = cudf::cast(inputFloat32Col, GDF_INT32) );

		// Output column
		auto outputInt32Col = cudf::test::column_wrapper<int32_t>(output);
		EXPECT_TRUE( results == outputInt32Col );
	}

	// example for cudf::cast generic to i64 - upcast
	{
		auto results = cudf::test::column_wrapper<int64_t>(std::vector<int64_t>{
			-1528,
			1,
			19382,
		});

		// from int32
		gdf_column output;
		EXPECT_NO_THROW( output = cudf::cast(inputInt32Col, GDF_INT64) );

		// Output column
		auto outputInt64Col = cudf::test::column_wrapper<int64_t>(output);
		EXPECT_TRUE( results == outputInt64Col );
	}

	// example for cudf::cast generic to i32 - downcast
	{
		auto results = cudf::test::column_wrapper<int32_t>(std::vector<int32_t>{
			-1528,
			1,
			19382,
		});

		// from int64
		gdf_column output;
		EXPECT_NO_THROW( output = cudf::cast(inputInt64Col, GDF_INT32) );

		// Output column
		auto outputInt32Col = cudf::test::column_wrapper<int32_t>(output);
		EXPECT_TRUE( results == outputInt32Col );
	}

	// example for cudf::cast generic to i32
	{
		auto results = cudf::test::column_wrapper<int32_t>(std::vector<int32_t>{
			-1528,
			17716,
			19382,
		});

		// from date32
		gdf_column output;
		EXPECT_NO_THROW( output = cudf::cast(inputDate32Col, GDF_INT32) );

		// Output column
		auto outputInt32Col = cudf::test::column_wrapper<int32_t>(output);
		EXPECT_TRUE( results == outputInt32Col );
	}

	// example for cudf::cast generic to date32
	{
		auto results = cudf::test::column_wrapper<cudf::date32>(std::vector<cudf::date32>{
			cudf::date32{-1528},
			cudf::date32{1},
			cudf::date32{19382},
		});

		// from int32
		gdf_column output;
		EXPECT_NO_THROW( output = cudf::cast(inputInt32Col, GDF_DATE32) );

		// Output column
		auto outputDate32Col = cudf::test::column_wrapper<cudf::date32>(output);
		EXPECT_TRUE( results == outputDate32Col );
	}

	// example for cudf::cast generic to timestamp
	{
		auto results = cudf::test::column_wrapper<cudf::timestamp>(std::vector<cudf::timestamp>{
			cudf::timestamp{1528935590000000}, // '2018-06-14 00:19:50.000000'
			cudf::timestamp{1528935599999000}, // '2018-06-14 00:19:59.999000'
			cudf::timestamp{-1577923201000000},  // '1919-12-31 23:59:59.000000'
		});
		results.get()->dtype_info.time_unit = TIME_UNIT_us;

		// from date64
		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_us;
		EXPECT_NO_THROW( output = cudf::cast(inputDate64Col, GDF_TIMESTAMP, info) );

		// Output column
		auto outputTimestampMicroCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( results == outputTimestampMicroCol );
	}

	// example for cudf::cast generic to date32
	{
		auto results = cudf::test::column_wrapper<cudf::date32>(std::vector<cudf::date32>{
			cudf::date32{17696}, // '2018-06-14'
			cudf::date32{17696}, // '2018-06-14'
			cudf::date32{-18264}, // '1919-12-31'
		});

		// from timestamp in ms
		gdf_column output;
		EXPECT_NO_THROW( output = cudf::cast(inputTimestampMilliCol, GDF_DATE32) );

		// Output column
		auto outputDate32Col = cudf::test::column_wrapper<cudf::date32>(output);
		EXPECT_TRUE( results == outputDate32Col );
	}
}

TEST_F(col_cast_test, dataPtrFailureTest) {

	std::vector<int32_t> inputData = { -1528, 1, 19382 };
	auto inputCol = cudf::test::column_wrapper<int32_t>(inputData);

	auto indata = inputCol.get()->data;
	inputCol.get()->data = nullptr;

	// Pointer to data in input column is null
	EXPECT_THROW(cudf::cast(inputCol, GDF_FLOAT32), cudf::logic_error);
}

TEST_F(col_cast_test, inputValidMaskFailureTest) {

	std::vector<int32_t> inputData = { -1528, 1, 19382 };
	auto inputCol = cudf::test::column_wrapper<int32_t>(inputData);

	inputCol.get()->null_count = 1;

	// Pointer to input column's valid mask is null but null count > 0
	EXPECT_THROW(cudf::cast(inputCol, GDF_FLOAT32), cudf::logic_error);
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

struct col_cast_CPU_VS_GPU_TEST : public GdfTest {};

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
	TEST_F(col_cast_CPU_VS_GPU_TEST, VFROM##_to_##VTO) {						\
	{																			\
		int colSize = 1024;														\
		std::vector<TFROM> inputData(colSize);									\
		fill_with_random_values<TTO, TFROM>(inputData, colSize);				\
																				\
		auto inputCol = cudf::test::column_wrapper<TFROM>(inputData);			\
																				\
		gdf_column output;														\
		EXPECT_NO_THROW(output = cudf::cast(inputCol, cudf::gdf_dtype_of<TTO>()));\
																				\
		auto outputCol = cudf::test::column_wrapper<TTO>(output);				\
		auto results = std::get<0>(outputCol.to_host());						\
																				\
		std::vector<TTO> outputData(colSize);									\
		inputCol.get()->data = inputData.data();								\
		outputCol.get()->data = outputData.data();								\
																				\
		gdf_error gdfError = gdf_host_cast_##VFROM##_to_##VTO(inputCol, outputCol);\
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

struct col_cast_swap_TEST : public GdfTest {};

// Casting from T1 to T2, and then casting from T2 to T1 results in the same value 
#define DEF_CAST_SWAP_TEST(VFROM, VTO, VVFROM, VVTO, TFROM, TTO)				\
	TEST_F(col_cast_swap_TEST, VFROM##_to_##VTO) {								\
	{																			\
		int colSize = 1024;														\
		std::vector<TFROM> inputData(colSize);									\
		fill_with_random_values<TTO, TFROM>(inputData, colSize);				\
																				\
		auto inputCol = cudf::test::column_wrapper<TFROM>(inputData);			\
																				\
		gdf_column output = cudf::cast(inputCol, cudf::gdf_dtype_of<TTO>());\
																				\
		gdf_column originalOutput = cudf::cast(output, cudf::gdf_dtype_of<TFROM>());\
																				\
		auto originalOutputCol = cudf::test::column_wrapper<TFROM>(originalOutput);\
		auto results = std::get<0>(originalOutputCol.to_host());				\
																				\
		for (int i = 0; i < colSize; i++){										\
			EXPECT_TRUE( results[i] == inputData[i] );							\
		}																		\
	}																			\
}

// Casting from T1 to T2, and then casting from T2 to T1 results in the same value
#define DEF_CAST_SWAP_TEST_TO_TIMESTAMP(VFROM, VVFROM, TFROM)				\
	TEST_F(col_cast_swap_TEST, VFROM##_to_timestamp) {								\
	{																			\
		int colSize = 1024;														\
		std::vector<TFROM> inputData(colSize);									\
		fill_with_random_values<int64_t, TFROM>(inputData, colSize);			\
																				\
		auto inputCol = cudf::test::column_wrapper<TFROM>(inputData);			\
																				\
		gdf_dtype_extra_info info{};											\
		info.time_unit = TIME_UNIT_ms;											\
																				\
		gdf_column output = cudf::cast(inputCol, GDF_TIMESTAMP, info);		\
																				\
		gdf_column originalOutput = cudf::cast(output, cudf::gdf_dtype_of<TFROM>());\
																				\
		auto originalOutputCol = cudf::test::column_wrapper<TFROM>(originalOutput);\
		auto results = std::get<0>(originalOutputCol.to_host());				\
																				\
		for (int i = 0; i < colSize; i++){										\
			EXPECT_TRUE( results[i] == inputData[i] );							\
		}																		\
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
    __host__ __device__
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
		std::vector<float> inputData(colSize);
		fill_with_random_values<double, float>(inputData, colSize);

		auto inputCol = cudf::test::column_wrapper<float>(inputData); 

		EXPECT_NO_THROW(cudf::cast(inputCol, GDF_FLOAT64));
	}

	//The input and output valid bitmaps are equal
	{
		const int colSize = 1024;
		std::vector<float> inputData(colSize);
		fill_with_random_values<float, float>(inputData, colSize);

		auto inputCol = cudf::test::column_wrapper<float>(inputData, generateValidRandom{}); 

		gdf_column output;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_FLOAT32));

		auto outputCol = cudf::test::column_wrapper<float>(output);
		EXPECT_TRUE( outputCol.get()->dtype == GDF_FLOAT32 );
		EXPECT_TRUE( inputCol == outputCol );
	}

	//Testing with a colSize not divisible by 8
	{
		const int colSize = 1000;
		std::vector<float> inputData(colSize);
		fill_with_random_values<float, float>(inputData, colSize);

		auto inputCol = cudf::test::column_wrapper<float>(inputData, generateValidRandom{}); 

		gdf_column output;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_FLOAT32));

		auto outputCol = cudf::test::column_wrapper<float>(output);
		EXPECT_TRUE( inputCol == outputCol );
	}
}

struct gdf_date_casting_TEST : public GdfTest {};

TEST_F(gdf_date_casting_TEST, date32_to_date64) {

	//date32 to date64
	{
		std::vector<cudf::date32> inputData = {
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{17697},	// '2018-06-15'
			cudf::date32{-18264},	// '1919-12-31'
			cudf::date32{18321},   // '2020-02-29'
			cudf::date32{0},       // '1970-01-01'
			cudf::date32{26732},   // '2043-03-11'
			cudf::date32{10336},    // '1998-04-20'
			cudf::date32{-56374}  // '1815-08-28
		};

		std::vector<cudf::date64> outputData = {
			cudf::date64{1528934400000},	// '2018-06-14 00:00:00.000'
			cudf::date64{1529020800000},	// '2018-06-15 00:00:00.000'
			cudf::date64{-1578009600000},	// '1919-12-31 00:00:00.000'
			cudf::date64{1582934400000},   // '2020-02-29 00:00:00.000'
			cudf::date64{0},            // '1970-01-01 00:00:00.000'
			cudf::date64{2309644800000},   // '2043-03-11 00:00:00.000'
			cudf::date64{893030400000},    // '1998-04-20 00:00:00.000'
			cudf::date64{-4870713600000}  // '1815-08-28 00:00:00.000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::date32>(inputData,  allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::date64>(outputData, allValidFunctor);

		gdf_column output;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_DATE64));

		auto outputCol = cudf::test::column_wrapper<cudf::date64>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	// date64 to date32
	{
		// timestamps with milliseconds
		std::vector<cudf::date64> inputData = {
			cudf::date64{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::date64{1528935599999}, // '2018-06-14 00:19:59.999'
			cudf::date64{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::date64{1582934401123}, // '2020-02-29 00:00:01.123'
			cudf::date64{0},             // '1970-01-01 00:00:00.000'
			cudf::date64{2309653342222}, // '2043-03-11 02:22:22.222'
			cudf::date64{893075430345}, // '1998-04-20 12:30:30.345'
			cudf::date64{-4870653058987},  // '1815-08-28 16:49:01.013'
			cudf::date64{-4500},            // '1969-12-31 23:59:55.500'
			cudf::date64{-169138999},    // '1969-12-30 01:01:01.001'
			cudf::date64{-5999},        // '1969-12-31 23:59:54.001'
			cudf::date64{-1991063752000}, //	'1906-11-28 06:44:08'
			cudf::date64{-1954281039000}, //	'1908-01-28 00:09:21'
			cudf::date64{-1669612095000}, //	'1917-02-03 18:51:45'
			cudf::date64{-1184467876000}, //	'1932-06-19 21:08:44'
			cudf::date64{362079575000}, //	'1981-06-22 17:39:35'
			cudf::date64{629650040000}, //	'1989-12-14 14:47:20'
			cudf::date64{692074060000}, //	'1991-12-07 02:47:40'
			cudf::date64{734734764000}, //	'1993-04-13 20:59:24'
			cudf::date64{1230998894000}, //	'2009-01-03 16:08:14'
			cudf::date64{1521989991000}, //	'2018-03-25 14:59:51'
			cudf::date64{1726355294000}, //	'2024-09-14 23:08:14'
			cudf::date64{-1722880051000}, //	'1915-05-29 06:12:29'
			cudf::date64{-948235893000}, //	'1939-12-15 01:08:27'
			cudf::date64{-811926962000}, //	'1944-04-09 16:43:58'
			cudf::date64{-20852065000}, //	'1969-05-04 15:45:35'
			cudf::date64{191206704000}, //	'1976-01-23 00:58:24'
			cudf::date64{896735912000}, //	'1998-06-01 21:18:32'
			cudf::date64{1262903093000}, //	'2010-01-07 22:24:53'
			cudf::date64{1926203568000} //	'2031-01-15 00:32:48'
		};

		std::vector<cudf::date32> outputData = {
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{-18264},	// '1919-12-31'
			cudf::date32{18321},  // '2020-02-29'
			cudf::date32{0},      // '1970-01-01'
			cudf::date32{26732},  // '2043-03-11'
			cudf::date32{10336},  // '1998-04-20'
			cudf::date32{-56374}, // '1815-08-28'
			cudf::date32{-1},		// '1969-12-31'
			cudf::date32{-2},		// '1969-12-30'
			cudf::date32{-1},		// '1969-12-31'
			cudf::date32{-23045},	// '1906-11-28'
			cudf::date32{-22619},	// '1908-01-28'
			cudf::date32{-19325},	// '1917-02-03'
			cudf::date32{-13710},	// '1932-06-19'
			cudf::date32{4190},	// '1981-06-22'
			cudf::date32{7287},	// '1989-12-14'
			cudf::date32{8010},	// '1991-12-07'
			cudf::date32{8503},	// '1993-04-13'
			cudf::date32{14247},	// '2009-01-03'
			cudf::date32{17615},	// '2018-03-25'
			cudf::date32{19980},	// '2024-09-14'
			cudf::date32{-19941},	// '1915-05-29'
			cudf::date32{-10975},	// '1939-12-15'
			cudf::date32{-9398},	// '1944-04-09'
			cudf::date32{-242},	// '1969-05-04'
			cudf::date32{2213},	// '1976-01-23'
			cudf::date32{10378},	// '1998-06-01'
			cudf::date32{14616},	// '2010-01-07'
			cudf::date32{22294}	// '2031-01-15'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::date64>(inputData,  allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::date32>(outputData, allValidFunctor);

		gdf_column output;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_DATE32));

		auto outputCol = cudf::test::column_wrapper<cudf::date32>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}
}

TEST_F(gdf_date_casting_TEST, date32_to_date64_over_valid_bitmask) {

	//date32 to date64 over valid bitmask
	{
		std::vector<cudf::date32> inputData = {
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{17697},	// '2018-06-15'
			cudf::date32{-18264},	// '1919-12-31'
			cudf::date32{18321},   // '2020-02-29'
			cudf::date32{0},       // '1970-01-01'
			cudf::date32{26732},   // '2043-03-11'
			cudf::date32{10336},    // '1998-04-20'
			cudf::date32{-56374}  // '1815-08-28
		};

		std::vector<cudf::date64> outputData = {
			cudf::date64{1528934400000},	// '2018-06-14 00:00:00.000'
			cudf::date64{0}, // no operation
			cudf::date64{-1578009600000},	// '1919-12-31 00:00:00.000'
			cudf::date64{0}, // no operation
			cudf::date64{0},            // '1970-01-01 00:00:00.000'
			cudf::date64{0}, // no operation
			cudf::date64{893030400000},    // '1998-04-20 00:00:00.000'
			cudf::date64{0} // no operation
		};

		auto altValidFunctor = [](gdf_size_type row){return (row % 2 == 0);}; //01010101
		auto inputCol  = cudf::test::column_wrapper<cudf::date32>(inputData,  altValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::date64>(outputData, altValidFunctor);

		gdf_column output;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_DATE64));

		auto outputCol = cudf::test::column_wrapper<cudf::date64>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}
}

TEST_F(gdf_date_casting_TEST, date32_to_timestamp) {

	//date32 to timestamp s
	{
		std::vector<cudf::date32> inputData = {
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{17697},	// '2018-06-15'
			cudf::date32{-18264},	// '1919-12-31'
			cudf::date32{18321},   // '2020-02-29'
			cudf::date32{0},       // '1970-01-01'
			cudf::date32{26732},   // '2043-03-11'
			cudf::date32{10336},    // '1998-04-20'
			cudf::date32{-56374}  // '1815-08-28
		};

		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528934400},	// '2018-06-14 00:00:00'
			cudf::timestamp{1529020800},	// '2018-06-15 00:00:00'
			cudf::timestamp{-1578009600},	// '1919-12-31 00:00:00'
			cudf::timestamp{1582934400},   // '2020-02-29 00:00:00'
			cudf::timestamp{0},            // '1970-01-01 00:00:00'
			cudf::timestamp{2309644800},   // '2043-03-11 00:00:00'
			cudf::timestamp{893030400},    // '1998-04-20 00:00:00'
			cudf::timestamp{-4870713600}  // '1815-08-28 00:00:00'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::date32>   (inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		expectOut.get()->dtype_info.time_unit = TIME_UNIT_s;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_s;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp s to date32
	{
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528934400},	// '2018-06-14 00:00:00'
			cudf::timestamp{1529020800},	// '2018-06-15 00:00:00'
			cudf::timestamp{-1578009600},	// '1919-12-31 00:00:00'
			cudf::timestamp{1582934400},   // '2020-02-29 00:00:00'
			cudf::timestamp{0},            // '1970-01-01 00:00:00'
			cudf::timestamp{2309644800},   // '2043-03-11 00:00:00'
			cudf::timestamp{893030400},    // '1998-04-20 00:00:00'
			cudf::timestamp{-4870713600}  // '1815-08-28 00:00:00'
		};

		std::vector<cudf::date32> outputData = {
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{17697},	// '2018-06-15'
			cudf::date32{-18264},	// '1919-12-31'
			cudf::date32{18321},   // '2020-02-29'
			cudf::date32{0},       // '1970-01-01'
			cudf::date32{26732},   // '2043-03-11'
			cudf::date32{10336},    // '1998-04-20'
			cudf::date32{-56374}  // '1815-08-28
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::date32>   (outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_s;

		gdf_column output;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_DATE32));

		auto outputCol = cudf::test::column_wrapper<cudf::date32>   (output);
		EXPECT_TRUE( outputCol == expectOut );
	}
	
	//date32 to timestamp ms
	{
		std::vector<cudf::date32> inputData = {
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{17697},	// '2018-06-15'
			cudf::date32{-18264},	// '1919-12-31'
			cudf::date32{18321},   // '2020-02-29'
			cudf::date32{0},       // '1970-01-01'
			cudf::date32{26732},   // '2043-03-11'
			cudf::date32{10336},    // '1998-04-20'
			cudf::date32{-56374}  // '1815-08-28
		};

		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528934400000},	// '2018-06-14 00:00:00.000'
			cudf::timestamp{1529020800000},	// '2018-06-15 00:00:00.000'
			cudf::timestamp{-1578009600000},	// '1919-12-31 00:00:00.000'
			cudf::timestamp{1582934400000},   // '2020-02-29 00:00:00.000'
			cudf::timestamp{0},            // '1970-01-01 00:00:00.000'
			cudf::timestamp{2309644800000},   // '2043-03-11 00:00:00.000'
			cudf::timestamp{893030400000},    // '1998-04-20 00:00:00.000'
			cudf::timestamp{-4870713600000}  // '1815-08-28 00:00:00.000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::date32>   (inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		expectOut.get()->dtype_info.time_unit = TIME_UNIT_ms;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_ms;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp ms to date32
	{
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::timestamp{1528935599999}, // '2018-06-14 00:19:59.999'
			cudf::timestamp{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::timestamp{1582934401123}, // '2020-02-29 00:00:01.123'
			cudf::timestamp{0},             // '1970-01-01 00:00:00.000'
			cudf::timestamp{2309653342222}, // '2043-03-11 02:22:22.222'
			cudf::timestamp{893075430345}, // '1998-04-20 12:30:30.345'
			cudf::timestamp{-4870653058987},  // '1815-08-28 16:49:01.013'
		};

		std::vector<cudf::date32> outputData = {
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{-18264},	// '1919-12-31'
			cudf::date32{18321},   // '2020-02-29'
			cudf::date32{0},       // '1970-01-01'
			cudf::date32{26732},   // '2043-03-11'
			cudf::date32{10336},    // '1998-04-20'
			cudf::date32{-56374}  // '1815-08-28
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::date32>   (outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_ms;

		gdf_column output;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_DATE32));

		auto outputCol = cudf::test::column_wrapper<cudf::date32>   (output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//date32 to timestamp ns
	{
		std::vector<cudf::date32> inputData = {
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{17697},	// '2018-06-15'
			cudf::date32{-18264},	// '1919-12-31'
			cudf::date32{18321},   // '2020-02-29'
			cudf::date32{0},       // '1970-01-01'
			cudf::date32{26732},   // '2043-03-11'
			cudf::date32{10336},    // '1998-04-20'
			cudf::date32{-56374}  // '1815-08-28
		};

		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528934400000000000},	// '2018-06-14 00:00:00.000000000'
			cudf::timestamp{1529020800000000000},	// '2018-06-15 00:00:00.000000000'
			cudf::timestamp{-1578009600000000000},	// '1919-12-31 00:00:00.000000000'
			cudf::timestamp{1582934400000000000},	// '2020-02-29 00:00:00.000000000'
			cudf::timestamp{0},						// '1970-01-01 00:00:00.000000000'
			cudf::timestamp{2309644800000000000},	// '2043-03-11 00:00:00.000000000'
			cudf::timestamp{893030400000000000},		// '1998-04-20 00:00:00.000000000'
			cudf::timestamp{-4870713600000000000}	// '1815-08-28 00:00:00.000000000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::date32>   (inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		expectOut.get()->dtype_info.time_unit = TIME_UNIT_ns;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_ns;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp ns to date32
	{
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000000000}, // '2018-06-14 00:19:50.000000000'
			cudf::timestamp{1528935599999999999}, // '2018-06-14 00:19:59.999999999'
			cudf::timestamp{-1577923201000000000}, // '1919-12-31 23:59:59.000000000'
			cudf::timestamp{1582934401123123123}, // '2020-02-29 00:00:01.123123123'
			cudf::timestamp{0},             // '1970-01-01 00:00:00.000000000'
			cudf::timestamp{2309653342222222222}, // '2043-03-11 02:22:22.222222222'
			cudf::timestamp{893075430345543345}, // '1998-04-20 12:30:30.345543345'
			cudf::timestamp{-4870653058987789987},  // '1815-08-28 16:49:01.012210013'
		};

		std::vector<cudf::date32> outputData = {
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{-18264},	// '1919-12-31'
			cudf::date32{18321},   // '2020-02-29'
			cudf::date32{0},       // '1970-01-01'
			cudf::date32{26732},   // '2043-03-11'
			cudf::date32{10336},    // '1998-04-20'
			cudf::date32{-56374}  // '1815-08-28
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::date32>   (outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_ns;

		gdf_column output;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_DATE32));

		auto outputCol = cudf::test::column_wrapper<cudf::date32>   (output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//date32 to timestamp us
	{
		std::vector<cudf::date32> inputData = {
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{17697},	// '2018-06-15'
			cudf::date32{-18264},	// '1919-12-31'
			cudf::date32{18321},   // '2020-02-29'
			cudf::date32{00},       // '1970-01-01'
			cudf::date32{26732},   // '2043-03-11'
			cudf::date32{10336},    // '1998-04-20'
			cudf::date32{-56374}  // '1815-08-28
		};

		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528934400000000},	// '2018-06-14 00:00:00.000000'
			cudf::timestamp{1529020800000000},	// '2018-06-15 00:00:00.000000'
			cudf::timestamp{-1578009600000000},	// '1919-12-31 00:00:00.000000'
			cudf::timestamp{1582934400000000},   // '2020-02-29 00:00:00.000000'
			cudf::timestamp{00},            // '1970-01-01 00:00:00.000000'
			cudf::timestamp{2309644800000000},   // '2043-03-11 00:00:00.000000'
			cudf::timestamp{893030400000000},    // '1998-04-20 00:00:00.000000'
			cudf::timestamp{-4870713600000000}  // '1815-08-28 00:00:00.000000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::date32>   (inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		expectOut.get()->dtype_info.time_unit = TIME_UNIT_us;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_us;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp us to date32
	{
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000000}, // '2018-06-14 00:19:50.000000'
			cudf::timestamp{1528935599000000}, // '2018-06-14 00:19:59.000000'
			cudf::timestamp{-1577923201000000}, // '1919-12-31 23:59:59.000000'
			cudf::timestamp{1582934401000000}, // '2020-02-29 00:00:01.000000'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000'
			cudf::timestamp{2309653342000000}, // '2043-03-11 02:22:22.000000'
			cudf::timestamp{893075430000000}, // '1998-04-20 12:30:30.000000'
			cudf::timestamp{-4870653059000000},  // '1815-08-28 16:49:01.000000'
		};

		std::vector<cudf::date32> outputData = {
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{17696},	// '2018-06-14'
			cudf::date32{-18264},	// '1919-12-31'
			cudf::date32{18321},   // '2020-02-29'
			cudf::date32{00},       // '1970-01-01'
			cudf::date32{26732},   // '2043-03-11'
			cudf::date32{10336},    // '1998-04-20'
			cudf::date32{-56374}  // '1815-08-28
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::date32>   (outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_us;

		gdf_column output;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_DATE32));

		auto outputCol = cudf::test::column_wrapper<cudf::date32>   (output);
		EXPECT_TRUE( outputCol == expectOut );
	}
}

TEST_F(gdf_date_casting_TEST, date64_to_timestamp) {

	// date64 to timestamp ms, internally the output is equal to the input
	{
		int colSize = 30;

		// timestamps with milliseconds
		std::vector<int64_t> data = {
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

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::date64>(colSize,
			[data] (gdf_size_type row) { return cudf::date64{ data[row] }; },
			allValidFunctor
		); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(colSize,
			[data] (gdf_size_type row) { return cudf::timestamp{ data[row] }; },
			allValidFunctor
		);

		expectOut.get()->dtype_info.time_unit = TIME_UNIT_ms;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_ms;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	// timestamp ms to date64, internally the output is equal to the input
	{
		int colSize = 30;

		// timestamps with milliseconds
		std::vector<int64_t> data = {
			1528935590000, // '2018-06-14 00:19:50.000'
			1528935599999, // '2018-06-14 00:19:59.999'
			-1577923201000, // '1919-12-31 23:59:59.000'
			1582934401123, // '2020-02-29 00:00:01.123'
			00,             // '1970-01-01 00:00:00.000'
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

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(colSize,
			[data] (gdf_size_type row) { return cudf::timestamp{ data[row] }; },
			allValidFunctor
		); 
		auto expectOut = cudf::test::column_wrapper<cudf::date64>(colSize,
			[data] (gdf_size_type row) { return cudf::date64{ data[row] }; },
			allValidFunctor
		);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_ms;

		gdf_column output;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_DATE64));

		auto outputCol = cudf::test::column_wrapper<cudf::date64>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//date64 to timestamp s
	{
		std::vector<cudf::date64> inputData = {
			cudf::date64{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::date64{1528935599999}, // '2018-06-14 00:19:59.999'
			cudf::date64{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::date64{1582934401123}, // '2020-02-29 00:00:01.123'
			cudf::date64{00},             // '1970-01-01 00:00:00.000'
			cudf::date64{2309653342222}, // '2043-03-11 02:22:22.222'
			cudf::date64{893075430345}, // '1998-04-20 12:30:30.345'
			cudf::date64{-4870653058987},  // '1815-08-28 16:49:01.013'
		};

		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590}, // '2018-06-14 00:19:50'
			cudf::timestamp{1528935599}, // '2018-06-14 00:19:59'
			cudf::timestamp{-1577923201}, // '1919-12-31 23:59:59'
			cudf::timestamp{1582934401}, // '2020-02-29 00:00:01'
			cudf::timestamp{00},             // '1970-01-01 00:00:00'
			cudf::timestamp{2309653342}, // '2043-03-11 02:22:22'
			cudf::timestamp{893075430}, // '1998-04-20 12:30:30'
			cudf::timestamp{-4870653059},  // '1815-08-28 16:49:01'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::date64>   (inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		expectOut.get()->dtype_info.time_unit = TIME_UNIT_s;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_s;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp s to date64
	{
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590}, // '2018-06-14 00:19:50'
			cudf::timestamp{1528935599}, // '2018-06-14 00:19:59'
			cudf::timestamp{-1577923201}, // '1919-12-31 23:59:59'
			cudf::timestamp{1582934401}, // '2020-02-29 00:00:01'
			cudf::timestamp{00},             // '1970-01-01 00:00:00'
			cudf::timestamp{2309653342}, // '2043-03-11 02:22:22'
			cudf::timestamp{893075430}, // '1998-04-20 12:30:30'
			cudf::timestamp{-4870653059},  // '1815-08-28 16:49:01'
		};

		std::vector<cudf::date64> outputData = {
			cudf::date64{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::date64{1528935599000}, // '2018-06-14 00:19:59.000'
			cudf::date64{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::date64{1582934401000}, // '2020-02-29 00:00:01.000'
			cudf::date64{00},             // '1970-01-01 00:00:00.000'
			cudf::date64{2309653342000}, // '2043-03-11 02:22:22.000'
			cudf::date64{893075430000}, // '1998-04-20 12:30:30.000'
			cudf::date64{-4870653059000},  // '1815-08-28 16:49:01.000
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::date64>   (outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_s;

		gdf_column output;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_DATE64));

		auto outputCol = cudf::test::column_wrapper<cudf::date64>   (output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//date64 to timestamp us
	{
		std::vector<cudf::date64> inputData = {
			cudf::date64{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::date64{1528935599999}, // '2018-06-14 00:19:59.999'
			cudf::date64{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::date64{1582934401123}, // '2020-02-29 00:00:01.123'
			cudf::date64{00},             // '1970-01-01 00:00:00.000'
			cudf::date64{2309653342222}, // '2043-03-11 02:22:22.222'
			cudf::date64{893075430345}, // '1998-04-20 12:30:30.345'
			cudf::date64{-4870653058987},  // '1815-08-28 16:49:01.013'
		};

		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590000000}, // '2018-06-14 00:19:50.000000'
			cudf::timestamp{1528935599999000}, // '2018-06-14 00:19:59.999000'
			cudf::timestamp{-1577923201000000}, // '1919-12-31 23:59:59.000000'
			cudf::timestamp{1582934401123000}, // '2020-02-29 00:00:01.123000'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000'
			cudf::timestamp{2309653342222000}, // '2043-03-11 02:22:22.222000'
			cudf::timestamp{893075430345000}, // '1998-04-20 12:30:30.345000'
			cudf::timestamp{-4870653058987000},  // '1815-08-28 16:49:01.013000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::date64>   (inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		expectOut.get()->dtype_info.time_unit = TIME_UNIT_us;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_us;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp us to date64
	{
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000000}, // '2018-06-14 00:19:50.000000'
			cudf::timestamp{1528935599999999}, // '2018-06-14 00:19:59.999999'
			cudf::timestamp{-1577923201000000}, // '1919-12-31 23:59:59.000000'
			cudf::timestamp{1582934401123123}, // '2020-02-29 00:00:01.123123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000'
			cudf::timestamp{2309653342222222}, // '2043-03-11 02:22:22.222222'
			cudf::timestamp{893075430345543}, // '1998-04-20 12:30:30.345543'
			cudf::timestamp{-4870653058987789},  // '1815-08-28 16:49:01.012211'
		};

		std::vector<cudf::date64> outputData = {
			cudf::date64{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::date64{1528935599999}, // '2018-06-14 00:19:59.999'
			cudf::date64{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::date64{1582934401123}, // '2020-02-29 00:00:01.123'
			cudf::date64{00},             // '1970-01-01 00:00:00.000'
			cudf::date64{2309653342222}, // '2043-03-11 02:22:22.222'
			cudf::date64{893075430345}, // '1998-04-20 12:30:30.345'
			cudf::date64{-4870653058988},  // '1815-08-28 16:49:01.012'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::date64>   (outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_us;

		gdf_column output;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_DATE64));

		auto outputCol = cudf::test::column_wrapper<cudf::date64>   (output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//date64 to timestamp ns
	{
		std::vector<cudf::date64> inputData = {
			cudf::date64{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::date64{1528935599999}, // '2018-06-14 00:19:59.999'
			cudf::date64{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::date64{1582934401123}, // '2020-02-29 00:00:01.123'
			cudf::date64{00},             // '1970-01-01 00:00:00.000'
			cudf::date64{2309653342222}, // '2043-03-11 02:22:22.222'
			cudf::date64{893075430345}, // '1998-04-20 12:30:30.345'
			cudf::date64{-4870653058987},  // '1815-08-28 16:49:01.013'
		};

		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590000000000}, // '2018-06-14 00:19:50.000000000'
			cudf::timestamp{1528935599999000000}, // '2018-06-14 00:19:59.999000000'
			cudf::timestamp{-1577923201000000000}, // '1919-12-31 23:59:59.000000000'
			cudf::timestamp{1582934401123000000}, // '2020-02-29 00:00:01.123000000'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000000'
			cudf::timestamp{2309653342222000000}, // '2043-03-11 02:22:22.222000000'
			cudf::timestamp{893075430345000000}, // '1998-04-20 12:30:30.345000000'
			cudf::timestamp{-4870653058987000000},  // '1815-08-28 16:49:01.013000000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::date64>   (inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		expectOut.get()->dtype_info.time_unit = TIME_UNIT_ns;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_ns;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp ns to date64
	{
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000000000}, // '2018-06-14 00:19:50.000000000'
			cudf::timestamp{1528935599999999999}, // '2018-06-14 00:19:59.999999999'
			cudf::timestamp{-1577923201000000000}, // '1919-12-31 23:59:59.000000000'
			cudf::timestamp{1582934401123123123}, // '2020-02-29 00:00:01.123123123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000000'
			cudf::timestamp{2309653342222222222}, // '2043-03-11 02:22:22.222222222'
			cudf::timestamp{893075430345543345}, // '1998-04-20 12:30:30.345543345'
			cudf::timestamp{-4870653058987789987},  // '1815-08-28 16:49:01.012210013'
		};

		std::vector<cudf::date64> outputData = {
			cudf::date64{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::date64{1528935599999}, // '2018-06-14 00:19:59.999'
			cudf::date64{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::date64{1582934401123}, // '2020-02-29 00:00:01.123'
			cudf::date64{00},             // '1970-01-01 00:00:00.000'
			cudf::date64{2309653342222}, // '2043-03-11 02:22:22.222'
			cudf::date64{893075430345}, // '1998-04-20 12:30:30.345'
			cudf::date64{-4870653058988},  // '1815-08-28 16:49:01.012'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::date64>   (outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_ns;

		gdf_column output;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_DATE64));

		auto outputCol = cudf::test::column_wrapper<cudf::date64>   (output);
		EXPECT_TRUE( outputCol == expectOut );
	}
}

struct gdf_timestamp_casting_TEST : public GdfTest {};

TEST_F(gdf_timestamp_casting_TEST, timestamp_to_timestamp) {

	//timestamp to timestamp from s to ms
	{
		// timestamps with seconds
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590}, // '2018-06-14 00:19:50'
			cudf::timestamp{1528935599}, // '2018-06-14 00:19:59'
			cudf::timestamp{-1577923201}, // '1919-12-31 23:59:59'
			cudf::timestamp{1582934401}, // '2020-02-29 00:00:01'
			cudf::timestamp{00},             // '1970-01-01 00:00:00'
			cudf::timestamp{2309653342}, // '2043-03-11 02:22:22'
			cudf::timestamp{893075430}, // '1998-04-20 12:30:30'
			cudf::timestamp{-4870653059},  // '1815-08-28 16:49:01'
			cudf::timestamp{-5},            // '1969-12-31 23:59:55'
			cudf::timestamp{-169139},    // '1969-12-30 01:01:01'
			cudf::timestamp{-6},        // '1969-12-31 23:59:54'
			cudf::timestamp{-1991063752}, //	'1906-11-28 06:44:08'
			cudf::timestamp{-1954281039}, //	'1908-01-28 00:09:21'
			cudf::timestamp{-1669612095}, //	'1917-02-03 18:51:45'
			cudf::timestamp{-1184467876}, //	'1932-06-19 21:08:44'
			cudf::timestamp{362079575}, //	'1981-06-22 17:39:35'
			cudf::timestamp{629650040}, //	'1989-12-14 14:47:20'
			cudf::timestamp{692074060}, //	'1991-12-07 02:47:40'
			cudf::timestamp{734734764}, //	'1993-04-13 20:59:24'
			cudf::timestamp{1230998894}, //	'2009-01-03 16:08:14'
			cudf::timestamp{1521989991}, //	'2018-03-25 14:59:51'
			cudf::timestamp{1726355294}, //	'2024-09-14 23:08:14'
			cudf::timestamp{-1722880051}, //	'1915-05-29 06:12:29'
			cudf::timestamp{-948235893}, //	'1939-12-15 01:08:27'
			cudf::timestamp{-811926962}, //	'1944-04-09 16:43:58'
			cudf::timestamp{-20852065}, //	'1969-05-04 15:45:35'
			cudf::timestamp{191206704}, //	'1976-01-23 00:58:24'
			cudf::timestamp{896735912}, //	'1998-06-01 21:18:32'
			cudf::timestamp{1262903093}, //	'2010-01-07 22:24:53'
			cudf::timestamp{1926203568} //	'2031-01-15 00:32:48'
		};

		// timestamps with milliseconds
		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::timestamp{1528935599000}, // '2018-06-14 00:19:59.000'
			cudf::timestamp{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::timestamp{1582934401000}, // '2020-02-29 00:00:01.000'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000'
			cudf::timestamp{2309653342000}, // '2043-03-11 02:22:22.000'
			cudf::timestamp{893075430000}, // '1998-04-20 12:30:30.000'
			cudf::timestamp{-4870653059000},  // '1815-08-28 16:49:01.000
			cudf::timestamp{-5000},            // '1969-12-31 23:59:55.000'
			cudf::timestamp{-169139000},    // '1969-12-30 01:01:01.000
			cudf::timestamp{-6000},        // '1969-12-31 23:59:54.000'
			cudf::timestamp{-1991063752000}, //	1906-11-28 06:44:08.000
			cudf::timestamp{-1954281039000}, //	1908-01-28 00:09:21.000
			cudf::timestamp{-1669612095000}, //	1917-02-03 18:51:45.000
			cudf::timestamp{-1184467876000}, //	1932-06-19 21:08:44.000
			cudf::timestamp{362079575000}, //	1981-06-22 17:39:35.000
			cudf::timestamp{629650040000}, //	1989-12-14 14:47:20.000
			cudf::timestamp{692074060000}, //	1991-12-07 02:47:40.000
			cudf::timestamp{734734764000}, //	1993-04-13 20:59:24.000
			cudf::timestamp{1230998894000}, //	2009-01-03 16:08:14.000
			cudf::timestamp{1521989991000}, //	2018-03-25 14:59:51.000
			cudf::timestamp{1726355294000}, //	2024-09-14 23:08:14.000
			cudf::timestamp{-1722880051000}, //	1915-05-29 06:12:29.000
			cudf::timestamp{-948235893000}, //	1939-12-15 01:08:27.000
			cudf::timestamp{-811926962000}, //	1944-04-09 16:43:58.000
			cudf::timestamp{-20852065000}, //	1969-05-04 15:45:35.000
			cudf::timestamp{191206704000}, //	1976-01-23 00:58:24.000
			cudf::timestamp{896735912000}, //	1998-06-01 21:18:32.000
			cudf::timestamp{1262903093000}, //	2010-01-07 22:24:53.000
			cudf::timestamp{1926203568000} //	2031-01-15 00:32:48.000
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_s;
		expectOut.get()->dtype_info.time_unit = TIME_UNIT_ms;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_ms;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp to timestamp from ms to s
	{
		// timestamps with milliseconds
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::timestamp{1528935599999}, // '2018-06-14 00:19:59.999'
			cudf::timestamp{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::timestamp{1582934401123}, // '2020-02-29 00:00:01.123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000'
			cudf::timestamp{2309653342222}, // '2043-03-11 02:22:22.222'
			cudf::timestamp{893075430345}, // '1998-04-20 12:30:30.345'
			cudf::timestamp{-4870653058987},  // '1815-08-28 16:49:01.013'
			cudf::timestamp{-4500},            // '1969-12-31 23:59:55.500'
			cudf::timestamp{-169138999},    // '1969-12-30 01:01:01.001'
			cudf::timestamp{-5999},        // '1969-12-31 23:59:54.001'
			cudf::timestamp{-1991063752000}, //	'1906-11-28 06:44:08.000'
			cudf::timestamp{-1954281039000}, //	'1908-01-28 00:09:21.000'
			cudf::timestamp{-1669612095000}, //	'1917-02-03 18:51:45.000'
			cudf::timestamp{-1184467876000}, //	'1932-06-19 21:08:44.000'
			cudf::timestamp{362079575000}, //	'1981-06-22 17:39:35.000'
			cudf::timestamp{629650040000}, //	'1989-12-14 14:47:20.000'
			cudf::timestamp{692074060000}, //	'1991-12-07 02:47:40.000'
			cudf::timestamp{734734764000}, //	'1993-04-13 20:59:24.000'
			cudf::timestamp{1230998894000}, //	'2009-01-03 16:08:14.000'
			cudf::timestamp{1521989991000}, //	'2018-03-25 14:59:51.000'
			cudf::timestamp{1726355294000}, //	'2024-09-14 23:08:14.000'
			cudf::timestamp{-1722880051000}, //	'1915-05-29 06:12:29.000'
			cudf::timestamp{-948235893000}, //	'1939-12-15 01:08:27.000'
			cudf::timestamp{-811926962000}, //	'1944-04-09 16:43:58.000'
			cudf::timestamp{-20852065000}, //	'1969-05-04 15:45:35.000'
			cudf::timestamp{191206704000}, //	'1976-01-23 00:58:24.000'
			cudf::timestamp{896735912000}, //	'1998-06-01 21:18:32.000'
			cudf::timestamp{1262903093000}, //	'2010-01-07 22:24:53.000'
			cudf::timestamp{1926203568000} //	'2031-01-15 00:32:48.000'
		};

		// timestamps with seconds
		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590}, // '2018-06-14 00:19:50'
			cudf::timestamp{1528935599}, // '2018-06-14 00:19:59'
			cudf::timestamp{-1577923201}, // '1919-12-31 23:59:59'
			cudf::timestamp{1582934401}, // '2020-02-29 00:00:01'
			cudf::timestamp{00},             // '1970-01-01 00:00:00'
			cudf::timestamp{2309653342}, // '2043-03-11 02:22:22'
			cudf::timestamp{893075430}, // '1998-04-20 12:30:30'
			cudf::timestamp{-4870653059},  // '1815-08-28 16:49:01'
			cudf::timestamp{-5},            // '1969-12-31 23:59:55'
			cudf::timestamp{-169139},    // '1969-12-30 01:01:01'
			cudf::timestamp{-6},        // '1969-12-31 23:59:54'
			cudf::timestamp{-1991063752}, //	'1906-11-28 06:44:08'
			cudf::timestamp{-1954281039}, //	'1908-01-28 00:09:21'
			cudf::timestamp{-1669612095}, //	'1917-02-03 18:51:45'
			cudf::timestamp{-1184467876}, //	'1932-06-19 21:08:44'
			cudf::timestamp{362079575}, //	'1981-06-22 17:39:35'
			cudf::timestamp{629650040}, //	'1989-12-14 14:47:20'
			cudf::timestamp{692074060}, //	'1991-12-07 02:47:40'
			cudf::timestamp{734734764}, //	'1993-04-13 20:59:24'
			cudf::timestamp{1230998894}, //	'2009-01-03 16:08:14'
			cudf::timestamp{1521989991}, //	'2018-03-25 14:59:51'
			cudf::timestamp{1726355294}, //	'2024-09-14 23:08:14'
			cudf::timestamp{-1722880051}, //	'1915-05-29 06:12:29'
			cudf::timestamp{-948235893}, //	'1939-12-15 01:08:27'
			cudf::timestamp{-811926962}, //	'1944-04-09 16:43:58'
			cudf::timestamp{-20852065}, //	'1969-05-04 15:45:35'
			cudf::timestamp{191206704}, //	'1976-01-23 00:58:24'
			cudf::timestamp{896735912}, //	'1998-06-01 21:18:32'
			cudf::timestamp{1262903093}, //	'2010-01-07 22:24:53'
			cudf::timestamp{1926203568} //	'2031-01-15 00:32:48'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_ms;
		expectOut.get()->dtype_info.time_unit = TIME_UNIT_s;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_s;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp to timestamp from s to us
	{
		// timestamps with seconds
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590}, // '2018-06-14 00:19:50'
			cudf::timestamp{1528935599}, // '2018-06-14 00:19:59'
			cudf::timestamp{-1577923201}, // '1919-12-31 23:59:59'
			cudf::timestamp{1582934401}, // '2020-02-29 00:00:01'
			cudf::timestamp{00},             // '1970-01-01 00:00:00'
			cudf::timestamp{2309653342}, // '2043-03-11 02:22:22'
			cudf::timestamp{893075430}, // '1998-04-20 12:30:30'
			cudf::timestamp{-4870653059},  // '1815-08-28 16:49:01'
			cudf::timestamp{-5},            // '1969-12-31 23:59:55'
			cudf::timestamp{-169139},    // '1969-12-30 01:01:01'
			cudf::timestamp{-6},        // '1969-12-31 23:59:54'
			cudf::timestamp{-1991063752}, //	'1906-11-28 06:44:08'
			cudf::timestamp{-1954281039}, //	'1908-01-28 00:09:21'
			cudf::timestamp{-1669612095}, //	'1917-02-03 18:51:45'
			cudf::timestamp{-1184467876}, //	'1932-06-19 21:08:44'
			cudf::timestamp{362079575}, //	'1981-06-22 17:39:35'
			cudf::timestamp{629650040}, //	'1989-12-14 14:47:20'
			cudf::timestamp{692074060}, //	'1991-12-07 02:47:40'
			cudf::timestamp{734734764}, //	'1993-04-13 20:59:24'
			cudf::timestamp{1230998894}, //	'2009-01-03 16:08:14'
			cudf::timestamp{1521989991}, //	'2018-03-25 14:59:51'
			cudf::timestamp{1726355294}, //	'2024-09-14 23:08:14'
			cudf::timestamp{-1722880051}, //	'1915-05-29 06:12:29'
			cudf::timestamp{-948235893}, //	'1939-12-15 01:08:27'
			cudf::timestamp{-811926962}, //	'1944-04-09 16:43:58'
			cudf::timestamp{-20852065}, //	'1969-05-04 15:45:35'
			cudf::timestamp{191206704}, //	'1976-01-23 00:58:24'
			cudf::timestamp{896735912}, //	'1998-06-01 21:18:32'
			cudf::timestamp{1262903093}, //	'2010-01-07 22:24:53'
			cudf::timestamp{1926203568} //	'2031-01-15 00:32:48'
		};

		// timestamps with microseconds
		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590000000}, // '2018-06-14 00:19:50.000000'
			cudf::timestamp{1528935599000000}, // '2018-06-14 00:19:59.000000'
			cudf::timestamp{-1577923201000000}, // '1919-12-31 23:59:59.000000'
			cudf::timestamp{1582934401000000}, // '2020-02-29 00:00:01.000000'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000'
			cudf::timestamp{2309653342000000}, // '2043-03-11 02:22:22.000000'
			cudf::timestamp{893075430000000}, // '1998-04-20 12:30:30.000000'
			cudf::timestamp{-4870653059000000},  // '1815-08-28 16:49:01.000000'
			cudf::timestamp{-5000000},            // '1969-12-31 23:59:55.000000'
			cudf::timestamp{-169139000000},    // '1969-12-30 01:01:01.000000'
			cudf::timestamp{-6000000},        // '1969-12-31 23:59:54.000000'
			cudf::timestamp{-1991063752000000}, //	'1906-11-28 06:44:08.000000'
			cudf::timestamp{-1954281039000000}, //	'1908-01-28 00:09:21.000000'
			cudf::timestamp{-1669612095000000}, //	'1917-02-03 18:51:45.000000'
			cudf::timestamp{-1184467876000000}, //	'1932-06-19 21:08:44.000000'
			cudf::timestamp{362079575000000}, //	'1981-06-22 17:39:35.000000'
			cudf::timestamp{629650040000000}, //	'1989-12-14 14:47:20.000000'
			cudf::timestamp{692074060000000}, //	'1991-12-07 02:47:40.000000'
			cudf::timestamp{734734764000000}, //	'1993-04-13 20:59:24.000000'
			cudf::timestamp{1230998894000000}, //	'2009-01-03 16:08:14.000000'
			cudf::timestamp{1521989991000000}, //	'2018-03-25 14:59:51.000000'
			cudf::timestamp{1726355294000000}, //	'2024-09-14 23:08:14.000000'
			cudf::timestamp{-1722880051000000}, //	'1915-05-29 06:12:29.000000'
			cudf::timestamp{-948235893000000}, //	'1939-12-15 01:08:27.000000'
			cudf::timestamp{-811926962000000}, //	'1944-04-09 16:43:58.000000'
			cudf::timestamp{-20852065000000}, //	'1969-05-04 15:45:35.000000'
			cudf::timestamp{191206704000000}, //	'1976-01-23 00:58:24.000000'
			cudf::timestamp{896735912000000}, //	'1998-06-01 21:18:32.000000'
			cudf::timestamp{1262903093000000}, //	'2010-01-07 22:24:53.000000'
			cudf::timestamp{1926203568000000} //	'2031-01-15 00:32:48.000000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_s;
		expectOut.get()->dtype_info.time_unit = TIME_UNIT_us;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_us;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp to timestamp from us to s
	{
		// timestamps with microseconds
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000000}, // '2018-06-14 00:19:50.000000'
			cudf::timestamp{1528935599999999}, // '2018-06-14 00:19:59.999999'
			cudf::timestamp{-1577923201000000}, // '1919-12-31 23:59:59.000000'
			cudf::timestamp{1582934401123123}, // '2020-02-29 00:00:01.123123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000'
			cudf::timestamp{2309653342222222}, // '2043-03-11 02:22:22.222222'
			cudf::timestamp{893075430345543}, // '1998-04-20 12:30:30.345543'
			cudf::timestamp{-4870653058987789},  // '1815-08-28 16:49:01.012211'
			cudf::timestamp{-4500005},            // '1969-12-31 23:59:55.499995'
			cudf::timestamp{-169138999999},    // '1969-12-30 01:01:01.000001'
			cudf::timestamp{-5999999},        // '1969-12-31 23:59:54.000001'
			cudf::timestamp{-1991063752000000}, //	'1906-11-28 06:44:08.000000'
			cudf::timestamp{-1954281039000000}, //	'1908-01-28 00:09:21.000000'
			cudf::timestamp{-1669612095000000}, //	'1917-02-03 18:51:45.000000'
			cudf::timestamp{-1184467876000000}, //	'1932-06-19 21:08:44.000000'
			cudf::timestamp{362079575000000}, //	'1981-06-22 17:39:35.000000'
			cudf::timestamp{629650040000000}, //	'1989-12-14 14:47:20.000000'
			cudf::timestamp{692074060000000}, //	'1991-12-07 02:47:40.000000'
			cudf::timestamp{734734764000000}, //	'1993-04-13 20:59:24.000000'
			cudf::timestamp{1230998894000000}, //	'2009-01-03 16:08:14.000000'
			cudf::timestamp{1521989991000000}, //	'2018-03-25 14:59:51.000000'
			cudf::timestamp{1726355294000000}, //	'2024-09-14 23:08:14.000000'
			cudf::timestamp{-1722880051000000}, //	'1915-05-29 06:12:29.000000'
			cudf::timestamp{-948235893000000}, //	'1939-12-15 01:08:27.000000'
			cudf::timestamp{-811926962000000}, //	'1944-04-09 16:43:58.000000'
			cudf::timestamp{-20852065000000}, //	'1969-05-04 15:45:35.000000'
			cudf::timestamp{191206704000000}, //	'1976-01-23 00:58:24.000000'
			cudf::timestamp{896735912000000}, //	'1998-06-01 21:18:32.000000'
			cudf::timestamp{1262903093000000}, //	'2010-01-07 22:24:53.000000'
			cudf::timestamp{1926203568000000} //	'2031-01-15 00:32:48.000000'
		};

		// timestamps with seconds
		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590}, // '2018-06-14 00:19:50'
			cudf::timestamp{1528935599}, // '2018-06-14 00:19:59'
			cudf::timestamp{-1577923201}, // '1919-12-31 23:59:59'
			cudf::timestamp{1582934401}, // '2020-02-29 00:00:01'
			cudf::timestamp{00},             // '1970-01-01 00:00:00'
			cudf::timestamp{2309653342}, // '2043-03-11 02:22:22'
			cudf::timestamp{893075430}, // '1998-04-20 12:30:30'
			cudf::timestamp{-4870653059},  // '1815-08-28 16:49:01'
			cudf::timestamp{-5},            // '1969-12-31 23:59:55'
			cudf::timestamp{-169139},    // '1969-12-30 01:01:01'
			cudf::timestamp{-6},        // '1969-12-31 23:59:54'
			cudf::timestamp{-1991063752}, //	'1906-11-28 06:44:08'
			cudf::timestamp{-1954281039}, //	'1908-01-28 00:09:21'
			cudf::timestamp{-1669612095}, //	'1917-02-03 18:51:45'
			cudf::timestamp{-1184467876}, //	'1932-06-19 21:08:44'
			cudf::timestamp{362079575}, //	'1981-06-22 17:39:35'
			cudf::timestamp{629650040}, //	'1989-12-14 14:47:20'
			cudf::timestamp{692074060}, //	'1991-12-07 02:47:40'
			cudf::timestamp{734734764}, //	'1993-04-13 20:59:24'
			cudf::timestamp{1230998894}, //	'2009-01-03 16:08:14'
			cudf::timestamp{1521989991}, //	'2018-03-25 14:59:51'
			cudf::timestamp{1726355294}, //	'2024-09-14 23:08:14'
			cudf::timestamp{-1722880051}, //	'1915-05-29 06:12:29'
			cudf::timestamp{-948235893}, //	'1939-12-15 01:08:27'
			cudf::timestamp{-811926962}, //	'1944-04-09 16:43:58'
			cudf::timestamp{-20852065}, //	'1969-05-04 15:45:35'
			cudf::timestamp{191206704}, //	'1976-01-23 00:58:24'
			cudf::timestamp{896735912}, //	'1998-06-01 21:18:32'
			cudf::timestamp{1262903093}, //	'2010-01-07 22:24:53'
			cudf::timestamp{1926203568} //	'2031-01-15 00:32:48'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_us;
		expectOut.get()->dtype_info.time_unit = TIME_UNIT_s;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_s;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));
		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);

		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp to timestamp from s to ns
	{
		// timestamps with seconds
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590}, // '2018-06-14 00:19:50'
			cudf::timestamp{1528935599}, // '2018-06-14 00:19:59'
			cudf::timestamp{-1577923201}, // '1919-12-31 23:59:59'
			cudf::timestamp{1582934401}, // '2020-02-29 00:00:01'
			cudf::timestamp{00},             // '1970-01-01 00:00:00'
			cudf::timestamp{2309653342}, // '2043-03-11 02:22:22'
			cudf::timestamp{893075430}, // '1998-04-20 12:30:30'
			cudf::timestamp{-4870653059},  // '1815-08-28 16:49:01'
			cudf::timestamp{-5},            // '1969-12-31 23:59:55'
			cudf::timestamp{-169139},    // '1969-12-30 01:01:01'
			cudf::timestamp{-6},        // '1969-12-31 23:59:54'
			cudf::timestamp{-1991063752}, //	'1906-11-28 06:44:08'
			cudf::timestamp{-1954281039}, //	'1908-01-28 00:09:21'
			cudf::timestamp{-1669612095}, //	'1917-02-03 18:51:45'
			cudf::timestamp{-1184467876}, //	'1932-06-19 21:08:44'
			cudf::timestamp{362079575}, //	'1981-06-22 17:39:35'
			cudf::timestamp{629650040}, //	'1989-12-14 14:47:20'
			cudf::timestamp{692074060}, //	'1991-12-07 02:47:40'
			cudf::timestamp{734734764}, //	'1993-04-13 20:59:24'
			cudf::timestamp{1230998894}, //	'2009-01-03 16:08:14'
			cudf::timestamp{1521989991}, //	'2018-03-25 14:59:51'
			cudf::timestamp{1726355294}, //	'2024-09-14 23:08:14'
			cudf::timestamp{-1722880051}, //	'1915-05-29 06:12:29'
			cudf::timestamp{-948235893}, //	'1939-12-15 01:08:27'
			cudf::timestamp{-811926962}, //	'1944-04-09 16:43:58'
			cudf::timestamp{-20852065}, //	'1969-05-04 15:45:35'
			cudf::timestamp{191206704}, //	'1976-01-23 00:58:24'
			cudf::timestamp{896735912}, //	'1998-06-01 21:18:32'
			cudf::timestamp{1262903093}, //	'2010-01-07 22:24:53'
			cudf::timestamp{1926203568} //	'2031-01-15 00:32:48'
		};

		// timestamps with nanoseconds
		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590000000000}, // '2018-06-14 00:19:50.000000000'
			cudf::timestamp{1528935599000000000}, // '2018-06-14 00:19:59.000000000'
			cudf::timestamp{-1577923201000000000}, // '1919-12-31 23:59:59.000000000'
			cudf::timestamp{1582934401000000000}, // '2020-02-29 00:00:01.000000000'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000000'
			cudf::timestamp{2309653342000000000}, // '2043-03-11 02:22:22.000000000'
			cudf::timestamp{893075430000000000}, // '1998-04-20 12:30:30.000000000'
			cudf::timestamp{-4870653059000000000},  // '1815-08-28 16:49:01.000000000'
			cudf::timestamp{-5000000000},            // '1969-12-31 23:59:55.000000000'
			cudf::timestamp{-169139000000000},    // '1969-12-30 01:01:01.000000000'
			cudf::timestamp{-6000000000},        // '1969-12-31 23:59:54.000000000'
			cudf::timestamp{-1991063752000000000}, //	'1906-11-28 06:44:08.000000000'
			cudf::timestamp{-1954281039000000000}, //	'1908-01-28 00:09:21.000000000'
			cudf::timestamp{-1669612095000000000}, //	'1917-02-03 18:51:45.000000000'
			cudf::timestamp{-1184467876000000000}, //	'1932-06-19 21:08:44.000000000'
			cudf::timestamp{362079575000000000}, //	'1981-06-22 17:39:35.000000000'
			cudf::timestamp{629650040000000000}, //	'1989-12-14 14:47:20.000000000'
			cudf::timestamp{692074060000000000}, //	'1991-12-07 02:47:40.000000000'
			cudf::timestamp{734734764000000000}, //	'1993-04-13 20:59:24.000000000'
			cudf::timestamp{1230998894000000000}, //	'2009-01-03 16:08:14.000000000'
			cudf::timestamp{1521989991000000000}, //	'2018-03-25 14:59:51.000000000'
			cudf::timestamp{1726355294000000000}, //	'2024-09-14 23:08:14.000000000'
			cudf::timestamp{-1722880051000000000}, //	'1915-05-29 06:12:29.000000000'
			cudf::timestamp{-948235893000000000}, //	'1939-12-15 01:08:27.000000000'
			cudf::timestamp{-811926962000000000}, //	'1944-04-09 16:43:58.000000000'
			cudf::timestamp{-20852065000000000}, //	'1969-05-04 15:45:35.000000000'
			cudf::timestamp{191206704000000000}, //	'1976-01-23 00:58:24.000000000'
			cudf::timestamp{896735912000000000}, //	'1998-06-01 21:18:32.000000000'
			cudf::timestamp{1262903093000000000}, //	'2010-01-07 22:24:53.000000000'
			cudf::timestamp{1926203568000000000} //	'2031-01-15 00:32:48.000000000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_s;
		expectOut.get()->dtype_info.time_unit = TIME_UNIT_ns;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_ns;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp to timestamp from ns to s
	{
		// timestamps with nanoseconds
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000000000}, // '2018-06-14 00:19:50.000000000'
			cudf::timestamp{1528935599999999999}, // '2018-06-14 00:19:59.999999999'
			cudf::timestamp{-1577923201000000000}, // '1919-12-31 23:59:59.000000000'
			cudf::timestamp{1582934401123123123}, // '2020-02-29 00:00:01.123123123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000000'
			cudf::timestamp{2309653342222222222}, // '2043-03-11 02:22:22.222222222'
			cudf::timestamp{893075430345543345}, // '1998-04-20 12:30:30.345543345'
			cudf::timestamp{-4870653058987789987},  // '1815-08-28 16:49:01.012210013'
			cudf::timestamp{-4500000005},            // '1969-12-31 23:59:55.499999995'
			cudf::timestamp{-169138999999999},    // '1969-12-30 01:01:01.000000001'
			cudf::timestamp{-5999999999},        // '1969-12-31 23:59:54.000000001'
			cudf::timestamp{-1991063752000000000}, //	'1906-11-28 06:44:08.000000000'
			cudf::timestamp{-1954281039000000000}, //	'1908-01-28 00:09:21.000000000'
			cudf::timestamp{-1669612095000000000}, //	'1917-02-03 18:51:45.000000000'
			cudf::timestamp{-1184467876000000000}, //	'1932-06-19 21:08:44.000000000'
			cudf::timestamp{362079575000000000}, //	'1981-06-22 17:39:35.000000000'
			cudf::timestamp{629650040000000000}, //	'1989-12-14 14:47:20.000000000'
			cudf::timestamp{692074060000000000}, //	'1991-12-07 02:47:40.000000000'
			cudf::timestamp{734734764000000000}, //	'1993-04-13 20:59:24.000000000'
			cudf::timestamp{1230998894000000000}, //	'2009-01-03 16:08:14.000000000'
			cudf::timestamp{1521989991000000000}, //	'2018-03-25 14:59:51.000000000'
			cudf::timestamp{1726355294000000000}, //	'2024-09-14 23:08:14.000000000'
			cudf::timestamp{-1722880051000000000}, //	'1915-05-29 06:12:29.000000000'
			cudf::timestamp{-948235893000000000}, //	'1939-12-15 01:08:27.000000000'
			cudf::timestamp{-811926962000000000}, //	'1944-04-09 16:43:58.000000000'
			cudf::timestamp{-20852065000000000}, //	'1969-05-04 15:45:35.000000000'
			cudf::timestamp{191206704000000000}, //	'1976-01-23 00:58:24.000000000'
			cudf::timestamp{896735912000000000}, //	'1998-06-01 21:18:32.000000000'
			cudf::timestamp{1262903093000000000}, //	'2010-01-07 22:24:53.000000000'
			cudf::timestamp{1926203568000000000} //	'2031-01-15 00:32:48.000000000'
		};

		// timestamps with seconds
		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590}, // '2018-06-14 00:19:50'
			cudf::timestamp{1528935599}, // '2018-06-14 00:19:59'
			cudf::timestamp{-1577923201}, // '1919-12-31 23:59:59'
			cudf::timestamp{1582934401}, // '2020-02-29 00:00:01'
			cudf::timestamp{00},             // '1970-01-01 00:00:00'
			cudf::timestamp{2309653342}, // '2043-03-11 02:22:22'
			cudf::timestamp{893075430}, // '1998-04-20 12:30:30'
			cudf::timestamp{-4870653059},  // '1815-08-28 16:49:01'
			cudf::timestamp{-5},            // '1969-12-31 23:59:55'
			cudf::timestamp{-169139},    // '1969-12-30 01:01:01'
			cudf::timestamp{-6},        // '1969-12-31 23:59:54'
			cudf::timestamp{-1991063752}, //	'1906-11-28 06:44:08'
			cudf::timestamp{-1954281039}, //	'1908-01-28 00:09:21'
			cudf::timestamp{-1669612095}, //	'1917-02-03 18:51:45'
			cudf::timestamp{-1184467876}, //	'1932-06-19 21:08:44'
			cudf::timestamp{362079575}, //	'1981-06-22 17:39:35'
			cudf::timestamp{629650040}, //	'1989-12-14 14:47:20'
			cudf::timestamp{692074060}, //	'1991-12-07 02:47:40'
			cudf::timestamp{734734764}, //	'1993-04-13 20:59:24'
			cudf::timestamp{1230998894}, //	'2009-01-03 16:08:14'
			cudf::timestamp{1521989991}, //	'2018-03-25 14:59:51'
			cudf::timestamp{1726355294}, //	'2024-09-14 23:08:14'
			cudf::timestamp{-1722880051}, //	'1915-05-29 06:12:29'
			cudf::timestamp{-948235893}, //	'1939-12-15 01:08:27'
			cudf::timestamp{-811926962}, //	'1944-04-09 16:43:58'
			cudf::timestamp{-20852065}, //	'1969-05-04 15:45:35'
			cudf::timestamp{191206704}, //	'1976-01-23 00:58:24'
			cudf::timestamp{896735912}, //	'1998-06-01 21:18:32'
			cudf::timestamp{1262903093}, //	'2010-01-07 22:24:53'
			cudf::timestamp{1926203568} //	'2031-01-15 00:32:48'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_ns;
		expectOut.get()->dtype_info.time_unit = TIME_UNIT_s;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_s;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp to timestamp from us to ns
	{
		// timestamps with microseconds
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000000}, // '2018-06-14 00:19:50.000000'
			cudf::timestamp{1528935599999999}, // '2018-06-14 00:19:59.999999'
			cudf::timestamp{-1577923201000000}, // '1919-12-31 23:59:59.000000'
			cudf::timestamp{1582934401123123}, // '2020-02-29 00:00:01.123123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000'
			cudf::timestamp{2309653342222222}, // '2043-03-11 02:22:22.222222'
			cudf::timestamp{893075430345543}, // '1998-04-20 12:30:30.345543'
			cudf::timestamp{-4870653058987789},  // '1815-08-28 16:49:01.012211'
			cudf::timestamp{-4500005},            // '1969-12-31 23:59:55.499995'
			cudf::timestamp{-169138999999},    // '1969-12-30 01:01:01.000001'
			cudf::timestamp{-5999999},        // '1969-12-31 23:59:54.000001'
			cudf::timestamp{-1991063752000000}, //	'1906-11-28 06:44:08.000000'
			cudf::timestamp{-1954281039000000}, //	'1908-01-28 00:09:21.000000'
			cudf::timestamp{-1669612095000000}, //	'1917-02-03 18:51:45.000000'
			cudf::timestamp{-1184467876000000}, //	'1932-06-19 21:08:44.000000'
			cudf::timestamp{362079575000000}, //	'1981-06-22 17:39:35.000000'
			cudf::timestamp{629650040000000}, //	'1989-12-14 14:47:20.000000'
			cudf::timestamp{692074060000000}, //	'1991-12-07 02:47:40.000000'
			cudf::timestamp{734734764000000}, //	'1993-04-13 20:59:24.000000'
			cudf::timestamp{1230998894000000}, //	'2009-01-03 16:08:14.000000'
			cudf::timestamp{1521989991000000}, //	'2018-03-25 14:59:51.000000'
			cudf::timestamp{1726355294000000}, //	'2024-09-14 23:08:14.000000'
			cudf::timestamp{-1722880051000000}, //	'1915-05-29 06:12:29.000000'
			cudf::timestamp{-948235893000000}, //	'1939-12-15 01:08:27.000000'
			cudf::timestamp{-811926962000000}, //	'1944-04-09 16:43:58.000000'
			cudf::timestamp{-20852065000000}, //	'1969-05-04 15:45:35.000000'
			cudf::timestamp{191206704000000}, //	'1976-01-23 00:58:24.000000'
			cudf::timestamp{896735912000000}, //	'1998-06-01 21:18:32.000000'
			cudf::timestamp{1262903093000000}, //	'2010-01-07 22:24:53.000000'
			cudf::timestamp{1926203568000000} //	'2031-01-15 00:32:48.000000'
		};

		// timestamps with nanoseconds
		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590000000000}, // '2018-06-14 00:19:50.000000000'
			cudf::timestamp{1528935599999999000}, // '2018-06-14 00:19:59.999999000'
			cudf::timestamp{-1577923201000000000}, // '1919-12-31 23:59:59.000000000'
			cudf::timestamp{1582934401123123000}, // '2020-02-29 00:00:01.123123000'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000000'
			cudf::timestamp{2309653342222222000}, // '2043-03-11 02:22:22.222222000'
			cudf::timestamp{893075430345543000}, // '1998-04-20 12:30:30.345543000'
			cudf::timestamp{-4870653058987789000},  // '1815-08-28 16:49:01.012211000'
			cudf::timestamp{-4500005000},            // '1969-12-31 23:59:55.499995000'
			cudf::timestamp{-169138999999000},    // '1969-12-30 01:01:01.000001000'
			cudf::timestamp{-5999999000},        // '1969-12-31 23:59:54.000001000'
			cudf::timestamp{-1991063752000000000}, //	'1906-11-28 06:44:08.000000000'
			cudf::timestamp{-1954281039000000000}, //	'1908-01-28 00:09:21.000000000'
			cudf::timestamp{-1669612095000000000}, //	'1917-02-03 18:51:45.000000000'
			cudf::timestamp{-1184467876000000000}, //	'1932-06-19 21:08:44.000000000'
			cudf::timestamp{362079575000000000}, //	'1981-06-22 17:39:35.000000000'
			cudf::timestamp{629650040000000000}, //	'1989-12-14 14:47:20.000000000'
			cudf::timestamp{692074060000000000}, //	'1991-12-07 02:47:40.000000000'
			cudf::timestamp{734734764000000000}, //	'1993-04-13 20:59:24.000000000'
			cudf::timestamp{1230998894000000000}, //	'2009-01-03 16:08:14.000000000'
			cudf::timestamp{1521989991000000000}, //	'2018-03-25 14:59:51.000000000'
			cudf::timestamp{1726355294000000000}, //	'2024-09-14 23:08:14.000000000'
			cudf::timestamp{-1722880051000000000}, //	'1915-05-29 06:12:29.000000000'
			cudf::timestamp{-948235893000000000}, //	'1939-12-15 01:08:27.000000000'
			cudf::timestamp{-811926962000000000}, //	'1944-04-09 16:43:58.000000000'
			cudf::timestamp{-20852065000000000}, //	'1969-05-04 15:45:35.000000000'
			cudf::timestamp{191206704000000000}, //	'1976-01-23 00:58:24.000000000'
			cudf::timestamp{896735912000000000}, //	'1998-06-01 21:18:32.000000000'
			cudf::timestamp{1262903093000000000}, //	'2010-01-07 22:24:53.000000000'
			cudf::timestamp{1926203568000000000} //	'2031-01-15 00:32:48.000000000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_us;
		expectOut.get()->dtype_info.time_unit = TIME_UNIT_ns;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_ns;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp to timestamp from ns to us
	{
		// timestamps with nanoseconds
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000000000}, // '2018-06-14 00:19:50.000000000'
			cudf::timestamp{1528935599999999999}, // '2018-06-14 00:19:59.999999999'
			cudf::timestamp{-1577923201000000000}, // '1919-12-31 23:59:59.000000000'
			cudf::timestamp{1582934401123123123}, // '2020-02-29 00:00:01.123123123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000000'
			cudf::timestamp{2309653342222222222}, // '2043-03-11 02:22:22.222222222'
			cudf::timestamp{893075430345543345}, // '1998-04-20 12:30:30.345543345'
			cudf::timestamp{-4870653058987789987},  // '1815-08-28 16:49:01.012210013'
			cudf::timestamp{-4500000005},            // '1969-12-31 23:59:55.499999995'
			cudf::timestamp{-169138999999999},    // '1969-12-30 01:01:01.000000001'
			cudf::timestamp{-5999999999},        // '1969-12-31 23:59:54.000000001'
			cudf::timestamp{-1991063752000000000}, //	'1906-11-28 06:44:08.000000000'
			cudf::timestamp{-1954281039000000000}, //	'1908-01-28 00:09:21.000000000'
			cudf::timestamp{-1669612095000000000}, //	'1917-02-03 18:51:45.000000000'
			cudf::timestamp{-1184467876000000000}, //	'1932-06-19 21:08:44.000000000'
			cudf::timestamp{362079575000000000}, //	'1981-06-22 17:39:35.000000000'
			cudf::timestamp{629650040000000000}, //	'1989-12-14 14:47:20.000000000'
			cudf::timestamp{692074060000000000}, //	'1991-12-07 02:47:40.000000000'
			cudf::timestamp{734734764000000000}, //	'1993-04-13 20:59:24.000000000'
			cudf::timestamp{1230998894000000000}, //	'2009-01-03 16:08:14.000000000'
			cudf::timestamp{1521989991000000000}, //	'2018-03-25 14:59:51.000000000'
			cudf::timestamp{1726355294000000000}, //	'2024-09-14 23:08:14.000000000'
			cudf::timestamp{-1722880051000000000}, //	'1915-05-29 06:12:29.000000000'
			cudf::timestamp{-948235893000000000}, //	'1939-12-15 01:08:27.000000000'
			cudf::timestamp{-811926962000000000}, //	'1944-04-09 16:43:58.000000000'
			cudf::timestamp{-20852065000000000}, //	'1969-05-04 15:45:35.000000000'
			cudf::timestamp{191206704000000000}, //	'1976-01-23 00:58:24.000000000'
			cudf::timestamp{896735912000000000}, //	'1998-06-01 21:18:32.000000000'
			cudf::timestamp{1262903093000000000}, //	'2010-01-07 22:24:53.000000000'
			cudf::timestamp{1926203568000000000} //	'2031-01-15 00:32:48.000000000'
		};

		// timestamps with microseconds
		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590000000}, // '2018-06-14 00:19:50.000000'
			cudf::timestamp{1528935599999999}, // '2018-06-14 00:19:59.999999'
			cudf::timestamp{-1577923201000000}, // '1919-12-31 23:59:59.000000'
			cudf::timestamp{1582934401123123}, // '2020-02-29 00:00:01.123123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000'
			cudf::timestamp{2309653342222222}, // '2043-03-11 02:22:22.222222'
			cudf::timestamp{893075430345543}, // '1998-04-20 12:30:30.345543'
			cudf::timestamp{-4870653058987790},  // '1815-08-28 16:49:01.012210'
			cudf::timestamp{-4500001},            // '1969-12-31 23:59:55.499999'
			cudf::timestamp{-169139000000},    // '1969-12-30 01:01:01.000000'
			cudf::timestamp{-6000000},        // '1969-12-31 23:59:54.000000'
			cudf::timestamp{-1991063752000000}, //	'1906-11-28 06:44:08.000000'
			cudf::timestamp{-1954281039000000}, //	'1908-01-28 00:09:21.000000'
			cudf::timestamp{-1669612095000000}, //	'1917-02-03 18:51:45.000000'
			cudf::timestamp{-1184467876000000}, //	'1932-06-19 21:08:44.000000'
			cudf::timestamp{362079575000000}, //	'1981-06-22 17:39:35.000000'
			cudf::timestamp{629650040000000}, //	'1989-12-14 14:47:20.000000'
			cudf::timestamp{692074060000000}, //	'1991-12-07 02:47:40.000000'
			cudf::timestamp{734734764000000}, //	'1993-04-13 20:59:24.000000'
			cudf::timestamp{1230998894000000}, //	'2009-01-03 16:08:14.000000'
			cudf::timestamp{1521989991000000}, //	'2018-03-25 14:59:51.000000'
			cudf::timestamp{1726355294000000}, //	'2024-09-14 23:08:14.000000'
			cudf::timestamp{-1722880051000000}, //	'1915-05-29 06:12:29.000000'
			cudf::timestamp{-948235893000000}, //	'1939-12-15 01:08:27.000000'
			cudf::timestamp{-811926962000000}, //	'1944-04-09 16:43:58.000000'
			cudf::timestamp{-20852065000000}, //	'1969-05-04 15:45:35.000000'
			cudf::timestamp{191206704000000}, //	'1976-01-23 00:58:24.000000'
			cudf::timestamp{896735912000000}, //	'1998-06-01 21:18:32.000000'
			cudf::timestamp{1262903093000000}, //	'2010-01-07 22:24:53.000000'
			cudf::timestamp{1926203568000000} //	'2031-01-15 00:32:48.000000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_ns;
		expectOut.get()->dtype_info.time_unit = TIME_UNIT_us;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_us;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp to timestamp from ms to ns
	{
		// timestamps with milliseconds
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::timestamp{1528935599999}, // '2018-06-14 00:19:59.999'
			cudf::timestamp{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::timestamp{1582934401123}, // '2020-02-29 00:00:01.123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000'
			cudf::timestamp{2309653342222}, // '2043-03-11 02:22:22.222'
			cudf::timestamp{893075430345}, // '1998-04-20 12:30:30.345'
			cudf::timestamp{-4870653058987},  // '1815-08-28 16:49:01.013'
			cudf::timestamp{-4500},            // '1969-12-31 23:59:55.500'
			cudf::timestamp{-169138999},    // '1969-12-30 01:01:01.001'
			cudf::timestamp{-5999},        // '1969-12-31 23:59:54.001'
			cudf::timestamp{-1991063752000}, //	'1906-11-28 06:44:08.000'
			cudf::timestamp{-1954281039000}, //	'1908-01-28 00:09:21.000'
			cudf::timestamp{-1669612095000}, //	'1917-02-03 18:51:45.000'
			cudf::timestamp{-1184467876000}, //	'1932-06-19 21:08:44.000'
			cudf::timestamp{362079575000}, //	'1981-06-22 17:39:35.000'
			cudf::timestamp{629650040000}, //	'1989-12-14 14:47:20.000'
			cudf::timestamp{692074060000}, //	'1991-12-07 02:47:40.000'
			cudf::timestamp{734734764000}, //	'1993-04-13 20:59:24.000'
			cudf::timestamp{1230998894000}, //	'2009-01-03 16:08:14.000'
			cudf::timestamp{1521989991000}, //	'2018-03-25 14:59:51.000'
			cudf::timestamp{1726355294000}, //	'2024-09-14 23:08:14.000'
			cudf::timestamp{-1722880051000}, //	'1915-05-29 06:12:29.000'
			cudf::timestamp{-948235893000}, //	'1939-12-15 01:08:27.000'
			cudf::timestamp{-811926962000}, //	'1944-04-09 16:43:58.000'
			cudf::timestamp{-20852065000}, //	'1969-05-04 15:45:35.000'
			cudf::timestamp{191206704000}, //	'1976-01-23 00:58:24.000'
			cudf::timestamp{896735912000}, //	'1998-06-01 21:18:32.000'
			cudf::timestamp{1262903093000}, //	'2010-01-07 22:24:53.000'
			cudf::timestamp{1926203568000} //	'2031-01-15 00:32:48.000'
		};

		// timestamps with nanoseconds
		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590000000000}, // '2018-06-14 00:19:50.000000000'
			cudf::timestamp{1528935599999000000}, // '2018-06-14 00:19:59.999000000'
			cudf::timestamp{-1577923201000000000}, // '1919-12-31 23:59:59.000000000'
			cudf::timestamp{1582934401123000000}, // '2020-02-29 00:00:01.123000000'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000000'
			cudf::timestamp{2309653342222000000}, // '2043-03-11 02:22:22.222000000'
			cudf::timestamp{893075430345000000}, // '1998-04-20 12:30:30.345000000'
			cudf::timestamp{-4870653058987000000},  // '1815-08-28 16:49:01.013000000'
			cudf::timestamp{-4500000000},            // '1969-12-31 23:59:55.500000000'
			cudf::timestamp{-169138999000000},    // '1969-12-30 01:01:01.001000000'
			cudf::timestamp{-5999000000},        // '1969-12-31 23:59:54.001000000'
			cudf::timestamp{-1991063752000000000}, //	'1906-11-28 06:44:08.000000000'
			cudf::timestamp{-1954281039000000000}, //	'1908-01-28 00:09:21.000000000'
			cudf::timestamp{-1669612095000000000}, //	'1917-02-03 18:51:45.000000000'
			cudf::timestamp{-1184467876000000000}, //	'1932-06-19 21:08:44.000000000'
			cudf::timestamp{362079575000000000}, //	'1981-06-22 17:39:35.000000000'
			cudf::timestamp{629650040000000000}, //	'1989-12-14 14:47:20.000000000'
			cudf::timestamp{692074060000000000}, //	'1991-12-07 02:47:40.000000000'
			cudf::timestamp{734734764000000000}, //	'1993-04-13 20:59:24.000000000'
			cudf::timestamp{1230998894000000000}, //	'2009-01-03 16:08:14.000000000'
			cudf::timestamp{1521989991000000000}, //	'2018-03-25 14:59:51.000000000'
			cudf::timestamp{1726355294000000000}, //	'2024-09-14 23:08:14.000000000'
			cudf::timestamp{-1722880051000000000}, //	'1915-05-29 06:12:29.000000000'
			cudf::timestamp{-948235893000000000}, //	'1939-12-15 01:08:27.000000000'
			cudf::timestamp{-811926962000000000}, //	'1944-04-09 16:43:58.000000000'
			cudf::timestamp{-20852065000000000}, //	'1969-05-04 15:45:35.000000000'
			cudf::timestamp{191206704000000000}, //	'1976-01-23 00:58:24.000000000'
			cudf::timestamp{896735912000000000}, //	'1998-06-01 21:18:32.000000000'
			cudf::timestamp{1262903093000000000}, //	'2010-01-07 22:24:53.000000000'
			cudf::timestamp{1926203568000000000} //	'2031-01-15 00:32:48.000000000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_ms;
		expectOut.get()->dtype_info.time_unit = TIME_UNIT_ns;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_ns;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp to timestamp from ns to ms
	{
		// timestamps with nanoseconds
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000000000}, // '2018-06-14 00:19:50.000000000'
			cudf::timestamp{1528935599999999999}, // '2018-06-14 00:19:59.999999999'
			cudf::timestamp{-1577923201000000000}, // '1919-12-31 23:59:59.000000000'
			cudf::timestamp{1582934401123123123}, // '2020-02-29 00:00:01.123123123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000000'
			cudf::timestamp{2309653342222222222}, // '2043-03-11 02:22:22.222222222'
			cudf::timestamp{893075430345543345}, // '1998-04-20 12:30:30.345543345'
			cudf::timestamp{-4870653058987789987},  // '1815-08-28 16:49:01.012210013'
			cudf::timestamp{-4500000005},            // '1969-12-31 23:59:55.499999995'
			cudf::timestamp{-169138999999999},    // '1969-12-30 01:01:01.000000001'
			cudf::timestamp{-5999999999},        // '1969-12-31 23:59:54.000000001'
			cudf::timestamp{-1991063752000000000}, //	'1906-11-28 06:44:08.000000000'
			cudf::timestamp{-1954281039000000000}, //	'1908-01-28 00:09:21.000000000'
			cudf::timestamp{-1669612095000000000}, //	'1917-02-03 18:51:45.000000000'
			cudf::timestamp{-1184467876000000000}, //	'1932-06-19 21:08:44.000000000'
			cudf::timestamp{362079575000000000}, //	'1981-06-22 17:39:35.000000000'
			cudf::timestamp{629650040000000000}, //	'1989-12-14 14:47:20.000000000'
			cudf::timestamp{692074060000000000}, //	'1991-12-07 02:47:40.000000000'
			cudf::timestamp{734734764000000000}, //	'1993-04-13 20:59:24.000000000'
			cudf::timestamp{1230998894000000000}, //	'2009-01-03 16:08:14.000000000'
			cudf::timestamp{1521989991000000000}, //	'2018-03-25 14:59:51.000000000'
			cudf::timestamp{1726355294000000000}, //	'2024-09-14 23:08:14.000000000'
			cudf::timestamp{-1722880051000000000}, //	'1915-05-29 06:12:29.000000000'
			cudf::timestamp{-948235893000000000}, //	'1939-12-15 01:08:27.000000000'
			cudf::timestamp{-811926962000000000}, //	'1944-04-09 16:43:58.000000000'
			cudf::timestamp{-20852065000000000}, //	'1969-05-04 15:45:35.000000000'
			cudf::timestamp{191206704000000000}, //	'1976-01-23 00:58:24.000000000'
			cudf::timestamp{896735912000000000}, //	'1998-06-01 21:18:32.000000000'
			cudf::timestamp{1262903093000000000}, //	'2010-01-07 22:24:53.000000000'
			cudf::timestamp{1926203568000000000} //	'2031-01-15 00:32:48.000000000'
		};

		// timestamps with milliseconds
		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::timestamp{1528935599999}, // '2018-06-14 00:19:59.999'
			cudf::timestamp{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::timestamp{1582934401123}, // '2020-02-29 00:00:01.123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000'
			cudf::timestamp{2309653342222}, // '2043-03-11 02:22:22.222'
			cudf::timestamp{893075430345}, // '1998-04-20 12:30:30.345'
			cudf::timestamp{-4870653058988},  // '1815-08-28 16:49:01.012'
			cudf::timestamp{-4501},            // '1969-12-31 23:59:55.499'
			cudf::timestamp{-169139000},    // '1969-12-30 01:01:01.000'
			cudf::timestamp{-6000},        // '1969-12-31 23:59:54.000'
			cudf::timestamp{-1991063752000}, //	'1906-11-28 06:44:08.000'
			cudf::timestamp{-1954281039000}, //	'1908-01-28 00:09:21.000'
			cudf::timestamp{-1669612095000}, //	'1917-02-03 18:51:45.000'
			cudf::timestamp{-1184467876000}, //	'1932-06-19 21:08:44.000'
			cudf::timestamp{362079575000}, //	'1981-06-22 17:39:35.000'
			cudf::timestamp{629650040000}, //	'1989-12-14 14:47:20.000'
			cudf::timestamp{692074060000}, //	'1991-12-07 02:47:40.000'
			cudf::timestamp{734734764000}, //	'1993-04-13 20:59:24.000'
			cudf::timestamp{1230998894000}, //	'2009-01-03 16:08:14.000'
			cudf::timestamp{1521989991000}, //	'2018-03-25 14:59:51.000'
			cudf::timestamp{1726355294000}, //	'2024-09-14 23:08:14.000'
			cudf::timestamp{-1722880051000}, //	'1915-05-29 06:12:29.000'
			cudf::timestamp{-948235893000}, //	'1939-12-15 01:08:27.000'
			cudf::timestamp{-811926962000}, //	'1944-04-09 16:43:58.000'
			cudf::timestamp{-20852065000}, //	'1969-05-04 15:45:35.000'
			cudf::timestamp{191206704000}, //	'1976-01-23 00:58:24.000'
			cudf::timestamp{896735912000}, //	'1998-06-01 21:18:32.000'
			cudf::timestamp{1262903093000}, //	'2010-01-07 22:24:53.000'
			cudf::timestamp{1926203568000} //	'2031-01-15 00:32:48.000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_ns;
		expectOut.get()->dtype_info.time_unit = TIME_UNIT_ms;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_ms;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp to timestamp from us to ms
	{
		// timestamps with microseconds
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000000}, // '2018-06-14 00:19:50.000000'
			cudf::timestamp{1528935599999999}, // '2018-06-14 00:19:59.999999'
			cudf::timestamp{-1577923201000000}, // '1919-12-31 23:59:59.000000'
			cudf::timestamp{1582934401123123}, // '2020-02-29 00:00:01.123123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000'
			cudf::timestamp{2309653342222222}, // '2043-03-11 02:22:22.222222'
			cudf::timestamp{893075430345543}, // '1998-04-20 12:30:30.345543'
			cudf::timestamp{-4870653058987789},  // '1815-08-28 16:49:01.012211'
			cudf::timestamp{-4500005},            // '1969-12-31 23:59:55.499995'
			cudf::timestamp{-169138999999},    // '1969-12-30 01:01:01.000001'
			cudf::timestamp{-5999999},        // '1969-12-31 23:59:54.000001'
			cudf::timestamp{-1991063752000000}, //	'1906-11-28 06:44:08.000000'
			cudf::timestamp{-1954281039000000}, //	'1908-01-28 00:09:21.000000'
			cudf::timestamp{-1669612095000000}, //	'1917-02-03 18:51:45.000000'
			cudf::timestamp{-1184467876000000}, //	'1932-06-19 21:08:44.000000'
			cudf::timestamp{362079575000000}, //	'1981-06-22 17:39:35.000000'
			cudf::timestamp{629650040000000}, //	'1989-12-14 14:47:20.000000'
			cudf::timestamp{692074060000000}, //	'1991-12-07 02:47:40.000000'
			cudf::timestamp{734734764000000}, //	'1993-04-13 20:59:24.000000'
			cudf::timestamp{1230998894000000}, //	'2009-01-03 16:08:14.000000'
			cudf::timestamp{1521989991000000}, //	'2018-03-25 14:59:51.000000'
			cudf::timestamp{1726355294000000}, //	'2024-09-14 23:08:14.000000'
			cudf::timestamp{-1722880051000000}, //	'1915-05-29 06:12:29.000000'
			cudf::timestamp{-948235893000000}, //	'1939-12-15 01:08:27.000000'
			cudf::timestamp{-811926962000000}, //	'1944-04-09 16:43:58.000000'
			cudf::timestamp{-20852065000000}, //	'1969-05-04 15:45:35.000000'
			cudf::timestamp{191206704000000}, //	'1976-01-23 00:58:24.000000'
			cudf::timestamp{896735912000000}, //	'1998-06-01 21:18:32.000000'
			cudf::timestamp{1262903093000000}, //	'2010-01-07 22:24:53.000000'
			cudf::timestamp{1926203568000000} //	'2031-01-15 00:32:48.000000'
		};

		// timestamps with milliseconds
		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::timestamp{1528935599999}, // '2018-06-14 00:19:59.999'
			cudf::timestamp{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::timestamp{1582934401123}, // '2020-02-29 00:00:01.123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000'
			cudf::timestamp{2309653342222}, // '2043-03-11 02:22:22.222'
			cudf::timestamp{893075430345}, // '1998-04-20 12:30:30.345'
			cudf::timestamp{-4870653058988},  // '1815-08-28 16:49:01.012'
			cudf::timestamp{-4501},            // '1969-12-31 23:59:55.499'
			cudf::timestamp{-169139000},    // '1969-12-30 01:01:01.000'
			cudf::timestamp{-6000},        // '1969-12-31 23:59:54.000'
			cudf::timestamp{-1991063752000}, //	'1906-11-28 06:44:08.000'
			cudf::timestamp{-1954281039000}, //	'1908-01-28 00:09:21.000'
			cudf::timestamp{-1669612095000}, //	'1917-02-03 18:51:45.000'
			cudf::timestamp{-1184467876000}, //	'1932-06-19 21:08:44.000'
			cudf::timestamp{362079575000}, //	'1981-06-22 17:39:35.000'
			cudf::timestamp{629650040000}, //	'1989-12-14 14:47:20.000'
			cudf::timestamp{692074060000}, //	'1991-12-07 02:47:40.000'
			cudf::timestamp{734734764000}, //	'1993-04-13 20:59:24.000'
			cudf::timestamp{1230998894000}, //	'2009-01-03 16:08:14.000'
			cudf::timestamp{1521989991000}, //	'2018-03-25 14:59:51.000'
			cudf::timestamp{1726355294000}, //	'2024-09-14 23:08:14.000'
			cudf::timestamp{-1722880051000}, //	'1915-05-29 06:12:29.000'
			cudf::timestamp{-948235893000}, //	'1939-12-15 01:08:27.000'
			cudf::timestamp{-811926962000}, //	'1944-04-09 16:43:58.000'
			cudf::timestamp{-20852065000}, //	'1969-05-04 15:45:35.000'
			cudf::timestamp{191206704000}, //	'1976-01-23 00:58:24.000'
			cudf::timestamp{896735912000}, //	'1998-06-01 21:18:32.000'
			cudf::timestamp{1262903093000}, //	'2010-01-07 22:24:53.000'
			cudf::timestamp{1926203568000} //	'2031-01-15 00:32:48.000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_us;
		expectOut.get()->dtype_info.time_unit = TIME_UNIT_ms;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_ms;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}

	//timestamp to timestamp from ms to us
	{
		// timestamps with milliseconds
		std::vector<cudf::timestamp> inputData = {
			cudf::timestamp{1528935590000}, // '2018-06-14 00:19:50.000'
			cudf::timestamp{1528935599999}, // '2018-06-14 00:19:59.999'
			cudf::timestamp{-1577923201000}, // '1919-12-31 23:59:59.000'
			cudf::timestamp{1582934401123}, // '2020-02-29 00:00:01.123'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000'
			cudf::timestamp{2309653342222}, // '2043-03-11 02:22:22.222'
			cudf::timestamp{893075430345}, // '1998-04-20 12:30:30.345'
			cudf::timestamp{-4870653058987},  // '1815-08-28 16:49:01.013'
			cudf::timestamp{-4500},            // '1969-12-31 23:59:55.500'
			cudf::timestamp{-169138999},    // '1969-12-30 01:01:01.001'
			cudf::timestamp{-5999},        // '1969-12-31 23:59:54.001'
			cudf::timestamp{-1991063752000}, //	'1906-11-28 06:44:08.000'
			cudf::timestamp{-1954281039000}, //	'1908-01-28 00:09:21.000'
			cudf::timestamp{-1669612095000}, //	'1917-02-03 18:51:45.000'
			cudf::timestamp{-1184467876000}, //	'1932-06-19 21:08:44.000'
			cudf::timestamp{362079575000}, //	'1981-06-22 17:39:35.000'
			cudf::timestamp{629650040000}, //	'1989-12-14 14:47:20.000'
			cudf::timestamp{692074060000}, //	'1991-12-07 02:47:40.000'
			cudf::timestamp{734734764000}, //	'1993-04-13 20:59:24.000'
			cudf::timestamp{1230998894000}, //	'2009-01-03 16:08:14.000'
			cudf::timestamp{1521989991000}, //	'2018-03-25 14:59:51.000'
			cudf::timestamp{1726355294000}, //	'2024-09-14 23:08:14.000'
			cudf::timestamp{-1722880051000}, //	'1915-05-29 06:12:29.000'
			cudf::timestamp{-948235893000}, //	'1939-12-15 01:08:27.000'
			cudf::timestamp{-811926962000}, //	'1944-04-09 16:43:58.000'
			cudf::timestamp{-20852065000}, //	'1969-05-04 15:45:35.000'
			cudf::timestamp{191206704000}, //	'1976-01-23 00:58:24.000'
			cudf::timestamp{896735912000}, //	'1998-06-01 21:18:32.000'
			cudf::timestamp{1262903093000}, //	'2010-01-07 22:24:53.000'
			cudf::timestamp{1926203568000} //	'2031-01-15 00:32:48.000'
		};

		// timestamps with microseconds
		std::vector<cudf::timestamp> outputData = {
			cudf::timestamp{1528935590000000}, // '2018-06-14 00:19:50.000000'
			cudf::timestamp{1528935599999000}, // '2018-06-14 00:19:59.999000'
			cudf::timestamp{-1577923201000000}, // '1919-12-31 23:59:59.000000'
			cudf::timestamp{1582934401123000}, // '2020-02-29 00:00:01.123000'
			cudf::timestamp{00},             // '1970-01-01 00:00:00.000000'
			cudf::timestamp{2309653342222000}, // '2043-03-11 02:22:22.222000'
			cudf::timestamp{893075430345000}, // '1998-04-20 12:30:30.345000'
			cudf::timestamp{-4870653058987000},  // '1815-08-28 16:49:01.013000'
			cudf::timestamp{-4500000},            // '1969-12-31 23:59:55.500000'
			cudf::timestamp{-169138999000},    // '1969-12-30 01:01:01.001000'
			cudf::timestamp{-5999000},        // '1969-12-31 23:59:54.001000'
			cudf::timestamp{-1991063752000000}, //	'1906-11-28 06:44:08.000000'
			cudf::timestamp{-1954281039000000}, //	'1908-01-28 00:09:21.000000'
			cudf::timestamp{-1669612095000000}, //	'1917-02-03 18:51:45.000000'
			cudf::timestamp{-1184467876000000}, //	'1932-06-19 21:08:44.000000'
			cudf::timestamp{362079575000000}, //	'1981-06-22 17:39:35.000000'
			cudf::timestamp{629650040000000}, //	'1989-12-14 14:47:20.000000'
			cudf::timestamp{692074060000000}, //	'1991-12-07 02:47:40.000000'
			cudf::timestamp{734734764000000}, //	'1993-04-13 20:59:24.000000'
			cudf::timestamp{1230998894000000}, //	'2009-01-03 16:08:14.000000'
			cudf::timestamp{1521989991000000}, //	'2018-03-25 14:59:51.000000'
			cudf::timestamp{1726355294000000}, //	'2024-09-14 23:08:14.000000'
			cudf::timestamp{-1722880051000000}, //	'1915-05-29 06:12:29.000000'
			cudf::timestamp{-948235893000000}, //	'1939-12-15 01:08:27.000000'
			cudf::timestamp{-811926962000000}, //	'1944-04-09 16:43:58.000000'
			cudf::timestamp{-20852065000000}, //	'1969-05-04 15:45:35.000000'
			cudf::timestamp{191206704000000}, //	'1976-01-23 00:58:24.000000'
			cudf::timestamp{896735912000000}, //	'1998-06-01 21:18:32.000000'
			cudf::timestamp{1262903093000000}, //	'2010-01-07 22:24:53.000000'
			cudf::timestamp{1926203568000000} //	'2031-01-15 00:32:48.000000'
		};

		auto allValidFunctor = [](gdf_size_type row){return true;};
		auto inputCol  = cudf::test::column_wrapper<cudf::timestamp>(inputData, allValidFunctor); 
		auto expectOut = cudf::test::column_wrapper<cudf::timestamp>(outputData,allValidFunctor);

		inputCol.get()->dtype_info.time_unit = TIME_UNIT_ms;
		expectOut.get()->dtype_info.time_unit = TIME_UNIT_us;

		gdf_column output;
		gdf_dtype_extra_info info{};
		info.time_unit = TIME_UNIT_us;
		EXPECT_NO_THROW(output = cudf::cast(inputCol, GDF_TIMESTAMP, info));

		auto outputCol = cudf::test::column_wrapper<cudf::timestamp>(output);
		EXPECT_TRUE( outputCol == expectOut );
	}
}

template <typename T>
struct gdf_logical_test : public GdfTest {};

using type_list = ::testing::Types<
	int8_t, int16_t, int32_t, int64_t, float, double, cudf::bool8>;

TYPED_TEST_CASE(gdf_logical_test, type_list);

TYPED_TEST(gdf_logical_test, LogicalNot) {
	const int colSize = 1000;

	// init input vector
	std::vector<TypeParam> h_input_v(colSize);
	initialize_vector(h_input_v, colSize, 10, false);

	auto inputCol = cudf::test::column_wrapper<TypeParam>{h_input_v}; 

	std::vector<cudf::bool8> h_expect_v{colSize};

	// compute NOT
	for (gdf_size_type i = 0; i < colSize; ++i)
		h_expect_v[i] = static_cast<cudf::bool8>( !h_input_v[i] );

	// Use vector to build expected output
	auto expectCol = cudf::test::column_wrapper<cudf::bool8>{h_expect_v};

	auto output = cudf::unary_operation(inputCol, cudf::unary_op::NOT);

	auto outputCol = cudf::test::column_wrapper<cudf::bool8>(output);

	EXPECT_EQ(expectCol, outputCol);
}