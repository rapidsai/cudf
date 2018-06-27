#include <cstdlib>
#include <iostream>
#include <vector>
#include <numeric>
#include <limits>
#include <random>
#include <algorithm>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>

// Generates random values between 0 and the maximum possible value of the data type with the minimum max() value
template<typename TOUT, typename TFROM>
void fill_with_random_values(std::vector<TFROM>& input, size_t size)
{
	std::random_device rd;
	std::default_random_engine eng(rd());

	if (sizeof(TOUT) > 8 || sizeof(TFROM) > 8){
		std::uniform_real_distribution<double> floating_dis;
		if( std::numeric_limits<TFROM>::max() < std::numeric_limits<TOUT>::max() )
			floating_dis = std::uniform_real_distribution<double>(std::numeric_limits<TFROM>::min(), std::numeric_limits<TFROM>::max());
		else
			floating_dis = std::uniform_real_distribution<double>(std::numeric_limits<TOUT>::min(), std::numeric_limits<TOUT>::max());

		std::generate(input.begin(), input.end(), [floating_dis, eng]() mutable {
			return static_cast<TFROM>(floating_dis(eng));
		});
	} else {
		std::uniform_real_distribution<float> floating_dis;
		if( std::numeric_limits<TFROM>::max() < std::numeric_limits<TOUT>::max() )
			floating_dis = std::uniform_real_distribution<float>(std::numeric_limits<TFROM>::min(), std::numeric_limits<TFROM>::max());
		else
			floating_dis = std::uniform_real_distribution<float>(std::numeric_limits<TOUT>::min(), std::numeric_limits<TOUT>::max());

		std::generate(input.begin(), input.end(), [floating_dis, eng]() mutable {
			return static_cast<TFROM>(floating_dis(eng));
		});
	}

}

// CPU casting

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
	TEST(gdf_cast_CPU_VS_GPU_TEST, VFROM##_to_##VTO) {							\
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
		thrust::device_vector<TFROM> intputDataDev(inputData);					\
		thrust::device_vector<TTO> outDataDev(colSize);							\
																				\
		inputCol.data = thrust::raw_pointer_cast(intputDataDev.data());			\
		inputCol.valid = nullptr;												\
		outputCol.data = thrust::raw_pointer_cast(outDataDev.data());			\
		outputCol.valid = nullptr;												\
																				\
		gdf_error gdfError = gdf_cast_##VFROM##_to_##VTO(&inputCol, &outputCol);\
		EXPECT_TRUE( gdfError == GDF_SUCCESS );									\
																				\
		std::vector<TTO> results(colSize);										\
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());	\
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
DEF_CAST_TYPE_TEST(date32, GDF_DATE32, int32_t)
DEF_CAST_TYPE_TEST(date64, GDF_DATE64, int64_t)
DEF_CAST_TYPE_TEST(timestamp, GDF_TIMESTAMP, int64_t)

// Casting from T1 to T2, and then casting from T2 to T1 results in the same value 
#define DEF_CAST_SWAP_TEST(VFROM, VTO, VVFROM, VVTO, TFROM, TTO)				\
	TEST(gdf_cast_swap_TEST, VFROM##_to_##VTO) {								\
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
		thrust::device_vector<TFROM> intputDataDev(inputData);					\
		thrust::device_vector<TTO> outDataDev(colSize);							\
		thrust::device_vector<TFROM> origOutDataDev(colSize);					\
																				\
		inputCol.data = thrust::raw_pointer_cast(intputDataDev.data());			\
		inputCol.valid = nullptr;												\
		outputCol.data = thrust::raw_pointer_cast(outDataDev.data());			\
		outputCol.valid = nullptr;												\
		originalOutputCol.data = thrust::raw_pointer_cast(origOutDataDev.data());\
		originalOutputCol.valid = nullptr;										\
																				\
		gdf_error gdfError = gdf_cast_##VFROM##_to_##VTO(&inputCol, &outputCol);\
		EXPECT_TRUE( gdfError == GDF_SUCCESS );									\
		gdfError = gdf_cast_##VTO##_to_##VFROM(&outputCol, &originalOutputCol);\
		EXPECT_TRUE( gdfError == GDF_SUCCESS );									\
																				\
		std::vector<TFROM> results(colSize);									\
		thrust::copy(origOutDataDev.begin(), origOutDataDev.end(), results.begin());\
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
DEF_CAST_SWAP_TEST(i8, timestamp, GDF_INT8, GDF_TIMESTAMP,  int8_t, int64_t)
DEF_CAST_SWAP_TEST(i32, i64, GDF_INT32, GDF_INT64,  int32_t, int64_t)
DEF_CAST_SWAP_TEST(i32, f64, GDF_INT32, GDF_FLOAT64,  int32_t, double)
DEF_CAST_SWAP_TEST(i32, f32, GDF_INT32, GDF_FLOAT32,  int32_t, float)
DEF_CAST_SWAP_TEST(f32, f64, GDF_FLOAT32, GDF_FLOAT64,  float, double)
DEF_CAST_SWAP_TEST(date32, date64, GDF_DATE32, GDF_DATE64,  int32_t, int64_t)
DEF_CAST_SWAP_TEST(date32, timestamp, GDF_DATE32, GDF_TIMESTAMP,  int32_t, int64_t)
DEF_CAST_SWAP_TEST(date64, timestamp, GDF_DATE64, GDF_TIMESTAMP,  int64_t, int64_t)
