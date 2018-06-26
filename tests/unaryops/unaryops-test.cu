#include <cstdlib>
#include <iostream>
#include <vector>
#include <numeric>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>

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

#define DEF_CAST_IMPL_TEST(VFROM, VTO, VVFROM, VVTO, TFROM, TTO)				\
	TEST(gdf_cast_CPU_VS_GPU_TEST, VFROM##_to_##VTO) {							\
	{																			\
		int colSize = 128;														\
		gdf_column inputCol;													\
		gdf_column outputCol;													\
																				\
		inputCol.dtype = VVFROM;												\
		inputCol.size = colSize;												\
		outputCol.dtype = VVTO;													\
		outputCol.size = colSize;												\
																				\
		std::vector<TFROM> inputData(colSize);									\
		std::iota(inputData.begin(), inputData.end(), 0);						\
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
		std::vector<int64_t> results(colSize);									\
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());	\
																				\
		EXPECT_TRUE( gdfError == GDF_SUCCESS );									\
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
																				\
		for (int i = 0; i < colSize; i++){										\
			EXPECT_TRUE( results[i] == outputData[i]);							\
		}																		\
																				\
		EXPECT_TRUE( gdfError == GDF_SUCCESS );									\
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
