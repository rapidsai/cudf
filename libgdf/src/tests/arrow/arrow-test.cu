#include "gtest/gtest.h"
#include "gdf/cffi/types.h"
#include "gdf/arrow.hpp"
#include <arrow/api.h>
#include <thrust/device_vector.h>
#include <arrow/gpu/cuda_api.h>

using namespace arrow;
using namespace arrow::gpu;

TEST(ArrowTests, ArrowToGDF){
  gdf_column inputCol, outputCol, outputCol2;
  std::vector<int32_t> inputData = {
    17696,
    17697,
  };
  inputCol.dtype = GDF_INT32;
  inputCol.size = 2;
  inputCol.valid = NULL;
  inputCol.null_count = 0;
  thrust::device_vector<int32_t> inputDataDev(inputData);
  inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
  auto arrow_dtype = int32();
  CudaDeviceManager* manager_;
  CudaDeviceManager::GetInstance(&manager_);
  std::shared_ptr<arrow::gpu::CudaContext> context_;
  manager_->GetContext(0, &context_);
  uint8_t * casted_data = (uint8_t *) inputCol.data;
  auto buffer = std::make_shared<CudaBuffer>(casted_data,
                                             4 * inputCol.size,
                                             context_,
                                             false,
                                             false);
  auto array = std::make_shared<PrimitiveArray>(arrow_dtype, int64_t(inputCol.size), buffer);
  
  arrow_to_gdf(array.get(), &outputCol);
  
  EXPECT_TRUE(outputCol.size == inputCol.size);
  EXPECT_TRUE(inputCol.data == outputCol.data);
  EXPECT_TRUE(inputCol.valid == outputCol.valid);
  EXPECT_TRUE(inputCol.dtype == outputCol.dtype);  

  thrust::device_vector<gdf_valid_type> inputValidDev(1,255);
  inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());
  auto null_bitmap = std::make_shared<CudaBuffer>((uint8_t *)inputCol.valid,
                                                  inputCol.size,
                                                  context_,
                                                  false,
                                                  false);
  auto array2 = std::make_shared<PrimitiveArray>(arrow_dtype,
                                                 int64_t(inputCol.size),
                                                 buffer,
                                                 null_bitmap);
  arrow_to_gdf(array2.get(), &outputCol2);
  EXPECT_TRUE(outputCol2.size == inputCol.size);
  EXPECT_TRUE(inputCol.data == outputCol2.data);
  EXPECT_TRUE(inputCol.valid == outputCol2.valid);
  EXPECT_TRUE(inputCol.dtype == outputCol2.dtype);  
  
}

TEST(ArrowTests, GdfToArrow){
  gdf_column inputCol, outputCol, outputCol2;
  std::vector<int32_t> inputData = {
    17696,
    17697,
  };
  inputCol.dtype = GDF_INT32;
  inputCol.size = 2;
  inputCol.valid = NULL;
  inputCol.null_count = 0;
  thrust::device_vector<int32_t> inputDataDev(inputData);
  inputCol.data = thrust::raw_pointer_cast(inputDataDev.data());
  std::shared_ptr<arrow::PrimitiveArray> array = gdf_to_arrow(&inputCol);
  arrow_to_gdf(array.get(), &outputCol);
  
  EXPECT_TRUE(outputCol.size == inputCol.size);
  EXPECT_TRUE(inputCol.data == outputCol.data);
  EXPECT_TRUE(inputCol.valid == outputCol.valid);
  EXPECT_TRUE(inputCol.dtype == outputCol.dtype);  

  thrust::device_vector<gdf_valid_type> inputValidDev(1,255);
  inputCol.valid = thrust::raw_pointer_cast(inputValidDev.data());

  std::shared_ptr<arrow::PrimitiveArray> array2 = gdf_to_arrow(&inputCol);
  arrow_to_gdf(array2.get(), &outputCol2);
  EXPECT_TRUE(outputCol2.size == inputCol.size);
  EXPECT_TRUE(inputCol.data == outputCol2.data);
  EXPECT_TRUE(inputCol.valid == outputCol2.valid);
  EXPECT_TRUE(inputCol.dtype == outputCol2.dtype);  
}

