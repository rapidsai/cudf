/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <type_traits>
#include <algorithm>

#include "cudf.h"
#include "dlpack/dlpack.h"

#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/utilities/column_wrapper.cuh"

template <class TestParameters>
struct DLPackTypedTest : public GdfTest
{
  using TestParam = TestParameters;
};

struct DLPackTest : public GdfTest
{
};

using Types = testing::Types<int8_t,
                             int16_t,
                             int32_t,
                             int64_t,
                             float,
                             double>;
TYPED_TEST_CASE(DLPackTypedTest, Types);

namespace{
  static inline size_t tensor_size(const DLTensor& t)
  {
    size_t size = 1;
    for (int i = 0; i < t.ndim; ++i) size *= t.shape[i];
    size *= (t.dtype.bits * t.dtype.lanes + 7) / 8;
    return size;
  }

  template <typename T>
  DLDataType get_DLDataType()
  {
    DLDataType type;
    if (std::is_integral<T>::value) {
      if (std::is_signed<T>::value) type.code = kDLInt;
      else                          type.code = kDLUInt;
    }
    else if (std::is_floating_point<T>::value) type.code = kDLFloat;
    else type.code = 3U; // error!
    
    type.bits = sizeof(T) * 8;
    type.lanes = 1;
    return type;
  }

  void deleter(DLManagedTensor * arg) {
    if (arg->dl_tensor.ctx.device_type == kDLGPU)
      RMM_FREE(arg->dl_tensor.data, 0);
    else if (arg->dl_tensor.ctx.device_type == kDLCPUPinned)
      cudaFree(arg->dl_tensor.data);
    else
      free(arg->dl_tensor.data);
    delete [] arg->dl_tensor.shape;
    delete [] arg->dl_tensor.strides;
    delete arg;
  }  

  template <typename T>
  __global__ void foo(T *in, int size) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < size;
         i += gridDim.x * blockDim.x) 
         in[i] = i;
  }

  template <typename T>
  DLManagedTensor* create_DLTensor(gdf_size_type ncols, 
                                   gdf_size_type nrows,
                                   DLDeviceType device_type = kDLGPU) 
  {
    DLManagedTensor *mng_tensor = new DLManagedTensor;
    DLTensor &tensor = mng_tensor->dl_tensor;
    tensor.data = 0;
    tensor.ndim = (ncols > 1) ? 2 : 1;
    tensor.dtype = get_DLDataType<T>();
    if (tensor.dtype.code > kDLFloat) return nullptr;
    
    tensor.shape = new int64_t[tensor.ndim];
    tensor.shape[0] = nrows;
    if (tensor.ndim > 1) tensor.shape[1] = ncols;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    tensor.ctx.device_id = 0;
    tensor.ctx.device_type = device_type;

    T *data = nullptr;
    const size_t N = nrows * ncols;
    size_t bytesize = tensor_size(mng_tensor->dl_tensor);

    T *init = new T[N];
    for (gdf_size_type c = 0; c < ncols; ++c)
      for (gdf_size_type i = 0; i < nrows; ++i) init[c*nrows + i] = i;

    if (kDLGPU == device_type) {
      EXPECT_EQ(RMM_ALLOC(&data, bytesize, 0), RMM_SUCCESS);
      cudaMemcpy(data, init, bytesize, cudaMemcpyDefault);
    } else {
      data = static_cast<T*>(malloc(bytesize));
      memcpy(data, init, bytesize);
    }
    delete [] init;

    EXPECT_NE(data, nullptr);
    if (data == nullptr) return nullptr;
    tensor.data = data;

    mng_tensor->manager_ctx = nullptr;
    mng_tensor->deleter = deleter;

    return mng_tensor;
  }
}

TEST_F(DLPackTest, InvalidDeviceType)
{
  using T = int32_t;

  constexpr int64_t length = 100;

  DLManagedTensor *mng_tensor = 
    create_DLTensor<T>(1, length, kDLCPU);
  ASSERT_NE(mng_tensor, nullptr);

  gdf_column *columns = nullptr;
  int num_columns = 0;

  // We support kDLGPU, kDLCPU, and kDLCPUPinned
  for (int i = kDLOpenCL; i <= kDLExtDev; i++) {
    mng_tensor->dl_tensor.ctx.device_type = static_cast<DLDeviceType>(i);
    ASSERT_EQ(gdf_from_dlpack(&columns, &num_columns, mng_tensor), 
                              GDF_INVALID_API_CALL);
  }
  EXPECT_EQ(nullptr, columns);
  EXPECT_EQ(num_columns, 0);

  deleter(mng_tensor);
}

TEST_F(DLPackTest, InvalidDevice)
{
  using T = int32_t;
  constexpr int64_t length = 100;

  int device_id = 0;
  ASSERT_EQ(cudaGetDevice(&device_id), cudaSuccess);

  DLManagedTensor *mng_tensor = 
    create_DLTensor<T>(1, length);

  // spoof the wrong device ID
  mng_tensor->dl_tensor.ctx.device_id = device_id + 1;

  gdf_column *columns = nullptr;
  int num_columns = 0;
  
  ASSERT_EQ(gdf_from_dlpack(&columns, &num_columns, mng_tensor), 
                            GDF_INVALID_API_CALL);

  EXPECT_EQ(nullptr, columns);
  EXPECT_EQ(num_columns, 0);
                          
  deleter(mng_tensor);
}

TEST_F(DLPackTest, UnsupportedDimensions) {
  using T = int32_t;
  constexpr int64_t length = 100;

  DLManagedTensor *mng_tensor = 
    create_DLTensor<T>(2, length);

  gdf_column *columns = nullptr;
  int num_columns = 0;
  
  // too many dimensions
  mng_tensor->dl_tensor.ndim = 3;
  ASSERT_EQ(gdf_from_dlpack(&columns, &num_columns, mng_tensor), 
                            GDF_NOTIMPLEMENTED_ERROR);

  mng_tensor->dl_tensor.ndim = 0;
  ASSERT_EQ(gdf_from_dlpack(&columns, &num_columns, mng_tensor), 
                            GDF_DATASET_EMPTY);

  mng_tensor->dl_tensor.ndim = 1;  
  mng_tensor->dl_tensor.shape[0] = 0;
  ASSERT_EQ(gdf_from_dlpack(&columns, &num_columns, mng_tensor), 
                            GDF_DATASET_EMPTY);

  mng_tensor->dl_tensor.ndim = 1;  
  mng_tensor->dl_tensor.shape[0] = std::numeric_limits<gdf_size_type>::max();
  ASSERT_EQ(gdf_from_dlpack(&columns, &num_columns, mng_tensor), 
                            GDF_COLUMN_SIZE_TOO_BIG);

  EXPECT_EQ(nullptr, columns);
  EXPECT_EQ(num_columns, 0);
                          
  deleter(mng_tensor);
}

TEST_F(DLPackTest, UnsupportedDataType)
{
  using T = uint32_t; // unsigned types not supported yet
  constexpr int64_t length = 100;

  DLManagedTensor *mng_tensor = 
    create_DLTensor<T>(1, length);

  gdf_column *columns = nullptr;
  int num_columns = 0;

  ASSERT_EQ(gdf_from_dlpack(&columns, &num_columns, mng_tensor), 
                            GDF_UNSUPPORTED_DTYPE);

  EXPECT_EQ(nullptr, columns);
  EXPECT_EQ(num_columns, 0);
                          
  deleter(mng_tensor);
}

TEST_F(DLPackTest, ToDLPack_EmptyDataset)
{
  ASSERT_EQ(gdf_to_dlpack(nullptr, nullptr, 1), GDF_DATASET_EMPTY);
  DLManagedTensor *tensor = new DLManagedTensor;
  ASSERT_EQ(gdf_to_dlpack(tensor, nullptr, 1), GDF_DATASET_EMPTY);

  gdf_column **columns = new gdf_column*[2];
  ASSERT_EQ(gdf_to_dlpack(tensor, columns, 0), GDF_DATASET_EMPTY);

  columns[0] = new gdf_column;
  columns[0]->dtype = GDF_FLOAT32;
  columns[0]->size = 0;

  ASSERT_EQ(gdf_to_dlpack(tensor, columns, 1), GDF_DATASET_EMPTY);

  delete tensor;
  delete columns[0];
  delete [] columns;
}

TEST_F(DLPackTest, ToDLPack_ColumnMismatch)
{
  gdf_column **columns = new gdf_column*[2];
  columns[0] = new gdf_column;
  columns[1] = new gdf_column;
  columns[0]->size = columns[1]->size = 1;
  columns[0]->dtype = GDF_INT32;
  columns[1]->dtype = GDF_FLOAT32;

  DLManagedTensor *tensor = new DLManagedTensor;
  ASSERT_EQ(gdf_to_dlpack(tensor, columns, 2), GDF_DTYPE_MISMATCH);
  columns[1]->dtype = GDF_INT32;
  columns[1]->size = 2;
  ASSERT_EQ(gdf_to_dlpack(tensor, columns, 2), GDF_COLUMN_SIZE_MISMATCH);

  delete tensor;
  delete columns[0];
  delete columns[1];
  delete [] columns;
}

TEST_F(DLPackTest, ToDLPack_NonNumerical)
{
  gdf_column **columns = new gdf_column*[1];
  columns[0] = new gdf_column;
  columns[0]->size = 1;

  DLManagedTensor *tensor = new DLManagedTensor;

  // all non-numeric gdf_dtype enums results in GDF_UNSUPPORTED_TYPE
  columns[0]->dtype = GDF_invalid;
  ASSERT_EQ(gdf_to_dlpack(tensor, columns, 1), GDF_UNSUPPORTED_DTYPE);

  columns[0]->dtype = GDF_DATE32;
  ASSERT_EQ(gdf_to_dlpack(tensor, columns, 1), GDF_UNSUPPORTED_DTYPE);

  columns[0]->dtype = GDF_DATE64;
  ASSERT_EQ(gdf_to_dlpack(tensor, columns, 1), GDF_UNSUPPORTED_DTYPE);

  columns[0]->dtype = GDF_TIMESTAMP;
  ASSERT_EQ(gdf_to_dlpack(tensor, columns, 1), GDF_UNSUPPORTED_DTYPE);

  columns[0]->dtype = GDF_CATEGORY;
  ASSERT_EQ(gdf_to_dlpack(tensor, columns, 1), GDF_UNSUPPORTED_DTYPE);

  delete tensor;
  delete columns[0];
  delete [] columns;
}

TYPED_TEST(DLPackTypedTest, FromDLPack_SingleColumn)
{
  using T = typename TestFixture::TestParam;
  constexpr int64_t length = 100;

  DLManagedTensor *mng_tensor = create_DLTensor<T>(1, length);
  ASSERT_NE(mng_tensor, nullptr);

  gdf_column *columns = nullptr;
  int num_columns = 0;
  ASSERT_EQ(gdf_from_dlpack(&columns, &num_columns, mng_tensor), GDF_SUCCESS);
  ASSERT_NE(columns, nullptr);

  // We currently only support 1D Tensors
  ASSERT_EQ(num_columns, 1);
  ASSERT_EQ(columns[0].size, length);

  T *output = new T[length];
  cudaMemcpy(output, columns[0].data, length * sizeof(T), cudaMemcpyDefault);
  for (int64_t i = 0; i < length; i++)
    EXPECT_EQ(static_cast<T>(i), output[i]);
  
  delete [] output;

  gdf_column_free(&columns[0]);
  delete [] columns;
}

TYPED_TEST(DLPackTypedTest, FromDLPack_MultiColumn)
{
  using T = typename TestFixture::TestParam;
  constexpr int64_t length = 100;
  constexpr int64_t width = 3;

  DLManagedTensor *mng_tensor = create_DLTensor<T>(width, length);
  ASSERT_NE(mng_tensor, nullptr);

  gdf_column *columns = nullptr;
  int num_columns = 0;
  ASSERT_EQ(gdf_from_dlpack(&columns, &num_columns, mng_tensor), GDF_SUCCESS);
  ASSERT_NE(columns, nullptr);

  ASSERT_EQ(num_columns, width);

  for (int64_t c = 0; c < num_columns; ++c)
  {
    ASSERT_EQ(columns[c].size, length);

    T *output = new T[length];
    cudaMemcpy(output, columns[c].data, length * sizeof(T), cudaMemcpyDefault);
    for (int64_t i = 0; i < length; i++)
      EXPECT_EQ(static_cast<T>(i), output[i]);

    delete [] output;
    gdf_column_free(&columns[c]);
  }

  delete [] columns;
}

TYPED_TEST(DLPackTypedTest, ToDLPack_SingleColumn)
{
  using T = typename TestFixture::TestParam;

  constexpr int64_t length = 100;
  cudf::test::column_wrapper<T> col0(length,
                                     [](gdf_index_type i) { return i; },
                                     [](gdf_index_type i) { return true; });

  gdf_column **columns = new gdf_column*[1];
  columns[0] = col0.get();

  DLManagedTensor *tensor = new DLManagedTensor;

  ASSERT_EQ(gdf_to_dlpack(tensor, columns, 1), GDF_SUCCESS);

  ASSERT_EQ(tensor->dl_tensor.ndim, 1);
  ASSERT_EQ(tensor->dl_tensor.shape[0], length);

  T *output = new T[length];
  cudaMemcpy(output, tensor->dl_tensor.data, length * sizeof(T), cudaMemcpyDefault);
  for (int64_t i = 0; i < length; i++)
    EXPECT_EQ(static_cast<T>(i), output[i]);
  delete [] output;

  tensor->deleter(tensor);
  delete [] columns;
}

TYPED_TEST(DLPackTypedTest, ToDLPack_MultiColumn)
{
  using T = typename TestFixture::TestParam;

  constexpr int64_t length = 100;
  constexpr int64_t width = 3;
  cudf::test::column_wrapper<T>* cols[width];
  gdf_column *columns[width];

  for (int64_t c = 0; c < width; c++) {
    cols[c] = new cudf::test::column_wrapper<T>(length,
                                                [c](gdf_index_type i) { return i*(c+1); },
                                                [](gdf_index_type i) { return true; });
    columns[c] = cols[c]->get();
  }

  DLManagedTensor *tensor = new DLManagedTensor;

  ASSERT_EQ(gdf_to_dlpack(tensor, columns, width), GDF_SUCCESS);

  ASSERT_EQ(tensor->dl_tensor.ndim, 2);
  ASSERT_EQ(tensor->dl_tensor.shape[0], length);
  ASSERT_EQ(tensor->dl_tensor.shape[1], width);

  T *output = new T[tensor_size(tensor->dl_tensor)/sizeof(T)];
  cudaMemcpy(output, tensor->dl_tensor.data, width * length * sizeof(T), cudaMemcpyDefault);
  for (int64_t c = 0; c < width; c++) {
    T *o = &output[c * length];
    for (int64_t i = 0; i < length; i++)
      EXPECT_EQ(static_cast<T>(i*(c+1)), o[i]);
  }
  delete [] output;
  
  tensor->deleter(tensor);
  for (int64_t c = 0; c < width; c++) delete cols[c];

}
