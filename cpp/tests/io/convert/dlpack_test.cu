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
#include <thrust/sequence.h>
#include <thrust/equal.h>

#include "cudf.h"
#include "dlpack/dlpack.h"

#include "rmm/thrust_rmm_allocator.h"
#include "tests/utilities/cudf_test_fixtures.h"

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
  static inline size_t tensor_size(const DLTensor *t)
  {
    size_t size = 1;
    for (int i = 0; i < t->ndim; ++i) size *= t->shape[i];
    size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
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
    size_t bytesize = tensor_size(&(mng_tensor->dl_tensor));

    if (kDLGPU == device_type) {  
      EXPECT_EQ(RMM_ALLOC(&data, bytesize, 0), RMM_SUCCESS);
      
      // For some reason this raises an invalid device pointer exception...
      //thrust::sequence(rmm::exec_policy(0)->on(0), data, data + N);

      T *init = new T[N];
      for (size_t i = 0; i < N; i++) init[i] = i;
      cudaMemcpy(data, init, bytesize, cudaMemcpyDefault);
      delete [] init;
    } else {
      data = static_cast<T*>(malloc(bytesize));
      thrust::sequence(thrust::host, data, data + N);
    }
    
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
  using TensorType = int32_t;

  constexpr int64_t length = 100;

  DLManagedTensor *mng_tensor = 
    create_DLTensor<TensorType>(1, length, kDLCPU);
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
  using TensorType = int32_t;
  constexpr int64_t length = 100;

  int device_id = 0;
  ASSERT_EQ(cudaGetDevice(&device_id), cudaSuccess);

  DLManagedTensor *mng_tensor = 
    create_DLTensor<TensorType>(1, length);

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
  using TensorType = int32_t;
  constexpr int64_t length = 100;

  DLManagedTensor *mng_tensor = 
    create_DLTensor<TensorType>(2, length);

  gdf_column *columns = nullptr;
  int num_columns = 0;
  
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
  using TensorType = uint32_t; // unsigned types not supported yet
  constexpr int64_t length = 100;

  DLManagedTensor *mng_tensor = 
    create_DLTensor<TensorType>(1, length);

  gdf_column *columns = nullptr;
  int num_columns = 0;

  ASSERT_EQ(gdf_from_dlpack(&columns, &num_columns, mng_tensor), 
                            GDF_UNSUPPORTED_DTYPE);

  EXPECT_EQ(nullptr, columns);
  EXPECT_EQ(num_columns, 0);
                          
  deleter(mng_tensor);
}

TYPED_TEST(DLPackTypedTest, FromDLPack)
{
  using TensorType = typename TestFixture::TestParam;
  constexpr int64_t length = 100;

  DLManagedTensor *mng_tensor = create_DLTensor<TensorType>(1, length);
  ASSERT_NE(mng_tensor, nullptr);

  gdf_column *columns = nullptr;
  int num_columns = 0;
  ASSERT_EQ(gdf_from_dlpack(&columns, &num_columns, mng_tensor), GDF_SUCCESS);
  ASSERT_NE(columns, nullptr);

  // We currently only support 1D Tensors
  ASSERT_EQ(num_columns, 1);
  ASSERT_EQ(columns[0].size, length);

  TensorType *output = new TensorType[length];
  cudaMemcpy(output, columns[0].data, length * sizeof(TensorType), cudaMemcpyDefault);
  for (int64_t i = 0; i < length; i++)
    EXPECT_EQ(static_cast<TensorType>(i), output[i]);
  
  delete [] output;

  // This causes an invalid device pointer exception
  /*ASSERT_TRUE(thrust::equal(rmm::exec_policy(0)->on(0), 
                            thrust::make_counting_iterator<TensorType>(0), 
                            thrust::make_counting_iterator<TensorType>(length), 
                            reinterpret_cast<TensorType*>(columns[0].data)));*/
  
  gdf_column_free(&columns[0]);
  delete [] columns;
}

TYPED_TEST(DLPackTypedTest, ToDLPack)
{
  ASSERT_EQ(gdf_to_dlpack(nullptr, nullptr, 1), GDF_SUCCESS);
}