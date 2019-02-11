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

#include <algorithm>
#include <thrust/sequence.h>
#include <thrust/equal.h>

#include "cudf.h"
#include "dlpack/dlpack.h"

#include "rmm/thrust_rmm_allocator.h"
#include "tests/utilities/cudf_test_fixtures.h"

template <class T>
struct DLPackTest : public GdfTest
{

};

using Types = testing::Types<int32_t,
                             float>;

TYPED_TEST_CASE(DLPackTest, Types);

void deleter(DLManagedTensor * arg) {
  //RMM_FREE(arg->dl_tensor.data, 0);
  delete [] arg->dl_tensor.shape;
  delete arg;
  arg->dl_tensor.data = nullptr;
}

TYPED_TEST(DLPackTest, FromDLPack)
{
  constexpr int64_t length = 100;

  DLManagedTensor *mng_tensor = new DLManagedTensor;
  DLTensor &tensor = mng_tensor->dl_tensor;
  tensor.data = 0;
  tensor.ndim = 1;
  tensor.dtype.code = kDLInt;
  tensor.dtype.bits = 32;
  tensor.dtype.lanes = 1;
  tensor.shape = new int64_t[tensor.ndim];
  tensor.shape[0] = length;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;

  rmm::device_vector<int> data(length);
  thrust::sequence(rmm::exec_policy(0)->on(0), data.begin(), data.end());

  tensor.data = thrust::raw_pointer_cast(data.data());
  tensor.ctx.device_type = kDLGPU;
  ASSERT_EQ(cudaGetDevice(&tensor.ctx.device_id), cudaSuccess);

  mng_tensor->manager_ctx = nullptr;
  mng_tensor->deleter = deleter;

  gdf_column *columns = nullptr;
  int num_columns = 0;
  ASSERT_EQ(gdf_from_dlpack(&columns, &num_columns, mng_tensor), GDF_SUCCESS);
  ASSERT_EQ(mng_tensor->dl_tensor.data, nullptr); // since deleter is called
  ASSERT_NE(columns, nullptr);

  // We currently only support 1D Tensors
  ASSERT_EQ(num_columns, 1);
  ASSERT_EQ(columns[0].size, length);

  ASSERT_TRUE(thrust::equal(rmm::exec_policy(0)->on(0), 
                            data.begin(), data.end(), 
                            reinterpret_cast<int*>(columns[0].data)));
  
  gdf_column_free(&columns[0]);
  delete [] columns;
}

TYPED_TEST(DLPackTest, ToDLPack)
{
  ASSERT_EQ(gdf_to_dlpack(nullptr, nullptr, 1), GDF_SUCCESS);
}