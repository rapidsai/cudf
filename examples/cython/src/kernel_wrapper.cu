/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <kernel.cu>
#include <kernel_wrapper.hh>
#include <assert.h>
#include <iostream>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>


CudfWrapper::CudfWrapper(cudf::mutable_table_view table_view) {
  mtv = table_view;
}

void CudfWrapper::tenth_mm_to_inches(int column_index) {

  printf("kernel_wrapper.cu # of columns: %lu\n", mtv.num_columns());
  printf("kernel_wrapper.cu # of rows: %lu\n", mtv.num_rows());

  for( cudf::size_type c=0; c < mtv.num_columns(); c++ ) {
    printf("%d type=%d, ptr=%p\n", c, (int)mtv.column(c).type().id(), mtv.column(c).data<char>() );
  }

  std::unique_ptr<cudf::mutable_column_device_view, std::function<void(cudf::mutable_column_device_view*)>> 
  mutable_device_column = cudf::mutable_column_device_view::create(mtv.column(4));

  printf("kernel_wrapper.cu column.data<int64_t>(): %p\n", mutable_device_column->data<int64_t>());
  kernel_tenth_mm_to_inches<<<1, 1>>>(*mutable_device_column, mtv.num_rows());
  cudaDeviceSynchronize();
}

void CudfWrapper::mm_to_inches(int column_index) {
  printf("mm_to_inches");
}

CudfWrapper::~CudfWrapper() {
  // It is important to note that CudfWrapper does not own the underlying Dataframe 
  // object and that will be freed by the Python/Cython layer later.
}