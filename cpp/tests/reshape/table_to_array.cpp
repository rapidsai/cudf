/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/io/types.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/reshape.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

struct TableToDeviceArrayTest : public cudf::test::BaseFixture {};

TEST(TableToDeviceArrayTest, Int32Columns)
{
  auto stream = cudf::get_default_stream();

  std::vector<int32_t> col0{1, 2, 3};
  std::vector<int32_t> col1{4, 5, 6};
  std::vector<int32_t> col2{7, 8, 9};
  std::vector<int32_t> col3{10, 11, 12};

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, 3));
  columns.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, 3));
  columns.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, 3));
  columns.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, 3));

  cudaMemcpy(columns[0]->mutable_view().data<int32_t>(),
             col0.data(),
             3 * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(columns[1]->mutable_view().data<int32_t>(),
             col1.data(),
             3 * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(columns[2]->mutable_view().data<int32_t>(),
             col2.data(),
             3 * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(columns[3]->mutable_view().data<int32_t>(),
             col3.data(),
             3 * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  cudf::table_view input_table(
    {columns[0]->view(), columns[1]->view(), columns[2]->view(), columns[3]->view()});

  size_t num_elements = 3 * 4;
  rmm::device_buffer output(num_elements * sizeof(int32_t), stream);

  cudf::table_to_device_array(input_table,
                              output.data(),
                              cudf::data_type{cudf::type_id::INT32},
                              stream,
                              rmm::mr::get_current_device_resource());

  std::vector<int32_t> host_result(num_elements);
  cudaMemcpy(
    host_result.data(), output.data(), num_elements * sizeof(int32_t), cudaMemcpyDeviceToHost);

  std::vector<int32_t> expected{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  EXPECT_EQ(host_result, expected);
}
