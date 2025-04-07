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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/reshape.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

template <typename T>
struct TableToDeviceArrayTypedTest : public cudf::test::BaseFixture {};

using SupportedTypes = cudf::test::Types<int8_t,
                                         int16_t,
                                         int32_t,
                                         int64_t,
                                         uint8_t,
                                         uint16_t,
                                         uint32_t,
                                         uint64_t,
                                         float,
                                         double,
                                         cudf::timestamp_D,
                                         cudf::timestamp_s,
                                         cudf::timestamp_ms,
                                         cudf::timestamp_us,
                                         cudf::timestamp_ns,
                                         cudf::duration_D,
                                         cudf::duration_s,
                                         cudf::duration_ms,
                                         cudf::duration_us,
                                         cudf::duration_ns>;

TYPED_TEST_SUITE(TableToDeviceArrayTypedTest, SupportedTypes);

TYPED_TEST(TableToDeviceArrayTypedTest, SupportedTypes)
{
  using T     = TypeParam;
  auto stream = cudf::get_default_stream();

  auto const dtype = cudf::data_type{cudf::type_to_id<T>()};

  auto const col0 = cudf::test::make_type_param_vector<T>({1, 2, 3});
  auto const col1 = cudf::test::make_type_param_vector<T>({4, 5, 6});
  auto const col2 = cudf::test::make_type_param_vector<T>({7, 8, 9});
  auto const col3 = cudf::test::make_type_param_vector<T>({10, 11, 12});

  std::vector<std::unique_ptr<cudf::column>> cols;
  auto make_col = [&](auto const& data) {
    return std::make_unique<cudf::column>(
      cudf::test::fixed_width_column_wrapper<T>(data.begin(), data.end()));
  };

  cols.push_back(make_col(col0));
  cols.push_back(make_col(col1));
  cols.push_back(make_col(col2));
  cols.push_back(make_col(col3));

  cudf::table_view input({cols[0]->view(), cols[1]->view(), cols[2]->view(), cols[3]->view()});
  size_t num_elements = 3 * 4;
  rmm::device_buffer output(num_elements * sizeof(T), stream);

  cudf::table_to_device_array(
    input, output.data(), dtype, stream, rmm::mr::get_current_device_resource());

  std::vector<T> host_result(num_elements);
  CUDF_CUDA_TRY(cudaMemcpy(
    host_result.data(), output.data(), num_elements * sizeof(T), cudaMemcpyDeviceToHost));

  auto const expected_data =
    cudf::test::make_type_param_vector<T>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  std::vector<T> expected(expected_data.begin(), expected_data.end());

  EXPECT_EQ(host_result, expected);
}

template <typename T>
struct FixedPointTableToDeviceArrayTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointTableToDeviceArrayTest, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTableToDeviceArrayTest, SupportedFixedPointTypes)
{
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto stream = cudf::get_default_stream();
  auto scale  = numeric::scale_type{-2};
  auto dtype  = cudf::data_type{cudf::type_to_id<decimalXX>(), scale};

  fp_wrapper col0({123, 456, 789}, scale);
  fp_wrapper col1({321, 654, 987}, scale);

  cudf::table_view input({col0, col1});
  rmm::device_buffer output(2 * 3 * sizeof(RepType), stream);

  cudf::table_to_device_array(
    input, output.data(), dtype, stream, rmm::mr::get_current_device_resource());

  std::vector<RepType> host_result(6);
  CUDF_CUDA_TRY(cudaMemcpy(host_result.data(),
                           output.data(),
                           host_result.size() * sizeof(RepType),
                           cudaMemcpyDeviceToHost));

  std::vector<RepType> expected{123, 456, 789, 321, 654, 987};
  EXPECT_EQ(host_result, expected);
}

struct TableToDeviceArrayTest : public cudf::test::BaseFixture {};

TEST(TableToDeviceArrayTest, UnsupportedStringType)
{
  auto stream = cudf::get_default_stream();
  auto col    = cudf::test::strings_column_wrapper({"a", "b", "c"});
  cudf::table_view input_table({col});
  rmm::device_buffer output(3 * sizeof(int32_t), stream);

  EXPECT_THROW(cudf::table_to_device_array(input_table,
                                           output.data(),
                                           cudf::data_type{cudf::type_id::STRING},
                                           stream,
                                           rmm::mr::get_current_device_resource()),
               cudf::logic_error);
}
