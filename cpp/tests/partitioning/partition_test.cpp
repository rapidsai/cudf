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
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

class PartitionTest : public cudf::test::BaseFixture {};

using cudf::test::fixed_width_column_wrapper;

TEST_F(PartitionTest, First) {
  fixed_width_column_wrapper<int32_t> ints{0, 1, 2, 3, 4, 5};
  fixed_width_column_wrapper<int32_t> map{0, 1, 2, 3, 4, 5};

  auto result = cudf::experimental::partition(cudf::table_view{{ints}}, map, 6);

  auto offsets = cudf::test::to_host<int32_t>(map);
  std::vector<int32_t> expected_offsets{0, 1, 2, 3, 4, 5};

  EXPECT_TRUE(std::equal(expected_offsets.begin(), expected_offsets.end(),
                         offsets.first.begin()));
  cudf::test::expect_columns_equal(ints, result.first->get_column(0));
}

TEST_F(PartitionTest, Second) {
  fixed_width_column_wrapper<int32_t> ints{13, 5, 7, 3, 1, 0};
  fixed_width_column_wrapper<int32_t> map{5, 4, 3, 2, 1, 0};

  auto result = cudf::experimental::partition(cudf::table_view{{ints}}, map, 6);

  auto offsets = cudf::test::to_host<int32_t>(map);
  std::vector<int32_t> expected_offsets{0, 1, 2, 3, 4, 5, 6};
  fixed_width_column_wrapper<int32_t> expected_ints{0, 1, 3, 7, 5, 13};

  auto result_offsets = result.second;

  EXPECT_TRUE(std::equal(expected_offsets.begin(), expected_offsets.end(),
                         result_offsets.begin()));

  cudf::test::expect_columns_equal(expected_ints, result.first->get_column(0));
}

TEST_F(PartitionTest, EmptyInputs) {
  fixed_width_column_wrapper<int32_t> empty_column{};
  fixed_width_column_wrapper<int32_t> empty_map{};

  auto result = cudf::experimental::partition(cudf::table_view{{empty_column}},
                                              empty_map, 10);

  auto result_offsets = result.second;

  EXPECT_TRUE(result_offsets.empty());

  std::copy(result_offsets.begin(), result_offsets.end(),
            std::ostream_iterator<int32_t>(std::cout, ","));

  cudf::test::expect_columns_equal(empty_column, result.first->get_column(0));
}

TEST_F(PartitionTest, MapInputSizeMismatch) {
  fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  fixed_width_column_wrapper<int32_t> map{1, 2};

  EXPECT_THROW(cudf::experimental::partition(cudf::table_view{{input}}, map, 3),
               cudf::logic_error);
}