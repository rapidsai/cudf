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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

template <typename T>
class RoundRobinTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(RoundRobinTest, cudf::test::FixedWidthTypes);

TYPED_TEST(RoundRobinTest, RoundRobinPartitions13_3)
{
  strings_column_wrapper rrColWrap1(
    {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::test::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      bool ret = (row % 2 == 0);
      return static_cast<TypeParam>(ret);
    } else
      return static_cast<TypeParam>(row);
  });

  fixed_width_column_wrapper<TypeParam> rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions = 3;

  cudf::size_type start_partition = 0;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"a", "d", "g", "j", "m", "b", "e", "h", "k", "c", "f", "i", "l"},
      {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2(
        {1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2(
        {0, 3, 6, 9, 12, 1, 4, 7, 10, 2, 5, 8, 11});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 5, 9};
    EXPECT_EQ(num_partitions, expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  start_partition = 1;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"c", "f", "i", "l", "a", "d", "g", "j", "m", "b", "e", "h", "k"},
      {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2(
        {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2(
        {2, 5, 8, 11, 0, 3, 6, 9, 12, 1, 4, 7, 10});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 4, 9};
    EXPECT_EQ(num_partitions, expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  start_partition = 2;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"b", "e", "h", "k", "c", "f", "i", "l", "a", "d", "g", "j", "m"},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2(
        {0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2(
        {1, 4, 7, 10, 2, 5, 8, 11, 0, 3, 6, 9, 12});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 4, 8};
    EXPECT_EQ(num_partitions, expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }
}

TYPED_TEST(RoundRobinTest, RoundRobinPartitions11_3)
{
  strings_column_wrapper rrColWrap1({"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
                                    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::test::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      bool ret = (row % 2 == 0);
      return static_cast<TypeParam>(ret);
    } else
      return static_cast<TypeParam>(row);
  });

  fixed_width_column_wrapper<TypeParam> rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions = 3;

  cudf::size_type start_partition = 0;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"a", "d", "g", "j", "b", "e", "h", "k", "c", "f", "i"}, {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 4, 8};
    EXPECT_EQ(num_partitions, expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  start_partition = 1;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"c", "f", "i", "a", "d", "g", "j", "b", "e", "h", "k"}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({2, 5, 8, 0, 3, 6, 9, 1, 4, 7, 10});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 3, 7};
    EXPECT_EQ(num_partitions, expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  start_partition = 2;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"b", "e", "h", "k", "c", "f", "i", "a", "d", "g", "j"}, {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({1, 4, 7, 10, 2, 5, 8, 0, 3, 6, 9});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 4, 7};
    EXPECT_EQ(num_partitions, expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }
}

TYPED_TEST(RoundRobinTest, RoundRobinDegeneratePartitions11_15)
{
  strings_column_wrapper rrColWrap1({"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
                                    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::test::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      bool ret = (row % 2 == 0);
      return static_cast<TypeParam>(ret);
    } else
      return static_cast<TypeParam>(row);
  });

  fixed_width_column_wrapper<TypeParam> rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions = 15;

  cudf::size_type start_partition = 2;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{
      0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11};
    EXPECT_EQ(num_partitions, expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  start_partition = 10;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"f", "g", "h", "i", "j", "k", "a", "b", "c", "d", "e"}, {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{
      0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 10};
    EXPECT_EQ(num_partitions, expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  start_partition = 14;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "a"}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10};
    EXPECT_EQ(num_partitions, expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }
}

TYPED_TEST(RoundRobinTest, RoundRobinDegeneratePartitions11_11)
{
  strings_column_wrapper rrColWrap1({"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
                                    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::test::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      bool ret = (row % 2 == 0);
      return static_cast<TypeParam>(ret);
    } else
      return static_cast<TypeParam>(row);
  });

  fixed_width_column_wrapper<TypeParam> rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions = 11;

  cudf::size_type start_partition = 2;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"j", "k", "a", "b", "c", "d", "e", "f", "g", "h", "i"}, {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_EQ(num_partitions, expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }
}

TYPED_TEST(RoundRobinTest, RoundRobinNPartitionsDivideNRows)
{
  // test the case when nrows `mod` npartitions = 0
  //
  // input:
  // strings: ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u"],
  // bools:   [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
  // nulls:   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0])
  //
  strings_column_wrapper rrColWrap1(
    {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
     "l", "m", "n", "o", "p", "q", "r", "s", "t", "u"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::test::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      bool ret = (row % 2 == 0);
      return static_cast<TypeParam>(ret);
    } else
      return static_cast<TypeParam>(row);
  });

  fixed_width_column_wrapper<TypeParam> rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions = 3;

  // expected:
  // indxs:   [0,3,6,9,12,15,18,1,4,7,10,13,16,19,2,5,8,11,14,17,20],
  // strings: ["a","d","g","j","m","p","s","b","e","h","k","n","q","t","c","f","i","l","o","r","u"],
  // bools:   [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
  // nulls:   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
  // offsets: [0,7,14]
  //
  cudf::size_type start_partition = 0;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"a", "d", "g", "j", "m", "p", "s", "b", "e", "h", "k",
       "n", "q", "t", "c", "f", "i", "l", "o", "r", "u"},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2(
        {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2(
        {0, 3, 6, 9, 12, 15, 18, 1, 4, 7, 10, 13, 16, 19, 2, 5, 8, 11, 14, 17, 20});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 7, 14};
    EXPECT_EQ(num_partitions, expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  // expected:
  // indxs:   [2,5,8,11,14,17,20,0,3,6,9,12,15,18,1,4,7,10,13,16,19],
  // strings: ["c","f","i","l","o","r","u","a","d","g","j","m","p","s","b","e","h","k","n","q","t"],
  // bools:   [1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
  // nulls:   [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  // offsets: [0,7,14]
  //
  start_partition = 1;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"c", "f", "i", "l", "o", "r", "u", "a", "d", "g", "j",
       "m", "p", "s", "b", "e", "h", "k", "n", "q", "t"},
      {1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2(
        {1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2(
        {2, 5, 8, 11, 14, 17, 20, 0, 3, 6, 9, 12, 15, 18, 1, 4, 7, 10, 13, 16, 19});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 7, 14};
    EXPECT_EQ(num_partitions, expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }
}

TYPED_TEST(RoundRobinTest, RoundRobinSinglePartition)
{
  strings_column_wrapper rrColWrap1({"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
                                    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::test::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      bool ret = (row % 2 == 0);
      return static_cast<TypeParam>(ret);
    } else
      return static_cast<TypeParam>(row);
  });

  fixed_width_column_wrapper<TypeParam> rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions  = 1;
  cudf::size_type start_partition = 0;
  std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
  EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

  auto p_outputTable = std::move(result.first);

  auto output_column_view1{p_outputTable->view().column(0)};
  auto output_column_view2{p_outputTable->view().column(1)};

  strings_column_wrapper expectedDataWrap1({"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
                                           {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

  if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
    fixed_width_column_wrapper<TypeParam> expectedDataWrap2({1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
    auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

    EXPECT_EQ(inputRows, expected_column_view2.size());
    EXPECT_EQ(inputRows, output_column_view2.size());

    cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
  } else {
    fixed_width_column_wrapper<TypeParam> expectedDataWrap2({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
    cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
  }

  std::vector<cudf::size_type> expected_partition_offsets{0};
  EXPECT_EQ(num_partitions, expected_partition_offsets.size());
  EXPECT_EQ(expected_partition_offsets, result.second);
}

TYPED_TEST(RoundRobinTest, RoundRobinIncorrectNumPartitions)
{
  strings_column_wrapper rrColWrap1({"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
                                    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::test::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      bool ret = (row % 2 == 0);
      return static_cast<TypeParam>(ret);
    } else
      return static_cast<TypeParam>(row);
  });

  fixed_width_column_wrapper<TypeParam> rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions  = 0;
  cudf::size_type start_partition = 0;

  EXPECT_THROW(cudf::round_robin_partition(rr_view, num_partitions, start_partition),
               cudf::logic_error);
}

TYPED_TEST(RoundRobinTest, RoundRobinIncorrectStartPartition)
{
  strings_column_wrapper rrColWrap1({"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
                                    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::test::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      bool ret = (row % 2 == 0);
      return static_cast<TypeParam>(ret);
    } else
      return static_cast<TypeParam>(row);
  });

  fixed_width_column_wrapper<TypeParam> rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions  = 4;
  cudf::size_type start_partition = 5;

  EXPECT_THROW(cudf::round_robin_partition(rr_view, num_partitions, start_partition),
               cudf::logic_error);
}
