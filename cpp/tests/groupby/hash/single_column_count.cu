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

#include <tests/utilities/cudf_test_fixtures.h>
#include <groupby.hpp>
#include <table.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/compare_column_wrappers.cuh>
#include <utilities/type_dispatcher.hpp>
#include "single_column_groupby_test.cuh"
#include "type_info.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

static constexpr cudf::groupby::hash::operators op{
    cudf::groupby::hash::operators::COUNT};

template <typename T>
struct SingleColumnCount : public GdfTest {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution{1000, 10000};
  int random_size() { return distribution(generator); }
};

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

// TODO: tests for cudf::bool8
using TestingTypes =
    ::testing::Types<int32_t, int8_t, int16_t, int32_t, int64_t, float, double,
                     cudf::date32, cudf::date64, cudf::category>;

TYPED_TEST_CASE(SingleColumnCount, TestingTypes);

TYPED_TEST(SingleColumnCount, OneGroupNoNulls) {
  constexpr int size{10};
  using ResultValue = cudf::test::expected_result_t<int, op>;
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<TypeParam>(size, [](auto index) { return TypeParam(42); }),
      column_wrapper<int>(size, [](auto index) { return int(index); }),
      column_wrapper<TypeParam>{TypeParam(42)},
      column_wrapper<ResultValue>{size});
}

TYPED_TEST(SingleColumnCount, OneGroupAllNullKeys) {
  constexpr int size{10};
  using ResultValue = cudf::test::expected_result_t<int, op>;

  // If all keys are null, then there should be no output
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<TypeParam>(size, [](auto index) { return TypeParam(42); },
                                [](auto index) { return false; }),
      column_wrapper<int>(size, [](auto index) { return int(index); }),
      column_wrapper<TypeParam>{}, column_wrapper<ResultValue>{});
}

TYPED_TEST(SingleColumnCount, OneGroupAllNullValues) {
  constexpr int size{10};
  using ResultValue = cudf::test::expected_result_t<int, op>;
  TypeParam key{42};
  // If all values are null, then the output count should be a non-null zero
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<TypeParam>(size, [key](auto index) { return key; }),
      column_wrapper<int>(size, [](auto index) { return int(index); },
                          [](auto index) { return false; }),
      column_wrapper<TypeParam>({key}),
      column_wrapper<ResultValue>({0}, [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnCount, OneGroupOddNullKeys) {
  constexpr int size{10};
  using ResultValue = cudf::test::expected_result_t<int, op>;

  EXPECT_EQ(size % 2, 0) << "Size must be multiple of 2 for this test.";
  TypeParam key{42};
  // Even index keys are null, means COUNT should be size/2
  // Output keys should be nullable
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<TypeParam>(size, [key](auto index) { return key; },
                                [](auto index) { return index % 2; }),
      column_wrapper<int>(size, [](auto index) { return int(index); }),
      column_wrapper<TypeParam>({key}, [](auto index) { return true; }),
      column_wrapper<ResultValue>({size / 2}));
}

TYPED_TEST(SingleColumnCount, OneGroupOddNullValues) {
  constexpr int size{10};
  using ResultValue = cudf::test::expected_result_t<int, op>;

  EXPECT_EQ(size % 2, 0) << "Size must be multiple of 2 for this test.";
  TypeParam key{42};
  // Even index values are null, means COUNT should be size/2
  // Output values should be nullable
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<TypeParam>(size, [key](auto index) { return key; }),
      column_wrapper<int>(size, [](auto index) { return int(index); },
                          [](auto index) { return index % 2; }),
      column_wrapper<TypeParam>({key}),
      column_wrapper<ResultValue>({size / 2}, [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnCount, FourGroupsNoNulls) {
  using ResultValue = cudf::test::expected_result_t<int, op>;
  using T = TypeParam;
  using R = ResultValue;

  cudf::test::single_column_groupby_test<op>(
      column_wrapper<TypeParam>{T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
      column_wrapper<int>(8, [](auto index) { return int(index); }),
      column_wrapper<TypeParam>{T(1), T(2), T(3), T(4)},
      column_wrapper<ResultValue>{R(2), R(2), R(2), R(2)});
}

TYPED_TEST(SingleColumnCount, FourGroupsOddNullKeys) {
  using ResultValue = cudf::test::expected_result_t<int, op>;
  using T = TypeParam;
  using R = ResultValue;

  // Even index keys are null, COUNT should be the count of each key / 2
  // Output keys should be nullable
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<TypeParam>(
          {T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
          [](auto index) { return index % 2; }),
      column_wrapper<int>(8, [](auto index) { return int(index); }),
      column_wrapper<TypeParam>({T(1), T(2), T(3), T(4)},
                                [](auto index) { return true; }),
      column_wrapper<ResultValue>{R(1), R(1), R(1), R(1)});
}

TYPED_TEST(SingleColumnCount, FourGroupsOddNullValues) {
  using ResultValue = cudf::test::expected_result_t<int, op>;
  using T = TypeParam;
  using R = ResultValue;

  // Even index values are null, COUNT should be the count of each key / 2
  // Output values should be nullable
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<TypeParam>{T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
      column_wrapper<int>(8, [](auto index) { return int(index); },
                          [](auto index) { return index % 2; }),
      column_wrapper<TypeParam>({T(1), T(2), T(3), T(4)}),
      column_wrapper<ResultValue>({R(1), R(1), R(1), R(1)},
                                  [](auto index) { return true; }));
}