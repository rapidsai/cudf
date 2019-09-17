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

#include "../single_column_groupby_test.cuh"
#include <cudf/groupby.hpp>
#include <cudf/legacy/table.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/compare_column_wrappers.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

static constexpr cudf::groupby::operators op{cudf::groupby::operators::SUM};

template <typename KV> struct SingleColumnSumSql : public GdfTest {
  using KeyType = typename KV::Key;
  using ValueType = typename KV::Value;
};

template <typename T> using column_wrapper = cudf::test::column_wrapper<T>;

template <typename K, typename V> struct KV {
  using Key = K;
  using Value = V;
};

using TestingTypes = ::testing::Types<KV<int8_t, int8_t>>;

// TODO: tests for cudf::bool8

TYPED_TEST_CASE(SingleColumnSumSql, TestingTypes);

TYPED_TEST(SingleColumnSumSql, HalfWithNullKeys) {
  using Key = typename SingleColumnSumSql<TypeParam>::KeyType;
  using Value = typename SingleColumnSumSql<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;
  bool ignore_null_keys = false;

  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>({T(1), T(1), T(0), T(0)},
                          [](auto index) { return index < 2; }),
      column_wrapper<Value>(4, [](auto index) { return Value(1); }),
      column_wrapper<Key>({T(1), T(0)}, [](auto index) { return index == 0; }),
      column_wrapper<ResultValue>({R(2), R(2)}), ignore_null_keys);
}

TYPED_TEST(SingleColumnSumSql, OneGroupNoNulls) {
  constexpr int size{10};
  using Key = typename SingleColumnSumSql<TypeParam>::KeyType;
  using Value = typename SingleColumnSumSql<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  ResultValue sum{((size - 1) * size) / 2};
  bool ignore_null_keys = false;
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>(size, [key](auto index) { return key; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); }),
      column_wrapper<Key>({key}), column_wrapper<ResultValue>({sum}),
      ignore_null_keys);
}

TYPED_TEST(SingleColumnSumSql, OneGroupAllNullKeys) {
  constexpr int size{10};
  using Key = typename SingleColumnSumSql<TypeParam>::KeyType;
  using Value = typename SingleColumnSumSql<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using R = ResultValue;
  Key key{42};
  bool ignore_null_keys = false;
  // If all keys are null, then there should be no output
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>(size, [key](auto index) { return key; },
                          [](auto index) { return false; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); }),
      column_wrapper<Key>({key}, [](auto index) { return false; }),
      column_wrapper<ResultValue>({R(45)}), ignore_null_keys);
}

TYPED_TEST(SingleColumnSumSql, OneGroupAllNullValues) {
  constexpr int size{10};
  using Key = typename SingleColumnSumSql<TypeParam>::KeyType;
  using Value = typename SingleColumnSumSql<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  bool ignore_null_keys = false;
  // If all values are null, then there should be a single NULL output value
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>(size, [key](auto index) { return key; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); },
                            [](auto index) { return false; }),
      column_wrapper<Key>({key}), column_wrapper<ResultValue>(1, true),
      ignore_null_keys);
}

TYPED_TEST(SingleColumnSumSql, OneGroupEvenNullKeys) {
  constexpr int size{10};
  using Key = typename SingleColumnSumSql<TypeParam>::KeyType;
  using Value = typename SingleColumnSumSql<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  bool ignore_null_keys = false;
  // The sum of n odd numbers is n^2
  ResultValue sum = (size / 2) * (size / 2);
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>(size, [key](auto index) { return key; },
                          [](auto index) { return index % 2; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); }),
      column_wrapper<Key>({key, 0}, [](auto index) { return index == 0; }),
      column_wrapper<ResultValue>({sum, 20}), ignore_null_keys);
}

TYPED_TEST(SingleColumnSumSql, OneGroupOddNullKeys) {
  constexpr int size{10};
  using Key = typename SingleColumnSumSql<TypeParam>::KeyType;
  using Value = typename SingleColumnSumSql<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  bool ignore_null_keys = false;
  // The number of even values in the range [0,n) is (n-1)/2
  int num_even_numbers = (size - 1) / 2;
  // The sum of n even numbers is n(n+1)
  ResultValue sum = num_even_numbers * (num_even_numbers + 1);
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>(size, [key](auto index) { return key; },
                          [](auto index) { return not(index % 2); }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); }),
      column_wrapper<Key>({key, 0}, [](auto index) { return index == 0; }),
      column_wrapper<ResultValue>({sum, 25}), ignore_null_keys);
}
