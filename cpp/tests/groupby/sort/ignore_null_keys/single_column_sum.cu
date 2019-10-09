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
#include "../../common/type_info.hpp"
#include <cudf/groupby.hpp>
#include <cudf/legacy/table.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/compare_column_wrappers.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

static constexpr cudf::groupby::operators op{cudf::groupby::operators::SUM};

template <typename KV> struct SingleColumnSum : public GdfTest {
  using KeyType = typename KV::Key;
  using ValueType = typename KV::Value;
};

template <typename T> using column_wrapper = cudf::test::column_wrapper<T>;

template <typename K, typename V> struct KV {
  using Key = K;
  using Value = V;
};

using TestingTypes =
    ::testing::Types<KV<int8_t, int8_t>, KV<int32_t, int32_t>,
                     KV<int64_t, int64_t>, KV<int32_t, float>,
                     KV<int32_t, double>, KV<cudf::category, int32_t>,
                     KV<cudf::date32, int8_t>, KV<cudf::date64, double>>;

// TODO: tests for cudf::bool8

TYPED_TEST_CASE(SingleColumnSum, TestingTypes);

TYPED_TEST(SingleColumnSum, OneGroupNoNulls) {
  constexpr int size{10};
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  ResultValue sum{((size - 1) * size) / 2};
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>(size, [key](auto index) { return key; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); }),
      column_wrapper<Key>({key}), column_wrapper<ResultValue>({sum}));
}

TYPED_TEST(SingleColumnSum, OneGroupAllNullKeys) {
  constexpr int size{10};
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};

  // If all keys are null, then there should be no output
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>(size, [key](auto index) { return key; },
                          [](auto index) { return false; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); }),
      column_wrapper<Key>{}, column_wrapper<ResultValue>{});
}

TYPED_TEST(SingleColumnSum, OneGroupAllNullValues) {
  constexpr int size{10};
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  // If all values are null, then there should be a single NULL output value
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>(size, [key](auto index) { return key; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); },
                            [](auto index) { return false; }),
      column_wrapper<Key>({key}), column_wrapper<ResultValue>(1, true));
}

TYPED_TEST(SingleColumnSum, OneGroupEvenNullKeys) {
  constexpr int size{10};
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  // The sum of n odd numbers is n^2
  ResultValue sum = (size / 2) * (size / 2);
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>(size, [key](auto index) { return key; },
                          [](auto index) { return index % 2; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); }),
      column_wrapper<Key>({key}, [](auto index) { return true; }),
      column_wrapper<ResultValue>({sum}));
}

TYPED_TEST(SingleColumnSum, OneGroupOddNullKeys) {
  constexpr int size{10};
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
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
      column_wrapper<Key>({key}, [](auto index) { return true; }),
      column_wrapper<ResultValue>({sum}));
}

TYPED_TEST(SingleColumnSum, OneGroupEvenNullValues) {
  constexpr int size{10};
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  // The sum of n odd numbers is n^2
  ResultValue sum = (size / 2) * (size / 2);
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>(size, [key](auto index) { return key; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); },
                            [](auto index) { return index % 2; }),
      column_wrapper<Key>({key}),
      column_wrapper<ResultValue>({sum}, [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnSum, OneGroupOddNullValues) {
  constexpr int size{10};
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  // The number of even values in the range [0,n) is (n-1)/2
  int num_even_numbers = (size - 1) / 2;
  // The sum of n even numbers is n(n+1)
  ResultValue sum = num_even_numbers * (num_even_numbers + 1);
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>(size, [key](auto index) { return key; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); },
                            [](auto index) { return not(index % 2); }),
      column_wrapper<Key>({key}),
      column_wrapper<ResultValue>({sum}, [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnSum, FourGroupsNoNulls) {
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  // Each value needs to be casted to avoid a narrowing conversion warning for
  // the wrapper types
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>{T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
      column_wrapper<Value>(8, [](auto index) { return Value(index); }),
      column_wrapper<Key>{T(1), T(2), T(3), T(4)},
      column_wrapper<ResultValue>{R(1), R(5), R(9), R(13)});
}

TYPED_TEST(SingleColumnSum, FourGroupsEvenNullKeys) {
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>({T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
                          [](auto index) { return index % 2; }),
      column_wrapper<Value>(8, [](auto index) { return Value(index); }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)},
                          [](auto index) { return true; }),
      column_wrapper<ResultValue>{R(1), R(3), R(5), R(7)});
}
TYPED_TEST(SingleColumnSum, FourGroupsOddNullKeys) {
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>({T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
                          [](auto index) { return not(index % 2); }),
      column_wrapper<Value>(8, [](auto index) { return Value(index); }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)},
                          [](auto index) { return true; }),
      column_wrapper<ResultValue>{R(0), R(2), R(4), R(6)});
}

TYPED_TEST(SingleColumnSum, FourGroupsEvenNullValues) {
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>{T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
      column_wrapper<Value>(8, [](auto index) { return Value(index); },
                            [](auto index) { return index % 2; }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)}),
      column_wrapper<ResultValue>({R(1), R(3), R(5), R(7)},
                                  [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnSum, FourGroupsOddNullValues) {
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>{T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
      column_wrapper<Value>(8, [](auto index) { return Value(index); },
                            [](auto index) { return not(index % 2); }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)}),
      column_wrapper<ResultValue>({R(0), R(2), R(4), R(6)},
                                  [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnSum, FourGroupsEvenNullValuesEvenNullKeys) {
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>({T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
                          [](auto index) { return index % 2; }),
      column_wrapper<Value>(8, [](auto index) { return Value(index); },
                            [](auto index) { return index % 2; }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)},
                          [](auto index) { return true; }),
      column_wrapper<ResultValue>({R(1), R(3), R(5), R(7)},
                                  [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnSum, FourGroupsOddNullValuesOddNullKeys) {
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>({T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
                          [](auto index) { return not(index % 2); }),
      column_wrapper<Value>(8, [](auto index) { return Value(index); },
                            [](auto index) { return not(index % 2); }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)},
                          [](auto index) { return true; }),
      column_wrapper<ResultValue>({R(0), R(2), R(4), R(6)},
                                  [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnSum, FourGroupsOddNullValuesEvenNullKeys) {
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  // Even index keys are null & odd index values are null
  // Output should be null for each key
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>({T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
                          [](auto index) { return not(index % 2); }),
      column_wrapper<Value>(8, [](auto index) { return Value(index); },
                            [](auto index) { return index % 2; }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)},
                          [](auto index) { return true; }),
      column_wrapper<ResultValue>(4, true));
}

TYPED_TEST(SingleColumnSum, EightKeysAllUnique) {
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)}),
      column_wrapper<Value>(8, [](auto index) { return Value(index); }),
      column_wrapper<Key>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)}),
      column_wrapper<ResultValue>(8, [](auto index) { return R(index); }));
}

TYPED_TEST(SingleColumnSum, EightKeysAllUniqueEvenKeysNull) {
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)},
                          [](auto index) { return index % 2; }),
      column_wrapper<Value>(8, [](auto index) { return Value(2 * index); }),
      column_wrapper<Key>({T(1), T(3), T(5), T(7)},
                          [](auto index) { return true; }),
      column_wrapper<ResultValue>({R(2), R(6), R(10), R(14)}));
}

TYPED_TEST(SingleColumnSum, EightKeysAllUniqueEvenValuesNull) {
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  // Even index result values should be null
  cudf::groupby::sort::operation operation_with_args{op, nullptr};
  cudf::test::single_column_groupby_test<op>(
      std::move(operation_with_args),
      column_wrapper<Key>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)}),
      column_wrapper<Value>(8, [](auto index) { return Value(2 * index); },
                            [](auto index) { return index % 2; }),
      column_wrapper<Key>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)}),
      column_wrapper<ResultValue>(
          {R(-1), R(2), R(-1), R(6), R(-1), R(10), R(-1), R(14)},
          [](auto index) { return index % 2; }));
}
