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
#include <cudf/groupby.hpp>
#include <cudf/legacy/table.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/compare_column_wrappers.cuh>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include "single_column_groupby_test.cuh"
#include "type_info.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

namespace {
/**---------------------------------------------------------------------------*
 * @brief Return last odd index in a container containing `size` items
 *---------------------------------------------------------------------------**/
auto constexpr last_odd_index(gdf_size_type size) {
  return size - (size + 1) % 2;
}
/**---------------------------------------------------------------------------*
 * @brief Return last even index in a container containing `size` items
*---------------------------------------------------------------------------**/
auto constexpr last_even_index(gdf_size_type size) {
  return (size - 1) - ((size - 1) % 2);
}
}  // namespace

static constexpr cudf::groupby::hash::operators op{
    cudf::groupby::hash::operators::MAX};

template <typename K, typename V>
struct KV {
  using Key = K;
  using Value = V;
};

template <typename KV>
struct SingleColumnMax : public GdfTest {
  using KeyType = typename KV::Key;
  using ValueType = typename KV::Value;
};

using TestingTypes = ::testing::Types<
    KV<int8_t, int8_t>, KV<int32_t, int32_t>, KV<int64_t, int64_t>,
    KV<int32_t, float>, KV<int32_t, double>, KV<cudf::category, cudf::category>,
    KV<cudf::date32, cudf::date32>, KV<cudf::date64, cudf::date64>>;

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

// TODO: tests for cudf::bool8

TYPED_TEST_CASE(SingleColumnMax, TestingTypes);

TYPED_TEST(SingleColumnMax, OneGroupNoNulls) {
  constexpr int size{10};
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>(size, [key](auto index) { return key; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); }),
      column_wrapper<Key>({key}),
      column_wrapper<ResultValue>({ResultValue(size - 1)}));
}

TYPED_TEST(SingleColumnMax, OneGroupAllNullKeys) {
  constexpr int size{10};
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};

  // If all keys are null, then there should be no output
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>(size, [key](auto index) { return key; },
                          [](auto index) { return false; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); }),
      column_wrapper<Key>{}, column_wrapper<ResultValue>{});
}

TYPED_TEST(SingleColumnMax, OneGroupAllNullValues) {
  constexpr int size{10};
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  // If all values are null, then there should be a single NULL output value
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>(size, [key](auto index) { return key; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); },
                            [](auto index) { return false; }),
      column_wrapper<Key>({key}), column_wrapper<ResultValue>(1, true));
}

TYPED_TEST(SingleColumnMax, OneGroupEvenNullKeys) {
  constexpr int size{10};
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>(size, [key](auto index) { return key; },
                          [](auto index) { return index % 2; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); }),
      column_wrapper<Key>({key}, [](auto index) { return true; }),
      column_wrapper<ResultValue>({Value(last_odd_index(size))}));
}

TYPED_TEST(SingleColumnMax, OneGroupOddNullKeys) {
  constexpr int size{10};
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>(size, [key](auto index) { return key; },
                          [](auto index) { return not(index % 2); }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); }),
      column_wrapper<Key>({key}, [](auto index) { return true; }),
      column_wrapper<ResultValue>({Value(last_even_index(size))}));
}

TYPED_TEST(SingleColumnMax, OneGroupEvenNullValues) {
  constexpr int size{10};
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>(size, [key](auto index) { return key; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); },
                            [](auto index) { return index % 2; }),
      column_wrapper<Key>({key}),
      column_wrapper<ResultValue>({Value(last_odd_index(size))},
                                  [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnMax, OneGroupOddNullValues) {
  constexpr int size{10};
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  Key key{42};
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>(size, [key](auto index) { return key; }),
      column_wrapper<Value>(size, [](auto index) { return Value(index); },
                            [](auto index) { return not(index % 2); }),
      column_wrapper<Key>({key}),
      column_wrapper<ResultValue>({Value(last_even_index(size))},
                                  [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnMax, FourGroupsNoNulls) {
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  // Each value needs to be casted to avoid a narrowing conversion warning for
  // the wrapper types
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>{T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
      column_wrapper<Value>(8, [](auto index) { return Value(index); }),
      column_wrapper<Key>{T(1), T(2), T(3), T(4)},
      column_wrapper<ResultValue>{R(1), R(3), R(5), R(7)});
}

TYPED_TEST(SingleColumnMax, FourGroupsEvenNullKeys) {
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>({T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
                          [](auto index) { return index % 2; }),
      column_wrapper<Value>(8, [](auto index) { return Value(index); }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)},
                          [](auto index) { return true; }),
      column_wrapper<ResultValue>{R(1), R(3), R(5), R(7)});
}

TYPED_TEST(SingleColumnMax, FourGroupsOddNullKeys) {
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>({T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
                          [](auto index) { return not(index % 2); }),
      column_wrapper<Value>(8, [](auto index) { return Value(index); }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)},
                          [](auto index) { return true; }),
      column_wrapper<ResultValue>{R(0), R(2), R(4), R(6)});
}

TYPED_TEST(SingleColumnMax, FourGroupsEvenNullValues) {
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>{T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
      column_wrapper<Value>(8, [](auto index) { return Value(index); },
                            [](auto index) { return index % 2; }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)}),
      column_wrapper<ResultValue>({R(1), R(3), R(5), R(7)},
                                  [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnMax, FourGroupsOddNullValues) {
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>{T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
      column_wrapper<Value>(8, [](auto index) { return Value(index); },
                            [](auto index) { return not(index % 2); }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)}),
      column_wrapper<ResultValue>({R(0), R(2), R(4), R(6)},
                                  [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnMax, FourGroupsEvenNullValuesEvenNullKeys) {
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>({T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
                          [](auto index) { return index % 2; }),
      column_wrapper<Value>(8, [](auto index) { return Value(index); },
                            [](auto index) { return index % 2; }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)},
                          [](auto index) { return true; }),
      column_wrapper<ResultValue>({R(1), R(3), R(5), R(7)},
                                  [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnMax, FourGroupsOddNullValuesOddNullKeys) {
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>({T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
                          [](auto index) { return not(index % 2); }),
      column_wrapper<Value>(8, [](auto index) { return Value(index); },
                            [](auto index) { return not(index % 2); }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)},
                          [](auto index) { return true; }),
      column_wrapper<ResultValue>({R(0), R(2), R(4), R(6)},
                                  [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnMax, FourGroupsOddNullValuesEvenNullKeys) {
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  // Even index keys are null & odd index values are null
  // Output should be null for each key
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>({T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
                          [](auto index) { return not(index % 2); }),
      column_wrapper<Value>(8, [](auto index) { return Value(index); },
                            [](auto index) { return index % 2; }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)},
                          [](auto index) { return true; }),
      column_wrapper<ResultValue>(4, true));
}

TYPED_TEST(SingleColumnMax, EightKeysAllUnique) {
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)}),
      column_wrapper<Value>(8, [](auto index) { return Value(index); }),
      column_wrapper<Key>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)}),
      column_wrapper<ResultValue>(8, [](auto index) { return R(index); }));
}

TYPED_TEST(SingleColumnMax, EightKeysAllUniqueEvenKeysNull) {
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)},
                          [](auto index) { return index % 2; }),
      column_wrapper<Value>(8, [](auto index) { return Value(2 * index); }),
      column_wrapper<Key>({T(1), T(3), T(5), T(7)},
                          [](auto index) { return true; }),
      column_wrapper<ResultValue>({R(2), R(6), R(10), R(14)}));
}

TYPED_TEST(SingleColumnMax, EightKeysAllUniqueEvenValuesNull) {
  using Key = typename SingleColumnMax<TypeParam>::KeyType;
  using Value = typename SingleColumnMax<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  // Even index result values should be null
  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)}),
      column_wrapper<Value>(8, [](auto index) { return Value(2 * index); },
                            [](auto index) { return index % 2; }),
      column_wrapper<Key>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)}),
      column_wrapper<ResultValue>(
          {R(-1), R(2), R(-1), R(6), R(-1), R(10), R(-1), R(14)},
          [](auto index) { return index % 2; }));
}
