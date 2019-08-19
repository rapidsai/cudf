
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
#include "single_column_groupby_test.cuh"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

template <typename KV>
struct SingleColumnGroupby : public GdfTest {
  using KeyType = typename KV::Key;
  using ValueType = typename KV::Value;
};

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

template <typename K, typename V>
struct KV {
  using Key = K;
  using Value = V;
};

using TestingTypes =
    ::testing::Types<KV<int8_t, int8_t>, KV<int32_t, int32_t>,
                     KV<int64_t, int64_t>, KV<int32_t, float>,
                     KV<int32_t, double>, KV<cudf::category, int32_t>,
                     KV<cudf::date32, int8_t>, KV<cudf::date64, double>>;;

TYPED_TEST_CASE(SingleColumnGroupby, TestingTypes);


TYPED_TEST(SingleColumnGroupby, OneGroupEvenNullKeys) {
  constexpr int size{10};
  using Key = typename SingleColumnGroupby<TypeParam>::KeyType;
  using Value = typename SingleColumnGroupby<TypeParam>::ValueType;
  
  Key key{42};
  // The sum of n odd numbers is n^2
  cudf::test::single_column_groupby_test(
      column_wrapper<Key>(size, [key](auto index) { return key; },
                          [](auto index) { return index % 2; }),
      column_wrapper<Key>({key}, [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnGroupby, OneGroupOddNullKeys) {
  constexpr int size{10};
  using Key = typename SingleColumnGroupby<TypeParam>::KeyType;
  using Value = typename SingleColumnGroupby<TypeParam>::ValueType;
  
  Key key{42};
  cudf::test::single_column_groupby_test(
      column_wrapper<Key>(size, [key](auto index) { return key; },
                          [](auto index) { return not(index % 2); }),
      column_wrapper<Key>({key}, [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnGroupby, OneGroupEvenNullValues) {
  constexpr int size{10};
  using Key = typename SingleColumnGroupby<TypeParam>::KeyType;
  using Value = typename SingleColumnGroupby<TypeParam>::ValueType;
  
  Key key{42};
  cudf::test::single_column_groupby_test(
      column_wrapper<Key>(size, [key](auto index) { return key; }),
      column_wrapper<Key>({key}));
}

TYPED_TEST(SingleColumnGroupby, OneGroupOddNullValues) {
  constexpr int size{10};
  using Key = typename SingleColumnGroupby<TypeParam>::KeyType;
  using Value = typename SingleColumnGroupby<TypeParam>::ValueType;
  
  Key key{42};
  cudf::test::single_column_groupby_test(
      column_wrapper<Key>(size, [key](auto index) { return key; }),
      column_wrapper<Key>({key}));
}

TYPED_TEST(SingleColumnGroupby, FourGroupsNoNulls) {
  using Key = typename SingleColumnGroupby<TypeParam>::KeyType;
  using Value = typename SingleColumnGroupby<TypeParam>::ValueType;
  
  using T = Key;

  // Each value needs to be casted to avoid a narrowing conversion warning for
  // the wrapper types
  cudf::test::single_column_groupby_test(
      column_wrapper<Key>{T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
      column_wrapper<Key>{T(1), T(2), T(3), T(4)});
}

TYPED_TEST(SingleColumnGroupby, FourGroupsEvenNullKeys) {
  using Key = typename SingleColumnGroupby<TypeParam>::KeyType;
  using Value = typename SingleColumnGroupby<TypeParam>::ValueType;
  
  using T = Key;

  cudf::test::single_column_groupby_test(
      column_wrapper<Key>({T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
                          [](auto index) { return index % 2; }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)},
                          [](auto index) { return true; }));
}
TYPED_TEST(SingleColumnGroupby, FourGroupsOddNullKeys) {
  using Key = typename SingleColumnGroupby<TypeParam>::KeyType;
  using Value = typename SingleColumnGroupby<TypeParam>::ValueType;
  
  using T = Key;
 
  cudf::test::single_column_groupby_test(
      column_wrapper<Key>({T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
                          [](auto index) { return not(index % 2); }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)},
                          [](auto index) { return true; }));
}

TYPED_TEST(SingleColumnGroupby, FourGroupsEvenNullValues) {
  using Key = typename SingleColumnGroupby<TypeParam>::KeyType;
  using Value = typename SingleColumnGroupby<TypeParam>::ValueType;
  
  using T = Key;
 
  cudf::test::single_column_groupby_test(
      column_wrapper<Key>{T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
      column_wrapper<Key>({T(1), T(2), T(3), T(4)}));
}

TYPED_TEST(SingleColumnGroupby, FourGroupsOddNullValues) {
  using Key = typename SingleColumnGroupby<TypeParam>::KeyType;
  using Value = typename SingleColumnGroupby<TypeParam>::ValueType;
  
  using T = Key;

  cudf::test::single_column_groupby_test(
      column_wrapper<Key>{T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
      column_wrapper<Key>({T(1), T(2), T(3), T(4)}));
}

TYPED_TEST(SingleColumnGroupby, FourGroupsEvenNullValuesEvenNullKeys) {
  using Key = typename SingleColumnGroupby<TypeParam>::KeyType;
  using Value = typename SingleColumnGroupby<TypeParam>::ValueType;
  
  using T = Key;

  cudf::test::single_column_groupby_test(
      column_wrapper<Key>({T(1), T(1), T(2), T(2), T(3), T(3), T(4), T(4)},
                          [](auto index) { return index % 2; }),
      column_wrapper<Key>({T(1), T(2), T(3), T(4)},
                          [](auto index) { return true; }));
}
 