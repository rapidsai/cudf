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

static constexpr cudf::groupby::operators op{
    cudf::groupby::operators::MEDIAN};

template <typename KV>
struct SingleColumnSum : public GdfTest {
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
    ::testing::Types< KV<int32_t, int32_t> >;

// TODO: tests for cudf::bool8

TYPED_TEST_CASE(SingleColumnSum, TestingTypes);
 
// TYPED_TEST(SingleColumnSum, FourGroupsNoNulls) {
//   using Key = typename SingleColumnSum<TypeParam>::KeyType;
//   using Value = typename SingleColumnSum<TypeParam>::ValueType;
//   using ResultValue = cudf::test::expected_result_t<Value, op>;
//   using T = Key;
//   using R = ResultValue;

//   // Each value needs to be casted to avoid a narrowing conversion warning for
//   // the wrapper types
//   cudf::test::single_column_groupby_test<op>(
//       column_wrapper<Key>{T(1), T(1), T(1), T(2), T(2), T(2), T(2), T(2)},
//       column_wrapper<Value>(8, [](auto index) { return Value(index); }),
//       column_wrapper<Key>{T(1), T(2)},
//       column_wrapper<ResultValue>{R(1), R(5)});
// } 

TYPED_TEST(SingleColumnSum, FourGroupsOddNullKeys) {
  using Key = typename SingleColumnSum<TypeParam>::KeyType;
  using Value = typename SingleColumnSum<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;

  cudf::test::single_column_groupby_test<op>(
      column_wrapper<Key>({T(3), T(2), T(1), T(1), T(2), T(3), T(3), T(2), T(1)}),
      column_wrapper<Value>({T(1), T(2), T(3), T(4), T(4), T(3), T(2), T(1), T(0)}),
      column_wrapper<Key>({T(1), T(2), T(3)}),
      column_wrapper<ResultValue>{R(3), R(2), R(2)});
}
 