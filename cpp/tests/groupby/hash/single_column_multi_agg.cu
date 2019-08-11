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

using namespace cudf::groupby::hash;

template <typename KV>
struct SingleColumnMultiAgg : public GdfTest {
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
                     KV<cudf::date32, int8_t>, KV<cudf::date64, double>>;

// TODO: tests for cudf::bool8

TYPED_TEST_CASE(SingleColumnMultiAgg, TestingTypes);

TYPED_TEST(SingleColumnMultiAgg, RepeatedAgg) {
  constexpr int size{10};
  using Key = typename SingleColumnMultiAgg<TypeParam>::KeyType;
  using Value = typename SingleColumnMultiAgg<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, SUM>;
  Key key{42};
  ResultValue sum{((size - 1) * size) / 2};

  auto key_col = column_wrapper<Key>(size, [key](auto index) { return key; });
  auto val_col = column_wrapper<Value>(size, [](auto index) { return Value(index); });
  auto expected_key_col = column_wrapper<Key>({key});
  auto expected_val_col = column_wrapper<ResultValue>({sum});
  cudf::test::multi_column_groupby_test(
      cudf::table{key_col.get()}, cudf::table{val_col.get(), val_col.get()},
      {SUM, SUM}, cudf::table{expected_key_col.get()},
      cudf::table{expected_val_col.get(), expected_val_col.get()});
}

TYPED_TEST(SingleColumnMultiAgg, SimpleAndCompound) {
  constexpr int size{10};
  using Key = typename SingleColumnMultiAgg<TypeParam>::KeyType;
  using Value = typename SingleColumnMultiAgg<TypeParam>::ValueType;
  using ResultValueSum = cudf::test::expected_result_t<Value, SUM>;
  using ResultValueAvg = cudf::test::expected_result_t<Value, MEAN>;
  Key key{42};
  ResultValueSum sum{((size - 1) * size) / 2};
  ResultValueAvg avg{ResultValueAvg(sum) / size};

  auto key_col = column_wrapper<Key>(size, [key](auto index) { return key; });
  auto val_col = column_wrapper<Value>(size, [](auto index) { return Value(index); });
  auto expected_key_col = column_wrapper<Key>({key});
  auto expected_val_sum_col = column_wrapper<ResultValueSum>({sum});
  auto expected_val_avg_col = column_wrapper<ResultValueAvg>({avg});
  cudf::test::multi_column_groupby_test(
      cudf::table{key_col.get()}, cudf::table{val_col.get(), val_col.get()},
      {SUM, MEAN}, cudf::table{expected_key_col.get()},
      cudf::table{expected_val_sum_col.get(), expected_val_avg_col.get()});
}
