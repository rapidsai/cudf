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
#include "../single_column_groupby_test.cuh"
#include "../../common/type_info.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

static constexpr cudf::groupby::operators op{
    cudf::groupby::operators::QUANTILE};

template <typename KV>
struct SingleColumnQuantile : public GdfTest {
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

using quantile_args =  cudf::groupby::sort::quantile_args;

// TODO: tests for cudf::bool8
TYPED_TEST_CASE(SingleColumnQuantile, TestingTypes);
 
TYPED_TEST(SingleColumnQuantile, TestQuantile0) {
  using Key = typename SingleColumnQuantile<TypeParam>::KeyType;
  using Value = typename SingleColumnQuantile<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;
  std::vector<double> quantiles = {0.5};
  auto method = cudf::interpolation::LINEAR;
  cudf::groupby::sort::operation operation_with_args {op,  std::make_unique<quantile_args>(quantiles, method)}; 
      
  cudf::test::single_column_groupby_test<op>(std::move(operation_with_args),
      column_wrapper<Key>{T(1), T(1), T(1), T(2), T(2), T(2), T(2), T(2)},
      column_wrapper<Value>(8, [](auto index) { return Value(index); }),
      column_wrapper<Key>{T(1), T(2)},
      column_wrapper<ResultValue>{R(1), R(5)});
}

TYPED_TEST(SingleColumnQuantile, TestQuantile1) {
  using Key = typename SingleColumnQuantile<TypeParam>::KeyType;
  using Value = typename SingleColumnQuantile<TypeParam>::ValueType;
  using ResultValue = cudf::test::expected_result_t<Value, op>;
  using T = Key;
  using R = ResultValue;
  std::vector<double> quantiles ={0.25, 0.75};
  auto method = cudf::interpolation::LINEAR;
  cudf::groupby::sort::operation operation_with_args {op,  std::make_unique<quantile_args>(quantiles, method)}; 
      
  cudf::test::single_column_groupby_test<op>(std::move(operation_with_args),
      column_wrapper<Key>{T(1), T(2), T(3), T(1), T(2), T(2), T(1), T(3), T(3), T(2)},
      column_wrapper<Value>(10, [](auto index) { return Value(index); }),
      column_wrapper<Key>{T(1), T(2), T{3}},
      column_wrapper<ResultValue>{R(1.5), R(4.5), R( 3.25), R( 6), R(4.5), R(7.5)});
}