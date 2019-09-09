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

#include <cudf/utilities/traits.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <tuple>

template <typename Tuple, typename F, std::size_t... Indices>
void tuple_for_each_impl(Tuple&& tuple, F&& f,
                         std::index_sequence<Indices...>) {
  (void)std::initializer_list<int>{
      ((void)(f(std::get<Indices>(std::forward<Tuple>(tuple)))), int{})...};
}

template <typename F, typename... Args>
void tuple_for_each(const std::tuple<Args...>& tuple, F&& f) {
  tuple_for_each_impl(tuple, std::forward<F>(f),
                      std::index_sequence_for<Args...>{});
}

class TraitsTest : public ::testing::Test {};

template <typename T>
class TypedTraitsTest : public TraitsTest {};

using TestTypes =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

static constexpr std::array<cudf::data_type, 6> numeric_data_types{
    cudf::data_type{cudf::INT8},    cudf::data_type{cudf::INT16},
    cudf::data_type{cudf::INT32},   cudf::data_type{cudf::INT64},
    cudf::data_type{cudf::FLOAT32}, cudf::data_type{cudf::FLOAT64}};

static constexpr std::array<cudf::data_type, 6> non_numeric_data_types{
    cudf::data_type{cudf::EMPTY},    cudf::data_type{cudf::BOOL8},
    cudf::data_type{cudf::DATE32},   cudf::data_type{cudf::TIMESTAMP},
    cudf::data_type{cudf::CATEGORY}, cudf::data_type{cudf::STRING}};

static_assert(cudf::type_id::NUM_TYPE_IDS ==
                  (non_numeric_data_types.size() + numeric_data_types.size()),
              "Mismatch in number of types");

TYPED_TEST_CASE(TypedTraitsTest, TestTypes);

TEST_F(TraitsTest, NumericDataTypesAreNumeric) {
  EXPECT_TRUE(std::all_of(
      numeric_data_types.begin(), numeric_data_types.end(),
      [](cudf::data_type dtype) { return cudf::is_numeric(dtype); }));
}

/*
These types are not yet supported by the type dispatcher
TEST_F(TraitsTest, NonNumericDataTypesAreNotNumeric) {
  EXPECT_TRUE(std::none_of(
      non_numeric_data_types.begin(), non_numeric_data_types.end(),
      [](cudf::data_type dtype) { return cudf::is_numeric(dtype); }));
}
*/

TYPED_TEST(TypedTraitsTest, RelationallyComparable) {
  // All the test types should be comparable with themselves
  bool comparable = cudf::is_relationally_comparable<TypeParam, TypeParam>();
  EXPECT_TRUE(comparable);
}

TYPED_TEST(TypedTraitsTest, NotRelationallyComparable) {
  // No type should be comparable with an empty dummy type
  struct foo {};
  bool comparable = cudf::is_relationally_comparable<foo, TypeParam>();
  EXPECT_FALSE(comparable);

  comparable = cudf::is_relationally_comparable<TypeParam, foo>();
  EXPECT_FALSE(comparable);
}