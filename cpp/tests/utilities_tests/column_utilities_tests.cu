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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

template <typename T>

struct ColumnUtilitiesTest
    : public cudf::test::BaseFixture,
      cudf::test::UniformRandomGenerator<cudf::size_type> {
  ColumnUtilitiesTest()
      : cudf::test::UniformRandomGenerator<cudf::size_type>{1000, 5000} {}

  auto size() { return this->generate(); }

  auto data_type() {
    return cudf::data_type{cudf::experimental::type_to_id<T>()};
  }
};

template <typename T>
struct ColumnUtilitiesTestNumeric : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(ColumnUtilitiesTest, cudf::test::FixedWidthTypes);
TYPED_TEST_CASE(ColumnUtilitiesTestNumeric, cudf::test::NumericTypes);

TYPED_TEST(ColumnUtilitiesTest, NonNullableToHost) {
  auto sequence = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return TypeParam(i); });

  auto size = this->size();

  std::vector<TypeParam> data(sequence, sequence + size);
  cudf::test::fixed_width_column_wrapper<TypeParam> col(
    data.begin(), data.end());

  auto host_data = cudf::test::to_host<TypeParam>(col);

  EXPECT_TRUE(std::equal(data.begin(), data.end(), host_data.first.begin()));
}

TYPED_TEST(ColumnUtilitiesTest, NullableToHostAllValid) {
  auto sequence = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return TypeParam(i); });

  auto all_valid = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });

  auto size = this->size();

  std::vector<TypeParam> data(sequence, sequence + size);
  cudf::test::fixed_width_column_wrapper<TypeParam> col(
    data.begin(), data.end(), all_valid);

  auto host_data = cudf::test::to_host<TypeParam>(col);

  EXPECT_TRUE(std::equal(data.begin(), data.end(), host_data.first.begin()));

  auto masks = cudf::test::detail::make_null_mask_vector(all_valid, all_valid+size);

  EXPECT_TRUE(std::equal(masks.begin(), masks.end(), host_data.second.begin()));
}

TYPED_TEST(ColumnUtilitiesTestNumeric, PrintColumn) {
  const char* delimiter = ",";

  cudf::test::fixed_width_column_wrapper<TypeParam> cudf_col({1, 2, 3, 4, 5});
  std::vector<TypeParam>                            std_col({1, 2, 3, 4, 5});

  std::ostringstream tmp;

  int index = 0;
  for (auto x : std_col) {
    tmp << ((index == 0) ? "" : delimiter);
    tmp << std::to_string(x);
    ++index;
  }

  EXPECT_EQ(cudf::test::to_string(cudf_col, delimiter), tmp.str());
}

TYPED_TEST(ColumnUtilitiesTestNumeric, PrintColumnWithInvalids) {
  const char* delimiter = ",";

  cudf::test::fixed_width_column_wrapper<TypeParam> cudf_col{ {1, 2, 3, 4, 5},
                                                              {1, 0, 1, 0, 1} };
  std::vector<TypeParam>                            std_col({1, 2, 3, 4, 5});

  std::ostringstream tmp;
  tmp << std::to_string(std_col[0])
      << delimiter << "@"
      << delimiter << std::to_string(std_col[2])
      << delimiter << "@"
      << delimiter << std::to_string(std_col[4]);
  
  EXPECT_EQ(cudf::test::to_string(cudf_col, delimiter), tmp.str());
}
