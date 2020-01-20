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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/copying.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <thrust/iterator/constant_iterator.h>

template <typename T>
struct ColumnUtilitiesTest : public cudf::test::BaseFixture
{
  cudf::test::UniformRandomGenerator<cudf::size_type> random;

  ColumnUtilitiesTest() : random{1000, 5000} {}

  auto size() { return random.generate(); }

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

TYPED_TEST(ColumnUtilitiesTest, NonNullableToHostWithOffset) {
  auto sequence = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return TypeParam(i); });

  auto size = this->size();
  auto split = 2;

  std::vector<TypeParam> data(sequence, sequence + size);
  std::vector<TypeParam> expected_data(sequence+split, sequence + size);
  cudf::test::fixed_width_column_wrapper<TypeParam> col(
    data.begin(), data.end());

  std::vector<cudf::size_type> splits{split};
  std::vector<cudf::column_view> result = cudf::experimental::split(col, splits);

  auto host_data = cudf::test::to_host<TypeParam>(result.back());

  EXPECT_TRUE(std::equal(expected_data.begin(), expected_data.end(), host_data.first.begin()));
}

TYPED_TEST(ColumnUtilitiesTest, NullableToHostWithOffset) {
  auto sequence = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return TypeParam(i); });

  auto split = 2;
  auto size = this->size();
  auto valid = cudf::test::make_counting_transform_iterator(0, [&split](auto i) { return (i < (split+1) or i > 10)? false: true;});
  std::vector<TypeParam> data(sequence, sequence + size);
  std::vector<TypeParam> expected_data(sequence + split, sequence + size);
  cudf::test::fixed_width_column_wrapper<TypeParam> col(
    data.begin(), data.end(), valid);

  std::vector<cudf::size_type> splits{split};
  std::vector<cudf::column_view> result = cudf::experimental::split(col, splits);

  auto host_data = cudf::test::to_host<TypeParam>(result.back());

  EXPECT_TRUE(std::equal(expected_data.begin(), expected_data.end(), host_data.first.begin()));

  auto masks = cudf::test::detail::make_null_mask_vector(valid + split, valid + size);

  EXPECT_TRUE(std::equal(masks.begin(), masks.end(), host_data.second.begin()));
}

TYPED_TEST(ColumnUtilitiesTest, NullableToHostAllValid) {
  auto sequence = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return TypeParam(i); });

  auto all_valid = thrust::make_constant_iterator<bool>(true);

  auto size = this->size();

  std::vector<TypeParam> data(sequence, sequence + size);
  cudf::test::fixed_width_column_wrapper<TypeParam> col(
    data.begin(), data.end(), all_valid);

  auto host_data = cudf::test::to_host<TypeParam>(col);

  EXPECT_TRUE(std::equal(data.begin(), data.end(), host_data.first.begin()));

  auto masks = cudf::test::detail::make_null_mask_vector(all_valid, all_valid + size);

  EXPECT_TRUE(std::equal(masks.begin(), masks.end(), host_data.second.begin()));
}

struct ColumnUtilitiesEquivalenceTest : public cudf::test::BaseFixture {};

TEST_F(ColumnUtilitiesEquivalenceTest, DoubleTest) {
  cudf::test::fixed_width_column_wrapper<double> col1 { 10./3, 22./7 };
  cudf::test::fixed_width_column_wrapper<double> col2 { 31./3 - 21./3, 19./7 + 3./7 };

  cudf::test::expect_columns_equivalent(col1, col2);
}

TEST_F(ColumnUtilitiesEquivalenceTest, NullabilityTest) {
  auto all_valid = cudf::test::make_counting_transform_iterator(
                                  0, [](auto i) { return true; });
  cudf::test::fixed_width_column_wrapper<double> col1 { 1, 2, 3 };
  cudf::test::fixed_width_column_wrapper<double> col2({ 1, 2, 3 }, all_valid);

  cudf::test::expect_columns_equivalent(col1, col2);
}

struct ColumnUtilitiesStringsTest: public cudf::test::BaseFixture {};

TEST_F(ColumnUtilitiesStringsTest, StringsToHost)
{
  std::vector<const char*> h_strings{ "eee", "bb", nullptr, "", "aa", "bbb", "ééé" };
  cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
  auto host_data = cudf::test::to_host<std::string>(strings);
  auto result_itr = host_data.first.begin();
  for( auto itr = h_strings.begin(); itr != h_strings.end(); ++itr, ++result_itr )
  {

    if(*itr)
      EXPECT_TRUE((*result_itr)==(*itr));
  }
}

TEST_F(ColumnUtilitiesStringsTest, StringsToHostAllNulls)
{
  std::vector<const char*> h_strings{ nullptr, nullptr, nullptr };
  cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
  auto host_data = cudf::test::to_host<std::string>(strings);
  EXPECT_TRUE( host_data.first.empty() );
}

TYPED_TEST(ColumnUtilitiesTestNumeric, PrintColumnNumeric) {
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
      << delimiter << "NULL"
      << delimiter << std::to_string(std_col[2])
      << delimiter << "NULL"
      << delimiter << std::to_string(std_col[4]);
  
  EXPECT_EQ(cudf::test::to_string(cudf_col, delimiter), tmp.str());
}

TEST_F(ColumnUtilitiesStringsTest, StringsToString) {
  const char* delimiter = ",";

  std::vector<const char*> h_strings{ "eee", "bb", nullptr, "", "aa", "bbb", "ééé" };
  cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));


  std::ostringstream tmp;
  tmp << h_strings[0]
      << delimiter << h_strings[1]
      << delimiter << "NULL"
      << delimiter << h_strings[3]
      << delimiter << h_strings[4]
      << delimiter << h_strings[5]
      << delimiter << h_strings[6];
  
  EXPECT_EQ(cudf::test::to_string(strings, delimiter), tmp.str());
}
