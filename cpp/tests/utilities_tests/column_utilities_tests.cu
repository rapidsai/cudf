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

TYPED_TEST_CASE(ColumnUtilitiesTest, cudf::test::FixedWidthTypes);

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
