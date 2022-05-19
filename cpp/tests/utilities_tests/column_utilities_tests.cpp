/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <type_traits>

template <typename T>
struct ColumnUtilitiesTest : public cudf::test::BaseFixture {
  cudf::test::UniformRandomGenerator<cudf::size_type> random;

  ColumnUtilitiesTest() : random{1000, 5000} {}

  auto size() { return random.generate(); }

  auto data_type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

template <typename T>
struct ColumnUtilitiesTestIntegral : public cudf::test::BaseFixture {
};
template <typename T>
struct ColumnUtilitiesTestFloatingPoint : public cudf::test::BaseFixture {
};

template <typename T>
struct ColumnUtilitiesTestFixedPoint : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(ColumnUtilitiesTest, cudf::test::FixedWidthTypes);
TYPED_TEST_SUITE(ColumnUtilitiesTestIntegral, cudf::test::IntegralTypes);
TYPED_TEST_SUITE(ColumnUtilitiesTestFloatingPoint, cudf::test::FloatingPointTypes);
TYPED_TEST_SUITE(ColumnUtilitiesTestFixedPoint, cudf::test::FixedPointTypes);

TYPED_TEST(ColumnUtilitiesTest, NonNullableToHost)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return cudf::test::make_type_param_scalar<TypeParam>(i); });

  auto size = this->size();

  std::vector<TypeParam> data(sequence, sequence + size);
  cudf::test::fixed_width_column_wrapper<TypeParam> col(data.begin(), data.end());

  auto host_data = cudf::test::to_host<TypeParam>(col);

  EXPECT_TRUE(std::equal(data.begin(), data.end(), host_data.first.begin()));
}

TYPED_TEST(ColumnUtilitiesTest, NonNullableToHostWithOffset)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return cudf::test::make_type_param_scalar<TypeParam>(i); });

  auto const size  = this->size();
  auto const split = 2;

  auto data          = std::vector<TypeParam>(sequence, sequence + size);
  auto expected_data = std::vector<TypeParam>(sequence + split, sequence + size);
  auto col           = cudf::test::fixed_width_column_wrapper<TypeParam>(data.begin(), data.end());

  auto const splits = std::vector<cudf::size_type>{split};
  auto result       = cudf::split(col, splits);

  auto host_data = cudf::test::to_host<TypeParam>(result.back());

  EXPECT_TRUE(std::equal(expected_data.begin(), expected_data.end(), host_data.first.begin()));
}

TYPED_TEST(ColumnUtilitiesTest, NullableToHostWithOffset)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return cudf::test::make_type_param_scalar<TypeParam>(i); });

  auto split = 2;
  auto size  = this->size();
  auto valid = cudf::detail::make_counting_transform_iterator(
    0, [&split](auto i) { return (i < (split + 1) or i > 10) ? false : true; });
  std::vector<TypeParam> data(sequence, sequence + size);
  std::vector<TypeParam> expected_data(sequence + split, sequence + size);
  cudf::test::fixed_width_column_wrapper<TypeParam> col(data.begin(), data.end(), valid);

  std::vector<cudf::size_type> splits{split};
  std::vector<cudf::column_view> result = cudf::split(col, splits);

  auto host_data = cudf::test::to_host<TypeParam>(result.back());

  EXPECT_TRUE(std::equal(expected_data.begin(), expected_data.end(), host_data.first.begin()));

  auto masks = cudf::test::detail::make_null_mask_vector(valid + split, valid + size);

  EXPECT_TRUE(cudf::test::validate_host_masks(masks, host_data.second, expected_data.size()));
}

TYPED_TEST(ColumnUtilitiesTest, NullableToHostAllValid)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return cudf::test::make_type_param_scalar<TypeParam>(i); });

  auto all_valid = thrust::make_constant_iterator<bool>(true);

  auto size = this->size();

  std::vector<TypeParam> data(sequence, sequence + size);
  cudf::test::fixed_width_column_wrapper<TypeParam> col(data.begin(), data.end(), all_valid);

  auto host_data = cudf::test::to_host<TypeParam>(col);

  EXPECT_TRUE(std::equal(data.begin(), data.end(), host_data.first.begin()));

  auto masks = cudf::test::detail::make_null_mask_vector(all_valid, all_valid + size);

  EXPECT_TRUE(cudf::test::validate_host_masks(masks, host_data.second, size));
}

struct ColumnUtilitiesEquivalenceTest : public cudf::test::BaseFixture {
};

TEST_F(ColumnUtilitiesEquivalenceTest, DoubleTest)
{
  cudf::test::fixed_width_column_wrapper<double> col1{10. / 3, 22. / 7};
  cudf::test::fixed_width_column_wrapper<double> col2{31. / 3 - 21. / 3, 19. / 7 + 3. / 7};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(col1, col2);
}

TEST_F(ColumnUtilitiesEquivalenceTest, NullabilityTest)
{
  auto all_valid = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });
  cudf::test::fixed_width_column_wrapper<double> col1{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<double> col2({1, 2, 3}, all_valid);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(col1, col2);
}

struct ColumnUtilitiesStringsTest : public cudf::test::BaseFixture {
};

TEST_F(ColumnUtilitiesStringsTest, StringsToHost)
{
  std::vector<const char*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto host_data  = cudf::test::to_host<std::string>(strings);
  auto result_itr = host_data.first.begin();
  for (auto itr = h_strings.begin(); itr != h_strings.end(); ++itr, ++result_itr) {
    if (*itr) { EXPECT_TRUE((*result_itr) == (*itr)); }
  }
}

TEST_F(ColumnUtilitiesStringsTest, StringsToHostAllNulls)
{
  std::vector<const char*> h_strings{nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto host_data = cudf::test::to_host<std::string>(strings);
  auto results   = host_data.first;
  EXPECT_EQ(std::size_t{3}, host_data.first.size());
  EXPECT_TRUE(std::all_of(results.begin(), results.end(), [](auto s) { return s.empty(); }));
}

TEST_F(ColumnUtilitiesStringsTest, PrintColumnDuration)
{
  const char* delimiter = ",";

  cudf::test::fixed_width_column_wrapper<cudf::duration_s, int32_t> cudf_col({100, 0, 7, 140000});

  auto expected = "100 seconds,0 seconds,7 seconds,140000 seconds";

  EXPECT_EQ(cudf::test::to_string(cudf_col, delimiter), expected);
}

TYPED_TEST(ColumnUtilitiesTestIntegral, PrintColumnNumeric)
{
  const char* delimiter = ",";

  cudf::test::fixed_width_column_wrapper<TypeParam> cudf_col({1, 2, 3, 4, 5});
  auto std_col = cudf::test::make_type_param_vector<TypeParam>({1, 2, 3, 4, 5});

  std::stringstream tmp;
  auto string_iter =
    thrust::make_transform_iterator(std::begin(std_col), [](auto e) { return std::to_string(e); });

  std::copy(string_iter,
            string_iter + std_col.size() - 1,
            std::ostream_iterator<std::string>(tmp, delimiter));

  tmp << std::to_string(std_col.back());

  EXPECT_EQ(cudf::test::to_string(cudf_col, delimiter), tmp.str());
}

TYPED_TEST(ColumnUtilitiesTestIntegral, PrintColumnWithInvalids)
{
  const char* delimiter = ",";

  cudf::test::fixed_width_column_wrapper<TypeParam> cudf_col{{1, 2, 3, 4, 5}, {1, 0, 1, 0, 1}};
  auto std_col = cudf::test::make_type_param_vector<TypeParam>({1, 2, 3, 4, 5});

  std::ostringstream tmp;
  tmp << std::to_string(std_col[0]) << delimiter << "NULL" << delimiter
      << std::to_string(std_col[2]) << delimiter << "NULL" << delimiter
      << std::to_string(std_col[4]);

  EXPECT_EQ(cudf::test::to_string(cudf_col, delimiter), tmp.str());
}

TYPED_TEST(ColumnUtilitiesTestFloatingPoint, PrintColumnNumeric)
{
  const char* delimiter = ",";

  cudf::test::fixed_width_column_wrapper<TypeParam> cudf_col(
    {10001523.25, 2.0, 3.75, 0.000000034, 5.3});

  auto expected = std::is_same_v<TypeParam, double>
                    ? "10001523.25,2,3.75,3.4e-08,5.2999999999999998"
                    : "10001523,2,3.75,3.39999993e-08,5.30000019";

  EXPECT_EQ(cudf::test::to_string(cudf_col, delimiter), expected);
}

TYPED_TEST(ColumnUtilitiesTestFloatingPoint, PrintColumnWithInvalids)
{
  const char* delimiter = ",";

  cudf::test::fixed_width_column_wrapper<TypeParam> cudf_col(
    {10001523.25, 2.0, 3.75, 0.000000034, 5.3}, {1, 0, 1, 0, 1});

  auto expected = std::is_same_v<TypeParam, double>
                    ? "10001523.25,NULL,3.75,NULL,5.2999999999999998"
                    : "10001523,NULL,3.75,NULL,5.30000019";

  EXPECT_EQ(cudf::test::to_string(cudf_col, delimiter), expected);
}

TEST_F(ColumnUtilitiesStringsTest, StringsToString)
{
  const char* delimiter = ",";

  std::vector<const char*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  std::ostringstream tmp;
  tmp << h_strings[0] << delimiter << h_strings[1] << delimiter << "NULL" << delimiter
      << h_strings[3] << delimiter << h_strings[4] << delimiter << h_strings[5] << delimiter
      << h_strings[6];

  EXPECT_EQ(cudf::test::to_string(strings, delimiter), tmp.str());
}

TYPED_TEST(ColumnUtilitiesTestFixedPoint, NonNullableToHost)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using rep       = cudf::device_storage_type_t<decimalXX>;

  auto const scale = scale_type{-2};
  auto to_fp       = [&](auto i) { return decimalXX{i, scale}; };
  auto to_rep      = [](auto i) { return i * 100; };
  auto fps         = cudf::detail::make_counting_transform_iterator(0, to_fp);
  auto reps        = cudf::detail::make_counting_transform_iterator(0, to_rep);

  auto const size      = 1000;
  auto const expected  = std::vector<decimalXX>(fps, fps + size);
  auto const col       = cudf::test::fixed_point_column_wrapper<rep>(reps, reps + size, scale);
  auto const host_data = cudf::test::to_host<decimalXX>(col);

  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), host_data.first.begin()));
}

TYPED_TEST(ColumnUtilitiesTestFixedPoint, NonNullableToHostWithOffset)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using rep       = cudf::device_storage_type_t<decimalXX>;

  auto const scale = scale_type{-2};
  auto to_fp       = [&](auto i) { return decimalXX{i, scale}; };
  auto to_rep      = [](auto i) { return i * 100; };
  auto fps         = cudf::detail::make_counting_transform_iterator(0, to_fp);
  auto reps        = cudf::detail::make_counting_transform_iterator(0, to_rep);

  auto const size  = 1000;
  auto const split = cudf::size_type{2};

  auto const expected = std::vector<decimalXX>(fps + split, fps + size);
  auto const col      = cudf::test::fixed_point_column_wrapper<rep>(reps, reps + size, scale);
  auto const splits   = std::vector<cudf::size_type>{split};
  auto result         = cudf::split(col, splits);

  auto host_data = cudf::test::to_host<decimalXX>(result.back());

  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), host_data.first.begin()));
}

struct ColumnUtilitiesListsTest : public cudf::test::BaseFixture {
};

TEST_F(ColumnUtilitiesListsTest, Equivalence)
{
  // list<int>, nullable vs. non-nullable
  {
    auto all_valid = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });
    cudf::test::lists_column_wrapper<int> a{{1, 2, 3}, {5, 6}, {8, 9}, {10}, {14, 15}};
    cudf::test::lists_column_wrapper<int> b{{{1, 2, 3}, {5, 6}, {8, 9}, {10}, {14, 15}}, all_valid};

    // properties
    cudf::test::expect_column_properties_equivalent(a, b);
    EXPECT_EQ(
      cudf::test::expect_column_properties_equal(a, b, cudf::test::debug_output_level::QUIET),
      false);

    // values
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(a, b);
    EXPECT_EQ(cudf::test::expect_columns_equal(a, b, cudf::test::debug_output_level::QUIET), false);
  }

  // list<list<int>>, nullable vs. non-nullable
  {
    auto all_valid = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });
    cudf::test::lists_column_wrapper<int> a{{{1, 2, 3}, {5, 6}}, {{8, 9}, {10}}, {{14, 15}}};
    cudf::test::lists_column_wrapper<int> b{{{{1, 2, 3}, {5, 6}}, {{8, 9}, {10}}, {{14, 15}}},
                                            all_valid};

    // properties
    cudf::test::expect_column_properties_equivalent(a, b);
    EXPECT_EQ(
      cudf::test::expect_column_properties_equal(a, b, cudf::test::debug_output_level::QUIET),
      false);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(a, b);
    EXPECT_EQ(cudf::test::expect_columns_equal(a, b, cudf::test::debug_output_level::QUIET), false);
  }
}

TEST_F(ColumnUtilitiesListsTest, UnsanitaryLists)
{
  // unsanitary
  //
  // List<int32_t>:
  //  Length : 1
  //  Offsets : 0, 3
  //  Null count: 1
  //  0
  //    0, 1, 2
  cudf::test::fixed_width_column_wrapper<cudf::offset_type> offsets{0, 3};
  cudf::test::fixed_width_column_wrapper<int> values{0, 1, 2};
  auto l0 = cudf::make_lists_column(1,
                                    offsets.release(),
                                    values.release(),
                                    1,
                                    cudf::create_null_mask(1, cudf::mask_state::ALL_NULL));

  // sanitary
  //
  // List<int32_t>:
  //  Length : 1
  //  Offsets : 0, 0
  //  Null count: 1
  //    0
  auto l1 = cudf::test::lists_column_wrapper<int>::make_one_empty_row_column(false);

  // equivalent, but not equal
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*l0, l1);
  EXPECT_FALSE(cudf::test::expect_columns_equal(*l0, l1, cudf::test::debug_output_level::QUIET));
}

TEST_F(ColumnUtilitiesListsTest, DifferentPhysicalStructure)
{
  // list<int>
  {
    std::vector<bool> valids = {0, 0, 1, 0, 1, 0, 0};

    cudf::test::fixed_width_column_wrapper<int> c0_offsets{0, 3, 6, 8, 11, 14, 16, 19};
    cudf::test::fixed_width_column_wrapper<int> c0_data{
      1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7};
    auto c0 = make_lists_column(7,
                                c0_offsets.release(),
                                c0_data.release(),
                                5,
                                cudf::test::detail::make_null_mask(valids.begin(), valids.end()));

    cudf::test::fixed_width_column_wrapper<int> c1_offsets{0, 0, 0, 2, 2, 5, 5, 5};
    cudf::test::fixed_width_column_wrapper<int> c1_data{3, 3, 5, 5, 5};
    auto c1 = make_lists_column(7,
                                c1_offsets.release(),
                                c1_data.release(),
                                5,
                                cudf::test::detail::make_null_mask(valids.begin(), valids.end()));

    // properties
    cudf::test::expect_column_properties_equivalent(*c0, *c1);
    EXPECT_EQ(
      cudf::test::expect_column_properties_equal(*c0, *c1, cudf::test::debug_output_level::QUIET),
      false);

    // values
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*c0, *c1);
    EXPECT_EQ(cudf::test::expect_columns_equal(*c0, *c1, cudf::test::debug_output_level::QUIET),
              false);
  }

  // list<list<struct<int, float>>>
  {
    std::vector<bool> level1_valids = {0, 0, 1, 0, 1, 0, 0};

    cudf::test::fixed_width_column_wrapper<int> c0_l1_offsets{0, 1, 2, 4, 4, 7, 7, 7};
    cudf::test::fixed_width_column_wrapper<int> c0_l2_offsets{0, 1, 2, 5, 6, 7, 10, 14};
    cudf::test::fixed_width_column_wrapper<int> c0_l3_ints{
      1, 1, -1, -2, -3, 1, 1, -4, -5, -6, -7, -8, -9, -10};
    cudf::test::fixed_width_column_wrapper<float> c0_l3_floats{
      1, 1, 10, 20, 30, 1, 1, 40, 50, 60, 70, 80, 90, 100};
    cudf::test::structs_column_wrapper c0_l2_data({c0_l3_ints, c0_l3_floats});
    std::vector<bool> c0_l2_valids = {1, 1, 1, 0, 0, 1, 1};
    auto c0_l2                     = make_lists_column(
      7,
      c0_l2_offsets.release(),
      c0_l2_data.release(),
      0,
      cudf::test::detail::make_null_mask(c0_l2_valids.begin(), c0_l2_valids.end()));
    auto c0 = make_lists_column(
      7,
      c0_l1_offsets.release(),
      std::move(c0_l2),
      5,
      cudf::test::detail::make_null_mask(level1_valids.begin(), level1_valids.end()));

    cudf::test::fixed_width_column_wrapper<int> c1_l1_offsets{0, 0, 0, 2, 2, 5, 5, 5};
    cudf::test::fixed_width_column_wrapper<int> c1_l2_offsets{0, 3, 3, 3, 6, 10};
    cudf::test::fixed_width_column_wrapper<int> c1_l3_ints{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
    cudf::test::fixed_width_column_wrapper<float> c1_l3_floats{
      10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    cudf::test::structs_column_wrapper c1_l2_data({c1_l3_ints, c1_l3_floats});
    std::vector<bool> c1_l2_valids = {1, 0, 0, 1, 1};
    auto c1_l2                     = make_lists_column(
      5,
      c1_l2_offsets.release(),
      c1_l2_data.release(),
      2,
      cudf::test::detail::make_null_mask(c1_l2_valids.begin(), c1_l2_valids.end()));
    auto c1 = make_lists_column(
      7,
      c1_l1_offsets.release(),
      std::move(c1_l2),
      5,
      cudf::test::detail::make_null_mask(level1_valids.begin(), level1_valids.end()));

    // properties
    cudf::test::expect_column_properties_equivalent(*c0, *c1);
    EXPECT_EQ(
      cudf::test::expect_column_properties_equal(*c0, *c1, cudf::test::debug_output_level::QUIET),
      false);

    // values
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*c0, *c1);
    EXPECT_EQ(cudf::test::expect_columns_equal(*c0, *c1, cudf::test::debug_output_level::QUIET),
              false);
  }
}

struct ColumnUtilitiesStructsTest : public cudf::test::BaseFixture {
};

TEST_F(ColumnUtilitiesStructsTest, Properties)
{
  cudf::test::strings_column_wrapper s0_scol0{"mno", "jkl", "ghi", "def", "abc"};
  cudf::test::fixed_width_column_wrapper<float> s0_scol1{5, 4, 3, 2, 1};
  cudf::test::strings_column_wrapper s0_sscol0{"5555", "4444", "333", "22", "1"};
  cudf::test::fixed_width_column_wrapper<float> s0_sscol1{50, 40, 30, 20, 10};
  cudf::test::lists_column_wrapper<int> s0_sscol2{{1, 2}, {3, 4}, {5}, {6, 7, 8}, {12, 12}};
  cudf::test::structs_column_wrapper s0_scol2({s0_sscol0, s0_sscol1, s0_sscol2});
  cudf::test::structs_column_wrapper s_col0({s0_scol0, s0_scol1, s0_scol2});

  auto all_valid = thrust::make_constant_iterator<bool>(true);

  cudf::test::strings_column_wrapper s1_scol0{"mno", "jkl", "ghi", "def", "abc"};
  cudf::test::fixed_width_column_wrapper<float> s1_scol1{5, 4, 3, 2, 1};
  cudf::test::strings_column_wrapper s1_sscol0{"5555", "4444", "333", "22", "1"};
  cudf::test::fixed_width_column_wrapper<float> s1_sscol1{50, 40, 30, 20, 10};
  cudf::test::lists_column_wrapper<int> s1_sscol2{{{1, 2}, {3, 4}, {5}, {6, 7, 8}, {12, 12}},
                                                  all_valid};
  cudf::test::structs_column_wrapper s1_scol2({s1_sscol0, s1_sscol1, s1_sscol2});
  cudf::test::structs_column_wrapper s_col1({s1_scol0, s1_scol1, s1_scol2});

  // equivalent, but not equal
  cudf::test::expect_column_properties_equivalent(s_col0, s_col1);
  EXPECT_EQ(cudf::test::expect_column_properties_equal(
              s_col0, s_col1, cudf::test::debug_output_level::QUIET),
            false);

  cudf::test::expect_column_properties_equal(s_col0, s_col0);
  cudf::test::expect_column_properties_equal(s_col1, s_col1);
}

TEST_F(ColumnUtilitiesStructsTest, Values)
{
  cudf::test::strings_column_wrapper s0_scol0{"mno", "jkl", "ghi", "def", "abc"};
  cudf::test::fixed_width_column_wrapper<float> s0_scol1{5, 4, 3, 2, 1};
  cudf::test::strings_column_wrapper s0_sscol0{"5555", "4444", "333", "22", "1"};
  cudf::test::fixed_width_column_wrapper<float> s0_sscol1{50, 40, 30, 20, 10};
  cudf::test::lists_column_wrapper<int> s0_sscol2{{1, 2}, {3, 4}, {5}, {6, 7, 8}, {12, 12}};
  cudf::test::structs_column_wrapper s0_scol2({s0_sscol0, s0_sscol1, s0_sscol2});
  cudf::test::structs_column_wrapper s_col0({s0_scol0, s0_scol1, s0_scol2});

  auto all_valid = thrust::make_constant_iterator<bool>(true);

  cudf::test::strings_column_wrapper s1_scol0{"mno", "jkl", "ghi", "def", "abc"};
  cudf::test::fixed_width_column_wrapper<float> s1_scol1{5, 4, 3, 2, 1};
  cudf::test::strings_column_wrapper s1_sscol0{"5555", "4444", "333", "22", "1"};
  cudf::test::fixed_width_column_wrapper<float> s1_sscol1{50, 40, 30, 20, 10};
  cudf::test::lists_column_wrapper<int> s1_sscol2{{{1, 2}, {3, 4}, {5}, {6, 7, 8}, {12, 12}},
                                                  all_valid};
  cudf::test::structs_column_wrapper s1_scol2({s1_sscol0, s1_sscol1, s1_sscol2});
  cudf::test::structs_column_wrapper s_col1({s1_scol0, s1_scol1, s1_scol2});

  // equivalent, but not equal
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(s_col0, s_col1);
  EXPECT_EQ(cudf::test::expect_columns_equal(s_col0, s_col1, cudf::test::debug_output_level::QUIET),
            false);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(s_col0, s_col0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(s_col1, s_col1);
}

CUDF_TEST_PROGRAM_MAIN()
