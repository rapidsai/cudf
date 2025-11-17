/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/stream_compaction.hpp>

namespace filters {

struct FilterTestFixture : public cudf::test::BaseFixture {
 protected:
  static constexpr char const* udf =
    R"***(
    template<typename T>
    __device__ void is_equal(bool * out, T a, T b) { *out = (a == b); }
    )***";
};

template <typename T>
struct FilterNumericTest : public FilterTestFixture {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(FilterNumericTest, NumericTypesNotBool);

TYPED_TEST(FilterNumericTest, NoAssertions)
{
  using T = TypeParam;

  auto a = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                     {1, 0, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto b = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 8, 5, 6, 7, 4, 9},
                                                     {0, 0, 1, 1, 1, 1, 1, 1, 0, 0}};

  auto expected = cudf::test::fixed_width_column_wrapper<T>{{2, 3, 5, 6, 7}, {1, 1, 1, 1, 1}};

  std::vector<std::unique_ptr<cudf::column>> results;

  EXPECT_NO_THROW(
    results = cudf::filter({a, b}, this->udf, {a}, false, std::nullopt, cudf::null_aware::NO));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results[0]->view());
}

template <typename T>
struct FilterChronoTest : public FilterTestFixture {};

TYPED_TEST_SUITE(FilterChronoTest, cudf::test::ChronoTypes);

TYPED_TEST(FilterChronoTest, NoAssertions)
{
  using T = TypeParam;

  auto a = cudf::test::fixed_width_column_wrapper<T>{
    {T{}, T{}, T{}, T{}, T{}, T{}, T{}, T{}, T{}, T{}}, {1, 0, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto b = cudf::test::fixed_width_column_wrapper<T>{
    {T{}, T{}, T{}, T{}, T{}, T{}, T{}, T{}, T{}, T{}}, {0, 0, 1, 1, 1, 1, 1, 1, 0, 0}};

  auto expected =
    cudf::test::fixed_width_column_wrapper<T>{{T{}, T{}, T{}, T{}, T{}, T{}}, {1, 1, 1, 1, 1, 1}};

  std::vector<std::unique_ptr<cudf::column>> results;
  EXPECT_NO_THROW(
    results = cudf::filter({a, b}, this->udf, {a}, false, std::nullopt, cudf::null_aware::NO));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results[0]->view());
}

template <typename T>
struct FilterFixedPointTest : public FilterTestFixture {};

TYPED_TEST_SUITE(FilterFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(FilterFixedPointTest, NoAssertions)
{
  using T = TypeParam;

  auto a = cudf::test::fixed_point_column_wrapper<typename T::rep>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 0, 1, 1, 1, 1, 1, 1, 0, 0}, numeric::scale_type{0}};
  auto b = cudf::test::fixed_point_column_wrapper<typename T::rep>{
    {0, 1, 2, 3, 8, 5, 6, 7, 4, 9}, {0, 0, 1, 1, 1, 1, 1, 1, 0, 0}, numeric::scale_type{0}};

  auto expected = cudf::test::fixed_point_column_wrapper<typename T::rep>{
    {2, 3, 5, 6, 7}, {1, 1, 1, 1, 1}, numeric::scale_type{0}};

  std::vector<std::unique_ptr<cudf::column>> results;

  EXPECT_NO_THROW(
    results = cudf::filter({a, b}, this->udf, {a}, false, std::nullopt, cudf::null_aware::NO));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results[0]->view());
}

TEST_F(FilterTestFixture, StringNoAssertions)
{
  auto a = cudf::test::strings_column_wrapper{{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"},
                                              {1, 0, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto b = cudf::test::strings_column_wrapper{{"0", "1", "2", "3", "8", "5", "6", "7", "8", "9"},
                                              {0, 0, 1, 1, 1, 1, 1, 1, 0, 0}};

  auto expected = cudf::test::strings_column_wrapper{{"2", "3", "5", "6", "7"}, {1, 1, 1, 1, 1}};

  std::vector<std::unique_ptr<cudf::column>> results;

  EXPECT_NO_THROW(
    results = cudf::filter({a, b}, this->udf, {a}, false, std::nullopt, cudf::null_aware::NO));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results[0]->view());
}

struct FilterAssertsTest : public FilterTestFixture {};

struct FilterTest : public FilterTestFixture {};

TEST_F(FilterTest, Basic)
{
  auto a = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                                           {1, 1, 1, 1, 1, 1, 1, 0, 0, 0});
  std::string cuda = R"***(
__device__ void is_even(bool* out, int32_t a) { *out = (a % 2 == 0); }
  )***";

  auto result   = cudf::filter({a}, cuda, {a}, false, std::nullopt, cudf::null_aware::NO);
  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>{{2, 4, 6}, {1, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result[0]->view());

  std::string null_cuda = R"***(
__device__ void is_even(bool* out, cuda::std::optional<int32_t> a) { *out = a.has_value() && (*a % 2 == 0); }
  )***";

  auto null_result = cudf::filter({a}, null_cuda, {a}, false, std::nullopt, cudf::null_aware::YES);
  auto null_expected = cudf::test::fixed_width_column_wrapper<int32_t>{{2, 4, 6}, {1, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(null_expected, null_result[0]->view());
}

TEST_F(FilterTest, ScalarFilter)
{
  auto a           = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto b           = cudf::test::fixed_width_column_wrapper<int32_t>{2};
  std::string cuda = R"***(
__device__ void is_divisible(bool* out, int32_t a, int32_t b) { *out = ((a % b) == 0); }
  )***";

  EXPECT_THROW(cudf::filter({a, b}, cuda, {a, b}, false, std::nullopt, cudf::null_aware::NO),
               std::invalid_argument);
}

TEST_F(FilterTest, MixedTypes)
{
  auto countries = cudf::test::strings_column_wrapper{
    "USA", "Canada", "Mexico", "Brazil", "Argentina", "France", "Germany", "Italy", "Spain"};
  auto average_tmp      = cudf::test::fixed_width_column_wrapper<float>{0, 0, 1, 1, 1, 1, 1, 1, 1};
  auto average_humidity = cudf::test::fixed_width_column_wrapper<float>{0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto timezones        = cudf::test::strings_column_wrapper{
    "EST", "EST", "CST", "BRT", "ART", "CET", "CET", "CET", "CET"};

  std::string cuda = R"***(
__device__ void filter(bool* out,
                       [[maybe_unused]] cudf::string_view country,
                       cudf::string_view tz,
                       float tmp,
                       float hum,
                       float min_tmp,
                       float max_tmp,
                       float min_hum,
                       float max_hum,
                       cudf::string_view tz1,
                       cudf::string_view tz2)
{
  *out = (tmp >= min_tmp && tmp <= max_tmp) && (hum >= min_hum && hum <= max_hum) &&
         (tz == tz1 || tz == tz2);
}
  )***";

  auto min_tmp   = cudf::test::fixed_width_column_wrapper<float>{0.5};
  auto max_tmp   = cudf::test::fixed_width_column_wrapper<float>{1};
  auto min_hum   = cudf::test::fixed_width_column_wrapper<float>{0.5};
  auto max_hum   = cudf::test::fixed_width_column_wrapper<float>{1};
  auto timezone1 = cudf::test::strings_column_wrapper{"CET"};
  auto timezone2 = cudf::test::strings_column_wrapper{"EST"};

  auto result = cudf::filter({countries,
                              timezones,
                              average_tmp,
                              average_humidity,
                              min_tmp,
                              max_tmp,
                              min_hum,
                              max_hum,
                              timezone1,
                              timezone2},
                             cuda,
                             {countries, timezones},
                             false,
                             std::nullopt,
                             cudf::null_aware::NO);

  EXPECT_EQ(result.size(), 2);

  auto expected_countries =
    cudf::test::strings_column_wrapper{"France", "Germany", "Italy", "Spain"};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_countries, result[0]->view());

  auto expected_timezones = cudf::test::strings_column_wrapper{"CET", "CET", "CET", "CET"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_timezones, result[1]->view());
}

TEST_F(FilterTest, NullableMixedTypes)
{
  auto countries = cudf::test::strings_column_wrapper(
    {"USA", "Canada", "Mexico", "Brazil", "Argentina", "France", "Germany", "Italy", "Spain"},
    {1, 1, 1, 1, 1, 0, 1, 0, 1});
  auto average_tmp      = cudf::test::fixed_width_column_wrapper<float>{0, 0, 1, 1, 1, 1, 1, 1, 1};
  auto average_humidity = cudf::test::fixed_width_column_wrapper<float>{0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto timezones        = cudf::test::strings_column_wrapper{
    "EST", "EST", "CST", "BRT", "ART", "CET", "CET", "CET", "CET"};

  std::string cuda = R"***(
__device__ void filter(bool* out,
                       [[maybe_unused]] cudf::string_view country,
                       cudf::string_view tz,
                       float tmp,
                       float hum,
                       float min_tmp,
                       float max_tmp,
                       float min_hum,
                       float max_hum,
                       cudf::string_view tz1,
                       cudf::string_view tz2)
{
  *out = (tmp >= min_tmp && tmp <= max_tmp) && (hum >= min_hum && hum <= max_hum) &&
         (tz == tz1 || tz == tz2);
}
)***";

  auto min_tmp   = cudf::test::fixed_width_column_wrapper<float>{0.5};
  auto max_tmp   = cudf::test::fixed_width_column_wrapper<float>{1};
  auto min_hum   = cudf::test::fixed_width_column_wrapper<float>{0.5};
  auto max_hum   = cudf::test::fixed_width_column_wrapper<float>{1};
  auto timezone1 = cudf::test::strings_column_wrapper{"CET"};
  auto timezone2 = cudf::test::strings_column_wrapper{"EST"};

  auto result = cudf::filter({countries,
                              timezones,
                              average_tmp,
                              average_humidity,
                              min_tmp,
                              max_tmp,
                              min_hum,
                              max_hum,
                              timezone1,
                              timezone2},
                             cuda,
                             {countries, timezones},
                             false,
                             std::nullopt,
                             cudf::null_aware::NO);

  auto expected_countries = cudf::test::strings_column_wrapper({"Germany", "Spain"}, {true, true});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_countries, result[0]->view());

  auto expected_timezones = cudf::test::strings_column_wrapper({"CET", "CET"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_timezones, result[1]->view());
}

}  // namespace filters

CUDF_TEST_PROGRAM_MAIN()
