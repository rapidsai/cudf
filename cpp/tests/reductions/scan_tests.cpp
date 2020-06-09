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

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <vector>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/reduction.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
using aggregation = cudf::aggregation;
using cudf::column_view;
using cudf::null_policy;
using cudf::scan_type;

void print_view(column_view const& view, const char* msg = nullptr)
{
  std::cout << msg << " {";
  cudf::test::print(view);
  std::cout << "}\n";
}

// This is the main test feature
template <typename T>
struct ScanTest : public cudf::test::BaseFixture {
  void scan_test(cudf::test::fixed_width_column_wrapper<T> const col_in,
                 cudf::test::fixed_width_column_wrapper<T> const expected_col_out,
                 std::unique_ptr<aggregation> const& agg,
                 scan_type inclusive)
  {
    bool do_print = false;

    auto int_values   = cudf::test::to_host<T>(col_in);
    auto exact_values = cudf::test::to_host<T>(expected_col_out);
    this->val_check(std::get<0>(int_values), do_print, "input = ");
    this->val_check(std::get<0>(exact_values), do_print, "exact = ");

    const column_view input_view = col_in;
    std::unique_ptr<cudf::column> col_out;

    CUDF_EXPECT_NO_THROW(col_out = cudf::scan(input_view, agg, inclusive));
    const column_view result_view = col_out->view();

    cudf::test::expect_column_properties_equal(input_view, result_view);
    cudf::test::expect_columns_equal(expected_col_out, result_view);

    auto host_result = cudf::test::to_host<T>(result_view);
    this->val_check(std::get<0>(host_result), do_print, "result = ");
  }

  template <typename Ti>
  void val_check(thrust::host_vector<Ti> const& v, bool do_print = false, const char* msg = nullptr)
  {
    if (do_print) {
      std::cout << msg << " {";
      std::for_each(v.begin(), v.end(), [](Ti i) { std::cout << ", " << i; });
      std::cout << "}" << std::endl;
    }
    range_check(v);
  }

  // make sure all elements in the range of sint8([-128, 127])
  template <typename Ti>
  void range_check(thrust::host_vector<Ti> const& v)
  {
    std::for_each(v.begin(), v.end(), [](Ti i) {
      ASSERT_GE(static_cast<int>(i), -128);
      ASSERT_LT(static_cast<int>(i), 128);
    });
  }
};

using Types = cudf::test::NumericTypes;

TYPED_TEST_CASE(ScanTest, Types);

// ------------------------------------------------------------------------

template <typename I, typename I2, typename O, typename ZipOp, typename BinOp>
void zip_scan(I first, I last, I2 first2, O output, ZipOp zipop, BinOp binop)
{
  // this could be implemented with a call to std::transform and then a
  // subsequent call to std::partial_sum but that you be less memory efficient
  if (first == last) return;
  auto acc = zipop(*first, *first2);
  *output  = acc;
  std::transform(std::next(first),
                 last,
                 std::next(first2),
                 std::next(output),
                 [&](auto const& e, auto const& mask) mutable {
                   acc = binop(acc, zipop(e, mask));
                   return acc;
                 });
}

template <typename T>
struct value_or {
  T _or;
  explicit value_or(T value) : _or{value} {}
  T operator()(T const& value, bool mask) { return mask ? value : _or; }
};

TYPED_TEST(ScanTest, Min)
{
  auto const v =
    cudf::test::make_type_param_vector<TypeParam>({123, 64, 63, 99, -5, 123, -16, -120, -111});
  auto const b = std::vector<bool>{1, 0, 1, 1, 1, 1, 0, 0, 1};
  std::vector<TypeParam> exact(v.size());

  std::partial_sum(
    v.cbegin(), v.cend(), exact.begin(), [](auto a, auto b) { return std::min(a, b); });

  this->scan_test(cudf::test::fixed_width_column_wrapper<TypeParam>(v.begin(), v.end()),
                  cudf::test::fixed_width_column_wrapper<TypeParam>(exact.begin(), exact.end()),
                  cudf::make_min_aggregation(),
                  scan_type::INCLUSIVE);

  zip_scan(v.cbegin(),
           v.cend(),
           b.cbegin(),
           exact.begin(),
           value_or<TypeParam>{std::numeric_limits<TypeParam>::max()},
           [](auto a, auto b) { return std::min(a, b); });

  this->scan_test(
    cudf::test::fixed_width_column_wrapper<TypeParam>(v.begin(), v.end(), b.begin()),
    cudf::test::fixed_width_column_wrapper<TypeParam>(exact.begin(), exact.end(), b.begin()),
    cudf::make_min_aggregation(),
    scan_type::INCLUSIVE);
}

TYPED_TEST(ScanTest, Max)
{
  auto const v =
    cudf::test::make_type_param_vector<TypeParam>({-120, 5, 0, -120, -111, 64, 63, 99, 123, -16});
  auto const b = std::vector<bool>{1, 0, 1, 1, 1, 1, 0, 1, 0, 1};
  std::vector<TypeParam> exact(v.size());

  std::partial_sum(
    v.cbegin(), v.cend(), exact.begin(), [](auto a, auto b) { return std::max(a, b); });

  this->scan_test(cudf::test::fixed_width_column_wrapper<TypeParam>(v.begin(), v.end()),
                  cudf::test::fixed_width_column_wrapper<TypeParam>(exact.begin(), exact.end()),
                  cudf::make_max_aggregation(),
                  scan_type::INCLUSIVE);

  zip_scan(v.cbegin(),
           v.cend(),
           b.cbegin(),
           exact.begin(),
           value_or<TypeParam>{std::numeric_limits<TypeParam>::lowest()},
           [](auto a, auto b) { return std::max(a, b); });

  this->scan_test(
    cudf::test::fixed_width_column_wrapper<TypeParam>(v.begin(), v.end(), b.begin()),
    cudf::test::fixed_width_column_wrapper<TypeParam>(exact.begin(), exact.end(), b.begin()),
    cudf::make_max_aggregation(),
    scan_type::INCLUSIVE);
}

TYPED_TEST(ScanTest, Product)
{
  auto const v = cudf::test::make_type_param_vector<TypeParam>({5, -1, 1, 3, -2, 4});
  auto const b = std::vector<bool>{1, 1, 1, 0, 1, 1};
  std::vector<TypeParam> exact(v.size());

  std::partial_sum(v.cbegin(), v.cend(), exact.begin(), std::multiplies<TypeParam>{});

  this->scan_test(cudf::test::fixed_width_column_wrapper<TypeParam>(v.begin(), v.end()),
                  cudf::test::fixed_width_column_wrapper<TypeParam>(exact.begin(), exact.end()),
                  cudf::make_product_aggregation(),
                  scan_type::INCLUSIVE);

  zip_scan(v.cbegin(),
           v.cend(),
           b.cbegin(),
           exact.begin(),
           value_or<TypeParam>{1},
           std::multiplies<TypeParam>{});

  this->scan_test(
    cudf::test::fixed_width_column_wrapper<TypeParam>(v.begin(), v.end(), b.begin()),
    cudf::test::fixed_width_column_wrapper<TypeParam>(exact.begin(), exact.end(), b.begin()),
    cudf::make_product_aggregation(),
    scan_type::INCLUSIVE);
}

TYPED_TEST(ScanTest, Sum)
{
  auto const v = [] {
    if (std::is_signed<TypeParam>::value)
      return cudf::test::make_type_param_vector<TypeParam>(
        {-120, 5, 6, 113, -111, 64, -63, 9, 34, -16});
    return cudf::test::make_type_param_vector<TypeParam>({12, 5, 6, 13, 11, 14, 3, 9, 34, 16});
  }();
  auto const b = std::vector<bool>{1, 0, 1, 1, 0, 0, 1, 1, 1, 1};
  std::vector<TypeParam> exact(v.size());

  std::partial_sum(v.cbegin(), v.cend(), exact.begin(), std::plus<TypeParam>{});

  this->scan_test(cudf::test::fixed_width_column_wrapper<TypeParam>(v.begin(), v.end()),
                  cudf::test::fixed_width_column_wrapper<TypeParam>(exact.begin(), exact.end()),
                  cudf::make_sum_aggregation(),
                  scan_type::INCLUSIVE);

  zip_scan(v.cbegin(),
           v.cend(),
           b.cbegin(),
           exact.begin(),
           value_or<TypeParam>{0},
           std::plus<TypeParam>{});

  this->scan_test(
    cudf::test::fixed_width_column_wrapper<TypeParam>(v.begin(), v.end(), b.begin()),
    cudf::test::fixed_width_column_wrapper<TypeParam>(exact.begin(), exact.end(), b.begin()),
    cudf::make_sum_aggregation(),
    scan_type::INCLUSIVE);
}

struct ScanStringTest : public cudf::test::BaseFixture {
  void scan_test(cudf::test::strings_column_wrapper const& col_in,
                 cudf::test::strings_column_wrapper const& expected_col_out,
                 std::unique_ptr<aggregation> const& agg,
                 scan_type inclusive)
  {
    bool do_print = false;
    if (do_print) {
      std::cout << "input = {";
      cudf::test::print(col_in);
      std::cout << "}\n";
      std::cout << "expect = {";
      cudf::test::print(expected_col_out);
      std::cout << "}\n";
    }

    const column_view input_view = col_in;
    std::unique_ptr<cudf::column> col_out;

    CUDF_EXPECT_NO_THROW(col_out = cudf::scan(input_view, agg, inclusive));
    const column_view result_view = col_out->view();

    cudf::test::expect_column_properties_equal(input_view, result_view);
    cudf::test::expect_columns_equal(expected_col_out, result_view);

    if (do_print) {
      std::cout << "result = {";
      cudf::test::print(result_view);
      std::cout << "}\n";
    }
  }
};

TEST_F(ScanStringTest, Min)
{
  std::vector<std::string> v = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
  std::vector<bool> b = {1, 0, 1, 1, 0, 0, 1, 0, 1};
  std::vector<std::string> exact(v.size());

  std::partial_sum(v.cbegin(), v.cend(), exact.begin(), [](auto const& a, auto const& b) {
    return std::min(a, b);
  });

  // string column without nulls
  cudf::test::strings_column_wrapper col_nonulls(v.begin(), v.end());
  cudf::test::strings_column_wrapper expected1(exact.begin(), exact.end());
  this->scan_test(col_nonulls, expected1, cudf::make_min_aggregation(), scan_type::INCLUSIVE);

  auto const STRING_MAX = std::string("\xF7\xBF\xBF\xBF");

  zip_scan(v.cbegin(),
           v.cend(),
           b.cbegin(),
           exact.begin(),
           value_or<std::string>{STRING_MAX},
           [](auto const& a, auto const& b) { return std::min(a, b); });

  // string column with nulls
  cudf::test::strings_column_wrapper col_nulls(v.begin(), v.end(), b.begin());
  cudf::test::strings_column_wrapper expected2(exact.begin(), exact.end(), b.begin());
  this->scan_test(col_nulls, expected2, cudf::make_min_aggregation(), scan_type::INCLUSIVE);
}

TEST_F(ScanStringTest, Max)
{
  std::vector<std::string> v = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
  std::vector<bool> b = {1, 0, 1, 1, 0, 0, 1, 1, 1};
  std::vector<std::string> exact(v.size());

  std::partial_sum(v.cbegin(), v.cend(), exact.begin(), [](auto const& a, auto const& b) {
    return std::max(a, b);
  });

  // string column without nulls
  cudf::test::strings_column_wrapper col_nonulls(v.begin(), v.end());
  cudf::test::strings_column_wrapper expected1(exact.begin(), exact.end());
  this->scan_test(col_nonulls, expected1, cudf::make_max_aggregation(), scan_type::INCLUSIVE);

  auto const STRING_MIN = std::string{};

  zip_scan(v.cbegin(),
           v.cend(),
           b.cbegin(),
           exact.begin(),
           value_or<std::string>{STRING_MIN},
           [](auto const& a, auto const& b) { return std::max(a, b); });

  // string column with nulls
  cudf::test::strings_column_wrapper col_nulls(v.begin(), v.end(), b.begin());
  cudf::test::strings_column_wrapper expected2(exact.begin(), exact.end(), b.begin());
  this->scan_test(col_nulls, expected2, cudf::make_max_aggregation(), scan_type::INCLUSIVE);
}

TYPED_TEST(ScanTest, skip_nulls)
{
  bool do_print = false;
  auto const v  = cudf::test::make_type_param_vector<TypeParam>({1, 2, 3, 4, 5, 6, 7, 8, 1, 1});
  auto const b  = std::vector<bool>{1, 1, 1, 1, 1, 0, 1, 0, 1, 1};
  cudf::test::fixed_width_column_wrapper<TypeParam> const col_in(v.begin(), v.end(), b.begin());
  const column_view input_view = col_in;
  std::unique_ptr<cudf::column> col_out;

  // test output calculation
  std::vector<TypeParam> out_v(input_view.size());
  std::vector<bool> out_b(input_view.size());

  zip_scan(v.cbegin(),
           v.cend(),
           b.cbegin(),
           out_v.begin(),
           value_or<TypeParam>{0},
           std::plus<TypeParam>{});

  std::partial_sum(b.cbegin(), b.cend(), out_b.begin(), std::logical_and<bool>{});

  // skipna=true (default)
  CUDF_EXPECT_NO_THROW(
    col_out = cudf::scan(
      input_view, cudf::make_sum_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE));
  cudf::test::fixed_width_column_wrapper<TypeParam> expected_col_out1(
    out_v.begin(), out_v.end(), b.cbegin());
  cudf::test::expect_column_properties_equal(expected_col_out1, col_out->view());
  cudf::test::expect_columns_equal(expected_col_out1, col_out->view());
  if (do_print) {
    print_view(expected_col_out1, "expect = ");
    print_view(col_out->view(), "result = ");
  }

  // skipna=false
  CUDF_EXPECT_NO_THROW(
    col_out = cudf::scan(
      input_view, cudf::make_sum_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE));
  cudf::test::fixed_width_column_wrapper<TypeParam> expected_col_out2(
    out_v.begin(), out_v.end(), out_b.begin());
  if (do_print) {
    print_view(expected_col_out2, "expect = ");
    print_view(col_out->view(), "result = ");
  }
  cudf::test::expect_column_properties_equal(expected_col_out2, col_out->view());
  cudf::test::expect_columns_equal(expected_col_out2, col_out->view());
}

TEST_F(ScanStringTest, skip_nulls)
{
  bool do_print = false;
  // data and valid arrays
  std::vector<std::string> v(
    {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"});
  std::vector<bool> b({1, 1, 1, 0, 0, 0, 1, 1, 1});
  std::vector<std::string> exact(v.size());
  std::vector<bool> out_b(v.size());

  auto const STRING_MIN = std::string(1, char(0));

  zip_scan(v.cbegin(),
           v.cend(),
           b.cbegin(),
           exact.begin(),
           value_or<std::string>{STRING_MIN},
           [](auto const& a, auto const& b) { return std::max(a, b); });

  std::partial_sum(b.cbegin(), b.cend(), out_b.begin(), std::logical_and<bool>{});

  // string column with nulls
  cudf::test::strings_column_wrapper col_nulls(v.begin(), v.end(), b.begin());
  cudf::test::strings_column_wrapper expected2(exact.begin(), exact.end(), out_b.begin());
  std::unique_ptr<cudf::column> col_out;
  // skipna=false
  CUDF_EXPECT_NO_THROW(
    col_out = cudf::scan(
      col_nulls, cudf::make_max_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE));
  if (do_print) {
    print_view(expected2, "expect = ");
    print_view(col_out->view(), "result = ");
  }
  cudf::test::expect_column_properties_equal(expected2, col_out->view());
  cudf::test::expect_columns_equal(expected2, col_out->view());

  // Exclusive scan string not supported.
  CUDF_EXPECT_THROW_MESSAGE(
    (cudf::scan(
      col_nulls, cudf::make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE)),
    "String types supports only inclusive min/max for `cudf::scan`");

  CUDF_EXPECT_THROW_MESSAGE(
    (cudf::scan(
      col_nulls, cudf::make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE)),
    "String types supports only inclusive min/max for `cudf::scan`");
}

TYPED_TEST(ScanTest, EmptyColumnskip_nulls)
{
  bool do_print = false;
  std::vector<TypeParam> v{};
  std::vector<bool> b{};
  cudf::test::fixed_width_column_wrapper<TypeParam> const col_in(v.begin(), v.end(), b.begin());
  std::unique_ptr<cudf::column> col_out;

  // test output calculation
  std::vector<TypeParam> out_v(v.size());
  std::vector<bool> out_b(v.size());

  // skipna=true (default)
  CUDF_EXPECT_NO_THROW(
    col_out =
      cudf::scan(col_in, cudf::make_sum_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE));
  cudf::test::fixed_width_column_wrapper<TypeParam> expected_col_out1(
    out_v.begin(), out_v.end(), b.cbegin());
  cudf::test::expect_column_properties_equal(expected_col_out1, col_out->view());
  cudf::test::expect_columns_equal(expected_col_out1, col_out->view());
  if (do_print) {
    print_view(expected_col_out1, "expect = ");
    print_view(col_out->view(), "result = ");
  }

  // skipna=false
  CUDF_EXPECT_NO_THROW(
    col_out =
      cudf::scan(col_in, cudf::make_sum_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE));
  cudf::test::fixed_width_column_wrapper<TypeParam> expected_col_out2(
    out_v.begin(), out_v.end(), out_b.begin());
  if (do_print) {
    print_view(expected_col_out2, "expect = ");
    print_view(col_out->view(), "result = ");
  }
  cudf::test::expect_column_properties_equal(expected_col_out2, col_out->view());
  cudf::test::expect_columns_equal(expected_col_out2, col_out->view());
}
