/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/reduction.hpp>

#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>
#include "cudf/strings/string_view.hpp"

using aggregation = cudf::aggregation;
using cudf::column_view;
using cudf::null_policy;
using cudf::scan_type;

template <typename T>
struct TypeParam_to_host_type {
  using type = T;
};

template <>
struct TypeParam_to_host_type<cudf::string_view> {
  using type = std::string;
};

template <typename TypeParam, typename T>
typename std::enable_if<std::is_same_v<TypeParam, cudf::string_view>,
                        thrust::host_vector<std::string>>::type
make_vector(std::initializer_list<T> const& init)
{
  return cudf::test::make_type_param_vector<std::string, T>(init);
}

template <typename TypeParam, typename T>
typename std::enable_if<not std::is_same_v<TypeParam, cudf::string_view>,
                        thrust::host_vector<TypeParam>>::type
make_vector(std::initializer_list<T> const& init)
{
  return cudf::test::make_type_param_vector<TypeParam, T>(init);
}

// This is the main test feature
template <typename T>
struct ScanTest : public cudf::test::BaseFixture {
  typedef typename TypeParam_to_host_type<T>::type HostType;

  void scan_test(cudf::host_span<HostType const> v,
                 cudf::host_span<bool const> b,
                 std::unique_ptr<aggregation> const& agg,
                 scan_type inclusive,
                 null_policy null_handling = null_policy::EXCLUDE)
  {
    bool const do_print = false;

    auto col_in           = this->make_column(v, b);
    auto expected_col_out = this->make_expected(v, b, agg, inclusive, null_handling);
    auto col_out          = cudf::scan(*col_in, agg, inclusive, null_handling);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_col_out, *col_out);

    if constexpr (do_print) {
      auto int_values      = cudf::test::to_host<T>(*col_in);
      auto expected_values = cudf::test::to_host<T>(*expected_col_out);
      auto host_result     = cudf::test::to_host<T>(*col_out);
      this->print(std::get<0>(int_values), "input = ");
      this->print(std::get<0>(expected_values), "expected = ");
      this->print(std::get<0>(host_result), "result = ");
    }
  }

  std::unique_ptr<cudf::column> make_column(cudf::host_span<HostType const> v,
                                            cudf::host_span<bool const> b = {})
  {
    if constexpr (std::is_same_v<T, cudf::string_view>) {
      auto col_in = (b.size() > 0)
                      ? cudf::test::strings_column_wrapper(v.begin(), v.end(), b.begin())
                      : cudf::test::strings_column_wrapper(v.begin(), v.end());
      return col_in.release();
    } else {
      auto col_in = (b.size() > 0)
                      ? cudf::test::fixed_width_column_wrapper<T>(v.begin(), v.end(), b.begin())
                      : cudf::test::fixed_width_column_wrapper<T>(v.begin(), v.end());
      return col_in.release();
    }
  }

  std::function<HostType(HostType, HostType)> make_agg(std::unique_ptr<aggregation> const& agg)
  {
    if constexpr (std::is_same_v<T, cudf::string_view>) {
      switch (agg->kind) {
        case cudf::aggregation::MIN: return [](HostType a, HostType b) { return std::min(a, b); };
        case cudf::aggregation::MAX: return [](HostType a, HostType b) { return std::max(a, b); };
        default: {
          CUDF_FAIL("Unsupported aggregation");
          return [](HostType a, HostType b) { return std::min(a, b); };
        }
      }
    } else {
      switch (agg->kind) {
        case cudf::aggregation::SUM: return std::plus<T>{};
        case cudf::aggregation::PRODUCT: return std::multiplies<T>{};
        case cudf::aggregation::MIN: return [](T a, T b) { return std::min(a, b); };
        case cudf::aggregation::MAX: return [](T a, T b) { return std::max(a, b); };
        default: {
          CUDF_FAIL("Unsupported aggregation");
          return [](HostType a, HostType b) { return std::min(a, b); };
        }
      }
    }
  }

  HostType make_identity(std::unique_ptr<aggregation> const& agg)
  {
    if constexpr (std::is_same_v<T, cudf::string_view>) {
      switch (agg->kind) {
        case cudf::aggregation::MIN: return std::string{"\xF7\xBF\xBF\xBF"};
        case cudf::aggregation::MAX: return std::string{};
        default: {
          CUDF_FAIL("Unsupported aggregation");
          return HostType{};
        }
      }
    } else {
      switch (agg->kind) {
        case cudf::aggregation::SUM: return HostType{0};
        case cudf::aggregation::PRODUCT: return HostType{1};
        case cudf::aggregation::MIN: return std::numeric_limits<HostType>::max();
        case cudf::aggregation::MAX: return std::numeric_limits<HostType>::lowest();
        default: {
          CUDF_FAIL("Unsupported aggregation");
          return HostType{};
        }
      }
    }
  }

  std::unique_ptr<cudf::column> make_expected(cudf::host_span<HostType const> v,
                                              cudf::host_span<bool const> b,
                                              std::unique_ptr<aggregation> const& agg,
                                              scan_type inclusive,
                                              null_policy null_handling)
  {
    auto op       = this->make_agg(agg);
    auto identity = this->make_identity(agg);

    thrust::host_vector<HostType> expected(v.size());
    thrust::host_vector<bool> b_out(b.begin(), b.end());

    bool const nullable = (b.size() > 0);

    auto masked_value = [identity](auto const& z) {
      return thrust::get<1>(z) ? thrust::get<0>(z) : identity;
    };

    if (inclusive == cudf::scan_type::INCLUSIVE) {
      if (nullable) {
        std::transform_inclusive_scan(
          thrust::make_zip_iterator(thrust::make_tuple(v.begin(), b.begin())),
          thrust::make_zip_iterator(thrust::make_tuple(v.end(), b.end())),
          expected.begin(),
          op,
          masked_value);

        if (null_handling == null_policy::INCLUDE) {
          std::inclusive_scan(b.begin(), b.end(), b_out.begin(), std::logical_and<bool>{});
        }
      } else {
        std::inclusive_scan(v.begin(), v.end(), expected.begin(), op);
      }
    } else {
      if (nullable) {
        std::transform_exclusive_scan(
          thrust::make_zip_iterator(thrust::make_tuple(v.begin(), b.begin())),
          thrust::make_zip_iterator(thrust::make_tuple(v.end(), b.end())),
          expected.begin(),
          identity,
          op,
          masked_value);

        if (null_handling == null_policy::INCLUDE) {
          std::exclusive_scan(b.begin(), b.end(), b_out.begin(), true, std::logical_and<bool>{});
        }
      } else {
        std::exclusive_scan(v.begin(), v.end(), expected.begin(), identity, op);
      }
    }

    return nullable ? this->make_column(expected, b_out) : this->make_column(expected);
  }

  template <typename Ti>
  void print(thrust::host_vector<Ti> const& v, const char* msg = nullptr)
  {
    std::cout << msg << " {";
    std::for_each(v.begin(), v.end(), [](Ti i) { std::cout << ", " << i; });
    std::cout << "}" << std::endl;
  }
};

using Types = cudf::test::Concat<cudf::test::NumericTypes, cudf::test::Types<cudf::string_view>>;

TYPED_TEST_CASE(ScanTest, Types);

TYPED_TEST(ScanTest, Min)
{
  auto const v = make_vector<TypeParam>({123, 64, 63, 99, -5, 123, -16, -120, -111});
  auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 0, 1, 1, 1, 1, 0, 0, 1});

  // no nulls
  this->scan_test(v, {}, cudf::make_min_aggregation(), scan_type::INCLUSIVE);
  // skipna = true (default)
  this->scan_test(v, b, cudf::make_min_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v, b, cudf::make_min_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);

  // strings only support inclusive scan
  if constexpr (not std::is_same_v<TypeParam, cudf::string_view>) {
    // no nulls
    this->scan_test(v, {}, cudf::make_min_aggregation(), scan_type::EXCLUSIVE);
    // skipna = true (default)
    this->scan_test(v, b, cudf::make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);
    // skipna = false
    this->scan_test(v, b, cudf::make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE);
  }
}

TYPED_TEST(ScanTest, Max)
{
  auto const v = make_vector<TypeParam>({-120, 5, 0, -120, -111, 64, 63, 99, 123, -16});
  auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 0, 1, 1, 1, 1, 0, 1, 0, 1});

  // no nulls
  this->scan_test(v, {}, cudf::make_max_aggregation(), scan_type::INCLUSIVE);
  // skipna = true (default)
  this->scan_test(v, b, cudf::make_max_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v, b, cudf::make_max_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);

  // strings only support inclusive scan
  if constexpr (not std::is_same_v<TypeParam, cudf::string_view>) {
    // no nulls
    this->scan_test(v, {}, cudf::make_max_aggregation(), scan_type::EXCLUSIVE);
    // skipna = true (default)
    this->scan_test(v, b, cudf::make_max_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);
    // skipna = false
    this->scan_test(v, b, cudf::make_max_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE);
  }
}

TYPED_TEST(ScanTest, Product)
{
  if constexpr (not std::is_same_v<cudf::string_view, TypeParam>) {
    auto const v = make_vector<TypeParam>({5, -1, 1, 3, -2, 4});
    auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 1, 1, 0, 1, 1});

    // no nulls
    this->scan_test(v, {}, cudf::make_product_aggregation(), scan_type::INCLUSIVE);
    this->scan_test(v, {}, cudf::make_product_aggregation(), scan_type::EXCLUSIVE);
    // skipna = true (default)
    this->scan_test(
      v, b, cudf::make_product_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);
    this->scan_test(
      v, b, cudf::make_product_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);
    // skipna = false
    this->scan_test(
      v, b, cudf::make_product_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
    this->scan_test(
      v, b, cudf::make_product_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE);
  }
}

TYPED_TEST(ScanTest, Sum)
{
  if constexpr (not std::is_same_v<cudf::string_view, TypeParam>) {
    auto const v = [] {
      if (std::is_signed<TypeParam>::value)
        return make_vector<TypeParam>({-120, 5, 6, 113, -111, 64, -63, 9, 34, -16});
      return make_vector<TypeParam>({12, 5, 6, 13, 11, 14, 3, 9, 34, 16});
    }();
    auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 0, 1, 1, 0, 0, 1, 1, 1, 1});

    // no nulls
    this->scan_test(v, {}, cudf::make_sum_aggregation(), scan_type::INCLUSIVE);
    // skipna = true (default)
    this->scan_test(v, b, cudf::make_sum_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);
    // skipna = false
    this->scan_test(v, b, cudf::make_sum_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);

    // no nulls
    this->scan_test(v, {}, cudf::make_sum_aggregation(), scan_type::EXCLUSIVE);
    // skipna = true (default)
    this->scan_test(v, b, cudf::make_sum_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);
    // skipna = false
    this->scan_test(v, b, cudf::make_sum_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE);
  }
}

TYPED_TEST(ScanTest, EmptyColumn)
{
  auto const v = thrust::host_vector<typename TypeParam_to_host_type<TypeParam>::type>{};
  auto const b = thrust::host_vector<bool>{};

  // skipna = true (default)
  this->scan_test(v, b, cudf::make_min_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v, b, cudf::make_min_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);

  if constexpr (not std::is_same_v<cudf::string_view, TypeParam>) {
    // skipna = true (default)
    this->scan_test(v, b, cudf::make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);
    // skipna = false
    this->scan_test(v, b, cudf::make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE);
  }
}

TYPED_TEST(ScanTest, LeadingNulls)
{
  auto const v = make_vector<TypeParam>({100, 200, 300});
  auto const b = thrust::host_vector<bool>(std::vector<bool>{0, 1, 1});

  // skipna = true (default)
  this->scan_test(v, b, cudf::make_min_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v, b, cudf::make_min_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);

  if constexpr (not std::is_same_v<cudf::string_view, TypeParam>) {
    // skipna = true (default)
    this->scan_test(v, b, cudf::make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);
    // skipna = false
    this->scan_test(v, b, cudf::make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE);
  }
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

    CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(input_view, result_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col_out, result_view);

    if (do_print) {
      std::cout << "result = {";
      cudf::test::print(result_view);
      std::cout << "}\n";
    }
  }
};

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, FixedPointScanSum)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3}) {
    auto const scale    = scale_type{i};
    auto const column   = fp_wrapper{{1, 2, 3, 4}, scale};
    auto const expected = fp_wrapper{{1, 3, 6, 10}, scale};
    auto const result   = cudf::scan(column, cudf::make_sum_aggregation(), scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);

    auto const with_nulls     = fp_wrapper({1, 2, 3, 0, 4, 0}, {1, 1, 1, 0, 1, 0}, scale);
    auto const expected_nulls = fp_wrapper({1, 3, 6, 0, 10, 0}, {1, 1, 1, 0, 1, 0}, scale);
    auto const result_nulls =
      cudf::scan(with_nulls, cudf::make_sum_aggregation(), scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_nulls->view(), expected_nulls);
  }
}

TYPED_TEST(FixedPointTestBothReps, FixedPointPreScanSum)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3}) {
    auto const scale    = scale_type{i};
    auto const column   = fp_wrapper{{1, 2, 3, 4}, scale};
    auto const expected = fp_wrapper{{0, 1, 3, 6}, scale};
    auto const result   = cudf::scan(column, cudf::make_sum_aggregation(), scan_type::EXCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);

    auto const with_nulls     = fp_wrapper({0, 1, 2, 3, 0, 4}, {0, 1, 1, 1, 0, 1}, scale);
    auto const expected_nulls = fp_wrapper({0, 0, 1, 3, 0, 6}, {0, 1, 1, 1, 0, 1}, scale);
    auto const result_nulls =
      cudf::scan(with_nulls, cudf::make_sum_aggregation(), scan_type::EXCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_nulls->view(), expected_nulls);
  }
}

TYPED_TEST(FixedPointTestBothReps, FixedPointScanProduct)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const scale  = scale_type{0};
  auto const column = fp_wrapper{{1, 2, 3, 4}, scale};
  EXPECT_THROW(cudf::scan(column, cudf::make_product_aggregation(), scan_type::INCLUSIVE),
               cudf::logic_error);
}

TYPED_TEST(FixedPointTestBothReps, FixedPointScanMin)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3}) {
    auto const scale    = scale_type{i};
    auto const column   = fp_wrapper{{1, 2, 3, 4}, scale};
    auto const expected = fp_wrapper{{1, 1, 1, 1}, scale};
    auto const result   = cudf::scan(column, cudf::make_min_aggregation(), scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);

    auto const with_nulls     = fp_wrapper({1, 0, 2, 0, 3, 4}, {1, 0, 1, 0, 1, 1}, scale);
    auto const expected_nulls = fp_wrapper({1, 0, 1, 0, 1, 1}, {1, 0, 1, 0, 1, 1}, scale);
    auto const result_nulls =
      cudf::scan(with_nulls, cudf::make_min_aggregation(), scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_nulls->view(), expected_nulls);
  }
}

TYPED_TEST(FixedPointTestBothReps, FixedPointScanMax)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3}) {
    auto const scale  = scale_type{i};
    auto const column = fp_wrapper{{1, 2, 3, 4}, scale};
    auto const result = cudf::scan(column, cudf::make_max_aggregation(), scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), column);

    auto const with_nulls = fp_wrapper({1, 0, 0, 2, 3, 4}, {1, 0, 0, 1, 1, 1}, scale);
    auto const result_nulls =
      cudf::scan(with_nulls, cudf::make_max_aggregation(), scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_nulls->view(), with_nulls);
  }
}
