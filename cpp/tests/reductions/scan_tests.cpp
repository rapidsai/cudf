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
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

using aggregation = cudf::aggregation;
using cudf::column_view;
using cudf::null_policy;
using cudf::scan_type;
using namespace cudf::test::iterators;

namespace cudf {
namespace test {

template <typename T>
struct TypeParam_to_host_type {
  using type = T;
};

template <>
struct TypeParam_to_host_type<string_view> {
  using type = std::string;
};

template <>
struct TypeParam_to_host_type<numeric::decimal32> {
  using type = numeric::decimal32::rep;
};

template <>
struct TypeParam_to_host_type<numeric::decimal64> {
  using type = numeric::decimal64::rep;
};

template <typename TypeParam, typename T>
typename std::enable_if<std::is_same_v<TypeParam, string_view>,
                        thrust::host_vector<std::string>>::type
make_vector(std::initializer_list<T> const& init)
{
  return make_type_param_vector<std::string, T>(init);
}

template <typename TypeParam, typename T>
typename std::enable_if<is_fixed_point<TypeParam>(),
                        thrust::host_vector<typename TypeParam::rep>>::type
make_vector(std::initializer_list<T> const& init)
{
  return make_type_param_vector<typename TypeParam::rep, T>(init);
}

template <typename TypeParam, typename T>
typename std::enable_if<not(std::is_same_v<TypeParam, string_view> || is_fixed_point<TypeParam>()),
                        thrust::host_vector<TypeParam>>::type
make_vector(std::initializer_list<T> const& init)
{
  return make_type_param_vector<TypeParam, T>(init);
}

// This is the main test feature
template <typename T>
struct ScanTest : public BaseFixture {
  typedef typename TypeParam_to_host_type<T>::type HostType;

  void scan_test(host_span<HostType const> v,
                 host_span<bool const> b,
                 std::unique_ptr<aggregation> const& agg,
                 scan_type inclusive,
                 null_policy null_handling,
                 numeric::scale_type scale)
  {
    bool const do_print = false;  // set true for debugging

    auto col_in = this->make_column(v, b, scale);
    std::unique_ptr<column> col_out;
    std::unique_ptr<column> expected_col_out;

    if (not this->params_supported(agg, inclusive)) {
      EXPECT_THROW(scan(*col_in, agg, inclusive, null_handling), logic_error);
    } else {
      expected_col_out = this->make_expected(v, b, agg, inclusive, null_handling, scale);
      col_out          = scan(*col_in, agg, inclusive, null_handling);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_col_out, *col_out);

      if constexpr (do_print) {
        std::cout << "input = ";
        print(*col_in);
        std::cout << "expected = ";
        print(*expected_col_out);
        std::cout << "result = ";
        print(*col_out);
        std::cout << std::endl;
      }
    }
  }

  // Overload to iterate the test over a few different scales for fixed-point tests
  void scan_test(host_span<HostType const> v,
                 host_span<bool const> b,
                 std::unique_ptr<aggregation> const& agg,
                 scan_type inclusive,
                 null_policy null_handling = null_policy::EXCLUDE)
  {
    if constexpr (is_fixed_point<T>()) {
      for (auto scale : {0, -1, -2, -3}) {
        scan_test(v, b, agg, inclusive, null_handling, numeric::scale_type{scale});
      }
    } else {
      scan_test(v, b, agg, inclusive, null_handling, numeric::scale_type{0});
    }
  }

  bool params_supported(std::unique_ptr<aggregation> const& agg, scan_type inclusive)
  {
    if constexpr (std::is_same_v<T, string_view>) {
      bool supported_agg = (agg->kind == aggregation::MIN || agg->kind == aggregation::MAX ||
                            agg->kind == aggregation::RANK || agg->kind == aggregation::DENSE_RANK);
      return supported_agg && (inclusive == scan_type::INCLUSIVE);
    } else if constexpr (is_fixed_point<T>()) {
      bool supported_agg = (agg->kind == aggregation::MIN || agg->kind == aggregation::MAX ||
                            agg->kind == aggregation::SUM || agg->kind == aggregation::RANK ||
                            agg->kind == aggregation::DENSE_RANK);
      return supported_agg;
    } else if constexpr (std::is_arithmetic<T>()) {
      bool supported_agg = (agg->kind == aggregation::MIN || agg->kind == aggregation::MAX ||
                            agg->kind == aggregation::SUM || agg->kind == aggregation::PRODUCT ||
                            agg->kind == aggregation::RANK || agg->kind == aggregation::DENSE_RANK);
      return supported_agg;
    } else {
      return false;
    }
  }

  std::unique_ptr<column> make_column(host_span<HostType const> v,
                                      host_span<bool const> b   = {},
                                      numeric::scale_type scale = numeric::scale_type{0})
  {
    if constexpr (std::is_same_v<T, string_view>) {
      auto col = (b.size() > 0) ? strings_column_wrapper(v.begin(), v.end(), b.begin())
                                : strings_column_wrapper(v.begin(), v.end());
      return col.release();
    } else if constexpr (is_fixed_point<T>()) {
      auto col =
        (b.size() > 0)
          ? fixed_point_column_wrapper<typename T::rep>(v.begin(), v.end(), b.begin(), scale)
          : fixed_point_column_wrapper<typename T::rep>(v.begin(), v.end(), scale);
      return col.release();
    } else {
      auto col = (b.size() > 0) ? fixed_width_column_wrapper<T>(v.begin(), v.end(), b.begin())
                                : fixed_width_column_wrapper<T>(v.begin(), v.end());
      return col.release();
    }
  }

  std::function<HostType(HostType, HostType)> make_agg(std::unique_ptr<aggregation> const& agg)
  {
    if constexpr (std::is_same_v<T, string_view>) {
      switch (agg->kind) {
        case aggregation::MIN: return [](HostType a, HostType b) { return std::min(a, b); };
        case aggregation::MAX: return [](HostType a, HostType b) { return std::max(a, b); };
        default: {
          CUDF_FAIL("Unsupported aggregation");
          return [](HostType a, HostType b) { return std::min(a, b); };
        }
      }
    } else {
      switch (agg->kind) {
        case aggregation::SUM: return std::plus<HostType>{};
        case aggregation::PRODUCT: return std::multiplies<HostType>{};
        case aggregation::MIN: return [](HostType a, HostType b) { return std::min(a, b); };
        case aggregation::MAX: return [](HostType a, HostType b) { return std::max(a, b); };
        default: {
          CUDF_FAIL("Unsupported aggregation");
          return [](HostType a, HostType b) { return std::min(a, b); };
        }
      }
    }
  }

  HostType make_identity(std::unique_ptr<aggregation> const& agg)
  {
    if constexpr (std::is_same_v<T, string_view>) {
      switch (agg->kind) {
        case aggregation::MIN: return std::string{"\xF7\xBF\xBF\xBF"};
        case aggregation::MAX: return std::string{};
        default: {
          CUDF_FAIL("Unsupported aggregation");
          return HostType{};
        }
      }
    } else {
      switch (agg->kind) {
        case aggregation::SUM: return HostType{0};
        case aggregation::PRODUCT: return HostType{1};
        case aggregation::MIN: return std::numeric_limits<HostType>::max();
        case aggregation::MAX: return std::numeric_limits<HostType>::lowest();
        default: {
          CUDF_FAIL("Unsupported aggregation");
          return HostType{};
        }
      }
    }
  }

  std::unique_ptr<column> make_expected(host_span<HostType const> v,
                                        host_span<bool const> b,
                                        std::unique_ptr<aggregation> const& agg,
                                        scan_type inclusive,
                                        null_policy null_handling,
                                        numeric::scale_type scale = numeric::scale_type{0})
  {
    auto op       = this->make_agg(agg);
    auto identity = this->make_identity(agg);

    thrust::host_vector<HostType> expected(v.size());
    thrust::host_vector<bool> b_out(b.begin(), b.end());

    bool const nullable = (b.size() > 0);

    auto masked_value = [identity](auto const& z) {
      return thrust::get<1>(z) ? thrust::get<0>(z) : identity;
    };

    if (inclusive == scan_type::INCLUSIVE) {
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

    return nullable ? this->make_column(expected, b_out, scale)
                    : this->make_column(expected, {}, scale);
  }
};

using TestTypes = Concat<NumericTypes, FixedPointTypes, Types<string_view>>;

TYPED_TEST_CASE(ScanTest, TestTypes);

TYPED_TEST(ScanTest, Min)
{
  auto const v = make_vector<TypeParam>({123, 64, 63, 99, -5, 123, -16, -120, -111});
  auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 0, 1, 1, 1, 1, 0, 0, 1});

  // no nulls
  this->scan_test(v, {}, make_min_aggregation(), scan_type::INCLUSIVE);
  this->scan_test(v, {}, make_min_aggregation(), scan_type::EXCLUSIVE);
  // skipna = true (default)
  this->scan_test(v, b, make_min_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);
  this->scan_test(v, b, make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v, b, make_min_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  this->scan_test(v, b, make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE);
}

TYPED_TEST(ScanTest, Max)
{
  auto const v = make_vector<TypeParam>({-120, 5, 0, -120, -111, 64, 63, 99, 123, -16});
  auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 0, 1, 1, 1, 1, 0, 1, 0, 1});

  // inclusive
  // no nulls
  this->scan_test(v, {}, make_max_aggregation(), scan_type::INCLUSIVE);
  this->scan_test(v, {}, make_max_aggregation(), scan_type::EXCLUSIVE);
  // skipna = true (default)
  this->scan_test(v, b, make_max_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);
  this->scan_test(v, b, make_max_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v, b, make_max_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  this->scan_test(v, b, make_max_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE);
}

TYPED_TEST(ScanTest, Product)
{
  auto const v = make_vector<TypeParam>({5, -1, 1, 3, -2, 4});
  auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 1, 1, 0, 1, 1});

  // no nulls
  this->scan_test(v, {}, make_product_aggregation(), scan_type::INCLUSIVE);
  this->scan_test(v, {}, make_product_aggregation(), scan_type::EXCLUSIVE);
  // skipna = true (default)
  this->scan_test(v, b, make_product_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);
  this->scan_test(v, b, make_product_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v, b, make_product_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  this->scan_test(v, b, make_product_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE);
}

TYPED_TEST(ScanTest, Sum)
{
  auto const v = [] {
    if (std::is_signed<TypeParam>::value)
      return make_vector<TypeParam>({-120, 5, 6, 113, -111, 64, -63, 9, 34, -16});
    return make_vector<TypeParam>({12, 5, 6, 13, 11, 14, 3, 9, 34, 16});
  }();
  auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 0, 1, 1, 0, 0, 1, 1, 1, 1});

  // no nulls
  this->scan_test(v, {}, make_sum_aggregation(), scan_type::INCLUSIVE);
  this->scan_test(v, {}, make_sum_aggregation(), scan_type::EXCLUSIVE);
  // skipna = true (default)
  this->scan_test(v, b, make_sum_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);
  this->scan_test(v, b, make_sum_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v, b, make_sum_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  this->scan_test(v, b, make_sum_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE);
}

TYPED_TEST(ScanTest, EmptyColumn)
{
  auto const v = thrust::host_vector<typename TypeParam_to_host_type<TypeParam>::type>{};
  auto const b = thrust::host_vector<bool>{};

  // skipna = true (default)
  this->scan_test(v, b, make_min_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);
  this->scan_test(v, b, make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v, b, make_min_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  this->scan_test(v, b, make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE);
}

TYPED_TEST(ScanTest, LeadingNulls)
{
  auto const v = make_vector<TypeParam>({100, 200, 300});
  auto const b = thrust::host_vector<bool>(std::vector<bool>{0, 1, 1});

  // skipna = true (default)
  this->scan_test(v, b, make_min_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);
  this->scan_test(v, b, make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v, b, make_min_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  this->scan_test(v, b, make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE);
}

class ScanStringsTest : public ScanTest<string_view> {
};

TEST_F(ScanStringsTest, MoreStringsMinMax)
{
  int row_count = 512;

  auto data_begin = cudf::detail::make_counting_transform_iterator(0, [](auto idx) {
    char const s[] = {static_cast<char>('a' + (idx % 26)), 0};
    return std::string(s);
  });
  auto validity   = cudf::detail::make_counting_transform_iterator(
    0, [](auto idx) -> bool { return (idx % 23) != 22; });
  strings_column_wrapper col(data_begin, data_begin + row_count, validity);

  thrust::host_vector<std::string> v(data_begin, data_begin + row_count);
  thrust::host_vector<bool> b(validity, validity + row_count);

  this->scan_test(v, {}, make_min_aggregation(), scan_type::INCLUSIVE);
  this->scan_test(v, b, make_min_aggregation(), scan_type::INCLUSIVE);
  this->scan_test(v, b, make_min_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);

  this->scan_test(v, {}, make_min_aggregation(), scan_type::EXCLUSIVE);
  this->scan_test(v, b, make_min_aggregation(), scan_type::EXCLUSIVE);
  this->scan_test(v, b, make_min_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);

  this->scan_test(v, {}, make_max_aggregation(), scan_type::INCLUSIVE);
  this->scan_test(v, b, make_max_aggregation(), scan_type::INCLUSIVE);
  this->scan_test(v, b, make_max_aggregation(), scan_type::INCLUSIVE, null_policy::EXCLUDE);

  this->scan_test(v, {}, make_max_aggregation(), scan_type::EXCLUSIVE);
  this->scan_test(v, b, make_max_aggregation(), scan_type::EXCLUSIVE);
  this->scan_test(v, b, make_max_aggregation(), scan_type::EXCLUSIVE, null_policy::EXCLUDE);
}

template <typename T>
struct ScanChronoTest : public BaseFixture {
};

TYPED_TEST_CASE(ScanChronoTest, ChronoTypes);

TYPED_TEST(ScanChronoTest, ChronoMinMax)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected_min({5, 4, 4, 0, 1, 1, 1, 1},
                                                                          {1, 1, 1, 0, 1, 1, 1, 1});

  auto result = cudf::scan(col, cudf::make_min_aggregation(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected_min);

  result = cudf::scan(
    col, cudf::make_min_aggregation(), cudf::scan_type::INCLUSIVE, cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected_min);

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected_max({5, 5, 6, 0, 6, 6, 6, 6},
                                                                          {1, 1, 1, 0, 1, 1, 1, 1});
  result = cudf::scan(col, cudf::make_max_aggregation(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected_max);

  result = cudf::scan(
    col, cudf::make_max_aggregation(), cudf::scan_type::INCLUSIVE, cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected_max);

  EXPECT_THROW(cudf::scan(col, cudf::make_max_aggregation(), cudf::scan_type::EXCLUSIVE),
               cudf::logic_error);
  EXPECT_THROW(cudf::scan(col, cudf::make_min_aggregation(), cudf::scan_type::EXCLUSIVE),
               cudf::logic_error);
}

template <typename T>
struct TypedRankScanTest : ScanTest<T> {
  inline void test_ungrouped_rank_scan(column_view const& input,
                                       column_view const& expect_vals,
                                       std::unique_ptr<aggregation> const& agg,
                                       null_policy null_handling)
  {
    auto col_out = scan(input, agg, scan_type::INCLUSIVE, null_handling);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_vals, col_out->view());
  }
};

using RankTypes =
  Concat<IntegralTypesNotBool, FloatingPointTypes, FixedPointTypes, ChronoTypes, StringTypes>;

TYPED_TEST_CASE(TypedRankScanTest, RankTypes);

TYPED_TEST(TypedRankScanTest, Rank)
{
  auto const v = [] {
    if (std::is_signed<TypeParam>::value)
      return make_vector<TypeParam>({-120, -120, -120, -16, -16, 5, 6, 6, 6, 6, 34, 113});
    return make_vector<TypeParam>({5, 5, 5, 6, 6, 9, 11, 11, 11, 11, 14, 34});
  }();
  auto col = this->make_column(v);

  auto const expected_dense_vals =
    fixed_width_column_wrapper<size_type>{1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5, 6};
  auto const expected_rank_vals =
    fixed_width_column_wrapper<size_type>{1, 1, 1, 4, 4, 6, 7, 7, 7, 7, 11, 12};
  this->test_ungrouped_rank_scan(
    *col, expected_dense_vals, make_dense_rank_aggregation(), null_policy::INCLUDE);
  this->test_ungrouped_rank_scan(
    *col, expected_rank_vals, make_rank_aggregation(), null_policy::INCLUDE);
}

TYPED_TEST(TypedRankScanTest, RankWithNulls)
{
  auto const v = [] {
    if (std::is_signed<TypeParam>::value)
      return make_vector<TypeParam>({-120, -120, -120, -16, -16, 5, 6, 6, 6, 6, 34, 113});
    return make_vector<TypeParam>({5, 5, 5, 6, 6, 9, 11, 11, 11, 11, 14, 34});
  }();
  auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0});
  auto col     = this->make_column(v, b);

  auto const expected_dense_vals =
    fixed_width_column_wrapper<size_type>{1, 1, 1, 2, 3, 4, 5, 5, 6, 6, 7, 8};
  auto const expected_rank_vals =
    fixed_width_column_wrapper<size_type>{1, 1, 1, 4, 5, 6, 7, 7, 9, 9, 11, 12};
  this->test_ungrouped_rank_scan(
    *col, expected_dense_vals, make_dense_rank_aggregation(), null_policy::INCLUDE);
  this->test_ungrouped_rank_scan(
    *col, expected_rank_vals, make_rank_aggregation(), null_policy::INCLUDE);
}

TYPED_TEST(TypedRankScanTest, mixedStructs)
{
  auto const v = [] {
    if (std::is_signed<TypeParam>::value)
      return make_vector<TypeParam>({-1, -1, -4, -4, -4, 5, 7, 7, 7, 9, 9, 9});
    return make_vector<TypeParam>({0, 0, 4, 4, 4, 5, 7, 7, 7, 9, 9, 9});
  }();
  auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1});
  auto col     = this->make_column(v, b);
  auto strings = strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};
  std::vector<std::unique_ptr<column>> vector_of_columns;
  vector_of_columns.push_back(std::move(col));
  vector_of_columns.push_back(strings.release());
  auto struct_col = structs_column_wrapper{std::move(vector_of_columns)}.release();

  auto expected_dense_vals =
    fixed_width_column_wrapper<size_type>{1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 8};
  auto expected_rank_vals =
    fixed_width_column_wrapper<size_type>{1, 1, 3, 3, 5, 6, 7, 7, 9, 10, 10, 12};

  this->test_ungrouped_rank_scan(
    *struct_col, expected_dense_vals, make_dense_rank_aggregation(), null_policy::INCLUDE);
  this->test_ungrouped_rank_scan(
    *struct_col, expected_rank_vals, make_rank_aggregation(), null_policy::INCLUDE);
}

TYPED_TEST(TypedRankScanTest, nestedStructs)
{
  auto const v = [] {
    if (std::is_signed<TypeParam>::value)
      return make_vector<TypeParam>({-1, -1, -4, -4, -4, 5, 7, 7, 7, 9, 9, 9});
    return make_vector<TypeParam>({0, 0, 4, 4, 4, 5, 7, 7, 7, 9, 9, 9});
  }();
  auto const b  = thrust::host_vector<bool>(std::vector<bool>{1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1});
  auto col1     = this->make_column(v, b);
  auto col2     = this->make_column(v, b);
  auto col3     = this->make_column(v, b);
  auto col4     = this->make_column(v, b);
  auto strings1 = strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};
  auto strings2 = strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};

  std::vector<std::unique_ptr<column>> struct_columns;
  struct_columns.push_back(std::move(col1));
  struct_columns.push_back(strings1.release());
  auto struct_col = structs_column_wrapper{std::move(struct_columns)};
  std::vector<std::unique_ptr<column>> nested_columns;
  nested_columns.push_back(struct_col.release());
  nested_columns.push_back(std::move(col2));
  auto nested_col = structs_column_wrapper{std::move(nested_columns)};
  std::vector<std::unique_ptr<column>> flat_columns;
  flat_columns.push_back(std::move(col3));
  flat_columns.push_back(strings2.release());
  flat_columns.push_back(std::move(col4));
  auto flat_col = structs_column_wrapper{std::move(flat_columns)};

  auto dense_out =
    scan(nested_col, make_dense_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto dense_expected =
    scan(flat_col, make_dense_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto rank_out =
    scan(nested_col, make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto rank_expected =
    scan(flat_col, make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(dense_out->view(), dense_expected->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(rank_out->view(), rank_expected->view());
}

TYPED_TEST(TypedRankScanTest, structsWithNullPushdown)
{
  auto const v = [] {
    if (std::is_signed<TypeParam>::value)
      return make_vector<TypeParam>({-1, -1, -4, -4, -4, 5, 7, 7, 7, 9, 9, 9});
    return make_vector<TypeParam>({0, 0, 4, 4, 4, 5, 7, 7, 7, 9, 9, 9});
  }();
  auto const b = thrust::host_vector<bool>(std::vector<bool>{1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1});
  auto col     = this->make_column(v, b);
  auto strings = strings_column_wrapper{
    {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};
  std::vector<std::unique_ptr<column>> struct_columns;
  struct_columns.push_back(std::move(col));
  struct_columns.push_back(strings.release());

  auto struct_col =
    cudf::make_structs_column(12, std::move(struct_columns), 0, rmm::device_buffer{});

  struct_col->set_null_mask(create_null_mask(12, cudf::mask_state::ALL_NULL));
  auto expected_null_result =
    fixed_width_column_wrapper<size_type>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto dense_null_out =
    scan(*struct_col, make_dense_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto rank_null_out =
    scan(*struct_col, make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(dense_null_out->view(), expected_null_result);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(rank_null_out->view(), expected_null_result);

  auto const struct_nulls =
    thrust::host_vector<bool>(std::vector<bool>{1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  struct_col->set_null_mask(
    cudf::test::detail::make_null_mask(struct_nulls.begin(), struct_nulls.end()));
  auto expected_dense_vals =
    fixed_width_column_wrapper<size_type>{1, 2, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9};
  auto expected_rank_vals =
    fixed_width_column_wrapper<size_type>{1, 2, 2, 4, 5, 6, 7, 7, 9, 10, 10, 12};
  auto dense_out =
    scan(*struct_col, make_dense_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto rank_out =
    scan(*struct_col, make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(dense_out->view(), expected_dense_vals);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(rank_out->view(), expected_rank_vals);
}

/* List support dependent on https://github.com/rapidsai/cudf/issues/8683
template <typename T>
struct ListRankScanTest : public BaseFixture {
};

using ListTestTypeSet = Concat<IntegralTypesNotBool,
                                           FloatingPointTypes,
                                           FixedPointTypes>;

TYPED_TEST_CASE(ListRankScanTest, ListTestTypeSet);

TYPED_TEST(ListRankScanTest, ListRank)
{
  auto list_col = lists_column_wrapper<TypeParam>{{0, 0},
                                                  {0, 0},
                                                  {7, 2},
                                                  {7, 2},
                                                  {7, 3},
                                                  {5, 5},
                                                  {4, 6},
                                                  {4, 6},
                                                  {4, 6},
                                                  {9, 9},
                                                  {9, 9},
                                                  {9, 10}};
  fixed_width_column_wrapper<TypeParam> element1{0, 0, 4, 4, 4, 5, 7, 7, 7, 9, 9, 9};
  fixed_width_column_wrapper<TypeParam> element2{0, 0, 2, 2, 3, 5, 6, 6, 6, 9, 9, 10};
  auto struct_col = structs_column_wrapper{element1, element2};

  auto dense_out      = scan(list_col->view(),
                              make_dense_rank_aggregation(),
                              scan_type::INCLUSIVE,
                              null_policy::INCLUDE);
  auto dense_expected = scan(
    struct_col, make_dense_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto rank_out = scan(
    list_col->view(), make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto rank_expected = scan(
    struct_col, make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(dense_out->view(), dense_expected->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(rank_out->view(), rank_expected->view());
}
*/

struct RankScanTest : public BaseFixture {
};

TEST(RankScanTest, BoolRank)
{
  fixed_width_column_wrapper<bool> vals{0, 0, 0, 6, 6, 9, 11, 11, 11, 11, 14, 34};
  fixed_width_column_wrapper<size_type> expected_dense_vals{1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  fixed_width_column_wrapper<size_type> expected_rank_vals{1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4};

  auto dense_out =
    scan(vals, make_dense_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto rank_out = scan(vals, make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_dense_vals, dense_out->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_rank_vals, rank_out->view());
}

TEST(RankScanTest, BoolRankWithNull)
{
  fixed_width_column_wrapper<bool> vals{{0, 0, 0, 6, 6, 9, 11, 11, 11, 11, 14, 34},
                                        {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}};
  table_view order_table{std::vector<column_view>{vals}};
  fixed_width_column_wrapper<size_type> expected_dense_vals{1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3};
  fixed_width_column_wrapper<size_type> expected_rank_vals{1, 1, 1, 4, 4, 4, 4, 4, 9, 9, 9, 9};

  auto nullable_dense_out =
    scan(vals, make_dense_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  auto nullable_rank_out =
    scan(vals, make_rank_aggregation(), scan_type::INCLUSIVE, null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_dense_vals, nullable_dense_out->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_rank_vals, nullable_rank_out->view());
}

TEST(RankScanTest, ExclusiveScan)
{
  fixed_width_column_wrapper<uint32_t> vals{3, 4, 5};
  fixed_width_column_wrapper<uint32_t> order_col{3, 3, 1};
  table_view order_table{std::vector<column_view>{order_col}};

  CUDF_EXPECT_THROW_MESSAGE(
    scan(vals, make_dense_rank_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE),
    "Unsupported dense rank aggregation operator for exclusive scan");
  CUDF_EXPECT_THROW_MESSAGE(
    scan(vals, make_rank_aggregation(), scan_type::EXCLUSIVE, null_policy::INCLUDE),
    "Unsupported rank aggregation operator for exclusive scan");
}

}  // namespace test
}  // namespace cudf
