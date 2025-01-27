/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <tests/reductions/scan_tests.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/reduction.hpp>

#include <thrust/host_vector.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <numeric>

using aggregation      = cudf::aggregation;
using scan_aggregation = cudf::scan_aggregation;
using cudf::null_policy;
using cudf::scan_type;

// This is the main test feature
template <typename T>
struct ScanTest : public BaseScanTest<T> {
  using HostType = typename BaseScanTest<T>::HostType;

  void scan_test(cudf::host_span<HostType const> v,
                 cudf::host_span<bool const> b,
                 scan_aggregation const& agg,
                 scan_type inclusive,
                 null_policy null_handling,
                 numeric::scale_type scale)
  {
    auto col_in = this->make_column(v, b, scale);

    if (not this->params_supported(agg, inclusive)) {
      EXPECT_THROW(scan(*col_in, agg, inclusive, null_handling), cudf::logic_error);
    } else {
      auto expected_col_out = this->make_expected(v, b, agg, inclusive, null_handling, scale);
      auto col_out          = scan(*col_in, agg, inclusive, null_handling);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_col_out, *col_out);
      EXPECT_FALSE(cudf::has_nonempty_nulls(col_out->view()));
    }
  }

  // Overload to iterate the test over a few different scales for fixed-point tests
  void scan_test(cudf::host_span<HostType const> v,
                 cudf::host_span<bool const> b,
                 scan_aggregation const& agg,
                 scan_type inclusive,
                 null_policy null_handling = null_policy::EXCLUDE)
  {
    if constexpr (cudf::is_fixed_point<T>()) {
      for (auto scale : {0, -1, -2, -3}) {
        scan_test(v, b, agg, inclusive, null_handling, numeric::scale_type{scale});
      }
    } else {
      scan_test(v, b, agg, inclusive, null_handling, numeric::scale_type{0});
    }
  }

  bool params_supported(scan_aggregation const& agg, scan_type inclusive)
  {
    bool supported = [&] {
      switch (agg.kind) {
        case aggregation::SUM: return std::is_invocable_v<cudf::DeviceSum, T, T>;
        case aggregation::PRODUCT: return std::is_invocable_v<cudf::DeviceProduct, T, T>;
        case aggregation::MIN: return std::is_invocable_v<cudf::DeviceMin, T, T>;
        case aggregation::MAX: return std::is_invocable_v<cudf::DeviceMax, T, T>;
        case aggregation::RANK: return std::is_invocable_v<cudf::DeviceMax, T, T>;  // comparable
        default: return false;
      }
      return false;
    }();

    // special cases for individual types
    if constexpr (cudf::is_fixed_point<T>()) return supported && (agg.kind != aggregation::PRODUCT);
    if constexpr (std::is_same_v<T, cudf::string_view> || cudf::is_timestamp<T>())
      return supported && (inclusive == scan_type::INCLUSIVE);
    return supported;
  }

  std::function<HostType(HostType, HostType)> make_agg(scan_aggregation const& agg)
  {
    if constexpr (std::is_same_v<T, cudf::string_view>) {
      switch (agg.kind) {
        case aggregation::MIN: return [](HostType a, HostType b) { return std::min(a, b); };
        case aggregation::MAX: return [](HostType a, HostType b) { return std::max(a, b); };
        default: {
          CUDF_FAIL("Unsupported aggregation");
          return [](HostType a, HostType b) { return std::min(a, b); };
        }
      }
    } else {
      switch (agg.kind) {
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

  HostType make_identity(scan_aggregation const& agg)
  {
    if constexpr (std::is_same_v<T, cudf::string_view>) {
      switch (agg.kind) {
        case aggregation::MIN: return std::string{"\xF7\xBF\xBF\xBF"};
        case aggregation::MAX: return std::string{};
        default: CUDF_FAIL("Unsupported aggregation");
      }
    } else {
      switch (agg.kind) {
        case aggregation::SUM: return HostType{0};
        case aggregation::PRODUCT: return HostType{1};
        case aggregation::MIN:
          if constexpr (std::numeric_limits<HostType>::has_infinity) {
            return std::numeric_limits<HostType>::infinity();
          } else {
            return std::numeric_limits<HostType>::max();
          }
        case aggregation::MAX:
          if constexpr (std::numeric_limits<HostType>::has_infinity) {
            return -std::numeric_limits<HostType>::infinity();
          } else {
            return std::numeric_limits<HostType>::lowest();
          }
        default: CUDF_FAIL("Unsupported aggregation");
      }
    }
  }

  std::unique_ptr<cudf::column> make_expected(cudf::host_span<HostType const> v,
                                              cudf::host_span<bool const> b,
                                              scan_aggregation const& agg,
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

using TestTypes = cudf::test::
  Concat<cudf::test::NumericTypes, cudf::test::FixedPointTypes, cudf::test::StringTypes>;

TYPED_TEST_SUITE(ScanTest, TestTypes);

TYPED_TEST(ScanTest, Min)
{
  auto const v = make_vector<TypeParam>({123, 64, 63, 99, -5, 123, -16, -120, -111});
  auto const b = thrust::host_vector<bool>(
    std::vector<bool>{true, false, true, true, true, true, false, false, true});

  // no nulls
  this->scan_test(v, {}, *cudf::make_min_aggregation<scan_aggregation>(), scan_type::INCLUSIVE);
  this->scan_test(v, {}, *cudf::make_min_aggregation<scan_aggregation>(), scan_type::EXCLUSIVE);
  // skipna = true (default)
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::EXCLUDE);
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::INCLUDE);
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::INCLUDE);
}

TYPED_TEST(ScanTest, Max)
{
  auto const v = make_vector<TypeParam>({-120, 5, 0, -120, -111, 64, 63, 99, 123, -16});
  auto const b = thrust::host_vector<bool>(
    std::vector<bool>{true, false, true, true, true, true, false, true, false, true});

  // inclusive
  // no nulls
  this->scan_test(v, {}, *cudf::make_max_aggregation<scan_aggregation>(), scan_type::INCLUSIVE);
  this->scan_test(v, {}, *cudf::make_max_aggregation<scan_aggregation>(), scan_type::EXCLUSIVE);
  // skipna = true (default)
  this->scan_test(v,
                  b,
                  *cudf::make_max_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::EXCLUDE);
  this->scan_test(v,
                  b,
                  *cudf::make_max_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v,
                  b,
                  *cudf::make_max_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::INCLUDE);
  this->scan_test(v,
                  b,
                  *cudf::make_max_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::INCLUDE);
}

TYPED_TEST(ScanTest, Product)
{
  auto const v = make_vector<TypeParam>({5, -1, 1, 3, -2, 4});
  auto const b = thrust::host_vector<bool>(std::vector<bool>{true, true, true, false, true, true});

  // no nulls
  this->scan_test(v, {}, *cudf::make_product_aggregation<scan_aggregation>(), scan_type::INCLUSIVE);
  this->scan_test(v, {}, *cudf::make_product_aggregation<scan_aggregation>(), scan_type::EXCLUSIVE);
  // skipna = true (default)
  this->scan_test(v,
                  b,
                  *cudf::make_product_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::EXCLUDE);
  this->scan_test(v,
                  b,
                  *cudf::make_product_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v,
                  b,
                  *cudf::make_product_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::INCLUDE);
  this->scan_test(v,
                  b,
                  *cudf::make_product_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::INCLUDE);
}

TYPED_TEST(ScanTest, Sum)
{
  auto const v = [] {
    if (std::is_signed_v<TypeParam>)
      return make_vector<TypeParam>({-120, 5, 6, 113, -111, 64, -63, 9, 34, -16});
    return make_vector<TypeParam>({12, 5, 6, 13, 11, 14, 3, 9, 34, 16});
  }();
  auto const b = thrust::host_vector<bool>(
    std::vector<bool>{true, false, true, true, false, false, true, true, true, true});

  // no nulls
  this->scan_test(v, {}, *cudf::make_sum_aggregation<scan_aggregation>(), scan_type::INCLUSIVE);
  this->scan_test(v, {}, *cudf::make_sum_aggregation<scan_aggregation>(), scan_type::EXCLUSIVE);
  // skipna = true (default)
  this->scan_test(v,
                  b,
                  *cudf::make_sum_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::EXCLUDE);
  this->scan_test(v,
                  b,
                  *cudf::make_sum_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v,
                  b,
                  *cudf::make_sum_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::INCLUDE);
  this->scan_test(v,
                  b,
                  *cudf::make_sum_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::INCLUDE);
}

TYPED_TEST(ScanTest, EmptyColumn)
{
  auto const v = thrust::host_vector<typename TypeParam_to_host_type<TypeParam>::type>{};
  auto const b = thrust::host_vector<bool>{};

  // skipna = true (default)
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::EXCLUDE);
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::INCLUDE);
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::INCLUDE);
}

TYPED_TEST(ScanTest, LeadingNulls)
{
  auto const v = make_vector<TypeParam>({100, 200, 300});
  auto const b = thrust::host_vector<bool>(std::vector<bool>{false, true, true});

  // skipna = true (default)
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::EXCLUDE);
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::EXCLUDE);
  // skipna = false
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::INCLUDE);
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::INCLUDE);
}

class ScanStringsTest : public ScanTest<cudf::string_view> {};

TEST_F(ScanStringsTest, MoreStringsMinMax)
{
  int row_count = 512;

  auto validity = cudf::detail::make_counting_transform_iterator(
    0, [](auto idx) -> bool { return (idx % 23) != 22; });
  auto data_begin = cudf::detail::make_counting_transform_iterator(0, [validity](auto idx) {
    if (validity[idx] == 0) return std::string{};
    char const s = static_cast<char>('a' + (idx % 26));
    return std::string{1, s};
  });
  cudf::test::strings_column_wrapper col(data_begin, data_begin + row_count, validity);

  thrust::host_vector<std::string> v(data_begin, data_begin + row_count);
  thrust::host_vector<bool> b(validity, validity + row_count);

  this->scan_test(v, {}, *cudf::make_min_aggregation<scan_aggregation>(), scan_type::INCLUSIVE);
  this->scan_test(v, b, *cudf::make_min_aggregation<scan_aggregation>(), scan_type::INCLUSIVE);
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::EXCLUDE);

  this->scan_test(v, {}, *cudf::make_min_aggregation<scan_aggregation>(), scan_type::EXCLUSIVE);
  this->scan_test(v, b, *cudf::make_min_aggregation<scan_aggregation>(), scan_type::EXCLUSIVE);
  this->scan_test(v,
                  b,
                  *cudf::make_min_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::EXCLUDE);

  this->scan_test(v, {}, *cudf::make_max_aggregation<scan_aggregation>(), scan_type::INCLUSIVE);
  this->scan_test(v, b, *cudf::make_max_aggregation<scan_aggregation>(), scan_type::INCLUSIVE);
  this->scan_test(v,
                  b,
                  *cudf::make_max_aggregation<scan_aggregation>(),
                  scan_type::INCLUSIVE,
                  null_policy::EXCLUDE);

  this->scan_test(v, {}, *cudf::make_max_aggregation<scan_aggregation>(), scan_type::EXCLUSIVE);
  this->scan_test(v, b, *cudf::make_max_aggregation<scan_aggregation>(), scan_type::EXCLUSIVE);
  this->scan_test(v,
                  b,
                  *cudf::make_max_aggregation<scan_aggregation>(),
                  scan_type::EXCLUSIVE,
                  null_policy::EXCLUDE);
}

template <typename T>
struct ScanChronoTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ScanChronoTest, cudf::test::ChronoTypes);

TYPED_TEST(ScanChronoTest, ChronoMinMax)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected_min({5, 4, 4, 0, 1, 1, 1, 1},
                                                                          {1, 1, 1, 0, 1, 1, 1, 1});

  auto result =
    cudf::scan(col, *cudf::make_min_aggregation<scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected_min);

  result = cudf::scan(col,
                      *cudf::make_min_aggregation<scan_aggregation>(),
                      cudf::scan_type::INCLUSIVE,
                      cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected_min);

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected_max({5, 5, 6, 0, 6, 6, 6, 6},
                                                                          {1, 1, 1, 0, 1, 1, 1, 1});
  result =
    cudf::scan(col, *cudf::make_max_aggregation<scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected_max);

  result = cudf::scan(col,
                      *cudf::make_max_aggregation<scan_aggregation>(),
                      cudf::scan_type::INCLUSIVE,
                      cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected_max);

  EXPECT_THROW(
    cudf::scan(col, *cudf::make_max_aggregation<scan_aggregation>(), cudf::scan_type::EXCLUSIVE),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::scan(col, *cudf::make_min_aggregation<scan_aggregation>(), cudf::scan_type::EXCLUSIVE),
    cudf::logic_error);
}

template <typename T>
struct ScanDurationTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ScanDurationTest, cudf::test::DurationTypes);

TYPED_TEST(ScanDurationTest, Sum)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({5, 4, 6, 0, 1, 6, 5, 3},
                                                                 {1, 1, 1, 0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> expected({5, 9, 15, 0, 16, 22, 27, 30},
                                                                      {1, 1, 1, 0, 1, 1, 1, 1});

  auto result =
    cudf::scan(col, *cudf::make_sum_aggregation<scan_aggregation>(), cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

  result = cudf::scan(col,
                      *cudf::make_sum_aggregation<scan_aggregation>(),
                      cudf::scan_type::INCLUSIVE,
                      cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);

  EXPECT_THROW(
    cudf::scan(col, *cudf::make_sum_aggregation<scan_aggregation>(), cudf::scan_type::EXCLUSIVE),
    cudf::logic_error);
}

struct StructScanTest : public cudf::test::BaseFixture {};

TEST_F(StructScanTest, StructScanMinMaxNoNull)
{
  using INTS_CW    = cudf::test::fixed_width_column_wrapper<int32_t>;
  using STRINGS_CW = cudf::test::strings_column_wrapper;
  using STRUCTS_CW = cudf::test::structs_column_wrapper;

  auto const input = [] {
    auto child1 = STRINGS_CW{"año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};
    auto child2 = INTS_CW{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    return STRUCTS_CW{{child1, child2}};
  }();

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{"año", "año", "año", "aaa", "aaa", "aaa", "aaa", "$1", "$1", "$1"};
      auto child2 = INTS_CW{1, 1, 1, 4, 4, 4, 4, 8, 8, 8};
      return STRUCTS_CW{{child1, child2}};
    }();
    auto const result = cudf::scan(
      input, *cudf::make_min_aggregation<scan_aggregation>(), cudf::scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{"año", "bit", "₹1", "₹1", "₹1", "₹1", "₹1", "₹1", "₹1", "₹1"};
      auto child2 = INTS_CW{1, 2, 3, 3, 3, 3, 3, 3, 3, 3};
      return STRUCTS_CW{{child1, child2}};
    }();
    auto const result = cudf::scan(
      input, *cudf::make_max_aggregation<scan_aggregation>(), cudf::scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TEST_F(StructScanTest, StructScanMinMaxSlicedInput)
{
  using INTS_CW    = cudf::test::fixed_width_column_wrapper<int>;
  using STRINGS_CW = cudf::test::strings_column_wrapper;
  using STRUCTS_CW = cudf::test::structs_column_wrapper;
  constexpr int32_t dont_care{1};

  auto const input_original = [] {
    auto child1 = STRINGS_CW{"$dont_care",
                             "$dont_care",
                             "año",
                             "bit",
                             "₹1",
                             "aaa",
                             "zit",
                             "bat",
                             "aab",
                             "$1",
                             "€1",
                             "wut",
                             "₹dont_care"};
    auto child2 = INTS_CW{dont_care, dont_care, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, dont_care};
    return STRUCTS_CW{{child1, child2}};
  }();

  auto const input = cudf::slice(input_original, {2, 12})[0];

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{"año", "año", "año", "aaa", "aaa", "aaa", "aaa", "$1", "$1", "$1"};
      auto child2 = INTS_CW{1, 1, 1, 4, 4, 4, 4, 8, 8, 8};
      return STRUCTS_CW{{child1, child2}};
    }();
    auto const result = cudf::scan(
      input, *cudf::make_min_aggregation<scan_aggregation>(), cudf::scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{"año", "bit", "₹1", "₹1", "₹1", "₹1", "₹1", "₹1", "₹1", "₹1"};
      auto child2 = INTS_CW{1, 2, 3, 3, 3, 3, 3, 3, 3, 3};
      return STRUCTS_CW{{child1, child2}};
    }();
    auto const result = cudf::scan(
      input, *cudf::make_max_aggregation<scan_aggregation>(), cudf::scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TEST_F(StructScanTest, StructScanMinMaxWithNulls)
{
  using INTS_CW    = cudf::test::fixed_width_column_wrapper<int>;
  using STRINGS_CW = cudf::test::strings_column_wrapper;
  using STRUCTS_CW = cudf::test::structs_column_wrapper;
  using cudf::test::iterators::null_at;
  using cudf::test::iterators::nulls_at;

  auto const input = [] {
    auto child1 = STRINGS_CW{{"año",
                              "bit",
                              "",     // child null
                              "aaa",  // parent null
                              "zit",
                              "bat",
                              "aab",
                              "",    // child null
                              "€1",  // parent null
                              "wut"},
                             nulls_at({2, 7})};
    auto child2 = INTS_CW{{1,
                           2,
                           0,  // child null
                           4,  // parent null
                           5,
                           6,
                           7,
                           0,  // child null
                           9,  // parent null
                           10},
                          nulls_at({2, 7})};
    return STRUCTS_CW{{child1, child2}, nulls_at({3, 8})};
  }();

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{{"año",
                                "año",
                                "" /*null*/,
                                "" /*null*/,
                                "" /*null*/,
                                "" /*null*/,
                                "" /*null*/,
                                "" /*null*/,
                                "" /*null*/,
                                "" /*null*/},
                               nulls_at({2, 3, 4, 5, 6, 7, 8, 9})};
      auto child2 = INTS_CW{{1,
                             1,
                             0 /*null*/,
                             0 /*null*/,
                             0 /*null*/,
                             0 /*null*/,
                             0 /*null*/,
                             0 /*null*/,
                             0 /*null*/,
                             0 /*null*/},
                            nulls_at({2, 3, 4, 5, 6, 7, 8, 9})};
      return STRUCTS_CW{{child1, child2}, nulls_at({3, 8})};
    }();

    auto const result = cudf::scan(input,
                                   *cudf::make_min_aggregation<scan_aggregation>(),
                                   cudf::scan_type::INCLUSIVE,
                                   null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{
        "año", "bit", "bit", "" /*NULL*/, "zit", "zit", "zit", "zit", "" /*NULL*/, "zit"};
      auto child2 = INTS_CW{1, 2, 2, 0 /*NULL*/, 5, 5, 5, 5, 0 /*NULL*/, 5};
      return STRUCTS_CW{{child1, child2}, nulls_at({3, 8})};
    }();

    auto const result = cudf::scan(input,
                                   *cudf::make_max_aggregation<scan_aggregation>(),
                                   cudf::scan_type::INCLUSIVE,
                                   null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{{"año",
                                "año",
                                "",   // child null
                                "",   // parent null
                                "",   // parent null
                                "",   // parent null
                                "",   // parent null
                                "",   // parent null
                                "",   // parent null
                                ""},  // parent null
                               null_at(2)};
      auto child2 = INTS_CW{{1,
                             1,
                             0,   // child null
                             0,   // parent null
                             0,   // parent null
                             0,   // parent null
                             0,   // parent null
                             0,   // parent null
                             0,   // parent null
                             0},  // parent null
                            null_at(2)};
      return STRUCTS_CW{{child1, child2}, nulls_at({3, 4, 5, 6, 7, 8, 9})};
    }();

    auto const result = cudf::scan(input,
                                   *cudf::make_min_aggregation<scan_aggregation>(),
                                   cudf::scan_type::INCLUSIVE,
                                   null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  {
    auto const expected = [] {
      auto child1 = STRINGS_CW{"año",
                               "bit",
                               "bit",
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/};
      auto child2 = INTS_CW{1,
                            2,
                            2,
                            0 /*NULL*/,
                            0 /*NULL*/,
                            0 /*NULL*/,
                            0 /*NULL*/,
                            0 /*NULL*/,
                            0 /*NULL*/,
                            0 /*NULL*/};
      return STRUCTS_CW{{child1, child2}, nulls_at({3, 4, 5, 6, 7, 8, 9})};
    }();

    auto const result = cudf::scan(input,
                                   *cudf::make_max_aggregation<scan_aggregation>(),
                                   cudf::scan_type::INCLUSIVE,
                                   null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}
