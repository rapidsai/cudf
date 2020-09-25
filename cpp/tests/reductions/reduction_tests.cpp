/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <iostream>
#include <vector>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <thrust/device_vector.h>

#include <cudf/detail/aggregation/aggregation.hpp>
#include "cudf/types.hpp"
using aggregation = cudf::aggregation;

template <typename T>
typename std::enable_if<!cudf::is_timestamp_t<T>::value, std::vector<T>>::type convert_values(
  std::vector<int> const &int_values)
{
  std::vector<T> v(int_values.size());
  std::transform(int_values.begin(), int_values.end(), v.begin(), [](int x) {
    if (std::is_unsigned<T>::value) x = std::abs(x);
    return static_cast<T>(x);
  });
  return v;
}

template <typename T>
typename std::enable_if<cudf::is_timestamp_t<T>::value, std::vector<T>>::type convert_values(
  std::vector<int> const &int_values)
{
  std::vector<T> v(int_values.size());
  std::transform(int_values.begin(), int_values.end(), v.begin(), [](int x) {
    if (std::is_unsigned<T>::value) x = std::abs(x);
    return T{typename T::duration(x)};
  });
  return v;
}

template <typename T>
cudf::test::fixed_width_column_wrapper<T> construct_null_column(std::vector<T> const &values,
                                                                std::vector<bool> const &bools)
{
  if (values.size() > bools.size()) { throw std::logic_error("input vector size mismatch."); }
  return cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end(), bools.begin());
}

template <typename T>
std::vector<T> replace_nulls(std::vector<T> const &values,
                             std::vector<bool> const &bools,
                             T identity)
{
  std::vector<T> v(values.size());
  std::transform(values.begin(), values.end(), bools.begin(), v.begin(), [identity](T x, bool b) {
    return (b) ? x : identity;
  });
  return v;
}

// ------------------------------------------------------------------------

// This is the main test feature
template <typename T>
struct ReductionTest : public cudf::test::BaseFixture {
  // Sum/Prod/SumOfSquare never support non arithmetics
  static constexpr bool ret_non_arithmetic =
    (std::is_arithmetic<T>::value || std::is_same<T, bool>::value) ? true : false;

  ReductionTest() {}

  ~ReductionTest() {}

  template <typename T_out>
  void reduction_test(const cudf::column_view underlying_column,
                      T_out expected_value,
                      bool succeeded_condition,
                      std::unique_ptr<aggregation> const &agg,
                      cudf::data_type output_dtype = cudf::data_type{},
                      bool expected_null           = false)
  {
    if (cudf::data_type{} == output_dtype) output_dtype = underlying_column.type();

    auto statement = [&]() {
      std::unique_ptr<cudf::scalar> result = cudf::reduce(underlying_column, agg, output_dtype);
      using ScalarType                     = cudf::scalar_type_t<T_out>;
      auto result1                         = static_cast<ScalarType *>(result.get());
      EXPECT_EQ(expected_null, !result1->is_valid());
      if (result1->is_valid()) { EXPECT_EQ(expected_value, result1->value()); }
    };

    if (succeeded_condition) {
      CUDF_EXPECT_NO_THROW(statement());
    } else {
      EXPECT_ANY_THROW(statement());
    }
  }
};

template <typename T>
struct MinMaxReductionTest : public ReductionTest<T> {
};

using MinMaxTypes = cudf::test::AllTypes;
TYPED_TEST_CASE(MinMaxReductionTest, MinMaxTypes);

// ------------------------------------------------------------------------
TYPED_TEST(MinMaxReductionTest, MinMax)
{
  using T = TypeParam;
  std::vector<int> int_values({5, 0, -120, -111, 0, 64, 63, 99, 123, -16});
  std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1, 0, 1});
  std::vector<bool> all_null({0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  std::vector<T> v = convert_values<T>(int_values);

  // Min/Max succeeds for any gdf types including
  // non-arithmetic types (date32, date64, timestamp, category)
  bool result_error(true);

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());

  T expected_min_result = *(std::min_element(v.begin(), v.end()));
  T expected_max_result = *(std::max_element(v.begin(), v.end()));
  this->reduction_test(col, expected_min_result, result_error, cudf::make_min_aggregation());
  this->reduction_test(col, expected_max_result, result_error, cudf::make_max_aggregation());

  auto res = cudf::minmax(col);

  using ScalarType = cudf::scalar_type_t<T>;
  auto min_result  = static_cast<ScalarType *>(res.first.get());
  auto max_result  = static_cast<ScalarType *>(res.second.get());
  EXPECT_EQ(min_result->value(), expected_min_result);
  EXPECT_EQ(max_result->value(), expected_max_result);

  // test with some nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  cudf::size_type valid_count =
    cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();

  auto r_min = replace_nulls(v, host_bools, std::numeric_limits<T>::max());
  auto r_max = replace_nulls(v, host_bools, std::numeric_limits<T>::lowest());

  T expected_min_null_result = *(std::min_element(r_min.begin(), r_min.end()));
  T expected_max_null_result = *(std::max_element(r_max.begin(), r_max.end()));

  this->reduction_test(
    col_nulls, expected_min_null_result, result_error, cudf::make_min_aggregation());
  this->reduction_test(
    col_nulls, expected_max_null_result, result_error, cudf::make_max_aggregation());

  auto null_res = cudf::minmax(col_nulls);

  using ScalarType     = cudf::scalar_type_t<T>;
  auto min_null_result = static_cast<ScalarType *>(null_res.first.get());
  auto max_null_result = static_cast<ScalarType *>(null_res.second.get());
  EXPECT_EQ(min_null_result->value(), expected_min_null_result);
  EXPECT_EQ(max_null_result->value(), expected_max_null_result);

  // test with some nulls
  cudf::test::fixed_width_column_wrapper<T> col_all_nulls = construct_null_column(v, all_null);
  cudf::size_type all_null_valid_count                    = 0;

  auto all_null_r_min = replace_nulls(v, all_null, std::numeric_limits<T>::max());
  auto all_null_r_max = replace_nulls(v, all_null, std::numeric_limits<T>::lowest());

  T expected_min_all_null_result =
    *(std::min_element(all_null_r_min.begin(), all_null_r_min.end()));
  T expected_max_all_null_result =
    *(std::max_element(all_null_r_max.begin(), all_null_r_max.end()));

  this->reduction_test(col_all_nulls,
                       expected_min_all_null_result,
                       result_error,
                       cudf::make_min_aggregation(),
                       cudf::data_type{},
                       true);
  this->reduction_test(col_all_nulls,
                       expected_max_all_null_result,
                       result_error,
                       cudf::make_max_aggregation(),
                       cudf::data_type{},
                       true);

  auto all_null_res = cudf::minmax(col_all_nulls);

  using ScalarType         = cudf::scalar_type_t<T>;
  auto min_all_null_result = static_cast<ScalarType *>(all_null_res.first.get());
  auto max_all_null_result = static_cast<ScalarType *>(all_null_res.second.get());
  EXPECT_EQ(min_all_null_result->is_valid(), false);
  EXPECT_EQ(max_all_null_result->is_valid(), false);
}

template <typename T>
struct SumReductionTest : public ReductionTest<T> {
};
using SumTypes = cudf::test::Concat<cudf::test::NumericTypes, cudf::test::DurationTypes>;
TYPED_TEST_CASE(SumReductionTest, SumTypes);

TYPED_TEST(SumReductionTest, Sum)
{
  using T = TypeParam;
  std::vector<int> int_values({6, -14, 13, 64, 0, -13, -20, 45});
  std::vector<bool> host_bools({1, 1, 0, 0, 1, 1, 1, 1});
  std::vector<T> v = convert_values<T>(int_values);

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  T expected_value = std::accumulate(v.begin(), v.end(), T{0});
  this->reduction_test(col, expected_value, this->ret_non_arithmetic, cudf::make_sum_aggregation());

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  cudf::size_type valid_count =
    cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
  auto r                = replace_nulls(v, host_bools, T{0});
  T expected_null_value = std::accumulate(r.begin(), r.end(), T{0});

  this->reduction_test(
    col_nulls, expected_null_value, this->ret_non_arithmetic, cudf::make_sum_aggregation());
}

TYPED_TEST_CASE(ReductionTest, cudf::test::NumericTypes);

TYPED_TEST(ReductionTest, Product)
{
  using T = TypeParam;
  std::vector<int> int_values({5, -1, 1, 0, 3, 2, 4});
  std::vector<bool> host_bools({1, 1, 0, 0, 1, 1, 1});
  std::vector<TypeParam> v = convert_values<TypeParam>(int_values);

  auto calc_prod = [](std::vector<T> &v) {
    T expected_value =
      std::accumulate(v.begin(), v.end(), T{1}, [](T acc, T i) { return acc * i; });
    return expected_value;
  };

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  TypeParam expected_value = calc_prod(v);

  this->reduction_test(
    col, expected_value, this->ret_non_arithmetic, cudf::make_product_aggregation());

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  cudf::size_type valid_count =
    cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
  auto r                        = replace_nulls(v, host_bools, T{1});
  TypeParam expected_null_value = calc_prod(r);

  this->reduction_test(
    col_nulls, expected_null_value, this->ret_non_arithmetic, cudf::make_product_aggregation());
}

TYPED_TEST(ReductionTest, SumOfSquare)
{
  using T = TypeParam;
  std::vector<int> int_values({-3, 2, 1, 0, 5, -3, -2});
  std::vector<bool> host_bools({1, 1, 0, 0, 1, 1, 1, 1});
  std::vector<T> v = convert_values<T>(int_values);

  auto calc_reduction = [](std::vector<T> &v) {
    T value = std::accumulate(v.begin(), v.end(), T{0}, [](T acc, T i) { return acc + i * i; });
    return value;
  };

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  T expected_value = calc_reduction(v);

  this->reduction_test(
    col, expected_value, this->ret_non_arithmetic, cudf::make_sum_of_squares_aggregation());

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  cudf::size_type valid_count =
    cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
  auto r                = replace_nulls(v, host_bools, T{0});
  T expected_null_value = calc_reduction(r);

  this->reduction_test(col_nulls,
                       expected_null_value,
                       this->ret_non_arithmetic,
                       cudf::make_sum_of_squares_aggregation());
}

template <typename T>
struct ReductionAnyAllTest : public ReductionTest<bool> {
};

TYPED_TEST_CASE(ReductionAnyAllTest, cudf::test::NumericTypes);

TYPED_TEST(ReductionAnyAllTest, AnyAllTrueTrue)
{
  using T = TypeParam;
  std::vector<int> int_values({true, true, true, true});
  std::vector<bool> host_bools({1, 1, 0, 1});
  std::vector<T> v = convert_values<T>(int_values);

  // Min/Max succeeds for any gdf types including
  // non-arithmetic types (date32, date64, timestamp, category)
  bool result_error = true;
  bool expected     = true;
  cudf::data_type output_dtype(cudf::type_id::BOOL8);

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());

  this->reduction_test(col, expected, result_error, cudf::make_any_aggregation(), output_dtype);
  this->reduction_test(col, expected, result_error, cudf::make_all_aggregation(), output_dtype);

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);

  this->reduction_test(
    col_nulls, expected, result_error, cudf::make_any_aggregation(), output_dtype);
  this->reduction_test(
    col_nulls, expected, result_error, cudf::make_all_aggregation(), output_dtype);
}

TYPED_TEST(ReductionAnyAllTest, AnyAllFalseFalse)
{
  using T = TypeParam;
  std::vector<int> int_values({false, false, false, false});
  std::vector<bool> host_bools({1, 1, 0, 1});
  std::vector<T> v = convert_values<T>(int_values);

  // Min/Max succeeds for any gdf types including
  // non-arithmetic types (date32, date64, timestamp, category)
  bool result_error = true;
  bool expected     = false;
  cudf::data_type output_dtype(cudf::type_id::BOOL8);

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());

  this->reduction_test(col, expected, result_error, cudf::make_any_aggregation(), output_dtype);
  this->reduction_test(col, expected, result_error, cudf::make_all_aggregation(), output_dtype);

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);

  this->reduction_test(
    col_nulls, expected, result_error, cudf::make_any_aggregation(), output_dtype);
  this->reduction_test(
    col_nulls, expected, result_error, cudf::make_all_aggregation(), output_dtype);
}

// ----------------------------------------------------------------------------

template <typename T>
struct MultiStepReductionTest : public ReductionTest<T> {
};

using MultiStepReductionTypes = cudf::test::NumericTypes;
TYPED_TEST_CASE(MultiStepReductionTest, MultiStepReductionTypes);

TYPED_TEST(MultiStepReductionTest, Mean)
{
  using T = TypeParam;
  std::vector<int> int_values({-3, 2, 1, 0, 5, -3, -2, 28});
  std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1});

  auto calc_mean = [](std::vector<T> &v, cudf::size_type valid_count) {
    double sum = std::accumulate(v.begin(), v.end(), double{0});
    return sum / valid_count;
  };

  // test without nulls
  std::vector<T> v = convert_values<T>(int_values);
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  double expected_value = calc_mean(v, v.size());
  this->reduction_test(col,
                       expected_value,
                       true,
                       cudf::make_mean_aggregation(),
                       cudf::data_type(cudf::type_id::FLOAT64));

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  cudf::size_type valid_count =
    cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
  auto replaced_array = replace_nulls(v, host_bools, T{0});

  double expected_value_nulls = calc_mean(replaced_array, valid_count);
  this->reduction_test(col_nulls,
                       expected_value_nulls,
                       true,
                       cudf::make_mean_aggregation(),
                       cudf::data_type(cudf::type_id::FLOAT64));
}

TYPED_TEST(MultiStepReductionTest, var_std)
{
  using T = TypeParam;
  std::vector<int> int_values({-3, 2, 1, 0, 5, -3, -2, 28});
  std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1});

  auto calc_var = [](std::vector<T> &v, cudf::size_type valid_count) {
    double mean = std::accumulate(v.begin(), v.end(), double{0});
    mean /= valid_count;

    double sum_of_sq = std::accumulate(
      v.begin(), v.end(), double{0}, [](double acc, TypeParam i) { return acc + i * i; });

    int ddof            = 1;
    cudf::size_type div = valid_count - ddof;

    double var = sum_of_sq / div - ((mean * mean) * valid_count) / div;
    return var;
  };

  // test without nulls
  std::vector<T> v = convert_values<T>(int_values);
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());

  double var   = calc_var(v, v.size());
  double std   = std::sqrt(var);
  auto var_agg = cudf::make_variance_aggregation(/*ddof =*/1);
  auto std_agg = cudf::make_std_aggregation(/*ddof =*/1);

  this->reduction_test(col, var, true, var_agg, cudf::data_type(cudf::type_id::FLOAT64));
  this->reduction_test(col, std, true, std_agg, cudf::data_type(cudf::type_id::FLOAT64));

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  cudf::size_type valid_count =
    cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
  auto replaced_array = replace_nulls(v, host_bools, T{0});

  double var_nulls = calc_var(replaced_array, valid_count);
  double std_nulls = std::sqrt(var_nulls);

  this->reduction_test(
    col_nulls, var_nulls, true, var_agg, cudf::data_type(cudf::type_id::FLOAT64));
  this->reduction_test(
    col_nulls, std_nulls, true, std_agg, cudf::data_type(cudf::type_id::FLOAT64));
}

// ----------------------------------------------------------------------------

template <typename T>
struct ReductionMultiStepErrorCheck : public ReductionTest<T> {
  void reduction_error_check(cudf::test::fixed_width_column_wrapper<T> &col,
                             bool succeeded_condition,
                             std::unique_ptr<aggregation> const &agg,
                             cudf::data_type output_dtype)
  {
    const cudf::column_view underlying_column = col;
    auto statement = [&]() { cudf::reduce(underlying_column, agg, output_dtype); };

    if (succeeded_condition) {
      CUDF_EXPECT_NO_THROW(statement());
    } else {
      EXPECT_ANY_THROW(statement());
    }
  }
};

TYPED_TEST_CASE(ReductionMultiStepErrorCheck, cudf::test::AllTypes);

TYPED_TEST(ReductionMultiStepErrorCheck, ErrorHandling)
{
  using T = TypeParam;
  std::vector<int> int_values({-3, 2});
  std::vector<bool> host_bools({1, 0});

  std::vector<T> v = convert_values<T>(int_values);
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);

  bool is_input_accpetable = this->ret_non_arithmetic;

  std::vector<cudf::data_type> dtypes(static_cast<int32_t>(cudf::type_id::NUM_TYPE_IDS) + 1);
  int i = 0;
  std::generate(dtypes.begin(), dtypes.end(), [&]() {
    return cudf::data_type(static_cast<cudf::type_id>(i++));
  });

  auto is_supported_outdtype = [](cudf::data_type dtype) {
    if (dtype == cudf::data_type(cudf::type_id::FLOAT32)) return true;
    if (dtype == cudf::data_type(cudf::type_id::FLOAT64)) return true;
    return false;
  };

  auto evaluate = [&](cudf::data_type dtype) mutable {
    bool expect_succeed = is_input_accpetable & is_supported_outdtype(dtype);
    auto var_agg        = cudf::make_variance_aggregation(/*ddof = 1*/);
    auto std_agg        = cudf::make_std_aggregation(/*ddof = 1*/);
    this->reduction_error_check(col, expect_succeed, cudf::make_mean_aggregation(), dtype);
    this->reduction_error_check(col, expect_succeed, var_agg, dtype);
    this->reduction_error_check(col, expect_succeed, std_agg, dtype);

    this->reduction_error_check(col_nulls, expect_succeed, cudf::make_mean_aggregation(), dtype);
    this->reduction_error_check(col_nulls, expect_succeed, var_agg, dtype);
    this->reduction_error_check(col_nulls, expect_succeed, std_agg, dtype);
    return;
  };

  std::for_each(dtypes.begin(), dtypes.end(), evaluate);
}

// ----------------------------------------------------------------------------

struct ReductionDtypeTest : public cudf::test::BaseFixture {
  template <typename T_in, typename T_out>
  void reduction_test(std::vector<int> &int_values,
                      T_out expected_value,
                      bool succeeded_condition,
                      std::unique_ptr<aggregation> const &agg,
                      cudf::data_type out_dtype,
                      bool expected_overflow = false)
  {
    std::vector<T_in> input_values = convert_values<T_in>(int_values);
    cudf::test::fixed_width_column_wrapper<T_in> const col(input_values.begin(),
                                                           input_values.end());

    auto statement = [&]() {
      std::unique_ptr<cudf::scalar> result = cudf::reduce(col, agg, out_dtype);
      using ScalarType                     = cudf::scalar_type_t<T_out>;
      auto result1                         = static_cast<ScalarType *>(result.get());
      if (result1->is_valid() && !expected_overflow) {
        EXPECT_EQ(expected_value, result1->value());
      }
    };

    if (succeeded_condition) {
      CUDF_EXPECT_NO_THROW(statement());
    } else {
      EXPECT_ANY_THROW(statement());
    }
  }
};

// test case for different output precision
TEST_F(ReductionDtypeTest, different_precision)
{
  constexpr bool expected_overflow = true;
  std::vector<int> int_values({6, -14, 13, 109, -13, -20, 0, 98, 122, 123});
  int expected_value = std::accumulate(int_values.begin(), int_values.end(), 0);
  auto sum_agg       = cudf::make_sum_aggregation();

  // over flow
  this->reduction_test<int8_t, int8_t>(int_values,
                                       static_cast<int8_t>(expected_value),
                                       true,
                                       sum_agg,
                                       cudf::data_type(cudf::type_id::INT8),
                                       expected_overflow);

  this->reduction_test<int8_t, int64_t>(int_values,
                                        static_cast<int64_t>(expected_value),
                                        true,
                                        sum_agg,
                                        cudf::data_type(cudf::type_id::INT64));

  this->reduction_test<int8_t, double>(int_values,
                                       static_cast<double>(expected_value),
                                       true,
                                       sum_agg,
                                       cudf::data_type(cudf::type_id::FLOAT64));

  // down cast (over flow)
  this->reduction_test<double, int8_t>(int_values,
                                       static_cast<int8_t>(expected_value),
                                       true,
                                       sum_agg,
                                       cudf::data_type(cudf::type_id::INT8),
                                       expected_overflow);

  // down cast (no over flow)
  this->reduction_test<double, int16_t>(int_values,
                                        static_cast<int16_t>(expected_value),
                                        true,
                                        sum_agg,
                                        cudf::data_type(cudf::type_id::INT16));

  // not supported case:
  // wrapper classes other than bool are not convertible
  this->reduction_test<cudf::timestamp_D, cudf::timestamp_s>(
    int_values,
    cudf::timestamp_s{cudf::duration_s(expected_value)},
    false,
    sum_agg,
    cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS));

  this->reduction_test<cudf::timestamp_s, cudf::timestamp_ns>(
    int_values,
    cudf::timestamp_ns{cudf::duration_ns(expected_value)},
    false,
    sum_agg,
    cudf::data_type(cudf::type_id::TIMESTAMP_NANOSECONDS));

  this->reduction_test<int8_t, cudf::timestamp_us>(
    int_values,
    cudf::timestamp_us{cudf::duration_us(expected_value)},
    false,
    sum_agg,
    cudf::data_type(cudf::type_id::TIMESTAMP_MICROSECONDS));

  /*TODO reimplement after Dictionary support
    this->reduction_test<cudf::timestamp_s, cudf::category>
        (int_values, static_cast<cudf::category>(expected_value), false,
         sum_agg, cudf::data_type(cudf::CATEGORY));

    this->reduction_test<int8_t, cudf::category>
        (int_values, static_cast<cudf::category>(expected_value), false,
         sum_agg, cudf::data_type(cudf::CATEGORY));

    this->reduction_test<bool, cudf::date32>
        (int_values, static_cast<cudf::date32>(expected_value), false,
         sum_agg, cudf::data_type(cudf::CATEGORY));
         )
    this->reduction_test<int8_t, cudf::nvstring_category>
        (int_values, static_cast<cudf::nvstring_category>(expected_value), false,
           sum_agg, GDF_STRING_CATEGORY);
     */

  std::vector<bool> v = convert_values<bool>(int_values);

  // When summing bool values into an non-bool arithmetic type,
  // it's an integer/float sum of ones and zeros.
  int expected = std::accumulate(v.begin(), v.end(), int{0});

  this->reduction_test<bool, int8_t>(
    int_values, static_cast<int8_t>(expected), true, sum_agg, cudf::data_type(cudf::type_id::INT8));
  this->reduction_test<bool, int16_t>(int_values,
                                      static_cast<int16_t>(expected),
                                      true,
                                      sum_agg,
                                      cudf::data_type(cudf::type_id::INT16));
  this->reduction_test<bool, int32_t>(int_values,
                                      static_cast<int32_t>(expected),
                                      true,
                                      sum_agg,
                                      cudf::data_type(cudf::type_id::INT32));
  this->reduction_test<bool, int64_t>(int_values,
                                      static_cast<int64_t>(expected),
                                      true,
                                      sum_agg,
                                      cudf::data_type(cudf::type_id::INT64));
  this->reduction_test<bool, float>(int_values,
                                    static_cast<float>(expected),
                                    true,
                                    sum_agg,
                                    cudf::data_type(cudf::type_id::FLOAT32));
  this->reduction_test<bool, double>(int_values,
                                     static_cast<double>(expected),
                                     true,
                                     sum_agg,
                                     cudf::data_type(cudf::type_id::FLOAT64));

  // make sure boolean arithmetic semantics are obeyed when reducing to a bool
  this->reduction_test<bool, bool>(
    int_values, true, true, sum_agg, cudf::data_type(cudf::type_id::BOOL8));

  this->reduction_test<int32_t, bool>(
    int_values, true, true, sum_agg, cudf::data_type(cudf::type_id::BOOL8));

  // cudf::timestamp_s and int64_t are not convertible types.
  this->reduction_test<cudf::timestamp_s, int64_t>(int_values,
                                                   static_cast<int64_t>(expected_value),
                                                   false,
                                                   sum_agg,
                                                   cudf::data_type(cudf::type_id::INT64));
}

struct ReductionErrorTest : public cudf::test::BaseFixture {
};

// test case for empty input cases
TEST_F(ReductionErrorTest, empty_column)
{
  using T        = int32_t;
  auto statement = [](const cudf::column_view col) {
    std::unique_ptr<cudf::scalar> result =
      cudf::reduce(col, cudf::make_sum_aggregation(), cudf::data_type(cudf::type_id::INT64));
    EXPECT_EQ(result->is_valid(), false);
  };

  // default column_view{} is an empty column
  // empty column_view
  CUDF_EXPECT_NO_THROW(statement(cudf::column_view{}));

  // test if the size of input column is zero
  // expect result.is_valid() is false
  std::vector<T> empty_data(0);
  cudf::test::fixed_width_column_wrapper<T> const col0(empty_data.begin(), empty_data.end());
  CUDF_EXPECT_NO_THROW(statement(col0));

  // test if null count is equal or greater than size of input
  // expect result.is_valid() is false
  int col_size = 5;
  std::vector<T> col_data(col_size);
  std::vector<bool> valids(col_size, 0);

  cudf::test::fixed_width_column_wrapper<T> col_empty = construct_null_column(col_data, valids);
  CUDF_EXPECT_NO_THROW(statement(col_empty));
}

// ----------------------------------------------------------------------------

struct ReductionParamTest : public ReductionTest<double>,
                            public ::testing::WithParamInterface<cudf::size_type> {
};

INSTANTIATE_TEST_CASE_P(ddofParam, ReductionParamTest, ::testing::Range(1, 5));

TEST_P(ReductionParamTest, std_var)
{
  int ddof = GetParam();
  std::vector<double> int_values({-3, 2, 1, 0, 5, -3, -2, 28});
  std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1});

  auto calc_var = [ddof](std::vector<double> &v, cudf::size_type valid_count) {
    double mean = std::accumulate(v.begin(), v.end(), double{0});
    mean /= valid_count;

    double sum_of_sq = std::accumulate(
      v.begin(), v.end(), double{0}, [](double acc, double i) { return acc + i * i; });

    cudf::size_type div = valid_count - ddof;

    double var = sum_of_sq / div - ((mean * mean) * valid_count) / div;
    return var;
  };

  // test without nulls
  cudf::test::fixed_width_column_wrapper<double> col(int_values.begin(), int_values.end());

  double var   = calc_var(int_values, int_values.size());
  double std   = std::sqrt(var);
  auto var_agg = cudf::make_variance_aggregation(/*ddof = 1*/ ddof);
  auto std_agg = cudf::make_std_aggregation(/*ddof = 1*/ ddof);

  this->reduction_test(col, var, true, var_agg, cudf::data_type(cudf::type_id::FLOAT64));
  this->reduction_test(col, std, true, std_agg, cudf::data_type(cudf::type_id::FLOAT64));

  // test with nulls
  cudf::test::fixed_width_column_wrapper<double> col_nulls =
    construct_null_column(int_values, host_bools);
  cudf::size_type valid_count =
    cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
  auto replaced_array = replace_nulls<double>(int_values, host_bools, int{0});

  double var_nulls = calc_var(replaced_array, valid_count);
  double std_nulls = std::sqrt(var_nulls);

  this->reduction_test(
    col_nulls, var_nulls, true, var_agg, cudf::data_type(cudf::type_id::FLOAT64));
  this->reduction_test(
    col_nulls, std_nulls, true, std_agg, cudf::data_type(cudf::type_id::FLOAT64));
}

//-------------------------------------------------------------------
struct StringReductionTest : public cudf::test::BaseFixture,
                             public testing::WithParamInterface<std::vector<std::string>> {
  // Min/Max

  void reduction_test(const cudf::column_view underlying_column,
                      std::string expected_value,
                      bool succeeded_condition,
                      std::unique_ptr<aggregation> const &agg,
                      cudf::data_type output_dtype = cudf::data_type{})
  {
    if (cudf::data_type{} == output_dtype) output_dtype = underlying_column.type();

    auto statement = [&]() {
      std::unique_ptr<cudf::scalar> result = cudf::reduce(underlying_column, agg, output_dtype);
      using ScalarType                     = cudf::scalar_type_t<cudf::string_view>;
      auto result1                         = static_cast<ScalarType *>(result.get());
      EXPECT_TRUE(result1->is_valid());
      if (!result1->is_valid())
        std::cout << "expected=" << expected_value << ",got=" << result1->to_string() << std::endl;
      EXPECT_EQ(expected_value, result1->to_string())
        << (agg->kind == aggregation::MIN ? "MIN" : "MAX");
    };

    if (succeeded_condition) {
      CUDF_EXPECT_NO_THROW(statement());
    } else {
      EXPECT_ANY_THROW(statement());
    }
  }
};

// ------------------------------------------------------------------------
std::vector<std::string> string_list[] = {
  {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"},
  {"", "two", "three", "four", "five", "six", "seven", "eight", "nine"},
  {"one", "", "three", "four", "five", "six", "seven", "eight", "nine"},
  {"", "", "", "four", "five", "six", "seven", "eight", "nine"},
  {"", "", "", "", "", "", "", "", ""},
  // DeviceMin identity sentinel test cases
  {"\xF7\xBF\xBF\xBF", "", "", "", "", "", "", "", ""},
  {"one", "two", "three", "four", "\xF7\xBF\xBF\xBF", "six", "seven", "eight", "nine"},
  {"one", "two", "\xF7\xBF\xBF\xBF", "four", "five", "six", "seven", "eight", "nine"},
};
INSTANTIATE_TEST_CASE_P(string_cases, StringReductionTest, testing::ValuesIn(string_list));
TEST_P(StringReductionTest, MinMax)
{
  // data and valid arrays
  std::vector<std::string> host_strings(GetParam());
  std::vector<bool> host_bools({1, 0, 1, 1, 1, 1, 0, 0, 1});
  bool succeed(true);

  // all valid string column
  cudf::test::strings_column_wrapper col(host_strings.begin(), host_strings.end());

  std::string expected_min_result = *(std::min_element(host_strings.begin(), host_strings.end()));
  std::string expected_max_result = *(std::max_element(host_strings.begin(), host_strings.end()));

  // string column with nulls
  cudf::test::strings_column_wrapper col_nulls(
    host_strings.begin(), host_strings.end(), host_bools.begin());

  std::vector<std::string> r_strings;
  std::copy_if(host_strings.begin(),
               host_strings.end(),
               std::back_inserter(r_strings),
               [host_bools, i = 0](auto s) mutable { return host_bools[i++]; });

  std::string expected_min_null_result = *(std::min_element(r_strings.begin(), r_strings.end()));
  std::string expected_max_null_result = *(std::max_element(r_strings.begin(), r_strings.end()));

  // MIN
  this->reduction_test(col, expected_min_result, succeed, cudf::make_min_aggregation());
  this->reduction_test(col_nulls, expected_min_null_result, succeed, cudf::make_min_aggregation());
  // MAX
  this->reduction_test(col, expected_max_result, succeed, cudf::make_max_aggregation());
  this->reduction_test(col_nulls, expected_max_null_result, succeed, cudf::make_max_aggregation());
}

TEST_F(StringReductionTest, AllNull)
{
  // data and all null arrays
  std::vector<std::string> host_strings(
    {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"});
  std::vector<bool> host_bools(host_strings.size(), false);

  // string column with nulls
  cudf::test::strings_column_wrapper col_nulls(
    host_strings.begin(), host_strings.end(), host_bools.begin());
  cudf::data_type output_dtype = cudf::column_view(col_nulls).type();

  // MIN
  std::unique_ptr<cudf::scalar> minresult =
    cudf::reduce(col_nulls, cudf::make_min_aggregation(), output_dtype);
  EXPECT_FALSE(minresult->is_valid());
  // MAX
  std::unique_ptr<cudf::scalar> maxresult =
    cudf::reduce(col_nulls, cudf::make_max_aggregation(), output_dtype);
  EXPECT_FALSE(maxresult->is_valid());
}

TYPED_TEST(ReductionTest, Median)
{
  using T = TypeParam;
  //{-20, -14, -13,  0, 6, 13, 45, 64/None} =  3.0, 0.0
  std::vector<int> int_values({6, -14, 13, 64, 0, -13, -20, 45});
  std::vector<bool> host_bools({1, 1, 1, 0, 1, 1, 1, 1});
  std::vector<T> v = convert_values<T>(int_values);

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  double expected_value = [] {
    if (std::is_same<T, bool>::value) return 1.0;
    if (std::is_signed<T>::value) return 3.0;
    return 13.5;
  }();
  this->reduction_test(
    col, expected_value, this->ret_non_arithmetic, cudf::make_median_aggregation());

  auto col_odd              = cudf::split(col, {1})[1];
  double expected_value_odd = [] {
    if (std::is_same<T, bool>::value) return 1.0;
    if (std::is_signed<T>::value) return 0.0;
    return 14.0;
  }();
  this->reduction_test(
    col_odd, expected_value_odd, this->ret_non_arithmetic, cudf::make_median_aggregation());
  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  double expected_null_value                          = [] {
    if (std::is_same<T, bool>::value) return 1.0;
    if (std::is_signed<T>::value) return 0.0;
    return 13.0;
  }();

  this->reduction_test(
    col_nulls, expected_null_value, this->ret_non_arithmetic, cudf::make_median_aggregation());

  auto col_nulls_odd             = cudf::split(col_nulls, {1})[1];
  double expected_null_value_odd = [] {
    if (std::is_same<T, bool>::value) return 1.0;
    if (std::is_signed<T>::value) return -6.5;
    return 13.5;
  }();
  this->reduction_test(col_nulls_odd,
                       expected_null_value_odd,
                       this->ret_non_arithmetic,
                       cudf::make_median_aggregation());
}

TYPED_TEST(ReductionTest, Quantile)
{
  using T = TypeParam;
  //{-20, -14, -13,  0, 6, 13, 45, 64/None}
  std::vector<int> int_values({6, -14, 13, 64, 0, -13, -20, 45});
  std::vector<bool> host_bools({1, 1, 1, 0, 1, 1, 1, 1});
  std::vector<T> v = convert_values<T>(int_values);
  cudf::interpolation interp{cudf::interpolation::LINEAR};

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  double expected_value0 = std::is_same<T, bool>::value || std::is_unsigned<T>::value ? v[4] : v[6];
  this->reduction_test(
    col, expected_value0, this->ret_non_arithmetic, cudf::make_quantile_aggregation({0.0}, interp));
  double expected_value1 = v[3];
  this->reduction_test(
    col, expected_value1, this->ret_non_arithmetic, cudf::make_quantile_aggregation({1.0}, interp));

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  double expected_null_value1                         = v[7];

  this->reduction_test(col_nulls,
                       expected_value0,
                       this->ret_non_arithmetic,
                       cudf::make_quantile_aggregation({0}, interp));
  this->reduction_test(col_nulls,
                       expected_null_value1,
                       this->ret_non_arithmetic,
                       cudf::make_quantile_aggregation({1}, interp));
}

TYPED_TEST(ReductionTest, UniqueCount)
{
  using T = TypeParam;
  std::vector<int> int_values({1, -3, 1, 2, 0, 2, -4, 45});  // 6 unique values
  std::vector<bool> host_bools({1, 1, 1, 0, 1, 1, 1, 1});
  std::vector<T> v = convert_values<T>(int_values);

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  cudf::size_type expected_value = std::is_same<T, bool>::value ? 2 : 6;
  this->reduction_test(col,
                       expected_value,
                       this->ret_non_arithmetic,
                       cudf::make_nunique_aggregation(cudf::null_policy::INCLUDE));
  this->reduction_test(col,
                       expected_value,
                       this->ret_non_arithmetic,
                       cudf::make_nunique_aggregation(cudf::null_policy::EXCLUDE));

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  cudf::size_type expected_null_value0                = std::is_same<T, bool>::value ? 3 : 7;
  cudf::size_type expected_null_value1                = std::is_same<T, bool>::value ? 2 : 6;

  this->reduction_test(col_nulls,
                       expected_null_value0,
                       this->ret_non_arithmetic,
                       cudf::make_nunique_aggregation(cudf::null_policy::INCLUDE));
  this->reduction_test(col_nulls,
                       expected_null_value1,
                       this->ret_non_arithmetic,
                       cudf::make_nunique_aggregation(cudf::null_policy::EXCLUDE));
}

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, FixedPointReductionProduct)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto const ONE   = decimalXX{1, scale_type{0}};
  auto const TWO   = decimalXX{2, scale_type{0}};
  auto const THREE = decimalXX{3, scale_type{0}};
  auto const FOUR  = decimalXX{4, scale_type{0}};
  // auto const _24   = decimalXX{24, scale_type{0}};

  auto const in     = std::vector<decimalXX>{ONE, TWO, THREE, FOUR};
  auto const column = cudf::test::fixed_width_column_wrapper<decimalXX>(in.cbegin(), in.cend());
  // auto const expected = std::accumulate(in.cbegin(), in.cend(), ONE,
  // std::multiplies<decimalXX>());
  auto const out_type = static_cast<cudf::column_view>(column).type();

  EXPECT_THROW(cudf::reduce(column, cudf::make_product_aggregation(), out_type), cudf::logic_error);

  // auto const result        = cudf::reduce(column, cudf::make_product_aggregation(), out_type);
  // auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

  // EXPECT_EQ(result_scalar->value(), expected);
  // EXPECT_EQ(result_scalar->value(), _24);
}

TYPED_TEST(FixedPointTestBothReps, FixedPointReductionSum)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  // auto const ZERO  = decimalXX{0, scale_type{0}};
  auto const ONE   = decimalXX{1, scale_type{0}};
  auto const TWO   = decimalXX{2, scale_type{0}};
  auto const THREE = decimalXX{3, scale_type{0}};
  auto const FOUR  = decimalXX{4, scale_type{0}};
  // auto const TEN   = decimalXX{10, scale_type{0}};

  auto const in     = std::vector<decimalXX>{ONE, TWO, THREE, FOUR};
  auto const column = cudf::test::fixed_width_column_wrapper<decimalXX>(in.cbegin(), in.cend());
  // auto const expected = std::accumulate(in.cbegin(), in.cend(), ZERO, std::plus<decimalXX>());
  auto const out_type = static_cast<cudf::column_view>(column).type();

  EXPECT_THROW(cudf::reduce(column, cudf::make_sum_aggregation(), out_type), cudf::logic_error);

  // auto const result        = cudf::reduce(column, cudf::make_sum_aggregation(), out_type);
  // auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

  // EXPECT_EQ(result_scalar->value(), expected);
  // EXPECT_EQ(result_scalar->value(), TEN);
}

TYPED_TEST(ReductionTest, NthElement)
{
  using T = TypeParam;
  std::vector<int> int_values(4000);
  std::iota(int_values.begin(), int_values.end(), 0);
  std::vector<bool> host_bools(int_values.size());
  auto valid_condition = [](auto i) { return (i % 3 and i % 7); };
  std::transform(int_values.begin(), int_values.end(), host_bools.begin(), valid_condition);

  cudf::size_type valid_count = std::count(host_bools.begin(), host_bools.end(), true);
  std::vector<int> int_values_valid(valid_count);
  std::copy_if(int_values.begin(), int_values.end(), int_values_valid.begin(), valid_condition);

  std::vector<T> v           = convert_values<T>(int_values);
  std::vector<T> v_valid     = convert_values<T>(int_values_valid);
  cudf::size_type input_size = v.size();

  auto mod = [](int a, int b) { return (a % b + b) % b; };
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  // without nulls
  for (cudf::size_type n :
       {-input_size, -input_size / 2, -2, -1, 0, 1, 2, input_size / 2, input_size - 1}) {
    auto const index         = mod(n, v.size());
    T expected_value_nonull  = v[index];
    bool const expected_null = !host_bools[index];
    this->reduction_test(col,
                         expected_value_nonull,
                         true,
                         cudf::make_nth_element_aggregation(n, cudf::null_policy::INCLUDE));
    this->reduction_test(col,
                         expected_value_nonull,
                         true,
                         cudf::make_nth_element_aggregation(n, cudf::null_policy::EXCLUDE));
    this->reduction_test(col_nulls,
                         expected_value_nonull,
                         true,
                         cudf::make_nth_element_aggregation(n, cudf::null_policy::INCLUDE),
                         cudf::data_type{},
                         expected_null);
  }
  // valid only
  for (cudf::size_type n :
       {-valid_count, -valid_count / 2, -2, -1, 0, 1, 2, valid_count / 2, valid_count - 1}) {
    T expected_value_null = v_valid[mod(n, v_valid.size())];
    this->reduction_test(col_nulls,
                         expected_value_null,
                         true,
                         cudf::make_nth_element_aggregation(n, cudf::null_policy::EXCLUDE));
  }
  // error cases
  for (cudf::size_type n : {-input_size - 1, input_size}) {
    this->reduction_test(
      col, T{}, false, cudf::make_nth_element_aggregation(n, cudf::null_policy::INCLUDE));
    this->reduction_test(
      col_nulls, T{}, false, cudf::make_nth_element_aggregation(n, cudf::null_policy::INCLUDE));
    this->reduction_test(
      col, T{}, false, cudf::make_nth_element_aggregation(n, cudf::null_policy::EXCLUDE));
    this->reduction_test(
      col_nulls, T{}, false, cudf::make_nth_element_aggregation(n, cudf::null_policy::EXCLUDE));
  }
}

CUDF_TEST_PROGRAM_MAIN()
