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

#include <iostream>
#include <vector>
#include <map>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
//TODO remove after PR 3490 merge
#include <tests/utilities/legacy/cudf_test_utils.cuh>

#include <cudf/cudf.h>
#include <cudf/reduction.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <thrust/device_vector.h>

#include <cudf/detail/aggregation/aggregation.hpp>
using aggregation = cudf::experimental::aggregation;

template <typename T>
std::vector<T> convert_values(std::vector<int> const & int_values)
{
    std::vector<T> v(int_values.size());
    std::transform(int_values.begin(), int_values.end(), v.begin(),
        [](int x) { return static_cast<T>(x); } );
    return v;
}

template <typename T>
cudf::test::fixed_width_column_wrapper<T> construct_null_column(std::vector<T> const & values,
    std::vector<bool> const & bools)
{
    if( values.size() > bools.size() ){
        throw std::logic_error("input vector size mismatch.");
    }
    return cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end(), bools.begin());
}

template <typename T>
std::vector<T> replace_nulls(
    std::vector<T> const & values,
    std::vector<bool> const & bools,
    T identity)
{
    std::vector<T> v(values.size());
    std::transform(values.begin(), values.end(), bools.begin(),
        v.begin(), [identity](T x, bool b) { return (b)? x : identity; } );
    return v;
}

// ------------------------------------------------------------------------

// This is the main test feature
template <typename T>
struct ReductionTest : public cudf::test::BaseFixture
{
    // Sum/Prod/SumOfSquare never support non arithmetics
    static constexpr bool ret_non_arithmetic =
        (std::is_arithmetic<T>::value || std::is_same<T, cudf::experimental::bool8>::value)
            ? true : false;

    ReductionTest(){}

    ~ReductionTest(){}

    template <typename T_out>
    void reduction_test(
        const cudf::column_view underlying_column,
        T_out expected_value, bool succeeded_condition,
        std::unique_ptr<aggregation> const &agg)
    {
        auto statement = [&]() {
          std::unique_ptr<cudf::scalar> result
                = cudf::experimental::reduce(underlying_column, agg);
            using ScalarType = cudf::experimental::scalar_type_t<T_out>;
            auto result1 = static_cast<ScalarType *>(result.get());
            EXPECT_EQ(expected_value, result1->value());
        };

        if( succeeded_condition ){
            CUDF_EXPECT_NO_THROW(statement());
        }else{
            EXPECT_ANY_THROW(statement());
        }
    }
};

template <typename T>
struct MinMaxReductionTest : public ReductionTest<T>  { };

using MinMaxTypes = cudf::test::AllTypes;
TYPED_TEST_CASE(MinMaxReductionTest, MinMaxTypes);

// ------------------------------------------------------------------------
TYPED_TEST(MinMaxReductionTest, MinMax)
{
   using T = TypeParam;
   std::vector<int> int_values({5, 0, -120, -111, 0, 64, 63, 99, 123, -16});
   std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1, 0, 1});
   std::vector<T> v = convert_values<T>(int_values);

   // Min/Max succeeds for any gdf types including
   // non-arithmetic types (date32, date64, timestamp, category)
   bool result_error(true);

   // test without nulls
   cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());

   T expected_min_result = *( std::min_element(v.begin(), v.end()) );
   T expected_max_result = *( std::max_element(v.begin(), v.end()) );
   this->reduction_test(col, expected_min_result, result_error, cudf::experimental::make_min_aggregation());
   this->reduction_test(col, expected_max_result, result_error, cudf::experimental::make_max_aggregation());

   // test with nulls
   cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
   cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();

   auto r_min = replace_nulls(v, host_bools, std::numeric_limits<T>::max() );
   auto r_max = replace_nulls(v, host_bools, std::numeric_limits<T>::lowest() );

   T expected_min_null_result = *( std::min_element(r_min.begin(), r_min.end()) );
   T expected_max_null_result = *( std::max_element(r_max.begin(), r_max.end()) );

   this->reduction_test(col_nulls, expected_min_null_result, result_error, cudf::experimental::make_min_aggregation());
   this->reduction_test(col_nulls, expected_max_null_result, result_error, cudf::experimental::make_max_aggregation());
}

TYPED_TEST_CASE(ReductionTest, cudf::test::NumericTypes);

TYPED_TEST(ReductionTest, Product)
{
    using T = TypeParam;
    std::vector<int> int_values({5, -1, 1, 0, 3, 2, 4});
    std::vector<bool> host_bools({1, 1, 0, 0, 1, 1, 1});
    std::vector<TypeParam> v = convert_values<TypeParam>(int_values);

    auto calc_prod = [](std::vector<T>& v){
        T expected_value = std::accumulate(v.begin(), v.end(), T{1},
        [](T acc, T i) { return acc * i; });
        return expected_value;
    };

    // test without nulls
    cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
    TypeParam expected_value = calc_prod(v);

    this->reduction_test(col, expected_value, this->ret_non_arithmetic,
                         cudf::experimental::make_product_aggregation());

    // test with nulls
    cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
    auto r = replace_nulls(v, host_bools, T{1});
    TypeParam expected_null_value = calc_prod(r);

    this->reduction_test(col_nulls, expected_null_value, this->ret_non_arithmetic,
        cudf::experimental::make_product_aggregation());

}

TYPED_TEST(ReductionTest, Sum)
{
    using T = TypeParam;
    std::vector<int> int_values({6, -14, 13, 64, 0, -13, -20, 45});
    std::vector<bool> host_bools({1, 1, 0, 0, 1, 1, 1, 1});
    std::vector<T> v = convert_values<T>(int_values);

    // test without nulls
    cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
    T expected_value = std::accumulate(v.begin(), v.end(), T{0});
    this->reduction_test(col, expected_value, this->ret_non_arithmetic, cudf::experimental::make_sum_aggregation());

    // test with nulls
    cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
    auto r = replace_nulls(v, host_bools, T{0});
    T expected_null_value = std::accumulate(r.begin(), r.end(), T{0});

    this->reduction_test(col_nulls, expected_null_value, this->ret_non_arithmetic,
        cudf::experimental::make_sum_aggregation());
}

TYPED_TEST(ReductionTest, SumOfSquare)
{
    using T = TypeParam;
    std::vector<int> int_values({-3, 2,  1, 0, 5, -3, -2});
    std::vector<bool> host_bools({1, 1, 0, 0, 1, 1, 1, 1});
    std::vector<T> v = convert_values<T>(int_values);

    auto calc_reduction = [](std::vector<T>& v){
        T value = std::accumulate(v.begin(), v.end(), T{0},
        [](T acc, T i) { return acc + i * i; });
        return value;
    };

    // test without nulls
    cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
    T expected_value = calc_reduction(v);

    this->reduction_test(col, expected_value, this->ret_non_arithmetic,
                         cudf::experimental::make_sum_of_squares_aggregation());

    // test with nulls
    cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
    auto r = replace_nulls(v, host_bools, T{0});
    T expected_null_value = calc_reduction(r);

    this->reduction_test(col_nulls, expected_null_value, this->ret_non_arithmetic,
        cudf::experimental::make_sum_of_squares_aggregation());
}

//TODO TYPED_TEST case for AllTypes
struct ReductionAnyAllTest : public ReductionTest<cudf::experimental::bool8> {
    ReductionAnyAllTest(){}
    ~ReductionAnyAllTest(){}
};

TEST_F(ReductionAnyAllTest, AnyAllTrueTrue)
{
   using T = cudf::experimental::bool8;
   std::vector<int> int_values({true, true, true, true});
   std::vector<bool> host_bools({1, 1, 0, 1});
   std::vector<T> v = convert_values<T>(int_values);

   // Min/Max succeeds for any gdf types including
   // non-arithmetic types (date32, date64, timestamp, category)
   bool result_error(true);
   T expected = true;

   // test without nulls
   cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());

   this->reduction_test(col, expected, result_error, cudf::experimental::make_any_aggregation());
   this->reduction_test(col, expected, result_error, cudf::experimental::make_all_aggregation());

   // test with nulls
   cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);

   this->reduction_test(col_nulls, expected, result_error, cudf::experimental::make_any_aggregation());
   this->reduction_test(col_nulls, expected, result_error, cudf::experimental::make_all_aggregation());
}

TEST_F(ReductionAnyAllTest, AnyAllFalseFalse)
{
   using T = cudf::experimental::bool8;
   std::vector<int> int_values({false, false, false, false});
   std::vector<bool> host_bools({1, 1, 0, 1});
   std::vector<T> v = convert_values<T>(int_values);

   // Min/Max succeeds for any gdf types including
   // non-arithmetic types (date32, date64, timestamp, category)
   bool result_error(true);
   T expected = false;

   // test without nulls
   cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());

   this->reduction_test(col, expected, result_error, cudf::experimental::make_any_aggregation());
   this->reduction_test(col, expected, result_error, cudf::experimental::make_all_aggregation());

   // test with nulls
   cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);

   this->reduction_test(col_nulls, expected, result_error, cudf::experimental::make_any_aggregation());
   this->reduction_test(col_nulls, expected, result_error, cudf::experimental::make_all_aggregation());
}

// ----------------------------------------------------------------------------

template <typename T>
struct MultiStepReductionTest : public ReductionTest<T>
{
    MultiStepReductionTest(){}
    ~MultiStepReductionTest(){}
};

using MultiStepReductionTypes = cudf::test::NumericTypes;
TYPED_TEST_CASE(MultiStepReductionTest, MultiStepReductionTypes);

TYPED_TEST(MultiStepReductionTest, Mean)
{
    using T = TypeParam;
    std::vector<int> int_values({-3, 2,  1, 0, 5, -3, -2, 28});
    std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1});

    auto calc_mean = [](std::vector<T>& v, cudf::size_type valid_count){
        double sum = std::accumulate(v.begin(), v.end(), double{0});
        return sum / valid_count ;
    };

    // test without nulls
    std::vector<T> v = convert_values<T>(int_values);
    cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
    double expected_value = calc_mean(v, v.size());
    this->reduction_test(col, expected_value, true,
        cudf::experimental::make_mean_aggregation());

    // test with nulls
    cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
    auto replaced_array = replace_nulls(v, host_bools, T{0});

    double expected_value_nulls = calc_mean(replaced_array, valid_count);
    this->reduction_test(col_nulls, expected_value_nulls, true,
        cudf::experimental::make_mean_aggregation());
}


TYPED_TEST(MultiStepReductionTest, var_std)
{
    using T = TypeParam;
    std::vector<int> int_values({-3, 2,  1, 0, 5, -3, -2, 28});
    std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1});

    auto calc_var = [](std::vector<T>& v, cudf::size_type valid_count){
        double mean = std::accumulate(v.begin(), v.end(), double{0});
        mean /= valid_count ;

        double sum_of_sq = std::accumulate(v.begin(), v.end(), double{0},
            [](double acc, TypeParam i) { return acc + i * i; });

        int ddof = 1;
        cudf::size_type div = valid_count - ddof;

        double var = sum_of_sq / div - ((mean * mean) * valid_count) /div;
        return var;
    };

    // test without nulls
    std::vector<T> v = convert_values<T>(int_values);
    cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());

    double var = calc_var(v, v.size());
    double std = std::sqrt(var);
    auto var_agg = cudf::experimental::make_variance_aggregation(/*ddof =*/ 1);
    auto std_agg = cudf::experimental::make_std_aggregation(/*ddof =*/ 1);

    this->reduction_test(col, var, true, var_agg);
    this->reduction_test(col, std, true, std_agg);

    // test with nulls
    cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
    auto replaced_array = replace_nulls(v, host_bools, T{0});

    double var_nulls = calc_var(replaced_array, valid_count);
    double std_nulls = std::sqrt(var_nulls);

    this->reduction_test(col_nulls, var_nulls, true, var_agg);
    this->reduction_test(col_nulls, std_nulls, true, std_agg);
}

// ----------------------------------------------------------------------------

template <typename T>
struct ReductionMultiStepErrorCheck : public ReductionTest<T>
{
    ReductionMultiStepErrorCheck(){}
    ~ReductionMultiStepErrorCheck(){}

    void reduction_error_check(cudf::test::fixed_width_column_wrapper<T> &col,
        bool succeeded_condition,
        std::unique_ptr<aggregation> const &agg)
    {
        const cudf::column_view underlying_column = col;
        auto statement = [&]() {
            cudf::experimental::reduce(underlying_column, agg);
        };

        if( succeeded_condition) {
            CUDF_EXPECT_NO_THROW(statement());
        }else{
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

    bool is_input_acceptable = this->ret_non_arithmetic;
    bool expect_succeed = is_input_acceptable;
    auto var_agg = cudf::experimental::make_variance_aggregation(/*ddof = 1*/);
    auto std_agg = cudf::experimental::make_std_aggregation(/*ddof = 1*/);
    this->reduction_error_check(col, expect_succeed, cudf::experimental::make_mean_aggregation());
    this->reduction_error_check(col, expect_succeed, var_agg);
    this->reduction_error_check(col, expect_succeed, std_agg);

    this->reduction_error_check(col_nulls, expect_succeed, cudf::experimental::make_mean_aggregation());
    this->reduction_error_check(col_nulls, expect_succeed, var_agg);
    this->reduction_error_check(col_nulls, expect_succeed, std_agg);
}


// ----------------------------------------------------------------------------

template<typename T>
struct ReductionDtypeTest : public ReductionTest<T>
{
    ReductionDtypeTest(){}
    ~ReductionDtypeTest(){}

    //reduction_error_check
    void reduction_test(cudf::test::fixed_width_column_wrapper<T> &col,
        std::string const& name,
        std::unique_ptr<aggregation> const &agg,
        bool succeeded_condition)
    {
        const cudf::column_view underlying_column = col;
        auto statement = [&]() {
            cudf::experimental::reduce(underlying_column, agg);
        };

        //std::cout<< name << ":" << std::endl;
        if( succeeded_condition) {
            EXPECT_NO_THROW(statement()) << name;
        }else{
            EXPECT_ANY_THROW(statement()) << name;
        }
    }
};

TYPED_TEST_CASE(ReductionDtypeTest, cudf::test::AllTypes);

// test case for only supported operations on different types
TYPED_TEST(ReductionDtypeTest, supported_aggregations)
{
    using T = TypeParam;
    //MIN, MAX - all
    //ANY, ALL - all except string, timestamp
    //SUM, MEAN, - all except string, timestamp
    //PRODUCT, SUM_OF_SQUARES, STD, VAR - numeric types only.
    std::vector<int> int_values({6, -14, 13, 109, -13, -20, 0, 98, 122, 123});
    std::vector<bool> host_bools({1,  1,  0,    1,   1,  1, 0,  1,   1,   1});

    std::vector<T> v = convert_values<T>(int_values);
    cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
    cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);

    // supported
    std::map<std::string, std::unique_ptr<cudf::experimental::aggregation>> supported_aggs;
    // unsupported
    std::map<std::string, std::unique_ptr<cudf::experimental::aggregation>> unsupported_aggs;
    supported_aggs["min"]=std::move(cudf::experimental::make_min_aggregation());
    supported_aggs["max"]=std::move(cudf::experimental::make_max_aggregation());
    supported_aggs["sum"]=std::move(cudf::experimental::make_sum_aggregation());

    (cudf::is_timestamp<T>() ? unsupported_aggs["any"] : supported_aggs["any"])
     =std::move(cudf::experimental::make_any_aggregation());
    (cudf::is_timestamp<T>() ? unsupported_aggs["all"] : supported_aggs["all"])
     =std::move(cudf::experimental::make_all_aggregation());
    (cudf::is_timestamp<T>() ? unsupported_aggs["mean"] : supported_aggs["mean"])
     =std::move(cudf::experimental::make_mean_aggregation());
    (cudf::is_timestamp<T>() ? unsupported_aggs["product"] : supported_aggs["product"])
     =std::move(cudf::experimental::make_product_aggregation());
    (cudf::is_timestamp<T>() ? unsupported_aggs["SoS"] : supported_aggs["SoS"])
     =std::move(cudf::experimental::make_sum_of_squares_aggregation());
    (cudf::is_timestamp<T>() ? unsupported_aggs["std"] : supported_aggs["std"])
     =std::move(cudf::experimental::make_std_aggregation(1));
    (cudf::is_timestamp<T>() ? unsupported_aggs["var"] : supported_aggs["var"])
     =std::move(cudf::experimental::make_variance_aggregation(1));

    for(auto& agg: supported_aggs)
        this->reduction_test(col, agg.first, agg.second, true);
        
    for(auto& agg: unsupported_aggs)
        this->reduction_test(col, agg.first, agg.second, false);
}

struct ReductionErrorTest : public cudf::test::BaseFixture {};

// test case for empty input cases
TEST_F(ReductionErrorTest, empty_column)
{
    using T = int32_t;
    auto statement = [](const cudf::column_view col) {
        std::unique_ptr<cudf::scalar> result =
            cudf::experimental::reduce(col, cudf::experimental::make_sum_aggregation());
        EXPECT_EQ( result->is_valid(), false );
    };

    // test input column is nullptr, reduction throws an error if input is nullptr
    //TODO(karthikeyann) invalid column_view
    //EXPECT_ANY_THROW(statement(nullptr));

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
                            public ::testing::WithParamInterface<cudf::size_type>
{
    ReductionParamTest(){}
    ~ReductionParamTest(){}
};

INSTANTIATE_TEST_CASE_P(ddofParam,
    ReductionParamTest,
    ::testing::Range(1,5));

TEST_P(ReductionParamTest, std_var)
{
    int ddof = GetParam();
    std::vector<double> int_values({-3, 2,  1, 0, 5, -3, -2, 28});
    std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1});

    auto calc_var = [ddof](std::vector<double>& v, cudf::size_type valid_count){
        double mean = std::accumulate(v.begin(), v.end(), double{0});
        mean /= valid_count ;

        double sum_of_sq = std::accumulate(v.begin(), v.end(), double{0},
            [](double acc, double i) { return acc + i * i; });

        cudf::size_type div = valid_count - ddof;

        double var = sum_of_sq / div - ((mean * mean) * valid_count) /div;
        return var;
    };

    // test without nulls
    cudf::test::fixed_width_column_wrapper<double> col(int_values.begin(), int_values.end());

    double var = calc_var(int_values, int_values.size());
    double std = std::sqrt(var);
    auto var_agg = cudf::experimental::make_variance_aggregation(/*ddof = 1*/ddof);
    auto std_agg = cudf::experimental::make_std_aggregation(/*ddof = 1*/ddof);

    this->reduction_test(col, var, true, var_agg);
    this->reduction_test(col, std, true, std_agg);

    // test with nulls
    cudf::test::fixed_width_column_wrapper<double> col_nulls = construct_null_column(int_values, host_bools);
    cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
    auto replaced_array = replace_nulls<double>(int_values, host_bools, int{0});

    double var_nulls = calc_var(replaced_array, valid_count);
    double std_nulls = std::sqrt(var_nulls);

    this->reduction_test(col_nulls, var_nulls, true, var_agg);
    this->reduction_test(col_nulls, std_nulls, true, std_agg);
}

//-------------------------------------------------------------------
struct StringReductionTest : public cudf::test::BaseFixture,
  public testing::WithParamInterface<std::vector<std::string> >
{
  // Min/Max
  StringReductionTest() {}

  ~StringReductionTest() {}

  void reduction_test(const cudf::column_view underlying_column,
                      std::string expected_value, bool succeeded_condition,
                      std::unique_ptr<aggregation> const &agg)
  {
    auto statement = [&]() {
      std::unique_ptr<cudf::scalar> result =
          cudf::experimental::reduce(underlying_column, agg);
      using ScalarType = cudf::experimental::scalar_type_t<cudf::string_view>;
      auto result1 = static_cast<ScalarType*>(result.get());
      EXPECT_TRUE(result1->is_valid());
      std::cout<<"expected="<<expected_value<<",got="<<result1->to_string()<<std::endl;
      EXPECT_EQ(expected_value, result1->to_string()) << (agg->kind == aggregation::MIN ? "MIN" : "MAX" );
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
//DeviceMin identity sentinel test cases
{"\xF7\xBF\xBF\xBF", "", "", "", "", "", "", "", ""},
{"one", "two", "three", "four", "\xF7\xBF\xBF\xBF", "six", "seven", "eight", "nine"},
{"one", "two", "\xF7\xBF\xBF\xBF", "four", "five", "six", "seven", "eight", "nine"},
};
INSTANTIATE_TEST_CASE_P(string_cases, StringReductionTest,
                         testing::ValuesIn(string_list));
TEST_P(StringReductionTest, MinMax)
{
  // data and valid arrays
  std::vector<std::string> host_strings(GetParam());
  std::vector<bool> host_bools(         {    1,     0,       1,      1,      1,     1,       0,       0,      1});
  bool succeed(true);

  // all valid string column
  cudf::test::strings_column_wrapper col(host_strings.begin(), host_strings.end());

  std::string expected_min_result = *( std::min_element(host_strings.begin(), host_strings.end()) );
  std::string expected_max_result = *( std::max_element(host_strings.begin(), host_strings.end()) );

  // string column with nulls
  cudf::test::strings_column_wrapper col_nulls(host_strings.begin(), host_strings.end(), host_bools.begin());

  std::vector<std::string> r_strings;
  std::copy_if(host_strings.begin(), host_strings.end(), 
    std::back_inserter(r_strings),
    [host_bools, i = 0](auto s) mutable { return host_bools[i++]; });

  std::string expected_min_null_result = *( std::min_element(r_strings.begin(), r_strings.end()) );
  std::string expected_max_null_result = *( std::max_element(r_strings.begin(), r_strings.end()) );

  //MIN
  this->reduction_test(col, expected_min_result, succeed, cudf::experimental::make_min_aggregation());
  this->reduction_test(col_nulls, expected_min_null_result, succeed, cudf::experimental::make_min_aggregation());
  //MAX
  this->reduction_test(col, expected_max_result, succeed,  cudf::experimental::make_max_aggregation());
  this->reduction_test(col_nulls, expected_max_null_result, succeed,  cudf::experimental::make_max_aggregation());
}

TEST_F(StringReductionTest, AllNull)
{
  // data and all null arrays
  std::vector<std::string> host_strings({"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"});
  std::vector<bool> host_bools(host_strings.size(), false);

  // string column with nulls
  cudf::test::strings_column_wrapper col_nulls(host_strings.begin(), host_strings.end(), host_bools.begin());

  //MIN
  std::unique_ptr<cudf::scalar> minresult = cudf::experimental::reduce(col_nulls, cudf::experimental::make_min_aggregation());
  EXPECT_FALSE(minresult->is_valid());
  //MAX
  std::unique_ptr<cudf::scalar> maxresult = cudf::experimental::reduce(col_nulls, cudf::experimental::make_max_aggregation());
  EXPECT_FALSE(maxresult->is_valid());
}
