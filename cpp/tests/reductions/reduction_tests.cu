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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
//TODO remove after PR 3490 merge
#include <tests/utilities/legacy/cudf_test_utils.cuh>

#include <cudf/cudf.h>
#include <cudf/reduction.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <thrust/device_vector.h>

using reduction_op = cudf::experimental::reduction_op;

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
        reduction_op op, cudf::data_type output_dtype = cudf::data_type{},
        cudf::size_type ddof = 1)
    {
        if( cudf::data_type{} == output_dtype) output_dtype = underlying_column.type();

        auto statement = [&]() {
          std::unique_ptr<cudf::scalar> result
                = cudf::experimental::reduce(underlying_column, op, output_dtype, ddof);
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
   this->reduction_test(col, expected_min_result, result_error, reduction_op::MIN);
   this->reduction_test(col, expected_max_result, result_error, reduction_op::MAX);

   // test with nulls
   cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
   cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();

   auto r_min = replace_nulls(v, host_bools, std::numeric_limits<T>::max() );
   auto r_max = replace_nulls(v, host_bools, std::numeric_limits<T>::lowest() );

   T expected_min_null_result = *( std::min_element(r_min.begin(), r_min.end()) );
   T expected_max_null_result = *( std::max_element(r_max.begin(), r_max.end()) );

   this->reduction_test(col_nulls, expected_min_null_result, result_error, reduction_op::MIN);
   this->reduction_test(col_nulls, expected_max_null_result, result_error, reduction_op::MAX);
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
                         reduction_op::PRODUCT);

    // test with nulls
    cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
    auto r = replace_nulls(v, host_bools, T{1});
    TypeParam expected_null_value = calc_prod(r);

    this->reduction_test(col_nulls, expected_null_value, this->ret_non_arithmetic,
        reduction_op::PRODUCT);

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
    this->reduction_test(col, expected_value, this->ret_non_arithmetic, reduction_op::SUM);

    // test with nulls
    cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
    auto r = replace_nulls(v, host_bools, T{0});
    T expected_null_value = std::accumulate(r.begin(), r.end(), T{0});

    this->reduction_test(col_nulls, expected_null_value, this->ret_non_arithmetic,
        reduction_op::SUM);
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
                         reduction_op::SUMOFSQUARES);

    // test with nulls
    cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
    auto r = replace_nulls(v, host_bools, T{0});
    T expected_null_value = calc_reduction(r);

    this->reduction_test(col_nulls, expected_null_value, this->ret_non_arithmetic,
        reduction_op::SUMOFSQUARES);
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

   this->reduction_test(col, expected, result_error, reduction_op::ANY);
   this->reduction_test(col, expected, result_error, reduction_op::ALL);

   // test with nulls
   cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);

   this->reduction_test(col_nulls, expected, result_error, reduction_op::ANY);
   this->reduction_test(col_nulls, expected, result_error, reduction_op::ALL);
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

   this->reduction_test(col, expected, result_error, reduction_op::ANY);
   this->reduction_test(col, expected, result_error, reduction_op::ALL);

   // test with nulls
   cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);

   this->reduction_test(col_nulls, expected, result_error, reduction_op::ANY);
   this->reduction_test(col_nulls, expected, result_error, reduction_op::ALL);
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
        reduction_op::MEAN, cudf::data_type(cudf::FLOAT64));

    // test with nulls
    cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
    auto replaced_array = replace_nulls(v, host_bools, T{0});

    double expected_value_nulls = calc_mean(replaced_array, valid_count);
    this->reduction_test(col_nulls, expected_value_nulls, true,
        reduction_op::MEAN, cudf::data_type(cudf::FLOAT64));
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

    this->reduction_test(col, var, true, reduction_op::VAR, cudf::data_type(cudf::FLOAT64));
    this->reduction_test(col, std, true, reduction_op::STD, cudf::data_type(cudf::FLOAT64));

    // test with nulls
    cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
    auto replaced_array = replace_nulls(v, host_bools, T{0});

    double var_nulls = calc_var(replaced_array, valid_count);
    double std_nulls = std::sqrt(var_nulls);

    this->reduction_test(col_nulls, var_nulls, true, reduction_op::VAR, cudf::data_type(cudf::FLOAT64));
    this->reduction_test(col_nulls, std_nulls, true, reduction_op::STD, cudf::data_type(cudf::FLOAT64));
}

// ----------------------------------------------------------------------------

template <typename T>
struct ReductionMultiStepErrorCheck : public ReductionTest<T>
{
    ReductionMultiStepErrorCheck(){}
    ~ReductionMultiStepErrorCheck(){}

    void reduction_error_check(cudf::test::fixed_width_column_wrapper<T> &col,
        bool succeeded_condition,
        reduction_op op, cudf::data_type output_dtype)
    {
        const cudf::column_view underlying_column = col;
        auto statement = [&]() {
            cudf::experimental::reduce(underlying_column, op, output_dtype);
        };

        if( succeeded_condition ){
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

    bool is_input_accpetable = this->ret_non_arithmetic;

    std::vector<cudf::data_type> dtypes(cudf::NUM_TYPE_IDS+1);
    int i=0;
    std::generate(dtypes.begin(), dtypes.end(), [&](){ return cudf::data_type(static_cast<cudf::type_id>(i++)); });

    auto is_supported_outdtype = [](cudf::data_type dtype) {
      if (dtype == cudf::data_type(cudf::FLOAT32)) return true;
      if (dtype == cudf::data_type(cudf::FLOAT64)) return true;
      return false;
    };

    auto evaluate = [&](cudf::data_type dtype) mutable {
        bool expect_succeed = is_input_accpetable & is_supported_outdtype(dtype);
        this->reduction_error_check(col, expect_succeed, reduction_op::MEAN, dtype);
        this->reduction_error_check(col, expect_succeed, reduction_op::VAR,  dtype);
        this->reduction_error_check(col, expect_succeed, reduction_op::STD,  dtype);

        this->reduction_error_check(col_nulls, expect_succeed, reduction_op::MEAN, dtype);
        this->reduction_error_check(col_nulls, expect_succeed, reduction_op::VAR,  dtype);
        this->reduction_error_check(col_nulls, expect_succeed, reduction_op::STD,  dtype);
        return;
    };

    std::for_each(dtypes.begin(), dtypes.end(), evaluate);
}


// ----------------------------------------------------------------------------

struct ReductionDtypeTest : public cudf::test::BaseFixture
{
    template <typename T_in, typename T_out>
    void reduction_test(std::vector<int> & int_values,
        T_out expected_value, bool succeeded_condition,
        reduction_op op, cudf::data_type out_dtype,
        bool expected_overflow = false)
    {
        std::vector<T_in> input_values = convert_values<T_in>(int_values);
        cudf::test::fixed_width_column_wrapper<T_in> const col(input_values.begin(), input_values.end());

        auto statement = [&]() {
            std::unique_ptr<cudf::scalar> result = 
                cudf::experimental::reduce(col, op, out_dtype);
            using ScalarType = cudf::experimental::scalar_type_t<T_out>;
            auto result1 = static_cast<ScalarType *>(result.get());
            if( result1->is_valid() && ! expected_overflow){
                EXPECT_EQ(expected_value, result1->value());
            }
        };

        if( succeeded_condition ){
            CUDF_EXPECT_NO_THROW(statement());
        }else{
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

    // over flow
    this->reduction_test<int8_t, int8_t>
        (int_values, static_cast<int8_t>(expected_value), true,
         reduction_op::SUM, cudf::data_type(cudf::INT8), expected_overflow);

    this->reduction_test<int8_t, int64_t>
        (int_values, static_cast<int64_t>(expected_value), true,
         reduction_op::SUM, cudf::data_type(cudf::INT64));

    this->reduction_test<int8_t, double>
        (int_values, static_cast<double>(expected_value), true,
         reduction_op::SUM, cudf::data_type(cudf::FLOAT64));

    // down cast (over flow)
    this->reduction_test<double, int8_t>
        (int_values, static_cast<int8_t>(expected_value), true,
         reduction_op::SUM, cudf::data_type(cudf::INT8), expected_overflow);

    // down cast (no over flow)
    this->reduction_test<double, int16_t>
        (int_values, static_cast<int16_t>(expected_value), true,
         reduction_op::SUM, cudf::data_type(cudf::INT16));

    // not supported case:
    // wrapper classes other than cudf::experimental::bool8 are not convertible
    this->reduction_test<cudf::timestamp_D, cudf::timestamp_s>
        (int_values, static_cast<cudf::timestamp_s>(expected_value), false,
         reduction_op::SUM, cudf::data_type(cudf::TIMESTAMP_SECONDS));

    this->reduction_test<cudf::timestamp_s, cudf::timestamp_ns>
        (int_values, static_cast<cudf::timestamp_ns>(expected_value), false,
         reduction_op::SUM, cudf::data_type(cudf::TIMESTAMP_NANOSECONDS));

    this->reduction_test<int8_t, cudf::timestamp_us>
        (int_values, static_cast<cudf::timestamp_us>(expected_value), false,
         reduction_op::SUM, cudf::data_type(cudf::TIMESTAMP_MICROSECONDS));

    /*TODO enable after category support
    this->reduction_test<cudf::timestamp_s, cudf::category>
        (int_values, static_cast<cudf::category>(expected_value), false,
         reduction_op::SUM, cudf::data_type(cudf::CATEGORY));

    this->reduction_test<int8_t, cudf::category>
        (int_values, static_cast<cudf::category>(expected_value), false,
         reduction_op::SUM, cudf::data_type(cudf::CATEGORY));

    this->reduction_test<cudf::experimental::bool8, cudf::date32>
        (int_values, static_cast<cudf::date32>(expected_value), false,
         reduction_op::SUM, cudf::data_type(cudf::CATEGORY));
         )
    this->reduction_test<int8_t, cudf::nvstring_category>
        (int_values, static_cast<cudf::nvstring_category>(expected_value), false,
           reduction_op::SUM, GDF_STRING_CATEGORY);
     */

    // supported case: cudf::experimental::bool8
    std::vector<bool> v = convert_values<bool>(int_values);

    // When summing bool8 values into an non-bool arithmetic type,
    // it's an integer/float sum of ones and zeros.
    int expected_bool8 = std::accumulate(v.begin(), v.end(), int{0});

    this->reduction_test<cudf::experimental::bool8, int8_t>
        (int_values, static_cast<int8_t>(expected_bool8), true,
         reduction_op::SUM, cudf::data_type(cudf::INT8));
    this->reduction_test<cudf::experimental::bool8, int16_t>
        (int_values, static_cast<int16_t>(expected_bool8), true,
         reduction_op::SUM, cudf::data_type(cudf::INT16));
    this->reduction_test<cudf::experimental::bool8, int32_t>
        (int_values, static_cast<int32_t>(expected_bool8), true,
         reduction_op::SUM, cudf::data_type(cudf::INT32));
    this->reduction_test<cudf::experimental::bool8, int64_t>
        (int_values, static_cast<int64_t>(expected_bool8), true,
         reduction_op::SUM, cudf::data_type(cudf::INT64));
    this->reduction_test<cudf::experimental::bool8, float>
        (int_values, static_cast<float>(expected_bool8), true,
         reduction_op::SUM, cudf::data_type(cudf::FLOAT32));
    this->reduction_test<cudf::experimental::bool8, double>
        (int_values, static_cast<double>(expected_bool8), true,
         reduction_op::SUM, cudf::data_type(cudf::FLOAT64));

    // make sure boolean arithmetic semantics are obeyed when
    // reducing to a bool
    this->reduction_test<cudf::experimental::bool8, cudf::experimental::bool8>
        (int_values, cudf::experimental::true_v, true, reduction_op::SUM, cudf::data_type(cudf::BOOL8));

    this->reduction_test<int32_t, cudf::experimental::bool8>
        (int_values, cudf::experimental::true_v, true, reduction_op::SUM, cudf::data_type(cudf::BOOL8));

    // cudf::timestamp_s and int64_t are not convertible types.
    this->reduction_test<cudf::timestamp_s, int64_t>
        (int_values, static_cast<int64_t>(expected_value), false,
         reduction_op::SUM, cudf::data_type(cudf::INT64));
}

struct ReductionErrorTest : public cudf::test::BaseFixture {};

// test case for empty input cases
TEST_F(ReductionErrorTest, empty_column)
{
    using T = int32_t;
    auto statement = [](const cudf::column_view col) {
        std::unique_ptr<cudf::scalar> result =
            cudf::experimental::reduce(col, reduction_op::SUM, cudf::data_type(cudf::INT64));
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

    this->reduction_test(col, var, true, reduction_op::VAR, cudf::data_type(cudf::FLOAT64), ddof);
    this->reduction_test(col, std, true, reduction_op::STD, cudf::data_type(cudf::FLOAT64), ddof);

    // test with nulls
    cudf::test::fixed_width_column_wrapper<double> col_nulls = construct_null_column(int_values, host_bools);
    cudf::size_type valid_count = cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
    auto replaced_array = replace_nulls<double>(int_values, host_bools, int{0});

    double var_nulls = calc_var(replaced_array, valid_count);
    double std_nulls = std::sqrt(var_nulls);

    this->reduction_test(col_nulls, var_nulls, true, reduction_op::VAR, cudf::data_type(cudf::FLOAT64), ddof);
    this->reduction_test(col_nulls, std_nulls, true, reduction_op::STD, cudf::data_type(cudf::FLOAT64), ddof);
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
                      cudf::experimental::reduction_op op,
                      cudf::data_type output_dtype = cudf::data_type{},
                      cudf::size_type ddof = 1)
  {
    if (cudf::data_type{} == output_dtype)
      output_dtype = underlying_column.type();

    auto statement = [&]() {
      std::unique_ptr<cudf::scalar> result =
          cudf::experimental::reduce(underlying_column, op, output_dtype, ddof);
      using ScalarType = cudf::experimental::scalar_type_t<cudf::string_view>;
      auto result1 = static_cast<ScalarType*>(result.get());
      EXPECT_TRUE(result1->is_valid());
      std::cout<<"expected="<<expected_value<<",got="<<result1->to_string()<<std::endl;
      EXPECT_EQ(expected_value, result1->to_string()) << (op == reduction_op::MIN ? "MIN" : "MAX" );
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
  this->reduction_test(col, expected_min_result, succeed, reduction_op::MIN);
  this->reduction_test(col_nulls, expected_min_null_result, succeed, reduction_op::MIN);
  //MAX
  this->reduction_test(col, expected_max_result, succeed, reduction_op::MAX);
  this->reduction_test(col_nulls, expected_max_null_result, succeed, reduction_op::MAX);
}

TEST_F(StringReductionTest, AllNull)
{
  // data and all null arrays
  std::vector<std::string> host_strings({"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"});
  std::vector<bool> host_bools(host_strings.size(), false);

  // string column with nulls
  cudf::test::strings_column_wrapper col_nulls(host_strings.begin(), host_strings.end(), host_bools.begin());
  cudf::data_type output_dtype = cudf::column_view(col_nulls).type();

  //MIN
  std::unique_ptr<cudf::scalar> minresult = cudf::experimental::reduce(col_nulls, reduction_op::MIN, output_dtype, 1);
  EXPECT_FALSE(minresult->is_valid());
  //MAX
  std::unique_ptr<cudf::scalar> maxresult = cudf::experimental::reduce(col_nulls, reduction_op::MAX, output_dtype, 1);
  EXPECT_FALSE(maxresult->is_valid());
}
