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

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <type_traits>
#include <stdlib.h>

#include <gtest/gtest.h>

#include <cudf/cudf.h>
#include <cudf/reduction.hpp>

#include <thrust/device_vector.h>

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/scalar_wrapper.cuh>

template <typename T>
std::vector<T> convert_values(std::vector<int> const & int_values)
{
    std::vector<T> v(int_values.size());
    std::transform(int_values.begin(), int_values.end(), v.begin(),
        [](int x) { return static_cast<T>(x); } );
    return v;
}

template <typename T>
cudf::test::column_wrapper<T> construct_null_column(std::vector<T> const & values,
    std::vector<bool> const & bools)
{
    if( values.size() > bools.size() ){
        throw std::logic_error("input vector size mismatch.");
    }
    return cudf::test::column_wrapper<T>(
        values.size(),
        [&values](gdf_index_type row) { return values[row]; },
        [&bools](gdf_index_type row) { return bools[row]; });
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
struct ReductionTest : public GdfTest
{
    // Sum/Prod/SumOfSquare never support non arithmetics
    static constexpr bool ret_non_arithmetic =
        (std::is_arithmetic<T>::value || std::is_same<T, cudf::bool8>::value)
            ? true : false;

    ReductionTest(){}

    ~ReductionTest(){}

    template <typename T_out>
    void reduction_test(cudf::test::column_wrapper<T> &col,
        T_out expected_value, bool succeeded_condition,
        cudf::reduction::operators op, gdf_dtype output_dtype = N_GDF_TYPES,
        gdf_size_type ddof = 1)
    {
        const gdf_column * underlying_column = col.get();
        thrust::device_vector<T_out> dev_result(1);

        if( N_GDF_TYPES == output_dtype) output_dtype = underlying_column->dtype;

        auto statement = [&]() {
            cudf::test::scalar_wrapper<T_out> result
                = cudf::reduce(underlying_column, op, output_dtype, ddof);
            EXPECT_EQ(expected_value, result.value());
        };

        if( succeeded_condition ){
            CUDF_EXPECT_NO_THROW(statement());
        }else{
            EXPECT_ANY_THROW(statement());
        }
    }
};

using AllTypes = testing::Types<
    int8_t,int16_t, int32_t, int64_t, float, double,
    cudf::bool8, cudf::date32, cudf::date64, cudf::timestamp,
    cudf::category, cudf::nvstring_category>;

TYPED_TEST_CASE(ReductionTest, AllTypes);

// ------------------------------------------------------------------------
TYPED_TEST(ReductionTest, MinMax)
{
   using T = TypeParam;
   std::vector<int> int_values({5, 0, -120, -111, 0, 64, 63, 99, 123, -16});
   std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1, 0, 1});
   std::vector<T> v = convert_values<T>(int_values);

   // Min/Max succeeds for any gdf types including
   // non-arithmetic types (date32, date64, timestamp, category)
   bool result_error(true);

   // test without nulls
   cudf::test::column_wrapper<T> col(v);

   T expected_min_result = *( std::min_element(v.begin(), v.end()) );
   T expected_max_result = *( std::max_element(v.begin(), v.end()) );
   this->reduction_test(col, expected_min_result, result_error, cudf::reduction::MIN);
   this->reduction_test(col, expected_max_result, result_error, cudf::reduction::MAX);

   // test with nulls
   cudf::test::column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
   gdf_size_type valid_count = col_nulls.size() - col_nulls.null_count();

   auto r_min = replace_nulls(v, host_bools, std::numeric_limits<T>::max() );
   auto r_max = replace_nulls(v, host_bools, std::numeric_limits<T>::lowest() );

   T expected_min_null_result = *( std::min_element(r_min.begin(), r_min.end()) );
   T expected_max_null_result = *( std::max_element(r_max.begin(), r_max.end()) );

   this->reduction_test(col_nulls, expected_min_null_result, result_error, cudf::reduction::MIN);
   this->reduction_test(col_nulls, expected_max_null_result, result_error, cudf::reduction::MAX);
}

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
    cudf::test::column_wrapper<T> col(v);
    TypeParam expected_value = calc_prod(v);

    this->reduction_test(col, expected_value, this->ret_non_arithmetic,
                         cudf::reduction::PRODUCT);

    // test with nulls
    cudf::test::column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    gdf_size_type valid_count = col_nulls.size() - col_nulls.null_count();
    auto r = replace_nulls(v, host_bools, T{1});
    TypeParam expected_null_value = calc_prod(r);

    this->reduction_test(col_nulls, expected_null_value, this->ret_non_arithmetic,
        cudf::reduction::PRODUCT);

}

TYPED_TEST(ReductionTest, Sum)
{
    using T = TypeParam;
    std::vector<int> int_values({6, -14, 13, 64, 0, -13, -20, 45});
    std::vector<bool> host_bools({1, 1, 0, 0, 1, 1, 1, 1});
    std::vector<T> v = convert_values<T>(int_values);

    // test without nulls
    cudf::test::column_wrapper<T> col(v);
    T expected_value = std::accumulate(v.begin(), v.end(), T{0});
    this->reduction_test(col, expected_value, this->ret_non_arithmetic, cudf::reduction::SUM);

    // test with nulls
    cudf::test::column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    gdf_size_type valid_count = col_nulls.size() - col_nulls.null_count();
    auto r = replace_nulls(v, host_bools, T{0});
    T expected_null_value = std::accumulate(r.begin(), r.end(), T{0});

    this->reduction_test(col_nulls, expected_null_value, this->ret_non_arithmetic,
        cudf::reduction::SUM);
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
    cudf::test::column_wrapper<T> col(v);
    T expected_value = calc_reduction(v);

    this->reduction_test(col, expected_value, this->ret_non_arithmetic,
                         cudf::reduction::SUMOFSQUARES);

    // test with nulls
    cudf::test::column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    gdf_size_type valid_count = col_nulls.size() - col_nulls.null_count();
    auto r = replace_nulls(v, host_bools, T{0});
    T expected_null_value = calc_reduction(r);

    this->reduction_test(col_nulls, expected_null_value, this->ret_non_arithmetic,
        cudf::reduction::SUMOFSQUARES);
}

// ----------------------------------------------------------------------------

template <typename T>
struct MultiStepReductionTest : public ReductionTest<T>
{
    MultiStepReductionTest(){}
    ~MultiStepReductionTest(){}
};

using MultiStepReductionTypes = testing::Types<
    int8_t,int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_CASE(MultiStepReductionTest, MultiStepReductionTypes);

TYPED_TEST(MultiStepReductionTest, Mean)
{
    using T = TypeParam;
    std::vector<int> int_values({-3, 2,  1, 0, 5, -3, -2, 28});
    std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1});

    auto calc_mean = [](std::vector<int>& v, gdf_size_type valid_count){
        double sum = std::accumulate(v.begin(), v.end(), double{0});
        return sum / valid_count ;
    };

    // test without nulls
    std::vector<T> v = convert_values<T>(int_values);
    cudf::test::column_wrapper<T> col(v);
    double expected_value = calc_mean(int_values, int_values.size());
    this->reduction_test(col, expected_value, true,
        cudf::reduction::MEAN, GDF_FLOAT64);

    // test with nulls
    cudf::test::column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    gdf_size_type valid_count = col_nulls.size() - col_nulls.null_count();
    auto replaced_array = replace_nulls(int_values, host_bools, int{0});

    double expected_value_nulls = calc_mean(replaced_array, valid_count);
    this->reduction_test(col_nulls, expected_value_nulls, true,
        cudf::reduction::MEAN, GDF_FLOAT64);
}


TYPED_TEST(MultiStepReductionTest, var_std)
{
    using T = TypeParam;
    std::vector<int> int_values({-3, 2,  1, 0, 5, -3, -2, 28});
    std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1});

    auto calc_var = [](std::vector<int>& v, gdf_size_type valid_count){
        double mean = std::accumulate(v.begin(), v.end(), double{0});
        mean /= valid_count ;

        double sum_of_sq = std::accumulate(v.begin(), v.end(), double{0},
            [](double acc, TypeParam i) { return acc + i * i; });

        int ddof = 1;
        gdf_size_type div = valid_count - ddof;

        double var = sum_of_sq / div - ((mean * mean) * valid_count) /div;
        return var;
    };

    // test without nulls
    std::vector<T> v = convert_values<T>(int_values);
    cudf::test::column_wrapper<T> col(v);

    double var = calc_var(int_values, int_values.size());
    double std = std::sqrt(var);

    this->reduction_test(col, var, true, cudf::reduction::VAR, GDF_FLOAT64);
    this->reduction_test(col, std, true, cudf::reduction::STD, GDF_FLOAT64);

    // test with nulls
    cudf::test::column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    gdf_size_type valid_count = col_nulls.size() - col_nulls.null_count();
    auto replaced_array = replace_nulls(int_values, host_bools, int{0});

    double var_nulls = calc_var(replaced_array, valid_count);
    double std_nulls = std::sqrt(var_nulls);

    this->reduction_test(col_nulls, var_nulls, true, cudf::reduction::VAR, GDF_FLOAT64);
    this->reduction_test(col_nulls, std_nulls, true, cudf::reduction::STD, GDF_FLOAT64);
}

// ----------------------------------------------------------------------------

template <typename T>
struct ReductionMultiStepErrorCheck : public ReductionTest<T>
{
    ReductionMultiStepErrorCheck(){}
    ~ReductionMultiStepErrorCheck(){}

    void reduction_error_check(cudf::test::column_wrapper<T> &col,
        bool succeeded_condition,
        cudf::reduction::operators op, gdf_dtype output_dtype)
    {
        const gdf_column * underlying_column = col.get();
        auto statement = [&]() {
            cudf::reduce(underlying_column, op, output_dtype);
        };

        if( succeeded_condition ){
            CUDF_EXPECT_NO_THROW(statement());
        }else{
            EXPECT_ANY_THROW(statement());
        }
    }
};

TYPED_TEST_CASE(ReductionMultiStepErrorCheck, AllTypes);

TYPED_TEST(ReductionMultiStepErrorCheck, ErrorHandling)
{
    using T = TypeParam;
    std::vector<int> int_values({-3, 2});
    std::vector<bool> host_bools({1, 0});

    std::vector<T> v = convert_values<T>(int_values);
    cudf::test::column_wrapper<T> col(v);
    cudf::test::column_wrapper<T> col_nulls = construct_null_column(v, host_bools);

    bool is_input_accpetable = this->ret_non_arithmetic;

    std::vector<gdf_dtype> dtypes(N_GDF_TYPES+1);
    int i=0;
    std::generate(dtypes.begin(), dtypes.end(), [&](){ return static_cast<gdf_dtype>(i++); });

    auto is_supported_outdtype = [](gdf_dtype dtype){
        if( dtype == GDF_FLOAT32)return true;
        if( dtype == GDF_FLOAT64)return true;
        return false;
    };

    auto evaluate = [&](gdf_dtype dtype) mutable {
        bool expect_succeed = is_input_accpetable & is_supported_outdtype(dtype);
        this->reduction_error_check(col, expect_succeed, cudf::reduction::MEAN, dtype);
        this->reduction_error_check(col, expect_succeed, cudf::reduction::VAR,  dtype);
        this->reduction_error_check(col, expect_succeed, cudf::reduction::STD,  dtype);

        this->reduction_error_check(col_nulls, expect_succeed, cudf::reduction::MEAN, dtype);
        this->reduction_error_check(col_nulls, expect_succeed, cudf::reduction::VAR,  dtype);
        this->reduction_error_check(col_nulls, expect_succeed, cudf::reduction::STD,  dtype);
        return;
    };

    std::for_each(dtypes.begin(), dtypes.end(), evaluate);
}


// ----------------------------------------------------------------------------

struct ReductionDtypeTest : public GdfTest
{
    template <typename T_in, typename T_out>
    void reduction_test(std::vector<int> & int_values,
        T_out expected_value, bool succeeded_condition,
        cudf::reduction::operators op, gdf_dtype out_dtype,
        bool expected_overflow = false)
    {
        std::vector<T_in> input_values = convert_values<T_in>(int_values);
        cudf::test::column_wrapper<T_in> const col(input_values);

        auto statement = [&]() {
            cudf::test::scalar_wrapper<T_out> result =
                cudf::reduce(col.get(), op, out_dtype);
            if( result.is_valid() && ! expected_overflow){
                EXPECT_EQ(expected_value, result.value());
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
         cudf::reduction::SUM, GDF_INT8, expected_overflow);

    this->reduction_test<int8_t, int64_t>
        (int_values, static_cast<int64_t>(expected_value), true,
         cudf::reduction::SUM, GDF_INT64);

    this->reduction_test<int8_t, double>
        (int_values, static_cast<double>(expected_value), true,
         cudf::reduction::SUM, GDF_FLOAT64);

    // down cast (over flow)
    this->reduction_test<double, int8_t>
        (int_values, static_cast<int8_t>(expected_value), true,
         cudf::reduction::SUM, GDF_INT8, expected_overflow);

    // down cast (no over flow)
    this->reduction_test<double, int16_t>
        (int_values, static_cast<int16_t>(expected_value), true,
         cudf::reduction::SUM, GDF_INT16);

    // not supported case:
    // wrapper classes other than cudf::bool8 are not convertible
    this->reduction_test<cudf::date64, cudf::timestamp>
        (int_values, static_cast<cudf::timestamp>(expected_value), false,
         cudf::reduction::SUM, GDF_TIMESTAMP);

    this->reduction_test<cudf::date32, cudf::category>
        (int_values, static_cast<cudf::category>(expected_value), false,
         cudf::reduction::SUM, GDF_CATEGORY);

    this->reduction_test<cudf::date32, cudf::date64>
        (int_values, static_cast<cudf::date64>(expected_value), false,
         cudf::reduction::SUM, GDF_DATE64);

    this->reduction_test<int8_t, cudf::timestamp>
        (int_values, static_cast<cudf::timestamp>(expected_value), false,
         cudf::reduction::SUM, GDF_TIMESTAMP);

    this->reduction_test<int8_t, cudf::nvstring_category>
        (int_values, static_cast<cudf::nvstring_category>(expected_value), false,
           cudf::reduction::SUM, GDF_STRING_CATEGORY);

    this->reduction_test<int8_t, cudf::category>
        (int_values, static_cast<cudf::category>(expected_value), false,
         cudf::reduction::SUM, GDF_CATEGORY);

    this->reduction_test<cudf::bool8, cudf::date32>
        (int_values, static_cast<cudf::date32>(expected_value), false,
         cudf::reduction::SUM, GDF_CATEGORY);

    // supported case: cudf::bool8
    std::vector<bool> v = convert_values<bool>(int_values);

    // When summing bool8 values into an non-bool arithmetic type,
    // it's an integer/float sum of ones and zeros.
    int expected_bool8 = std::accumulate(v.begin(), v.end(), int{0});

    this->reduction_test<cudf::bool8, int8_t>
        (int_values, static_cast<int8_t>(expected_bool8), true,
         cudf::reduction::SUM, GDF_INT8);
    this->reduction_test<cudf::bool8, int16_t>
        (int_values, static_cast<int16_t>(expected_bool8), true,
         cudf::reduction::SUM, GDF_INT16);
    this->reduction_test<cudf::bool8, int32_t>
        (int_values, static_cast<int32_t>(expected_bool8), true,
         cudf::reduction::SUM, GDF_INT32);
    this->reduction_test<cudf::bool8, int64_t>
        (int_values, static_cast<int64_t>(expected_bool8), true,
         cudf::reduction::SUM, GDF_INT64);
    this->reduction_test<cudf::bool8, float>
        (int_values, static_cast<float>(expected_bool8), true,
         cudf::reduction::SUM, GDF_FLOAT32);
    this->reduction_test<cudf::bool8, double>
        (int_values, static_cast<double>(expected_bool8), true,
         cudf::reduction::SUM, GDF_FLOAT64);

    // make sure boolean arithmetic semantics are obeyed when
    // reducing to a bool
    this->reduction_test<cudf::bool8, cudf::bool8>
        (int_values, cudf::true_v, true, cudf::reduction::SUM, GDF_BOOL8);

    // TODO: should this work? Currently "input type not convertible to output"
    /*this->reduction_test<int32_t, cudf::bool8>
        (int_values, cudf::true_v, false, cudf::reduction::SUM, GDF_BOOL8);*/

    // Though the underlying type of cudf::date64 is int64_t,
    // they are not convertible types.
    this->reduction_test<cudf::date64, int64_t>
        (int_values, static_cast<int64_t>(expected_value), false,
         cudf::reduction::SUM, GDF_INT64);

}

struct ReductionErrorTest : public GdfTest{};

// test case for empty input cases
TEST_F(ReductionErrorTest, empty_column)
{
    using T = int32_t;
    auto statement = [](const gdf_column* col) {
        cudf::test::scalar_wrapper<T> result =
            cudf::reduce(col, cudf::reduction::SUM, GDF_INT64);
        EXPECT_EQ( result.is_valid(), false );
    };

    // test input column is nullptr, reduction throws an error if input is nullptr
    EXPECT_ANY_THROW(statement(nullptr));

    // test if the size of input column is zero
    // expect result.is_valid() is false
    cudf::test::column_wrapper<T> const col0(0);
    CUDF_EXPECT_NO_THROW(statement(col0.get()));

    // test if null count is equal or greater than size of input
    // expect result.is_valid() is false
    int col_size = 5;
    std::vector<T> col_data(col_size);
    std::vector<gdf_valid_type> valids(gdf_valid_allocation_size(col_size));
    std::fill(valids.begin(), valids.end(), 0);

    cudf::test::column_wrapper<T> col_empty(col_data, valids);
    CUDF_EXPECT_NO_THROW(statement(col_empty.get()));
}


// ----------------------------------------------------------------------------

struct ReductionParamTest : public ReductionTest<double>,
                            public ::testing::WithParamInterface<gdf_size_type>
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

    auto calc_var = [ddof](std::vector<double>& v, gdf_size_type valid_count){
        double mean = std::accumulate(v.begin(), v.end(), double{0});
        mean /= valid_count ;

        double sum_of_sq = std::accumulate(v.begin(), v.end(), double{0},
            [](double acc, double i) { return acc + i * i; });

        gdf_size_type div = valid_count - ddof;

        double var = sum_of_sq / div - ((mean * mean) * valid_count) /div;
        return var;
    };

    // test without nulls
    cudf::test::column_wrapper<double> col(int_values);

    double var = calc_var(int_values, int_values.size());
    double std = std::sqrt(var);

    this->reduction_test(col, var, true, cudf::reduction::VAR, GDF_FLOAT64, ddof);
    this->reduction_test(col, std, true, cudf::reduction::STD, GDF_FLOAT64, ddof);

    // test with nulls
    cudf::test::column_wrapper<double> col_nulls = construct_null_column(int_values, host_bools);
    gdf_size_type valid_count = col_nulls.size() - col_nulls.null_count();
    auto replaced_array = replace_nulls<double>(int_values, host_bools, int{0});

    double var_nulls = calc_var(replaced_array, valid_count);
    double std_nulls = std::sqrt(var_nulls);

    this->reduction_test(col_nulls, var_nulls, true, cudf::reduction::VAR, GDF_FLOAT64, ddof);
    this->reduction_test(col_nulls, std_nulls, true, cudf::reduction::STD, GDF_FLOAT64, ddof);
}
