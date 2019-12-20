/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
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

#include <cudf/replace.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_utilities.hpp>

#include <gtest/gtest.h>

struct ClampErrorTest : public cudf::test::BaseFixture{};

TEST_F(ClampErrorTest, MisMatchingScalarTypes)
{
    auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::INT32));  
    lo->set_valid(true); 
    auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::INT64));
    hi->set_valid(true); 

    cudf::test::fixed_width_column_wrapper<int32_t> input({1, 2, 3, 4, 5, 6});

    EXPECT_THROW(cudf::experimental::clamp(input, *lo, *hi), cudf::logic_error);
}


TEST_F(ClampErrorTest, MisMatchingInputAndScalarTypes)
{
    auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::INT32));
    lo->set_valid(true);
    auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::INT32));
    hi->set_valid(true);

    cudf::test::fixed_width_column_wrapper<int64_t> input({1, 2, 3, 4, 5, 6});

    EXPECT_THROW(cudf::experimental::clamp(input, *lo, *hi), cudf::logic_error);
}

struct ClampEmptyCaseTest : public cudf::test::BaseFixture{};

TEST_F(ClampEmptyCaseTest, BothScalarEmptyInvalid)
{
    auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::INT32));  
    lo->set_valid(false); 
    auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::INT32));
    hi->set_valid(false); 

    cudf::test::fixed_width_column_wrapper<int32_t> input({1, 2, 3, 4, 5, 6});

    auto got = cudf::experimental::clamp(input, *lo, *hi);

    cudf::test::expect_columns_equal(input, got->view());
}

TEST_F(ClampEmptyCaseTest, EmptyInput)
{
    auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::INT32));
    lo->set_valid(true); 
    auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::INT32));
    hi->set_valid(true);

    cudf::test::fixed_width_column_wrapper<int32_t> input({});

    auto got = cudf::experimental::clamp(input, *lo, *hi);

    cudf::test::expect_columns_equal(input, got->view());
}

template <class T>
struct ClampTestNumeric : public cudf::test::BaseFixture{
    std::unique_ptr<cudf::column> run_clamp(std::vector<T> input, std::vector<cudf::size_type> input_validity,
                                            T lo, bool lo_validity, T hi, bool hi_validity) {

        using ScalarType = cudf::experimental::scalar_type_t<T>;
        std::unique_ptr<cudf::scalar> lo_scalar{nullptr};
        std::unique_ptr<cudf::scalar> hi_scalar{nullptr}; 
        if (cudf::is_numeric<T>()){
            lo_scalar = cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::experimental::type_to_id<T>()}));
            hi_scalar = cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::experimental::type_to_id<T>()}));
        } else if (cudf::is_timestamp<T>()) {
            lo_scalar = cudf::make_timestamp_scalar(cudf::data_type(cudf::data_type{cudf::experimental::type_to_id<T>()}));
            hi_scalar = cudf::make_timestamp_scalar(cudf::data_type(cudf::data_type{cudf::experimental::type_to_id<T>()}));
        } else if (cudf::is_string<T>())

        static_cast<ScalarType*>(lo_scalar.get())->set_value(lo);
        static_cast<ScalarType*>(lo_scalar.get())->set_valid(lo_validity);
        static_cast<ScalarType*>(hi_scalar.get())->set_value(hi);
        static_cast<ScalarType*>(hi_scalar.get())->set_valid(hi_validity);

        if (input.size() == input_validity.size()) {
            cudf::test::fixed_width_column_wrapper<T> input_column(input.begin(), input.end(), input_validity.begin());
            
            return cudf::experimental::clamp(input_column, *lo_scalar, *hi_scalar);
        } else {
            cudf::test::fixed_width_column_wrapper<T> input_column(input.begin(), input.end());
            return cudf::experimental::clamp(input_column, *lo_scalar, *hi_scalar);
        }

    }
};
using Types = cudf::test::FixedWidthTypes;

TYPED_TEST_CASE(ClampTestNumeric, Types);

TYPED_TEST (ClampTestNumeric, WithNoNull){
    using T = TypeParam;
   
    T lo = 2;
    T hi = 8;
    std::vector<T> input({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    auto got = this->run_clamp(input, {}, lo, true, hi, true); 

    cudf::test::fixed_width_column_wrapper<T> expected({2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8});

    cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST (ClampTestNumeric, LowerNull){
    using T = TypeParam;

    T lo = 2;
    T hi = 8;
    std::vector<T> input({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    auto got = this->run_clamp(input, {}, lo, false, hi, true);

    cudf::test::fixed_width_column_wrapper<T> expected({0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8});

    cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST (ClampTestNumeric, UpperNull){
    using T = TypeParam;

    T lo = 2;
    T hi = 8;
    std::vector<T> input({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    auto got = this->run_clamp(input, {}, lo, true, hi, false);

    cudf::test::fixed_width_column_wrapper<T> expected({2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST (ClampTestNumeric, InputNull){
    using T = TypeParam;

    T lo = 2;
    T hi = 8;
    std::vector<T> input         ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    std::vector<cudf::size_type> input_validity({0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});

    auto got = this->run_clamp(input, input_validity, lo, true, hi, true);

    cudf::test::fixed_width_column_wrapper<T> expected({2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8}, {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});

    cudf::test::expect_columns_equal(expected, got->view());
}

struct ClampStringTest : public cudf::test::BaseFixture{};

TEST_F(ClampStringTest, WithNullableColumn)
{
    std::vector<std::string> strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};
    std::vector<bool>        valids {  1,  1,    1,  0,   1,   1,   1,   1,   0,   1,   1};

    cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

    auto lo = cudf::make_string_scalar("B");
    auto hi = cudf::make_string_scalar("e");
    lo->set_valid(true);
    hi->set_valid(true);

    std::vector<std::string> expected_strings{"B", "b", "c", "D", "e", "F", "G", "H", "i", "e", "B"};

    cudf::test::strings_column_wrapper expected(expected_strings.begin(), expected_strings.end(), valids.begin());

    auto got = cudf::experimental::clamp(input, *lo, *hi);

    cudf::test::expect_columns_equal(expected, got->view());
}

TEST_F(ClampStringTest, WithNonNullableColumn)
{
    std::vector<std::string> strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};

    cudf::test::strings_column_wrapper input(strings.begin(), strings.end());

    auto lo = cudf::make_string_scalar("B");
    auto hi = cudf::make_string_scalar("e");
    lo->set_valid(true);
    hi->set_valid(true);

    std::vector<std::string> expected_strings{"B", "b", "c", "D", "e", "F", "G", "H", "e", "e", "B"};

    cudf::test::strings_column_wrapper expected(expected_strings.begin(), expected_strings.end());

    auto got = cudf::experimental::clamp(input, *lo, *hi);

    cudf::test::expect_columns_equal(expected, got->view());
}

TEST_F(ClampStringTest, WithNullableColumnNullLow)
{
    std::vector<std::string> strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};
    std::vector<bool>        valids {  1,  1,    1,  0,   1,   1,   1,   1,   0,   1,   1};

    cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

    auto lo = cudf::make_string_scalar("B");
    auto hi = cudf::make_string_scalar("e");
    lo->set_valid(false);
    hi->set_valid(true);

    std::vector<std::string> expected_strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "e", "B"};

    cudf::test::strings_column_wrapper expected(expected_strings.begin(), expected_strings.end(), valids.begin());

    auto got = cudf::experimental::clamp(input, *lo, *hi);

    cudf::test::expect_columns_equal(expected, got->view());
}

TEST_F(ClampStringTest, WithNullableColumnNullHigh)
{
    std::vector<std::string> strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};
    std::vector<bool>        valids {  1,  1,    1,  0,   1,   1,   1,   1,   0,   1,   1};

    cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

    auto lo = cudf::make_string_scalar("B");
    auto hi = cudf::make_string_scalar("e");
    lo->set_valid(true);
    hi->set_valid(false);

    std::vector<std::string> expected_strings{"B", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};

    cudf::test::strings_column_wrapper expected(expected_strings.begin(), expected_strings.end(), valids.begin());

    auto got = cudf::experimental::clamp(input, *lo, *hi);

    cudf::test::expect_columns_equal(expected, got->view());
}

TEST_F(ClampStringTest, WithNullableColumnBothLoAndHiNull)
{
    std::vector<std::string> strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};
    std::vector<bool>        valids {  1,  1,    1,  0,   1,   1,   1,   1,   0,   1,   1};

    cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

    auto lo = cudf::make_string_scalar("B");
    auto hi = cudf::make_string_scalar("e");
    lo->set_valid(false);
    hi->set_valid(false);

    auto got = cudf::experimental::clamp(input, *lo, *hi);

    cudf::test::expect_columns_equal(input, got->view());
}
