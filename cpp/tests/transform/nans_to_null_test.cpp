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

#include <cudf/types.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/transform.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>


template <typename T>
struct NaNsToNullTest : public cudf::test::BaseFixture {

    void run_test(cudf::column_view const& input, cudf::column_view const& expected) {

        auto got_mask = cudf::experimental::nans_to_nulls(input);
        cudf::column got (input);
        got.set_null_mask(std::move(*(got_mask.first)));

        EXPECT_EQ(expected.null_count(), got_mask.second);

        cudf::test::expect_columns_equal(expected, got.view());
    }

    std::unique_ptr<cudf::column> create_expected (std::vector<T> const& input, std::vector<bool> const& mask = {}) {
        std::vector<T> expected(input);
        std::vector<bool> expected_mask;

        if (mask.size() > 0) {
            std::transform(input.begin(), input.end(), mask.begin(), std::back_inserter(expected_mask), 
                    [](T val, bool validity){return (std::isnan(val) or validity == false)? false : true;});
        } else {
            std::transform(input.begin(), input.end(), std::back_inserter(expected_mask), 
                    [](T val){return (std::isnan(val))? false : true;});
        }

        return cudf::test::fixed_width_column_wrapper<T> (expected.begin(), expected.end(), expected_mask.begin()).release();

    }
};

using test_types = ::testing::Types<float, double>;

TYPED_TEST_CASE(NaNsToNullTest, test_types);

TYPED_TEST(NaNsToNullTest, WithMask){
    using T = TypeParam;

    std::vector<T> input = {1, NAN, 3, NAN, 5, NAN};
    std::vector<bool> mask = {1,   1, 1,   1, 0,   0};
    auto input_column = cudf::test::fixed_width_column_wrapper<T> (input.begin(), input.end(), mask.begin());
    auto expected_column = this->create_expected(input, mask);
    this->run_test(input_column, expected_column->view());
}

TYPED_TEST(NaNsToNullTest, WithNoMask){
    using T = TypeParam;

    std::vector<T> input = {1, NAN, 3, NAN, 5, NAN};
    auto input_column = cudf::test::fixed_width_column_wrapper<T> (input.begin(), input.end());
    auto expected_column = this->create_expected(input);
    this->run_test(input_column, expected_column->view());
}

TYPED_TEST(NaNsToNullTest, NoNANWithMask){
    using T = TypeParam;
    
    std::vector<T> input =   {1, 2, 3, 4, 5, 6};
    std::vector<bool> mask = {1, 1, 1, 1, 0, 0};
    auto input_column = cudf::test::fixed_width_column_wrapper<T> (input.begin(), input.end(), mask.begin());
    auto expected_column = this->create_expected(input, mask);
    this->run_test(input_column, expected_column->view());
}

TYPED_TEST(NaNsToNullTest, NoNANNoMask){
    using T = TypeParam;

    std::vector<T> input = {1, 2, 3, 4, 5, 6};
    auto input_column = cudf::test::fixed_width_column_wrapper<T> (input.begin(), input.end());
    auto expected_column = this->create_expected(input);
    this->run_test(input_column, expected_column->view());
}

TYPED_TEST(NaNsToNullTest, EmptyColumn){
    using T = TypeParam;

    std::vector<T> input = {};
    auto input_column = cudf::test::fixed_width_column_wrapper<T> (input.begin(), input.end());
    auto expected_column = this->create_expected(input);
    this->run_test(input_column, expected_column->view());
}

struct NaNsToNullFailTest : public cudf::test::BaseFixture{};

TEST_F(NaNsToNullFailTest, StringType){
    std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"};
    cudf::test::strings_column_wrapper input(strings.begin(), strings.end());

    EXPECT_THROW(cudf::experimental::nans_to_nulls(input), cudf::logic_error);
}

TEST_F(NaNsToNullFailTest, IntegerType){
    std::vector<int32_t> input = {1, 2, 3, 4, 5, 6};
    auto input_column = cudf::test::fixed_width_column_wrapper<int32_t> (input.begin(), input.end());
    
    EXPECT_THROW(cudf::experimental::nans_to_nulls(input_column), cudf::logic_error);
}
