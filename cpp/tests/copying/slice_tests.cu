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

#include <cudf/cudf.h>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/copying.hpp>
#include <cudf/column/column_factories.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/legacy/interop.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <string>
#include <vector>

#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <tests/copying/slice_tests.cuh>

template <typename T>
struct SliceTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SliceTest, cudf::test::NumericTypes);

TYPED_TEST(SliceTest, NumericColumnsWithNulls) {
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, valids);

    std::vector<cudf::size_type> indices{1, 3, 2, 2, 5, 9};
    std::vector<cudf::test::fixed_width_column_wrapper<T>> expected = create_expected_columns<T>(indices, true);
    std::vector<cudf::column_view> result = cudf::experimental::slice(col, indices);

    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
        cudf::test::expect_columns_equal(expected[index], result[index]);
    }
}

struct SliceStringTest : public SliceTest <std::string>{};

TEST_F(SliceStringTest, StringWithNulls) {
    std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"};
    auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i%2==0? true:false; });
    cudf::test::strings_column_wrapper s(strings.begin(), strings.end(), valids);

    std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};

    std::vector<cudf::test::strings_column_wrapper> expected = create_expected_string_columns(strings, indices, true);
    std::vector<cudf::column_view> result = cudf::experimental::slice(s, indices);

    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
        cudf::test::expect_column_properties_equal(expected[index], result[index]);
    }
}

struct SliceCornerCases : public SliceTest <int8_t>{};

TEST_F(SliceCornerCases, EmptyColumn) {
    cudf::column col {};
    std::vector<cudf::size_type> indices{1, 3, 2, 4, 5, 9};

    std::vector<cudf::column_view> result = cudf::experimental::slice(col.view(), indices);

    unsigned long expected = 0;

    EXPECT_EQ(expected, result.size());
}

TEST_F(SliceCornerCases, EmptyIndices) {
    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::test::fixed_width_column_wrapper<int8_t> col = create_fixed_columns<int8_t>(start, size, valids);
    std::vector<cudf::size_type> indices{};

    std::vector<cudf::column_view> result = cudf::experimental::slice(col, indices);
    
    unsigned long expected = 0;

    EXPECT_EQ(expected, result.size());
}

TEST_F(SliceCornerCases, InvalidSetOfIndices) {
    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });
    cudf::test::fixed_width_column_wrapper<int8_t> col = create_fixed_columns<int8_t>(start, size, valids);
    std::vector<cudf::size_type> indices{11, 12};

    EXPECT_THROW(cudf::experimental::slice(col, indices), cudf::logic_error);
}

TEST_F(SliceCornerCases, ImproperRange) {
    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::test::fixed_width_column_wrapper<int8_t> col = create_fixed_columns<int8_t>(start, size, valids);
    std::vector<cudf::size_type> indices{5, 4};

    EXPECT_THROW(cudf::experimental::slice(col, indices), cudf::logic_error);
}

TEST_F(SliceCornerCases, NegativeOffset) {
    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::test::fixed_width_column_wrapper<int8_t> col = create_fixed_columns<int8_t>(start, size, valids);
    std::vector<cudf::size_type> indices{-1, 4};

    EXPECT_THROW(cudf::experimental::slice(col, indices), cudf::logic_error);
}


template <typename T>
struct SliceTableTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SliceTableTest, cudf::test::NumericTypes);

TYPED_TEST(SliceTableTest, NumericColumnsWithNulls) {
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type col_size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::size_type num_cols = 5; 
    cudf::experimental::table src_table = create_fixed_table<T>(num_cols, start, col_size, valids);        

    std::vector<cudf::size_type> indices{1, 3, 2, 2, 5, 9};
    std::vector<cudf::experimental::table> expected = create_expected_tables<T>(num_cols, indices, true);

    std::vector<cudf::table_view> result = cudf::experimental::slice(src_table, indices);

    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {        
        cudf::test::expect_tables_equal(expected[index], result[index]);
    }        
}

struct SliceStringTableTest : public SliceTableTest <std::string>{};

TEST_F(SliceStringTableTest, StringWithNulls) {    
    auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i%2==0? true:false; });
    
    std::vector<std::string> strings[2]     = { {"", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"}, 
                                                {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"} };
    cudf::test::strings_column_wrapper sw[2] = { {strings[0].begin(), strings[0].end(), valids},
                                                {strings[1].begin(), strings[1].end(), valids} };
                                                
    std::vector<std::unique_ptr<cudf::column>> scols;
    scols.push_back(sw[0].release());
    scols.push_back(sw[1].release());    
    cudf::experimental::table src_table(std::move(scols));

    std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};

    std::vector<cudf::experimental::table> expected = create_expected_string_tables(strings, indices, true);
        
    std::vector<cudf::table_view> result = cudf::experimental::slice(src_table, indices);

    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {        
        cudf::test::expect_table_properties_equal(expected[index], result[index]);
    }        
}

struct SliceTableCornerCases : public SliceTableTest <int8_t>{};

TEST_F(SliceTableCornerCases, EmptyTable) {        
    std::vector<cudf::size_type> indices{1, 3, 2, 4, 5, 9};

    cudf::experimental::table src_table{};    
    std::vector<cudf::table_view> result = cudf::experimental::slice(src_table.view(), indices);

    unsigned long expected = 0;

    EXPECT_EQ(expected, result.size());
}

TEST_F(SliceTableCornerCases, EmptyIndices) {
    cudf::size_type start = 0;
    cudf::size_type col_size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::size_type num_cols = 5; 
    cudf::experimental::table src_table = create_fixed_table<int8_t>(num_cols, start, col_size, valids);    
    std::vector<cudf::size_type> indices{};

    std::vector<cudf::table_view> result = cudf::experimental::slice(src_table, indices);
    
    unsigned long expected = 0;

    EXPECT_EQ(expected, result.size());
}

TEST_F(SliceTableCornerCases, InvalidSetOfIndices) {
    cudf::size_type start = 0;
    cudf::size_type col_size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });
        
    cudf::size_type num_cols = 5; 
    cudf::experimental::table src_table = create_fixed_table<int8_t>(num_cols, start, col_size, valids);    
    
    std::vector<cudf::size_type> indices{11, 12};

    EXPECT_THROW(cudf::experimental::slice(src_table, indices), cudf::logic_error);
}

TEST_F(SliceTableCornerCases, ImproperRange) {
    cudf::size_type start = 0;
    cudf::size_type col_size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::size_type num_cols = 5; 
    cudf::experimental::table src_table = create_fixed_table<int8_t>(num_cols, start, col_size, valids);    
    
    std::vector<cudf::size_type> indices{5, 4};

    EXPECT_THROW(cudf::experimental::slice(src_table, indices), cudf::logic_error);
}

TEST_F(SliceTableCornerCases, NegativeOffset) {
    cudf::size_type start = 0;
    cudf::size_type col_size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::size_type num_cols = 5; 
    cudf::experimental::table src_table = create_fixed_table<int8_t>(num_cols, start, col_size, valids);    
    
    std::vector<cudf::size_type> indices{-1, 4};

    EXPECT_THROW(cudf::experimental::slice(src_table, indices), cudf::logic_error);
}
