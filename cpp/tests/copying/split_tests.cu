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

#include <tests/copying/slice_tests.cuh>

std::vector<cudf::size_type> splits_to_indices(std::vector<cudf::size_type> splits, cudf::size_type size){
    std::vector<cudf::size_type> indices{0};

    std::for_each(splits.begin(), splits.end(),
            [&indices](auto split) {
                indices.push_back(split); // This for end
                indices.push_back(split); // This for the start
            });

    if (splits.back() != size) {
        indices.push_back(size); // This to include rest of the elements
    } else {
        indices.pop_back(); // Not required as it is extra 
    }

    return indices;
}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns_for_splits(std::vector<cudf::size_type> const& splits, cudf::size_type size, bool nullable) {

    // convert splits to slice indices
    std::vector<cudf::size_type> indices = splits_to_indices(splits, size);
    return create_expected_columns<T>(indices, nullable);
}

std::vector<cudf::test::strings_column_wrapper> create_expected_string_columns_for_splits(std::vector<std::string> strings, std::vector<cudf::size_type> const& splits, bool nullable) {

    std::vector<cudf::size_type> indices = splits_to_indices(splits, strings.size());
    return create_expected_string_columns(strings, indices, nullable);   
}

template <typename T>
std::vector<cudf::experimental::table> create_expected_tables_for_splits(cudf::size_type num_cols, std::vector<cudf::size_type> const& splits, cudf::size_type col_size, bool nullable){
    std::vector<cudf::size_type> indices = splits_to_indices(splits, col_size);    
    return create_expected_tables<T>(num_cols, indices, nullable);
}

std::vector<cudf::experimental::table> create_expected_string_tables_for_splits(std::vector<std::string> const strings[2], std::vector<cudf::size_type> const& splits, bool nullable){    
    std::vector<cudf::size_type> indices = splits_to_indices(splits, strings[0].size());    
    return create_expected_string_tables(strings, indices, nullable);
}

template <typename T>
struct SplitTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SplitTest, cudf::test::NumericTypes);

TYPED_TEST(SplitTest, SplitEndLessThanSize) {
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, valids);

    std::vector<cudf::size_type> splits{2, 5, 7};
    std::vector<cudf::test::fixed_width_column_wrapper<T>> expected = create_expected_columns_for_splits<T>(splits, size, true);
    std::vector<cudf::column_view> result = cudf::experimental::split(col, splits);

    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
        cudf::test::expect_columns_equal(expected[index], result[index]);
    }
}

TYPED_TEST(SplitTest, SplitEndToSize) {
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, valids);

    std::vector<cudf::size_type> splits{2, 5, 10};
    std::vector<cudf::test::fixed_width_column_wrapper<T>> expected = create_expected_columns_for_splits<T>(splits, size, true);
    std::vector<cudf::column_view> result = cudf::experimental::split(col, splits);

    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
        cudf::test::expect_columns_equal(expected[index], result[index]);
    }
}

struct SplitStringTest : public SplitTest <std::string>{};

TEST_F(SplitStringTest, StringWithInvalids) {
    std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"};
    auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i%2==0? true:false; });
    cudf::test::strings_column_wrapper s(strings.begin(), strings.end(), valids);

    std::vector<cudf::size_type> splits{2, 5, 9};

    std::vector<cudf::test::strings_column_wrapper> expected = create_expected_string_columns_for_splits(strings, splits, true);
    std::vector<cudf::column_view> result = cudf::experimental::split(s, splits);

    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
        cudf::test::expect_column_properties_equal(expected[index], result[index]);
    }
}

struct SplitCornerCases : public SplitTest <int8_t>{};

TEST_F(SplitCornerCases, EmptyColumn) {
    cudf::column col {};
    std::vector<cudf::size_type> splits{2, 5, 9};

    std::vector<cudf::column_view> result = cudf::experimental::split(col.view(), splits);

    unsigned long expected = 0;

    EXPECT_EQ(expected, result.size());
}

TEST_F(SplitCornerCases, EmptyIndices) {
    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::test::fixed_width_column_wrapper<int8_t> col = create_fixed_columns<int8_t>(start, size, valids);
    std::vector<cudf::size_type> splits{};

    std::vector<cudf::column_view> result = cudf::experimental::split(col, splits);

    unsigned long expected = 0;

    EXPECT_EQ(expected, result.size());
}

TEST_F(SplitCornerCases, InvalidSetOfIndices) {
    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });
    cudf::test::fixed_width_column_wrapper<int8_t> col = create_fixed_columns<int8_t>(start, size, valids);
    std::vector<cudf::size_type> splits{11, 12};

    EXPECT_THROW(cudf::experimental::split(col, splits), cudf::logic_error);
}

TEST_F(SplitCornerCases, ImproperRange) {
    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::test::fixed_width_column_wrapper<int8_t> col = create_fixed_columns<int8_t>(start, size, valids);
    std::vector<cudf::size_type> splits{5, 4};

    EXPECT_THROW(cudf::experimental::split(col, splits), cudf::logic_error);
}

TEST_F(SplitCornerCases, NegativeValue) {
    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::test::fixed_width_column_wrapper<int8_t> col = create_fixed_columns<int8_t>(start, size, valids);
    std::vector<cudf::size_type> splits{-1, 4};

    EXPECT_THROW(cudf::experimental::split(col, splits), cudf::logic_error);
}

// common functions for testing split/contiguous_split
template<typename T, typename SplitFunc, typename CompareFunc>
void split_end_less_than_size(SplitFunc Split, CompareFunc Compare)
{
    cudf::size_type start = 0;
    cudf::size_type col_size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::size_type num_cols = 5;
    cudf::experimental::table src_table = create_fixed_table<T>(num_cols, start, col_size, valids);

    std::vector<cudf::size_type> splits{2, 5, 7};
    std::vector<cudf::experimental::table> expected = create_expected_tables_for_splits<T>(num_cols, splits, col_size, true);
    
    auto result = Split(src_table, splits);

    EXPECT_EQ(expected.size(), result.size());
        
    for (unsigned long index = 0; index < result.size(); index++) {
        Compare(expected[index], result[index]);
    }    
}

template<typename T, typename SplitFunc, typename CompareFunc>
void split_end_to_size(SplitFunc Split, CompareFunc Compare)
{
    cudf::size_type start = 0;
    cudf::size_type col_size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::size_type num_cols = 5;
    cudf::experimental::table src_table = create_fixed_table<T>(num_cols, start, col_size, valids);    

    std::vector<cudf::size_type> splits{2, 5, 10};
    std::vector<cudf::experimental::table> expected = create_expected_tables_for_splits<T>(num_cols, splits, col_size, true);
    
    auto result = Split(src_table, splits);

    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
        Compare(expected[index], result[index]);
    }
}

template<typename SplitFunc>
void split_empty_table(SplitFunc Split)
{
    std::vector<cudf::size_type> splits{2, 5, 9};

    cudf::experimental::table src_table{}; 
    auto result = Split(src_table, splits);

    unsigned long expected = 0;

    EXPECT_EQ(expected, result.size());
}

template<typename SplitFunc>
void split_empty_indices(SplitFunc Split)
{
    cudf::size_type start = 0;
    cudf::size_type col_size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::size_type num_cols = 5;    
    cudf::experimental::table src_table = create_fixed_table<int8_t>(num_cols, start, col_size, valids);
    std::vector<cudf::size_type> splits{};

    auto result = Split(src_table, splits);

    unsigned long expected = 0;

    EXPECT_EQ(expected, result.size());
}

template<typename SplitFunc>
void split_invalid_indices(SplitFunc Split)
{
    cudf::size_type start = 0;
    cudf::size_type col_size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::size_type num_cols = 5; 
    cudf::experimental::table src_table = create_fixed_table<int8_t>(num_cols, start, col_size, valids);
    
    std::vector<cudf::size_type> splits{11, 12};

    EXPECT_THROW(Split(src_table, splits), cudf::logic_error);
}

template<typename SplitFunc>
void split_improper_range(SplitFunc Split)
{
    cudf::size_type start = 0;
    cudf::size_type col_size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::size_type num_cols = 5; 
    cudf::experimental::table src_table = create_fixed_table<int8_t>(num_cols, start, col_size, valids);    

    std::vector<cudf::size_type> splits{5, 4};

    EXPECT_THROW(Split(src_table, splits), cudf::logic_error);
}

template<typename SplitFunc>
void split_negative_value(SplitFunc Split)
{
    cudf::size_type start = 0;
    cudf::size_type col_size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::size_type num_cols = 5; 
    cudf::experimental::table src_table = create_fixed_table<int8_t>(num_cols, start, col_size, valids);    

    std::vector<cudf::size_type> splits{-1, 4};

    EXPECT_THROW(Split(src_table, splits), cudf::logic_error);
}

// regular splits
template <typename T>
struct SplitTableTest : public cudf::test::BaseFixture {};
TYPED_TEST_CASE(SplitTableTest, cudf::test::NumericTypes);

TYPED_TEST(SplitTableTest, SplitEndLessThanSize) {
    split_end_less_than_size<TypeParam>(
        [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits){
            return cudf::experimental::split(t, splits);
        },
        [](cudf::table_view const& expected, cudf::table_view const& result){
            return cudf::test::expect_tables_equal(expected, result);
        }
    );
}

TYPED_TEST(SplitTableTest, SplitEndToSize) {
    split_end_to_size<TypeParam>(
        [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits){
            return cudf::experimental::split(t, splits);
        },
        [](cudf::table_view const& expected, cudf::table_view const& result){
            return cudf::test::expect_tables_equal(expected, result);
        }
    );
}

struct SplitTableCornerCases : public SplitTest <int8_t>{};

TEST_F(SplitTableCornerCases, EmptyTable) {
    split_empty_table([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits){
        return cudf::experimental::split(t, splits);
    });
}

TEST_F(SplitTableCornerCases, EmptyIndices) {
    split_empty_indices([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits){
        return cudf::experimental::split(t, splits);
    });
}

TEST_F(SplitTableCornerCases, InvalidSetOfIndices) {
    split_invalid_indices([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits){
        return cudf::experimental::split(t, splits);
    });
}

TEST_F(SplitTableCornerCases, ImproperRange) {
    split_improper_range([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits){
        return cudf::experimental::split(t, splits);
    });
}

TEST_F(SplitTableCornerCases, NegativeValue) {
    split_negative_value([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits){
        return cudf::experimental::split(t, splits);
    });
}

struct SplitStringTableTest : public SplitTest <std::string>{};

TEST_F(SplitStringTableTest, StringWithInvalids) {
    auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i%2==0? true:false; });

    std::vector<std::string> strings[2]     = { {"", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"}, 
                                                {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"} };
    cudf::test::strings_column_wrapper sw[2] = { {strings[0].begin(), strings[0].end(), valids},
                                                {strings[1].begin(), strings[1].end(), valids} };

    std::vector<std::unique_ptr<cudf::column>> scols;
    scols.push_back(sw[0].release());
    scols.push_back(sw[1].release());    
    cudf::experimental::table src_table(std::move(scols));

    std::vector<cudf::size_type> splits{2, 5, 9};
    
    std::vector<cudf::experimental::table> expected = create_expected_string_tables_for_splits(strings, splits, true);

    std::vector<cudf::table_view> result = cudf::experimental::split(src_table, splits);

    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
        cudf::test::expect_table_properties_equal(expected[index], result[index]);
    }
}


// contiguous splits
template <typename T>
struct ContiguousSplitTableTest : public cudf::test::BaseFixture {};
TYPED_TEST_CASE(ContiguousSplitTableTest, cudf::test::NumericTypes);

TYPED_TEST(ContiguousSplitTableTest, SplitEndLessThanSize) {
    split_end_less_than_size<TypeParam>(
        [](cudf::table_view const& t, std::vector<cudf::size_type> const &splits){
            return cudf::experimental::contiguous_split(t, splits);
        },
        [](cudf::table_view const& expected, cudf::experimental::contiguous_split_result const& result){
            return cudf::test::expect_tables_equal(expected, result.table);
        }
    );
}

TYPED_TEST(ContiguousSplitTableTest, SplitEndToSize) {
    split_end_to_size<TypeParam>(
        [](cudf::table_view const& t, std::vector<cudf::size_type> const &splits){
            return cudf::experimental::contiguous_split(t, splits);
        },
        [](cudf::table_view const& expected, cudf::experimental::contiguous_split_result const& result){
            return cudf::test::expect_tables_equal(expected, result.table);
        }
    );
}

struct ContiguousSplitTableCornerCases : public SplitTest <int8_t>{};

TEST_F(ContiguousSplitTableCornerCases, EmptyTable) {
    split_empty_table([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits){ 
       return cudf::experimental::contiguous_split(t, splits); 
    });
}

TEST_F(ContiguousSplitTableCornerCases, EmptyIndices) {   
    split_empty_indices([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits){ 
       return cudf::experimental::contiguous_split(t, splits); 
    }); 
}

TEST_F(ContiguousSplitTableCornerCases, InvalidSetOfIndices) {    
    split_invalid_indices([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits){ 
       return cudf::experimental::contiguous_split(t, splits); 
    }); 
}

TEST_F(ContiguousSplitTableCornerCases, ImproperRange) {    
    split_improper_range([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits){ 
       return cudf::experimental::contiguous_split(t, splits); 
    }); 
}

TEST_F(ContiguousSplitTableCornerCases, NegativeValue) {   
    split_negative_value([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits){ 
       return cudf::experimental::contiguous_split(t, splits); 
    }); 
}
