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
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/legacy/interop.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <string>
#include <vector>

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

template <typename T, typename InputIterator>
cudf::test::fixed_width_column_wrapper<T> create_fixed_columns_for_splits(cudf::size_type start, cudf::size_type size, InputIterator valids) {
    auto iter = cudf::test::make_counting_transform_iterator(start, [](auto i) { return T(i);});

        return cudf::test::fixed_width_column_wrapper<T> (iter, iter + size, valids);

}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns(std::vector<cudf::size_type> splits, cudf::size_type size, bool nullable) {

    std::vector<cudf::test::fixed_width_column_wrapper<T>> result = {};
    std::vector<cudf::size_type> indices = splits_to_indices(splits, size);

    for(unsigned long index = 0; index < indices.size(); index+=2) {
        auto iter = cudf::test::make_counting_transform_iterator(indices[index], [](auto i) { return T(i);});

        if(not nullable) {
            result.push_back(cudf::test::fixed_width_column_wrapper<T> (iter, iter + (indices[index+1] - indices[index])));
        } else {
            auto valids = cudf::test::make_counting_transform_iterator(indices[index], [](auto i) { return i%2==0? true:false; });
            result.push_back(cudf::test::fixed_width_column_wrapper<T> (iter, iter + (indices[index+1] - indices[index]), valids));
        }
    }

    return result;
}

std::vector<cudf::test::strings_column_wrapper> create_expected_string_columns_for_splits(std::vector<std::string> strings, std::vector<cudf::size_type> splits, bool nullable) {

    std::vector<cudf::test::strings_column_wrapper> result = {};
    std::vector<cudf::size_type> indices = splits_to_indices(splits, strings.size());

    for(unsigned long index = 0; index < indices.size(); index+=2) {
        if(not nullable) {
            result.push_back(cudf::test::strings_column_wrapper (strings.begin()+indices[index],  strings.begin()+indices[index+1]));
        } else {
            auto valids = cudf::test::make_counting_transform_iterator(indices[index], [](auto i) { return i%2==0? true:false; });
            result.push_back(cudf::test::strings_column_wrapper (strings.begin()+indices[index], strings.begin()+indices[index+1], valids));
        }
    }

    return result;
}

template <typename T>
struct SplitTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SplitTest, cudf::test::NumericTypes);

TYPED_TEST(SplitTest, SplitEndLessThanSize) {
    using T = TypeParam;

    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns_for_splits<T>(start, size, valids);

    std::vector<cudf::size_type> splits{2, 5, 7};
    std::vector<cudf::test::fixed_width_column_wrapper<T>> expected = create_expected_columns<T>(splits, size, true);
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

    cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns_for_splits<T>(start, size, valids);

    std::vector<cudf::size_type> splits{2, 5, 10};
    std::vector<cudf::test::fixed_width_column_wrapper<T>> expected = create_expected_columns<T>(splits, size, true);
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

    cudf::test::fixed_width_column_wrapper<int8_t> col = create_fixed_columns_for_splits<int8_t>(start, size, valids);
    std::vector<cudf::size_type> splits{};

    std::vector<cudf::column_view> result = cudf::experimental::split(col, splits);

    unsigned long expected = 0;

    EXPECT_EQ(expected, result.size());
}

TEST_F(SplitCornerCases, InvalidSetOfIndices) {
    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });
    cudf::test::fixed_width_column_wrapper<int8_t> col = create_fixed_columns_for_splits<int8_t>(start, size, valids);
    std::vector<cudf::size_type> splits{11, 12};

    EXPECT_THROW(cudf::experimental::split(col, splits), cudf::logic_error);
}

TEST_F(SplitCornerCases, ImproperRange) {
    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::test::fixed_width_column_wrapper<int8_t> col = create_fixed_columns_for_splits<int8_t>(start, size, valids);
    std::vector<cudf::size_type> splits{5, 4};

    EXPECT_THROW(cudf::experimental::split(col, splits), cudf::logic_error);
}

TEST_F(SplitCornerCases, NegativeValue) {
    cudf::size_type start = 0;
    cudf::size_type size = 10;
    auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return i%2==0? true:false; });

    cudf::test::fixed_width_column_wrapper<int8_t> col = create_fixed_columns_for_splits<int8_t>(start, size, valids);
    std::vector<cudf::size_type> splits{-1, 4};

    EXPECT_THROW(cudf::experimental::split(col, splits), cudf::logic_error);
}
