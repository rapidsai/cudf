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

template <typename T, typename InputIterator>
cudf::test::fixed_width_column_wrapper<T> create_fixed_columns(cudf::size_type start, cudf::size_type size, InputIterator valids) {
    auto iter = cudf::test::make_counting_transform_iterator(start, [](auto i) { return T(i);});

        return cudf::test::fixed_width_column_wrapper<T> (iter, iter + size, valids);

}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns(std::vector<cudf::size_type> indices, bool nullable) {

    std::vector<cudf::test::fixed_width_column_wrapper<T>> result = {};

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

std::vector<cudf::test::strings_column_wrapper> create_expected_string_columns(std::vector<std::string> strings, std::vector<cudf::size_type> indices, bool nullable) {

    std::vector<cudf::test::strings_column_wrapper> result = {};

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
struct SliceTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SliceTest, cudf::test::NumericTypes);

TYPED_TEST(SliceTest, NumericColumnsWithInValids) {
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

TEST_F(SliceStringTest, StringWithInvalids) {
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
