/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <tests/strings/utilities.h>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/scatter.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/lists_column_view.hpp>

template <typename T>
class TypedScatterListsTest : public cudf::test::BaseFixture {
};
using FixedWidthTypes= cudf::test::Concat<cudf::test::IntegralTypes,
                                          cudf::test::FloatingPointTypes,
                                          cudf::test::DurationTypes,
                                          cudf::test::TimestampTypes>;
TYPED_TEST_CASE(TypedScatterListsTest, FixedWidthTypes);

class ScatterListsTest : public cudf::test::BaseFixture {
};

TYPED_TEST(TypedScatterListsTest, ListsOfFixedWidth)
{
    using namespace cudf::test;
    using T = TypeParam;

    auto src_list_column = lists_column_wrapper<T, int32_t>{
        {9, 9, 9, 9}, {8, 8, 8}
    };

    auto target_list_column = lists_column_wrapper<T, int32_t>{
        {0,0}, {1,1}, {2,2}, {3,3}, {4,4}, {5,5}, {6,6}
    };

    auto scatter_map = fixed_width_column_wrapper<cudf::size_type>{2, 0};

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column}),
        scatter_map,
        cudf::table_view({target_list_column}));

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT (
        ret->get_column(0),
        lists_column_wrapper<T, int32_t> {{8,8,8}, {1,1}, {9,9,9,9}, {3,3}, {4,4}, {5,5}, {6,6}} 
    );
}

TYPED_TEST(TypedScatterListsTest, EmptyListsOfFixedWidth)
{
    using namespace cudf::test;
    using T = TypeParam;

    auto src_child = fixed_width_column_wrapper<T, int32_t> {
        {9, 9, 9, 9, 8, 8, 8},
    };

    // One null list row, and one row with nulls.
    auto src_list_column = cudf::make_lists_column(
        3,
        fixed_width_column_wrapper<cudf::size_type>{0, 4, 7, 7}.release(),
        src_child.release(),
        0,
        {}
    );

    auto target_list_column = lists_column_wrapper<T, int32_t>{
        {0,0}, {1,1}, {2,2}, {3,3}, {4,4}, {5,5}, {6,6}
    };

    auto scatter_map = fixed_width_column_wrapper<cudf::size_type>{2, 0, 5};

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column->view()}),
        scatter_map,
        cudf::table_view({target_list_column}));

    auto expected_child_ints = fixed_width_column_wrapper<T, int32_t> {
        {8,8,8, 1,1, 9,9,9,9, 3,3, 4,4, 6,6 }
    };
    auto expected_lists_column = cudf::make_lists_column(
        7,
        fixed_width_column_wrapper<cudf::size_type>{0, 3, 5, 9, 11, 13, 13, 15}.release(),
        expected_child_ints.release(), 
        0, 
        {}
    );

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
        expected_lists_column->view(),
        ret->get_column(0)
    );
}

TYPED_TEST(TypedScatterListsTest, EmptyListsOfNullableFixedWidth)
{
    using namespace cudf::test;
    using T = TypeParam;

    auto src_child = fixed_width_column_wrapper<T, int32_t> {
        {9, 9, 9, 9, 8, 8, 8},
        {1, 1, 1, 0, 1, 1, 1}
    };

    // One null list row, and one row with nulls.
    auto src_list_column = cudf::make_lists_column(
        3,
        fixed_width_column_wrapper<cudf::size_type>{0, 4, 7, 7}.release(),
        src_child.release(),
        0,
        {}
    );

    auto target_list_column = lists_column_wrapper<T, int32_t>{
        {0,0}, {1,1}, {2,2}, {3,3}, {4,4}, {5,5}, {6,6}
    };

    auto scatter_map = fixed_width_column_wrapper<cudf::size_type>{2, 0, 5};

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column->view()}),
        scatter_map,
        cudf::table_view({target_list_column}));

    auto expected_child_ints = fixed_width_column_wrapper<T, int32_t> {
        {8,8,8, 1,1, 9,9,9,9, 3,3, 4,4, 6,6 },
        {1,1,1, 1,1, 1,1,1,0, 1,1, 1,1, 1,1 }
    };
    auto expected_lists_column = cudf::make_lists_column(
        7,
        fixed_width_column_wrapper<cudf::size_type>{0, 3, 5, 9, 11, 13, 13, 15}.release(),
        expected_child_ints.release(), 
        0, 
        {}
    );

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
        expected_lists_column->view(),
        ret->get_column(0)
    );
}

TYPED_TEST(TypedScatterListsTest, NullableListsOfNullableFixedWidth)
{
    using namespace cudf::test;
    using T = TypeParam;

    auto src_child = fixed_width_column_wrapper<T, int32_t> {
        {9, 9, 9, 9, 8, 8, 8},
        {1, 1, 1, 0, 1, 1, 1}
    };

    auto src_list_validity = make_counting_transform_iterator(0, [](auto i) { return i != 2; });
    // One null list row, and one row with nulls.
    auto src_list_column = cudf::make_lists_column(
        3,
        fixed_width_column_wrapper<cudf::size_type>{0, 4, 7, 7}.release(),
        src_child.release(),
        1,
        detail::make_null_mask(src_list_validity, src_list_validity + 3)
    );

    auto target_list_column = lists_column_wrapper<T, int32_t>{
        {0,0}, {1,1}, {2,2}, {3,3}, {4,4}, {5,5}, {6,6}
    };

    auto scatter_map = fixed_width_column_wrapper<cudf::size_type>{2, 0, 5};

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column->view()}),
        scatter_map,
        cudf::table_view({target_list_column}));

    auto expected_child_ints = fixed_width_column_wrapper<T, int32_t> {
        {8,8,8, 1,1, 9,9,9,9, 3,3, 4,4, 6,6 },
        {1,1,1, 1,1, 1,1,1,0, 1,1, 1,1, 1,1 }
    };

    auto expected_validity = make_counting_transform_iterator(0, [](auto i) { return i != 5; });
    auto expected_lists_column = cudf::make_lists_column(
        7,
        fixed_width_column_wrapper<cudf::size_type>{0, 3, 5, 9, 11, 13, 13, 15}.release(),
        expected_child_ints.release(), 
        1, 
        detail::make_null_mask(expected_validity, expected_validity + 7)
    );

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
        expected_lists_column->view(),
        ret->get_column(0)
    );
}

TEST_F(ScatterListsTest, ListsOfStrings)
{
    using namespace cudf::test;

    auto src_list_column = lists_column_wrapper<cudf::string_view> {
        {"all", "the", "leaves", "are", "brown"},
        {"california", "dreaming"}
    };

    auto target_list_column = lists_column_wrapper<cudf::string_view> {
        {"zero"},
        {"one", "one"},
        {"two", "two"},
        {"three", "three", "three"},
        {"four", "four", "four", "four"}
    };

    auto scatter_map = fixed_width_column_wrapper<int32_t>{2, 0};

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column}),
        scatter_map,
        cudf::table_view({target_list_column})
    );

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
        lists_column_wrapper<cudf::string_view>{
            {"california", "dreaming"},
            {"one", "one"},
            {"all", "the", "leaves", "are", "brown"},
            {"three", "three", "three"},
            {"four", "four", "four", "four"}
        },
        ret->get_column(0)       
    );
}

TEST_F(ScatterListsTest, ListsOfNullableStrings)
{
    using namespace cudf::test;

    auto src_strings_column = strings_column_wrapper{
        {"all", "the", "leaves", "are", "brown", "california", "dreaming"},
        {    1,     1,        1,     0,       1,            0,          1}
    };

    auto src_list_column = cudf::make_lists_column(
        2, 
        fixed_width_column_wrapper<cudf::size_type>{0, 5, 7}.release(),
        src_strings_column.release(),
        0,
        {}
    );

    auto target_list_column = lists_column_wrapper<cudf::string_view> {
        {"zero"},
        {"one", "one"},
        {"two", "two"},
        {"three", "three"},
        {"four", "four"},
        {"five", "five"}
    };

    auto scatter_map = fixed_width_column_wrapper<int32_t>{2, 0};

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column->view()}),
        scatter_map,
        cudf::table_view({target_list_column})
    );

    auto expected_strings = strings_column_wrapper {
        {"california", "dreaming", "one", "one", "all", "the", "leaves", "are", "brown", 
         "three", "three", "four", "four", "five", "five"},
        make_counting_transform_iterator(0, [](auto i) {return i!=0 && i!=7;})
    };

    auto expected_lists = cudf::make_lists_column(
        6,
        fixed_width_column_wrapper<cudf::size_type>{0,2,4,9,11,13,15}.release(),
        expected_strings.release(),
        0,
        {}
    );

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
        expected_lists->view(),
        ret->get_column(0)
    );
}

TEST_F(ScatterListsTest, EmptyListsOfNullableStrings)
{
    using namespace cudf::test;

    auto src_strings_column = strings_column_wrapper{
        {"all", "the", "leaves", "are", "brown", "california", "dreaming"},
        {    1,     1,        1,     0,       1,            0,          1}
    };

    auto src_list_column = cudf::make_lists_column(
        3, 
        fixed_width_column_wrapper<cudf::size_type>{0, 5, 5, 7}.release(),
        src_strings_column.release(),
        0,
        {}
    );

    auto target_list_column = lists_column_wrapper<cudf::string_view> {
        {"zero"},
        {"one", "one"},
        {"two", "two"},
        {"three", "three"},
        {"four", "four"},
        {"five", "five"}
    };

    auto scatter_map = fixed_width_column_wrapper<int32_t>{2, 4, 0};

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column->view()}),
        scatter_map,
        cudf::table_view({target_list_column})
    );

    auto expected_strings = strings_column_wrapper {
        {"california", "dreaming", 
         "one", "one", 
         "all", "the", "leaves", "are", "brown", 
         "three", "three", 
         "five", "five"},
        make_counting_transform_iterator(0, [](auto i) {return i!=0 && i!=7;})
    };

    auto expected_lists = cudf::make_lists_column(
        6,
        fixed_width_column_wrapper<cudf::size_type>{0,2,4,9,11,11,13}.release(),
        expected_strings.release(),
        0,
        {}
    );

    std::cout << "Expected: " << std::endl; print(expected_lists->view());
    std::cout << "Received: " << std::endl; print(ret->get_column(0));

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
        expected_lists->view(),
        ret->get_column(0)
    );
}

TEST_F(ScatterListsTest, NullableListsOfNullableStrings)
{
    using namespace cudf::test;

    auto src_strings_column = strings_column_wrapper{
        {"all", "the", "leaves", "are", "brown", "california", "dreaming"},
        {    1,     1,        1,     0,       1,            0,          1}
    };

    auto src_validity = make_counting_transform_iterator(0, [](auto i) { return i != 1;});
    auto src_list_column = cudf::make_lists_column(
        3, 
        fixed_width_column_wrapper<cudf::size_type>{0, 5, 5, 7}.release(),
        src_strings_column.release(),
        1,
        detail::make_null_mask(src_validity, src_validity + 3)
    );

    auto target_list_column = lists_column_wrapper<cudf::string_view> {
        {"zero"},
        {"one", "one"},
        {"two", "two"},
        {"three", "three"},
        {"four", "four"},
        {"five", "five"}
    };

    auto scatter_map = fixed_width_column_wrapper<int32_t>{2, 4, 0};

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column->view()}),
        scatter_map,
        cudf::table_view({target_list_column})
    );

    auto expected_strings = strings_column_wrapper {
        {"california", "dreaming", 
         "one", "one", 
         "all", "the", "leaves", "are", "brown", 
         "three", "three", 
         "five", "five"},
        make_counting_transform_iterator(0, [](auto i) {return i!=0 && i!=7;})
    };

    auto expected_validity = make_counting_transform_iterator(0, [](auto i) { return i != 4; });
    auto expected_lists = cudf::make_lists_column(
        6,
        fixed_width_column_wrapper<cudf::size_type>{0,2,4,9,11,11,13}.release(),
        expected_strings.release(),
        1,
        detail::make_null_mask(expected_validity, expected_validity+6)
    );

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
        expected_lists->view(),
        ret->get_column(0)
    );
}

TYPED_TEST(TypedScatterListsTest, ListsOfLists)
{
    using namespace cudf::test;
    using T = TypeParam;

    auto src_list_column = lists_column_wrapper<T, int32_t> {
        { {1,1,1,1}, {2,2,2,2} },
        { {3,3,3,3}, {4,4,4,4} }
    };

    auto target_list_column = lists_column_wrapper<T, int32_t> {
        { {9,9,9}, {8,8,8}, {7,7,7} },
        { {6,6,6}, {5,5,5}, {4,4,4} },
        { {3,3,3}, {2,2,2}, {1,1,1} },
        { {9,9}, {8,8}, {7,7} },
        { {6,6}, {5,5}, {4,4} },
        { {3,3}, {2,2}, {1,1} }
    };

    auto scatter_map = fixed_width_column_wrapper<cudf::size_type>{2, 0};

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column}),
        scatter_map,
        cudf::table_view({target_list_column})
    );

    auto expected = lists_column_wrapper<T, int32_t> {
            { {3,3,3,3}, {4,4,4,4} },
            { {6,6,6}, {5,5,5}, {4,4,4} },
            { {1,1,1,1}, {2,2,2,2} },
            { {9,9}, {8,8}, {7,7} },
            { {6,6}, {5,5}, {4,4} },
            { {3,3}, {2,2}, {1,1} }
    };

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
        expected,
        ret->get_column(0)
    );
}

TYPED_TEST(TypedScatterListsTest, EmptyListsOfLists)
{
    using namespace cudf::test;
    using T = TypeParam;

    auto src_list_column = lists_column_wrapper<T, int32_t> {
        { {1,1,1,1}, {2,2,2,2} },
        { {3,3,3,3}, {} }, 
        {}
    };

    auto target_list_column = lists_column_wrapper<T, int32_t> {
        { {9,9,9}, {8,8,8}, {7,7,7} },
        { {6,6,6}, {5,5,5}, {4,4,4} },
        { {3,3,3}, {2,2,2}, {1,1,1} },
        { {9,9}, {8,8}, {7,7} },
        { {6,6}, {5,5}, {4,4} },
        { {3,3}, {2,2}, {1,1} }
    };

    auto scatter_map = fixed_width_column_wrapper<cudf::size_type>{2, 0, 4};

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column}),
        scatter_map,
        cudf::table_view({target_list_column})
    );

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
        lists_column_wrapper<T, int32_t> {
            { {3,3,3,3}, {} },
            { {6,6,6}, {5,5,5}, {4,4,4} },
            { {1,1,1,1}, {2,2,2,2} },
            { {9,9}, {8,8}, {7,7} },
            {  },
            { {3,3}, {2,2}, {1,1} }
        },
        ret->get_column(0)
    );
}

TYPED_TEST(TypedScatterListsTest, NullListsOfLists)
{
    using namespace cudf::test;
    using T = TypeParam;

    auto src_list_column = lists_column_wrapper<T, int32_t> {
        { 
            { {1,1,1,1}, {2,2,2,2} },
            { {3,3,3,3}, {} }, 
            {} 
        },
        make_counting_transform_iterator(0, [](auto i) { return i != 2; })
    };

    auto target_list_column = lists_column_wrapper<T, int32_t> {
        { {9,9,9}, {8,8,8}, {7,7,7} },
        { {6,6,6}, {5,5,5}, {4,4,4} },
        { {3,3,3}, {2,2,2}, {1,1,1} },
        { {9,9}, {8,8}, {7,7} },
        { {6,6}, {5,5}, {4,4} },
        { {3,3}, {2,2}, {1,1} }
    };

    auto scatter_map = fixed_width_column_wrapper<cudf::size_type>{2, 0, 4};

    auto ret = cudf::scatter(
        cudf::table_view({src_list_column}),
        scatter_map,
        cudf::table_view({target_list_column})
    );

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
        lists_column_wrapper<T, int32_t> {
            { 
                { {3,3,3,3}, {} },
                { {6,6,6}, {5,5,5}, {4,4,4} },
                { {1,1,1,1}, {2,2,2,2} },
                { {9,9}, {8,8}, {7,7} },
                {  },
                { {3,3}, {2,2}, {1,1} }
            },
            make_counting_transform_iterator(0, [](auto i) { return i != 4; })
        },
        ret->get_column(0)
    );
}
