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

//  #include <tests/utilities/base_fixture.hpp>
//  #include <gtest/gtest.h>
//  #include <gmock/gmock.h>
#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/predicates.hpp>
#include <cudf/sorting.hpp>

//  #include <cudf/utilities/error.hpp>

#include <cudf/column/column_factories.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <vector>


template<typename T>
using fixed_width_column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

template <typename T>
struct IsSortedNumeric : public cudf::test::BaseFixture {};

using test_types_numeric = ::testing::Types<int8_t, int16_t, int32_t, int64_t,
                                        float, double>;

TYPED_TEST_CASE(IsSortedNumeric, test_types_numeric);

TYPED_TEST(IsSortedNumeric, NoColumns)
{
    using T = TypeParam;

    cudf::table_view in{{ }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedNumeric, NoRows)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({ });
    fixed_width_column_wrapper<T> col2({ });

    cudf::table_view in{{ col1, col2 }};
    std::vector<cudf::order> order{ cudf::order::ASCENDING,
                                    cudf::order::DESCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedNumeric, Ascending)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({ -10, -10, -1, -1, 0, 1, 1, 10, 10 });

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::ASCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedNumeric, AscendingFalse)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({ 10, -10, -1, -1, 0, 1, 1, 10, -10 });

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::ASCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedNumeric, AscendingExtremes)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({ std::numeric_limits<T>::lowest(),
                                         0,
                                         std::numeric_limits<T>::max() });

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::ASCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedNumeric, AscendingExtremesFalse)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({ std::numeric_limits<T>::lowest(),
                                         0,
                                         std::numeric_limits<T>::lowest() });

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::ASCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedNumeric, Descending)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({ 10, 10, 1, 1, 0, -1, -1, -10, -10 });

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::DESCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedNumeric, DescendingFalse)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({ -10, 10, 1, 1, 0, -1, -1, -10, 10 });

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::DESCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedNumeric, DescendingExtremes)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({ std::numeric_limits<T>::max(),
                                         0,
                                         std::numeric_limits<T>::lowest() });

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::DESCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedNumeric, DescendingExtremesFalse)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({ std::numeric_limits<T>::max(),
                                         0,
                                         std::numeric_limits<T>::max() });

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::DESCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedNumeric, NullsAfter)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({0, 0}, {1, 0});

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ cudf::null_order::AFTER };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedNumeric, NullsAfterFalse)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({0, 0}, {0, 1});

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ cudf::null_order::AFTER };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedNumeric, NullsBefore)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({0, 0}, {0, 1});

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ cudf::null_order::BEFORE };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedNumeric, NullsBeforeFalse)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({0, 0}, {1, 0});

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ cudf::null_order::BEFORE };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedNumeric, OrderArgsTooFew)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({0, 0}, {1, 0});
    fixed_width_column_wrapper<T> col2({0, 0}, {1, 0});

    cudf::table_view in{{ col1, col2 }};
    std::vector<cudf::order> order{ cudf::order::ASCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    EXPECT_THROW(cudf::experimental::is_sorted(in, order, null_precedence),
                 cudf::logic_error);
}

TYPED_TEST(IsSortedNumeric, OrderArgsTooMany)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({0, 0}, {1, 0});

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::ASCENDING,
                                    cudf::order::ASCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    EXPECT_THROW(cudf::experimental::is_sorted(in, order, null_precedence),
                 cudf::logic_error);
}

TYPED_TEST(IsSortedNumeric, NullOrderArgsTooFew)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({0, 0}, {1, 0});
    fixed_width_column_wrapper<T> col2({0, 0}, {1, 0});

    cudf::table_view in{{ col1, col2 }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ cudf::null_order::BEFORE };

    EXPECT_THROW(cudf::experimental::is_sorted(in, order, null_precedence),
                 cudf::logic_error);
}

TYPED_TEST(IsSortedNumeric, NullOrderArgsTooMany)
{
    using T = TypeParam;

    fixed_width_column_wrapper<T> col1({0, 0}, {1, 0});

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ cudf::null_order::BEFORE,
                                                   cudf::null_order::BEFORE };

    EXPECT_THROW(cudf::experimental::is_sorted(in, order, null_precedence),
                 cudf::logic_error);
}

template <typename T>
struct IsSortedNvstringCategory : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(IsSortedNvstringCategory, ::testing::Types<cudf::nvstring_category>);

TYPED_TEST(IsSortedNvstringCategory, True)
{
    EXPECT_EQ(true, false); // todo
}