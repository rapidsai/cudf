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
#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/sorting.hpp>

#include <cudf/column/column_factories.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <vector>

template<typename T>
using fixed_width_column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

using strings_column_wrapper = cudf::test::strings_column_wrapper;

using bool8 = cudf::experimental::bool8;

// =============================================================================
// ---- test data --------------------------------------------------------------

namespace {
namespace testdata {

// ----- most numerics

template<typename T>
auto ascending() {
    return fixed_width_column_wrapper<T>({ std::numeric_limits<T>::lowest(),
                                            -100 -10, -1, 0, 1, 10, 100,
                                            std::numeric_limits<T>::max() });
}

template<typename T>
auto descending() {
    return fixed_width_column_wrapper<T>({ std::numeric_limits<T>::max(),
                                           100, 10, 1, 0, -1, -10, -100,
                                           std::numeric_limits<T>::lowest() });
}

template<typename T>
auto empty() {
    return fixed_width_column_wrapper<T>({ });
}

template<typename T>
auto nulls_after() {
    return fixed_width_column_wrapper<T>({ 0, 0 }, { 1, 0 });
}

template<typename T>
auto nulls_before() {
    return fixed_width_column_wrapper<T>({ 0, 0 }, { 0, 1 });
}

// ----- bool8

template<>
auto ascending<bool8>() {
    return fixed_width_column_wrapper<bool8>({ false, true });
}

template<>
auto descending<bool8>() {
    return fixed_width_column_wrapper<bool8>({ true, false });
}

// ----- std::string

template<>
auto ascending<std::string>() {
    return strings_column_wrapper({ "A", "B" });
}

template<>
auto descending<std::string>() {
    return strings_column_wrapper({ "B", "A" });
}

template<>
auto empty<std::string>() {
    return strings_column_wrapper({ });
}

template<>
auto nulls_after<std::string>() {
    return strings_column_wrapper({ "identical", "identical" }, { 1, 0 });
}

template<>
auto nulls_before<std::string>() {
    return strings_column_wrapper({ "identical", "identical" }, { 0, 1 });
}
    
} // namespace testdata
} // anonymous namespace

// =============================================================================
// ---- tests ------------------------------------------------------------------

template <typename T>
struct IsSortedNumeric : public cudf::test::BaseFixture {};

using test_types = ::testing::Types<int8_t, int16_t, int32_t, int64_t,
                                    float, double,
                                    cudf::experimental::bool8,
                                    std::string>;

// compiles and passes tests
TYPED_TEST_CASE(IsSortedNumeric, cudf::test::NumericTypes);

// compiles and passes tests
// TYPED_TEST_CASE(IsSortedStrings, ::testing::Types<std::string>);

// // compiles and passes tests if static_asserts for fixed-width types are removed
// TYPED_TEST_CASE(IsSortedNumeric, test_types); 

TYPED_TEST(IsSortedNumeric, NoColumns)
{
    using T = TypeParam;

    cudf::table_view in{{ }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

// TYPED_TEST(IsSortedNumeric, NoRows)
// {
//     using T = TypeParam;

//     auto col1 = testdata::empty<T>();
//     auto col2 = testdata::empty<T>();

//     cudf::table_view in{{ col1, col2 }};
//     std::vector<cudf::order> order{ cudf::order::ASCENDING,
//                                     cudf::order::DESCENDING };
//     std::vector<cudf::null_order> null_precedence{ };

//     auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

//     EXPECT_EQ(true, actual);
// }

TYPED_TEST(IsSortedNumeric, Ascending)
{
    using T = TypeParam;

    auto col1 = testdata::ascending<T>();
    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::ASCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedNumeric, AscendingFalse)
{
    using T = TypeParam;

    auto col1 = testdata::descending<T>();
    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::ASCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, { });

    EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedNumeric, Descending)
{
    using T = TypeParam;

    auto col1 = testdata::descending<T>();

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::DESCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedNumeric, DescendingFalse)
{
    using T = TypeParam;
    
    auto col1 = testdata::ascending<T>();

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::DESCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedNumeric, NullsAfter)
{
    using T = TypeParam;

    auto col1 = testdata::nulls_after<T>();

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ cudf::null_order::AFTER };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedNumeric, NullsAfterFalse)
{
    using T = TypeParam;

    auto col1 = testdata::nulls_before<T>();

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ cudf::null_order::AFTER };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedNumeric, NullsBefore)
{
    using T = TypeParam;

    auto col1 = testdata::nulls_before<T>();

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ cudf::null_order::BEFORE };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedNumeric, NullsBeforeFalse)
{
    using T = TypeParam;

    auto col1 = testdata::nulls_after<T>();

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ cudf::null_order::BEFORE };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(false, actual);
}

TYPED_TEST(IsSortedNumeric, OrderArgsTooFew)
{
    using T = TypeParam;

    auto col1 = testdata::ascending<T>();
    auto col2 = testdata::ascending<T>();

    cudf::table_view in{{ col1, col2 }};
    std::vector<cudf::order> order{ cudf::order::ASCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    EXPECT_THROW(cudf::experimental::is_sorted(in, order, null_precedence),
                 cudf::logic_error);
}

TYPED_TEST(IsSortedNumeric, OrderArgsTooMany)
{
    using T = TypeParam;

    auto col1 = testdata::ascending<T>();

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

    auto col1 = testdata::nulls_before<T>();
    auto col2 = testdata::nulls_before<T>();

    cudf::table_view in{{ col1, col2 }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ cudf::null_order::BEFORE };

    EXPECT_THROW(cudf::experimental::is_sorted(in, order, null_precedence),
                 cudf::logic_error);
}

TYPED_TEST(IsSortedNumeric, NullOrderArgsTooMany)
{
    using T = TypeParam;

    auto col1 = testdata::nulls_before<T>();

    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ };
    std::vector<cudf::null_order> null_precedence{ cudf::null_order::BEFORE,
                                                   cudf::null_order::BEFORE };

    EXPECT_THROW(cudf::experimental::is_sorted(in, order, null_precedence),
                 cudf::logic_error);
}

template <typename T>
struct IsSortedStrings : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(IsSortedStrings, ::testing::Types<std::string>);

TYPED_TEST(IsSortedStrings, Ascending)
{
    using T = TypeParam;

    auto col1 = testdata::ascending<T>();
    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::ASCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, null_precedence);

    EXPECT_EQ(true, actual);
}

TYPED_TEST(IsSortedStrings, AscendingFalse)
{
    using T = TypeParam;

    auto col1 = testdata::descending<T>();
    cudf::table_view in{{ col1 }};
    std::vector<cudf::order> order{ cudf::order::ASCENDING };
    std::vector<cudf::null_order> null_precedence{ };

    auto actual = cudf::experimental::is_sorted(in, order, { });

    EXPECT_EQ(false, actual);
}
