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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/sorting.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/utilities/error.hpp>

using namespace cudf;
using namespace test;

template <typename T>
struct DiscreteQuantilesTest : public BaseFixture {
};

using DiscreteQuantilesTestTypes = AllTypes;

TYPED_TEST_CASE(DiscreteQuantilesTest, DiscreteQuantilesTestTypes);

TYPED_TEST(DiscreteQuantilesTest, TestZeroColumns)
{
    auto input = table_view(std::vector<column_view>{ });

    auto interp = experimental::interpolation::NEAREST;

    EXPECT_THROW(experimental::quantiles(input, { 0.0f }, interp),
                 logic_error);
}

TYPED_TEST(DiscreteQuantilesTest, TestMultiColumnZeroRows)
{
    using T = TypeParam;

    auto input_a = fixed_width_column_wrapper<T>({ });
    auto input = table_view({ input_a });

    auto interp = experimental::interpolation::NEAREST;

    EXPECT_THROW(experimental::quantiles(input, { 0.0f }, interp, {}, true),
                 logic_error);
}

TYPED_TEST(DiscreteQuantilesTest, TestIncorrectIndicesType)
{
    using T = TypeParam;

    auto input_a = fixed_width_column_wrapper<T>({ });
    auto input = table_view({ input_a });
    auto sorted_indices = fixed_width_column_wrapper<int>({ });

    auto interp = experimental::interpolation::NEAREST;

    EXPECT_THROW(
        experimental::quantiles(input, { 0.0f }, interp, sorted_indices, false),
        logic_error
    );
}

TYPED_TEST(DiscreteQuantilesTest, TestZeroRequestedQuantiles)
{
    using T = TypeParam;

    auto input_a = fixed_width_column_wrapper<T>({ 1 }, { 1 });
    auto input = table_view(std::vector<column_view>{ input_a });

    auto interp = experimental::interpolation::NEAREST;

    auto actual = experimental::quantiles(input, { }, interp, {}, true);
    auto expected = experimental::empty_like(input);

    expect_tables_equal(expected->view(), actual->view());
}

TYPED_TEST(DiscreteQuantilesTest, TestMultiUsingSortmap)
{
    using T = TypeParam;

    auto input_a = strings_column_wrapper(
        { "C", "B", "A", "A", "D", "B", "D", "B", "D", "C", "C", "C", "D", "B", "D", "B", "C", "C", "A", "D", "B", "A", "A", "A" },
        {  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1  });

    auto input_b = fixed_width_column_wrapper<T>(
        {  4,   3,   5,   0,   1,   0,   4,   1,   5,   3,   0,   5,   2,   4,   3,   2,   1,   2,   3,   0,   5,   1,   4,   2 },
        {  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1 }
    );

    auto input = table_view({ input_a, input_b });

    auto sortmap = experimental::sorted_order(input, { order::ASCENDING, order::DESCENDING });

    auto actual = experimental::quantiles(input,
                                          { 0.0f, 0.5f, 0.7f, 0.25f, 1.0f },
                                          experimental::interpolation::NEAREST,
                                          sortmap->view(),
                                          true);

    auto expected_a = strings_column_wrapper(
        { "A", "C", "C", "B", "D" },
        {  1,   1,   1,   1,   1  });

    auto expected_b = fixed_width_column_wrapper<T>(
        {  5,   5,   1,   5,   0  },
        {  1,   1,   1,   1,   1  }
    );

    auto expected = table_view({ expected_a, expected_b });

    expect_tables_equal(expected, actual->view());
}

TYPED_TEST(DiscreteQuantilesTest, TestMultiColumnAssumedSorted)
{
    using T = TypeParam;

    auto input_a = strings_column_wrapper(
        { "C", "B", "A", "A", "D", "B", "D", "B", "D", "C", "C", "C", "D", "B", "D", "B", "C", "C", "A", "D", "B", "A", "A", "A" },
        {  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1  });

    auto input_b = fixed_width_column_wrapper<T>(
        {  4,   3,   5,   0,   1,   0,   4,   1,   5,   3,   0,   5,   2,   4,   3,   2,   1,   2,   3,   0,   5,   1,   4,   2 },
        {  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1 }
    );

    auto input = table_view({ input_a, input_b });

    auto actual = experimental::quantiles(input,
                                          { 0.0f, 0.5f, 0.7f, 0.25f, 1.0f },
                                          experimental::interpolation::NEAREST,
                                          {},
                                          true);

    auto expected_a = strings_column_wrapper(
        { "C", "D", "C", "D", "A" },
        {  1,   1,   1,   1,   1  });

    auto expected_b = fixed_width_column_wrapper<T>(
        {  4,   2,   1,   4,   2  },
        {  1,   1,   1,   1,   1  }
    );

    auto expected = table_view({ expected_a, expected_b });

    expect_tables_equal(expected, actual->view());
}


template <typename T>
struct ArithmeticQuantilesTest : public BaseFixture {
};

using ArithmeticQuantilesTestTypes = NumericTypes;

TYPED_TEST_CASE(ArithmeticQuantilesTest, ArithmeticQuantilesTestTypes);

TYPED_TEST(ArithmeticQuantilesTest, TestCastToDouble)
{
    using T = TypeParam;

    auto input_a = fixed_width_column_wrapper<T>({0});
    auto input = table_view({ input_a });
    auto expected_a = fixed_width_column_wrapper<double>({0});
    auto expected = table_view({ expected_a });
    auto actual = experimental::quantiles(input, {0}, experimental::interpolation::LINEAR);

    expect_tables_equal(expected, actual->view());
}

TYPED_TEST(ArithmeticQuantilesTest, TestRetainTypes)
{
    using T = TypeParam;

    auto input_a = fixed_width_column_wrapper<T>({0});
    auto input = table_view({ input_a });
    auto expected_a = fixed_width_column_wrapper<T>({0});
    auto expected = table_view({ expected_a });
    auto actual = experimental::quantiles(input, {0}, experimental::interpolation::LINEAR, {}, true);

    expect_tables_equal(expected, actual->view());
}

template<typename T>
struct NonNumericQuantilesTest : public BaseFixture {
};

using NonNumericQuantilesTestTypes = RemoveIf<ContainedIn<NumericTypes>, AllTypes>;

TYPED_TEST_CASE(NonNumericQuantilesTest, NonNumericQuantilesTestTypes);

TYPED_TEST(NonNumericQuantilesTest, TestNonNumericInterpolation)
{
    using T = TypeParam;

    auto input_a = fixed_width_column_wrapper<T>({ T{} });
    auto input = table_view({ input_a });

    auto interp = experimental::interpolation::LINEAR;

    EXPECT_THROW(
        experimental::quantiles(input, { 0.0f }, interp, {}, true),
        logic_error
    );
}

TYPED_TEST(NonNumericQuantilesTest, TestNonNumericWithoutRetainingTypes)
{
    using T = TypeParam;

    auto input_a = fixed_width_column_wrapper<T>({ T{} });
    auto input = table_view({ input_a });

    auto interp = experimental::interpolation::NEAREST;

    EXPECT_THROW(
        experimental::quantiles(input, { 0.0f }, interp, {}, false),
        logic_error
    );
}
