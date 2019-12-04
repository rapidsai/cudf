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

#include <cudf/wrappers/bool.hpp>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <algorithm>
#include <tests/utilities/cudf_gtest.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <memory>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/scalar_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include <vector>

#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <cudf/quantiles.hpp>

using namespace cudf::test;

using bool8 = cudf::experimental::bool8;


namespace {

// =============================================================================
// ---- helpers funcs ----------------------------------------------------------

template<typename T>
std::unique_ptr<cudf::scalar> make_numeric_scalar(T value)
{
    using TScalar = cudf::experimental::scalar_type_t<T>;

    cudf::data_type type{cudf::experimental::type_to_id<T>()};
    auto s = make_numeric_scalar(type);

    TScalar * s_sc = static_cast<TScalar *>(s.get());
    s_sc->set_valid(true);
    s_sc->set_value(value);

    return s;
}

template<typename T>
std::unique_ptr<cudf::scalar> make_numeric_scalar()
{
    cudf::data_type type{cudf::experimental::type_to_id<T>()};
    return make_numeric_scalar(type);
}

} // anonymous namespace

// =============================================================================
// ---- tests ------------------------------------------------------------------

template <typename T>
struct QuantilesTest : public BaseFixture {};

TYPED_TEST_CASE(QuantilesTest, cudf::test::NumericTypes);

// TYPED_TEST(QuantilesTest, TestScalar)
// {
//     using T = TypeParam;

//     auto val1 = make_numeric_scalar<T>(0);
//     auto val2 = make_numeric_scalar<T>(0);
//     auto val3 = make_numeric_scalar<T>(1);
//     auto val4 = make_numeric_scalar<T>(1);

//     cudf::test::expect_scalars_equal(*val1, *val2);
//     cudf::test::expect_scalars_equal(*val3, *val4);
// }

TYPED_TEST(QuantilesTest, TestColumnEmpty)
{
    using T = TypeParam;
    using TScalar = cudf::experimental::scalar_type_t<T>;

    auto in = fixed_width_column_wrapper<T>({ });

    auto expected = make_numeric_scalar<double>();
    auto actual = cudf::experimental::quantile(in, 0);

    cudf::test::expect_scalars_equal(*expected, *actual);
}

TYPED_TEST(QuantilesTest, TestColumnOneElement)
{
    using T = TypeParam;
    using TScalar = cudf::experimental::scalar_type_t<T>;

    auto in = fixed_width_column_wrapper<T>({ 5 });

    auto expected = make_numeric_scalar<double>(5.0);
    auto actual = cudf::experimental::quantile(in, 0, cudf::experimental::interpolation::NEAREST, true);

    cudf::test::expect_scalars_equal(*expected, *actual);
}

TYPED_TEST(QuantilesTest, TestColumnTwoElementsSortedLower)
{
    using T = TypeParam;
    using TScalar = cudf::experimental::scalar_type_t<T>;

    auto in = fixed_width_column_wrapper<T>({ -4, -1, 3, 2, 1 });

    auto expected_lower = make_numeric_scalar<double>(3);
    auto actual_lower = cudf::experimental::quantile(in, 0.5, cudf::experimental::interpolation::LOWER, true);

    cudf::test::expect_scalars_equal(*expected_lower, *actual_lower);
}

TYPED_TEST(QuantilesTest, TestColumnTwoElementsSortedHigher)
{
    using T = TypeParam;
    using TScalar = cudf::experimental::scalar_type_t<T>;

    const T higher = 5;
    const T lower = 0;

    auto in = fixed_width_column_wrapper<T>({ lower, higher });

    auto expected = make_numeric_scalar<double>(higher);
    auto actual = cudf::experimental::quantile(in, 0.5, cudf::experimental::interpolation::HIGHER, true);

    cudf::test::expect_scalars_equal(*expected, *actual);
}

TYPED_TEST(QuantilesTest, TestColumnTwoElementsSortedMidpoint)
{
    using T = TypeParam;
    using TScalar = cudf::experimental::scalar_type_t<T>;

    const T higher = 5;
    const T lower = 0;

    auto in = fixed_width_column_wrapper<T>({ lower, higher });

    auto expected = make_numeric_scalar<double>(2.5);
    auto actual = cudf::experimental::quantile(in, 0.5, cudf::experimental::interpolation::MIDPOINT, true);

    cudf::test::expect_scalars_equal(*expected, *actual);
}

TYPED_TEST(QuantilesTest, TestColumnOneElementNullMidpoint)
{
    using T = TypeParam;
    using TScalar = cudf::experimental::scalar_type_t<T>;

    auto in = fixed_width_column_wrapper<T>({ 0 }, { 0 });

    auto expected = make_numeric_scalar<double>();
    auto actual = cudf::experimental::quantile(in, 0.5, cudf::experimental::interpolation::MIDPOINT, true);

    cudf::test::expect_scalars_equal(*expected, *actual);
}

TYPED_TEST(QuantilesTest, TestColumnTwoElementsUnsorted)
{
    using T = TypeParam;
    using TScalar = cudf::experimental::scalar_type_t<T>;

    auto in = fixed_width_column_wrapper<T>({ 10, 5 });

    auto expected = make_numeric_scalar<double>(5);
    auto actual = cudf::experimental::quantile(in, 0.5, cudf::experimental::interpolation::LOWER, false);

    cudf::test::expect_scalars_equal(*expected, *actual);
}

TYPED_TEST(QuantilesTest, TestColumnTwoElementsNullMidpoint)
{
    using T = TypeParam;
    using TScalar = cudf::experimental::scalar_type_t<T>;

    auto in = fixed_width_column_wrapper<T>({ 0, 0 }, { 0, 0 });

    auto expected = make_numeric_scalar<double>();
    auto actual = cudf::experimental::quantile(in, 0.5, cudf::experimental::interpolation::MIDPOINT, true);

    cudf::test::expect_scalars_equal(*expected, *actual);
}


TYPED_TEST(QuantilesTest, TestColumnThreeElementsNullSandwichUnsorted)
{
    using T = TypeParam;
    using TScalar = cudf::experimental::scalar_type_t<T>;

    auto in = fixed_width_column_wrapper<T>({ 0, 7, 0 }, { 0, 1, 0 });

    auto expected = make_numeric_scalar<double>(7);
    auto actual = cudf::experimental::quantile(in, 0.5, cudf::experimental::interpolation::MIDPOINT, false);

    cudf::test::expect_scalars_equal(*expected, *actual);
}
