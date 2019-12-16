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

#include <tests/utilities/scalar_utilities.hpp>

#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <sstream>
#include <jit/type.h>
#include <type_traits>
#include "gtest/gtest.h"

using cudf::experimental::scalar_type_t;

namespace cudf {
namespace test {
namespace {

struct compare_scalar_functor
{
    template<typename T>
    typename std::enable_if_t<not std::is_floating_point<T>::value, void>
    operator()(cudf::scalar const& lhs, cudf::scalar const& rhs)
    {
        auto lhs_t = static_cast<scalar_type_t<T> const&>(lhs);
        auto rhs_t = static_cast<scalar_type_t<T> const&>(rhs);
        EXPECT_EQ(lhs_t.value(), rhs_t.value());
    }

    template<typename T>
    std::enable_if_t<std::is_same<T, float>::value>
    operator()(cudf::scalar const& lhs, cudf::scalar const& rhs)
    {
        auto lhs_t = static_cast<scalar_type_t<T> const&>(lhs);
        auto rhs_t = static_cast<scalar_type_t<T> const&>(rhs);
        EXPECT_FLOAT_EQ(lhs_t.value(), rhs_t.value());
    }

    template<typename T>
    std::enable_if_t<std::is_same<T, double>::value>
    operator()(cudf::scalar const& lhs, cudf::scalar const& rhs)
    {
        auto lhs_t = static_cast<scalar_type_t<T> const&>(lhs);
        auto rhs_t = static_cast<scalar_type_t<T> const&>(rhs);
        EXPECT_DOUBLE_EQ(lhs_t.value(), rhs_t.value());
    }
};

} // anonymous namespace

void expect_scalars_equal(cudf::scalar const& lhs,
                          cudf::scalar const& rhs)
{
    EXPECT_EQ(lhs.type(), rhs.type());
    EXPECT_EQ(lhs.is_valid(), rhs.is_valid());

    if (lhs.is_valid() && rhs.is_valid() && lhs.type() == rhs.type()) {
        experimental::type_dispatcher(lhs.type(), compare_scalar_functor{}, lhs, rhs);
    }
}

} // namespace test
} // namespace cudf
