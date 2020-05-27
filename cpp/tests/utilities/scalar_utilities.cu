/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <jit/type.h>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <sstream>
#include <tests/utilities/cudf_gtest.hpp>
#include <type_traits>
#include "gtest/gtest.h"

using cudf::scalar_type_t;

namespace cudf {
namespace test {
namespace {
struct compare_scalar_functor {
  template <typename T>
  void operator()(cudf::scalar const& lhs, cudf::scalar const& rhs)
  {
    auto lhs_t = static_cast<scalar_type_t<T> const&>(lhs);
    auto rhs_t = static_cast<scalar_type_t<T> const&>(rhs);
    EXPECT_EQ(lhs_t.value(), rhs_t.value());
  }
};

template <>
void compare_scalar_functor::operator()<float>(cudf::scalar const& lhs, cudf::scalar const& rhs)
{
  auto lhs_t = static_cast<scalar_type_t<float> const&>(lhs);
  auto rhs_t = static_cast<scalar_type_t<float> const&>(rhs);
  EXPECT_FLOAT_EQ(lhs_t.value(), rhs_t.value());
}

template <>
void compare_scalar_functor::operator()<double>(cudf::scalar const& lhs, cudf::scalar const& rhs)
{
  auto lhs_t = static_cast<scalar_type_t<double> const&>(lhs);
  auto rhs_t = static_cast<scalar_type_t<double> const&>(rhs);
  EXPECT_DOUBLE_EQ(lhs_t.value(), rhs_t.value());
}

template <>
void compare_scalar_functor::operator()<cudf::dictionary32>(cudf::scalar const& lhs,
                                                            cudf::scalar const& rhs)
{
  CUDF_FAIL("Unsupported scalar compare type: dictionary");
}

template <>
void compare_scalar_functor::operator()<cudf::list_view>(cudf::scalar const& lhs,
                                                         cudf::scalar const& rhs)
{
  CUDF_FAIL("Unsupported scalar compare type: list_view");
}

}  // anonymous namespace

void expect_scalars_equal(cudf::scalar const& lhs, cudf::scalar const& rhs)
{
  EXPECT_EQ(lhs.type(), rhs.type());
  EXPECT_EQ(lhs.is_valid(), rhs.is_valid());

  if (lhs.is_valid() && rhs.is_valid() && lhs.type() == rhs.type()) {
    type_dispatcher(lhs.type(), compare_scalar_functor{}, lhs, rhs);
  }
}

}  // namespace test
}  // namespace cudf
