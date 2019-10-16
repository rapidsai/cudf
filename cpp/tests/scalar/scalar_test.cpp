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

#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_list_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <thrust/sequence.h>
#include <random>

#include <gmock/gmock.h>

template <typename T>
struct TypedScalarTest : public cudf::test::BaseFixture {
  cudf::data_type type() { return cudf::data_type{cudf::experimental::type_to_id<T>()}; }
};

TYPED_TEST_CASE(TypedScalarTest, cudf::test::NumericTypes);


TYPED_TEST(TypedScalarTest, DefaultNullCountNoMask) {
  TypeParam value = 7;
  cudf::numeric_scalar<TypeParam> s(value);

  EXPECT_TRUE(s.is_valid());
  EXPECT_EQ(value, s.value());
}
