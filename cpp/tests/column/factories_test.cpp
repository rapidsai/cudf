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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/typed_tests.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

class ColumnFactoryTest : public ::testing::Test {
  cudf::size_type _size{1000};

 public:
  cudf::size_type size() { return _size; }
};

template <typename T>
class TypedColumnFactoryTest : public ColumnFactoryTest {};

TYPED_TEST_CASE(TypedColumnFactoryTest, cudf::test::NumericTypes);

/* TYPED_TEST(TypedColumnFactoryTest, NumericDefaultMask) {
  auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::exp::type_to_id<TypeParam>()}, this->size());

  EXPECT_EQ(column->type(),
            cudf::data_type{cudf::exp::type_to_id<TypeParam>()});
  EXPECT_EQ(column->size(), this->size());
} */