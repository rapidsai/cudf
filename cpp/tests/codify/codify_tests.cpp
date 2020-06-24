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
#include <cudf/transform.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

template <typename T>
class CodifyTypedTests : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(CodifyTypedTests, cudf::test::FixedWidthTypes);

TYPED_TEST(CodifyTypedTests, SingleNullCodify)
{
  using cudf::test::expect_tables_equal;
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<TypeParam> input({1}, {0});
  fixed_width_column_wrapper<cudf::size_type> expect({0});
  auto const result = cudf::codify(input);

  expect_columns_equal(result.second->view(), expect);
}

TYPED_TEST(CodifyTypedTests, EmptyCodify)
{
  using cudf::test::expect_tables_equal;
  using cudf::test::fixed_width_column_wrapper;

  fixed_width_column_wrapper<TypeParam> input({});
  fixed_width_column_wrapper<cudf::size_type> expect({});
  auto const result = cudf::codify(input);

  expect_columns_equal(result.second->view(), expect);
}

CUDF_TEST_PROGRAM_MAIN()
