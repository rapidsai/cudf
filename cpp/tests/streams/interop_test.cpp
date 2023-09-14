/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/interop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/default_stream.hpp>

struct ArrowTest : public cudf::test::BaseFixture {};

TEST_F(ArrowTest, ToArrow)
{
  int32_t const value{42};
  auto cudf_scalar =
    cudf::make_fixed_width_scalar<int32_t>(value, cudf::test::get_default_stream());

  cudf::column_metadata metadata{""};
  auto arrow_scalar = cudf::to_arrow(*cudf_scalar, metadata, cudf::test::get_default_stream());
}

TEST_F(ArrowTest, FromArrow)
{
  int32_t const value{42};
  auto arrow_scalar = arrow::MakeScalar(value);
  cudf::from_arrow(*arrow_scalar, cudf::test::get_default_stream());
}
