/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/round.hpp>

#include <vector>

class RoundTest : public cudf::test::BaseFixture {};

TEST_F(RoundTest, RoundHalfToEven)
{
  std::vector<double> vals = {1.729, 17.29, 172.9, 1729};
  cudf::test::fixed_width_column_wrapper<double> input(vals.begin(), vals.end());
  cudf::round(input, 0, cudf::rounding_method::HALF_UP, cudf::test::get_default_stream());
}

TEST_F(RoundTest, RoundHalfAwayFromEven)
{
  std::vector<double> vals = {1.5, 2.5, 1.35, 1.45, 15, 25};
  cudf::test::fixed_width_column_wrapper<double> input(vals.begin(), vals.end());
  cudf::round(input, -1, cudf::rounding_method::HALF_EVEN, cudf::test::get_default_stream());
}
