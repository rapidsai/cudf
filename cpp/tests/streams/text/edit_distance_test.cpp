/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <nvtext/edit_distance.hpp>

class TextEditDistanceTest : public cudf::test::BaseFixture {};

TEST_F(TextEditDistanceTest, EditDistance)
{
  auto const input       = cudf::test::strings_column_wrapper({"dog", "cat", "mouse", "pupper"});
  auto const input_view  = cudf::strings_column_view(input);
  auto const target      = cudf::test::strings_column_wrapper({"hog", "cake", "house", "puppy"});
  auto const target_view = cudf::strings_column_view(target);
  nvtext::edit_distance(input_view, target_view, cudf::test::get_default_stream());
  nvtext::edit_distance_matrix(input_view, cudf::test::get_default_stream());
}
