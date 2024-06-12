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

#include <cudf/labeling/label_bins.hpp>

class LabelingBinsStreamTest : public cudf::test::BaseFixture {};

TEST_F(LabelingBinsStreamTest, SimpleStringsTest)
{
  cudf::test::strings_column_wrapper left_edges{"a", "b", "c", "d", "e"};
  cudf::test::strings_column_wrapper right_edges{"b", "c", "d", "e", "f"};
  cudf::test::strings_column_wrapper input{"abc", "bcd", "cde", "def", "efg"};

  cudf::label_bins(input,
                   left_edges,
                   cudf::inclusive::YES,
                   right_edges,
                   cudf::inclusive::NO,
                   cudf::test::get_default_stream());
}
