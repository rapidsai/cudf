/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/bin.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

using namespace cudf::test;

namespace {

// =============================================================================
// ----- tests -----------------------------------------------------------------

TEST(BinColumnTest, TestSimple)
{
  fixed_width_column_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fixed_width_column_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fixed_width_column_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  auto result = cudf::bin::bin(input, left_edges, cudf::bin::inclusive::YES, right_edges, cudf::bin::inclusive::NO);
};

}  // anonymous namespace

CUDF_TEST_PROGRAM_MAIN()
