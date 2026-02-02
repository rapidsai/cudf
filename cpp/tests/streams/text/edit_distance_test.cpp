/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
}
