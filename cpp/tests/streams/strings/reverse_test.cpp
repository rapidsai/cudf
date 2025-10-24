/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/strings/reverse.hpp>

#include <string>

class StringsReverseTest : public cudf::test::BaseFixture {};

TEST_F(StringsReverseTest, Reverse)
{
  auto input = cudf::test::strings_column_wrapper({"aBcdef", "   ", "12345"});
  auto view  = cudf::strings_column_view(input);

  cudf::strings::reverse(view, cudf::test::get_default_stream());
}
