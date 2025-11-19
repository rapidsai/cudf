/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/split/partition.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/split/split_re.hpp>

#include <string>

class StringsSplitTest : public cudf::test::BaseFixture {};

TEST_F(StringsSplitTest, SplitAll)
{
  auto input = cudf::test::strings_column_wrapper({"Héllo thesé", "tést strings", ""});
  auto view  = cudf::strings_column_view(input);

  auto const delimiter = cudf::string_scalar("é", true, cudf::test::get_default_stream());
  cudf::strings::split(view, delimiter, -1, cudf::test::get_default_stream());
  cudf::strings::rsplit(view, delimiter, -1, cudf::test::get_default_stream());
  cudf::strings::split_record(view, delimiter, -1, cudf::test::get_default_stream());
  cudf::strings::rsplit_record(view, delimiter, -1, cudf::test::get_default_stream());
  cudf::strings::partition(view, delimiter, cudf::test::get_default_stream());
  cudf::strings::rpartition(view, delimiter, cudf::test::get_default_stream());
  cudf::strings::split_part(view, delimiter, 1, cudf::test::get_default_stream());

  auto const pattern = std::string("\\s");
  auto const prog    = cudf::strings::regex_program::create(pattern);
  cudf::strings::split_re(view, *prog, -1, cudf::test::get_default_stream());
  cudf::strings::split_record_re(view, *prog, -1, cudf::test::get_default_stream());
  cudf::strings::rsplit_re(view, *prog, -1, cudf::test::get_default_stream());
  cudf::strings::rsplit_record_re(view, *prog, -1, cudf::test::get_default_stream());
}
