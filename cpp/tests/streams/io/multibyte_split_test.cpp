/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/text/data_chunk_source_factories.hpp>
#include <cudf/io/text/multibyte_split.hpp>

#include <string>

class MultibyteSplitTest : public cudf::test::BaseFixture {};

TEST_F(MultibyteSplitTest, Reader)
{
  auto delimiter  = std::string(":");
  auto host_input = std::string("abc:def");
  auto source     = cudf::io::text::make_source(host_input);
  cudf::io::text::parse_options options{};
  auto result =
    cudf::io::text::multibyte_split(*source, delimiter, options, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
