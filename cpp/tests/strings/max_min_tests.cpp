/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

struct StringsMinMaxTest : public cudf::test::BaseFixture {};

TEST_F(StringsMinMaxTest, Basic)
{
  cudf::test::strings_column_wrapper strings({"hello", "world!", "a", "four"});
  auto view = cudf::strings_column_view(strings, cudf::get_default_stream());

  // "a" is 1 byte, "world!" is 6 bytes
  EXPECT_EQ(view.minimum(), 1L);
  EXPECT_EQ(view.maximum(), 6L);
}

TEST_F(StringsMinMaxTest, AllSameLength)
{
  cudf::test::strings_column_wrapper strings({"abc", "def", "ghi"});
  auto view = cudf::strings_column_view(strings, cudf::get_default_stream());

  EXPECT_EQ(view.minimum(), 3L);
  EXPECT_EQ(view.maximum(), 3L);
}

TEST_F(StringsMinMaxTest, WithNulls)
{
  // Null entries count as zero-length strings
  cudf::test::strings_column_wrapper strings({"hello", "world", "!"}, {true, false, true});
  auto view = cudf::strings_column_view(strings, cudf::get_default_stream());

  EXPECT_EQ(view.minimum(), 0L);
  EXPECT_EQ(view.maximum(), 5L);
}

TEST_F(StringsMinMaxTest, AllNulls)
{
  cudf::test::strings_column_wrapper strings({"a", "bb"}, {false, false});
  auto view = cudf::strings_column_view(strings, cudf::get_default_stream());

  EXPECT_EQ(view.minimum(), 0L);
  EXPECT_EQ(view.maximum(), 0L);
}

TEST_F(StringsMinMaxTest, EmptyColumn)
{
  cudf::test::strings_column_wrapper strings{};
  auto view = cudf::strings_column_view(strings, cudf::get_default_stream());

  EXPECT_EQ(view.minimum(), 0L);
  EXPECT_EQ(view.maximum(), 0L);
}

TEST_F(StringsMinMaxTest, SingleElement)
{
  cudf::test::strings_column_wrapper strings({"hello"});
  auto view = cudf::strings_column_view(strings, cudf::get_default_stream());

  EXPECT_EQ(view.minimum(), 5L);
  EXPECT_EQ(view.maximum(), 5L);
}

TEST_F(StringsMinMaxTest, EmptyStrings)
{
  // Empty strings have 0 bytes
  cudf::test::strings_column_wrapper strings({"", "abc", ""});
  auto view = cudf::strings_column_view(strings, cudf::get_default_stream());

  EXPECT_EQ(view.minimum(), 0L);
  EXPECT_EQ(view.maximum(), 3L);
}

TEST_F(StringsMinMaxTest, UnicodeMulitbyteChars)
{
  // "é" is 2 bytes in UTF-8, "日" is 3 bytes
  cudf::test::strings_column_wrapper strings({"é", "日", "a"});
  auto view = cudf::strings_column_view(strings, cudf::get_default_stream());

  EXPECT_EQ(view.minimum(), 1L);
  EXPECT_EQ(view.maximum(), 3L);
}

TEST_F(StringsMinMaxTest, NotComputedThrows)
{
  // Constructing without a stream leaves min/max uncomputed; accessors must throw.
  cudf::test::strings_column_wrapper strings({"hello", "world"});
  auto view = cudf::strings_column_view(strings);

  EXPECT_THROW((void)view.minimum(), cudf::logic_error);
  EXPECT_THROW((void)view.maximum(), cudf::logic_error);
}
