/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

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

CUDF_TEST_PROGRAM_MAIN()
