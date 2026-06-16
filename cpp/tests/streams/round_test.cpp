/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/round.hpp>

#include <vector>

class RoundTest : public cudf::test::BaseFixture {};

TEST_F(RoundTest, RoundHalfToEven)
{
  auto const input = cudf::test::fixed_point_column_wrapper<int32_t>{
    {1140, 1150, 1160, 1240, 1250, 1260, -1140, -1150, -1160, -1240, -1250, -1260},
    numeric::scale_type{0}};
  cudf::round_decimal(input, 0, cudf::rounding_method::HALF_UP, cudf::test::get_default_stream());
}

TEST_F(RoundTest, RoundHalfAwayFromEven)
{
  auto const input =
    cudf::test::fixed_width_column_wrapper<int32_t>{-12, -135, -145, -146, -1454, -1455, -1500};
  cudf::round_decimal(
    input, -1, cudf::rounding_method::HALF_EVEN, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
