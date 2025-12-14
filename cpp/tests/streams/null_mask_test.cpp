/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>

#include <vector>

class NullMaskTest : public cudf::test::BaseFixture {};

TEST_F(NullMaskTest, CreateNullMask)
{
  cudf::create_null_mask(10, cudf::mask_state::ALL_VALID, cudf::test::get_default_stream());
}

TEST_F(NullMaskTest, SetNullMask)
{
  cudf::test::fixed_width_column_wrapper<bool> col({0, 1, 0, 1, 1},
                                                   {true, false, true, false, false});

  cudf::set_null_mask(static_cast<cudf::mutable_column_view>(col).null_mask(),
                      0,
                      3,
                      false,
                      cudf::test::get_default_stream());
}

TEST_F(NullMaskTest, CopyBitmask)
{
  cudf::test::fixed_width_column_wrapper<bool> const col({0, 1, 0, 1, 1},
                                                         {true, false, true, false, false});

  cudf::copy_bitmask(
    static_cast<cudf::column_view>(col).null_mask(), 0, 3, cudf::test::get_default_stream());
}

TEST_F(NullMaskTest, CopyBitmaskFromColumn)
{
  cudf::test::fixed_width_column_wrapper<bool> const col({0, 1, 0, 1, 1},
                                                         {true, false, true, false, false});

  cudf::copy_bitmask(col, cudf::test::get_default_stream());
}

TEST_F(NullMaskTest, BitMaskAnd)
{
  cudf::test::fixed_width_column_wrapper<bool> const col1({0, 1, 0, 1, 1},
                                                          {true, false, true, false, false});
  cudf::test::fixed_width_column_wrapper<bool> const col2({0, 1, 0, 1, 1},
                                                          {true, true, false, false, true});

  auto tbl = cudf::table_view{{col1, col2}};
  cudf::bitmask_and(tbl, cudf::test::get_default_stream());
}

TEST_F(NullMaskTest, BitMaskOr)
{
  cudf::test::fixed_width_column_wrapper<bool> const col1({0, 1, 0, 1, 1},
                                                          {true, false, true, false, false});
  cudf::test::fixed_width_column_wrapper<bool> const col2({0, 1, 0, 1, 1},
                                                          {true, true, false, false, true});

  auto tbl = cudf::table_view{{col1, col2}};
  cudf::bitmask_or(tbl, cudf::test::get_default_stream());
}

TEST_F(NullMaskTest, NullCount)
{
  cudf::test::fixed_width_column_wrapper<bool> const col({0, 1, 0, 1, 1},
                                                         {true, true, false, false, true});

  cudf::null_count(
    static_cast<cudf::column_view>(col).null_mask(), 0, 4, cudf::test::get_default_stream());
}

TEST_F(NullMaskTest, BatchNullCount)
{
  auto stream = cudf::test::get_default_stream();

  cudf::test::fixed_width_column_wrapper<int32_t> const col1({0, 1, 2, 3}, {1, 0, 1, 0});
  cudf::test::fixed_width_column_wrapper<int32_t> const col2({0, 1, 2, 3}, {1, 1, 0, 0});
  cudf::test::fixed_width_column_wrapper<int32_t> const col3({0, 1, 2, 3});

  auto const col1_view = static_cast<cudf::column_view>(col1);
  auto const col2_view = static_cast<cudf::column_view>(col2);
  auto const col3_view = static_cast<cudf::column_view>(col3);

  std::vector<cudf::bitmask_type const*> bitmasks{
    col1_view.null_mask(), col2_view.null_mask(), col3_view.null_mask()};

  auto const counts = cudf::batch_null_count(bitmasks, 0, col1_view.size(), stream);

  ASSERT_EQ(counts.size(), bitmasks.size());
  EXPECT_EQ(counts[0], cudf::null_count(bitmasks[0], 0, col1_view.size(), stream));
  EXPECT_EQ(counts[1], cudf::null_count(bitmasks[1], 0, col1_view.size(), stream));
  EXPECT_EQ(counts[2], 0);
}

CUDF_TEST_PROGRAM_MAIN()
