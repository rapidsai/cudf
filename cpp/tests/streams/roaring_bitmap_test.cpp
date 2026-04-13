/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/roaring_bitmap_test_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/utilities/roaring_bitmap.hpp>

using cudf::test::serialize_roaring32;
using cudf::test::serialize_roaring64;

struct RoaringBitmapStreamTest : public cudf::test::BaseFixture,
                                 public ::testing::WithParamInterface<cudf::roaring_bitmap_type> {};

TEST_P(RoaringBitmapStreamTest, ContainsAsync)
{
  auto const bitmap_type = GetParam();
  auto const stream      = cudf::test::get_default_stream();
  auto const mr          = cudf::get_current_device_resource_ref();

  auto roaring_bitmap_data = (bitmap_type == cudf::roaring_bitmap_type::BITS_32)
                               ? serialize_roaring32({0, 2, 4})
                               : serialize_roaring64({0, 2, 4});
  auto const bitmap        = cudf::roaring_bitmap(bitmap_type, roaring_bitmap_data);
  bitmap.materialize(stream);

  cudf::size_type constexpr num_keys = 3;
  std::unique_ptr<cudf::column> result;

  if (bitmap_type == cudf::roaring_bitmap_type::BITS_32) {
    auto keys = cudf::test::fixed_width_column_wrapper<cuda::std::uint32_t>{0, 1, 2};
    result    = bitmap.contains_async(keys, stream, mr);
  } else {
    auto keys = cudf::test::fixed_width_column_wrapper<cuda::std::uint64_t>{0, 1, 2};
    result    = bitmap.contains_async(keys, stream, mr);
  }

  EXPECT_EQ(result->size(), num_keys);
}

INSTANTIATE_TEST_SUITE_P(BitmapTypes,
                         RoaringBitmapStreamTest,
                         ::testing::Values(cudf::roaring_bitmap_type::BITS_32,
                                           cudf::roaring_bitmap_type::BITS_64));

CUDF_TEST_PROGRAM_MAIN()
