/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/cudf_gtest.hpp>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cudf_streaming/detail/device_bloom_filter.hpp>
#include <cudf_streaming/detail/large_arrow_filter_policy.cuh>

#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

TEST(BloomFilterPolicyTest, UsesBlocksBeyondArrowLimit)
{
  using policy_type =
    cudf_streaming::detail::large_arrow_filter_policy<std::uint64_t, cuco::identity_hash>;

  constexpr auto arrow_max_blocks = std::size_t{4'194'304};
  constexpr auto num_blocks       = arrow_max_blocks + 1;
  constexpr auto hash             = std::uint64_t{std::numeric_limits<std::uint32_t>::max()} << 32;

  auto const index = policy_type{}.block_index(hash, cuco::extent<std::size_t>{num_blocks});

  EXPECT_EQ(index, arrow_max_blocks);
}

TEST(DeviceBloomFilterTest, RequiresAlignedStorageSize)
{
  constexpr auto unaligned_size = std::size_t{65};
  constexpr auto aligned_size   = std::size_t{64};
  auto const stream             = cudf::get_default_stream();

  EXPECT_THROW(cudf_streaming::detail::device_bloom_filter::storage(
                 unaligned_size, stream, cudf::get_current_device_resource_ref()),
               std::logic_error);

  auto storage = cudf_streaming::detail::device_bloom_filter::storage(
    aligned_size, stream, cudf::get_current_device_resource_ref());

  EXPECT_EQ(storage->size(), aligned_size);

  auto const filter =
    cudf_streaming::detail::device_bloom_filter{aligned_size, 0, storage->data(), stream};
  EXPECT_EQ(filter.size(), aligned_size);
}

}  // namespace
