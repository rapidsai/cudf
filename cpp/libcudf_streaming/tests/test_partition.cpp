/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/contiguous_split.hpp>
#include <cudf/utilities/traits.hpp>

#include <cudf_streaming/integrations/partition.hpp>
#include <cudf_streaming/integrations/utils.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include <memory>

using namespace cudf_streaming::integrations;

class NumOfPartitions : public cudf::test::BaseFixtureWithParam<std::tuple<int, int>> {};

// test different `num_partitions` and `num_rows`.
INSTANTIATE_TEST_SUITE_P(Partitions,
                         NumOfPartitions,
                         testing::Combine(testing::Range(1, 10),     // num_partitions
                                          testing::Range(1, 100, 9)  // num_rows
                                          ));

TEST_P(NumOfPartitions, partition_and_pack)
{
  int const num_partitions    = std::get<0>(GetParam());
  int const num_rows          = std::get<1>(GetParam());
  std::int64_t const seed     = 42;
  cudf::hash_id const hash_fn = cudf::hash_id::HASH_MURMUR3;
  auto stream                 = cudf::get_default_stream();
  auto br                     = rapidsmpf::BufferResource::create(mr());

  cudf::table expect = random_table_with_index(seed, static_cast<std::size_t>(num_rows), 0, 10);

  auto chunks = partition_and_pack(expect, {1}, num_partitions, hash_fn, seed, stream, br.get());

  // Convert to a vector
  std::vector<rapidsmpf::PackedData> chunks_vector;
  for (auto& [_, chunk] : chunks) {
    chunks_vector.push_back(std::move(chunk));
  }
  EXPECT_EQ(chunks_vector.size(), num_partitions);

  auto result = unpack_and_concat(std::move(chunks_vector), stream, br.get());

  // Compare the input table with the result. We ignore the row order by
  // sorting by their index (first column).
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(expect), sort_table(result));
}

TEST_P(NumOfPartitions, split_and_pack)
{
  int const num_partitions = std::get<0>(GetParam());
  int const num_rows       = std::get<1>(GetParam());
  std::int64_t const seed  = 42;
  auto stream              = cudf::get_default_stream();
  auto br = rapidsmpf::BufferResource::create(cudf::get_current_device_resource_ref());

  cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

  std::vector<cudf::size_type> splits;
  for (int i = 1; i < num_partitions; ++i) {
    splits.emplace_back(i * num_rows / num_partitions);
  }

  auto chunks = split_and_pack(expect, splits, stream, br.get());

  // Convert to a vector (restoring the original order).
  std::vector<rapidsmpf::PackedData> chunks_vector;
  for (int i = 0; i < num_partitions; ++i) {
    chunks_vector.emplace_back(std::move(chunks.at(i)));
  }
  EXPECT_EQ(chunks_vector.size(), num_partitions);

  auto result = unpack_and_concat(std::move(chunks_vector), stream, br.get());

  // Compare the input table with the result.
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect, *result);
}

class SpillingTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    br     = rapidsmpf::BufferResource::create(cudf::get_current_device_resource_ref());
    stream = cudf::get_default_stream();
  }

  std::shared_ptr<rapidsmpf::BufferResource> br;
  rmm::cuda_stream_view stream;
};

TEST_F(SpillingTest, SpillUnspillRoundtripPreservesDataAndMetadata)
{
  std::vector<std::uint8_t> metadata{42, 99};
  std::vector<std::uint8_t> payload{10, 20, 30};

  // Create device input.
  std::vector<rapidsmpf::PackedData> input;
  input.push_back(create_packed_data(metadata, payload, stream, br.get()));

  // Device -> Device (moves data)
  auto on_gpu = unspill_partitions(std::move(input), br.get(), rapidsmpf::AllowOverbooking::YES);
  ASSERT_EQ(on_gpu.size(), 1);
  EXPECT_EQ(on_gpu[0].data->mem_type(), rapidsmpf::MemoryType::DEVICE);
  EXPECT_EQ(*on_gpu[0].metadata, metadata);

  // Device -> Host
  auto back_on_host = spill_partitions(std::move(on_gpu), br.get());
  ASSERT_EQ(back_on_host.size(), 1);
  EXPECT_EQ(back_on_host[0].data->mem_type(), rapidsmpf::MemoryType::HOST);
  EXPECT_EQ(*back_on_host[0].metadata, metadata);

  // Check that contents match original
  auto res    = br->reserve_or_fail(back_on_host[0].data->size, rapidsmpf::MemoryType::HOST);
  auto actual = br->move_to_host_buffer(std::move(back_on_host[0].data), res);
  EXPECT_EQ(actual->copy_to_uint8_vector(), payload);
}
