/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

#include <cudf_test/table_utilities.hpp>

#include <cudf/copying.hpp>

#include <cudf_streaming/partition.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/streaming/coll/shuffler.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_actor.hpp>

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
using cudf_streaming::table_chunk;
namespace actor    = rapidsmpf::streaming::actor;
namespace cs_actor = cudf_streaming::actor;

class BaseStreamingShuffle : public BaseStreamingFixture {};

TEST_F(BaseStreamingShuffle, zero_owned_partitions_completes)
{
  auto comm = GlobalEnvironment->comm_;
  if (comm->nranks() < 2) {
    GTEST_SKIP() << "Need at least 2 ranks so that some rank owns 0 partitions";
  }
  constexpr Rank owner             = 0;
  auto collapse                    = [](std::shared_ptr<Communicator> const&,
                     shuffler::PartID,
                     shuffler::PartID) -> Rank { return owner; };
  constexpr OpID op_id             = 0;
  constexpr shuffler::PartID total = 4;
  auto shuffler = std::make_unique<ShufflerAsync>(ctx, comm, op_id, total, collapse);

  coro::sync_wait(shuffler->insert_finished());

  auto local_pids = shuffler->local_partitions();
  if (comm->rank() == owner) {
    EXPECT_EQ(local_pids.size(), total);
  } else {
    EXPECT_TRUE(local_pids.empty());
  }
}

class StreamingShuffler : public BaseStreamingShuffle, public ::testing::WithParamInterface<int> {
 public:
  const unsigned int num_partitions = 10;
  const unsigned int num_rows       = 1000;
  const unsigned int num_chunks     = 5;
  const unsigned int chunk_size     = num_rows / num_chunks;
  const std::int64_t seed           = 42;
  const cudf::hash_id hash_function = cudf::hash_id::HASH_MURMUR3;
  const OpID op_id                  = 0;

  void SetUp() override { BaseStreamingShuffle::SetUpWithThreads(GetParam()); }

  void TearDown() override { BaseStreamingShuffle::TearDown(); }

  void run_test(auto make_shuffler_actor_fn)
  {
    // Create the full input table and slice it into chunks.
    cudf::table full_input_table = random_table_with_index(seed, num_rows, 0, 10);
    std::vector<Message> input_chunks;
    for (unsigned int i = 0; i < num_chunks; ++i) {
      input_chunks.emplace_back(
        to_message(i,
                   std::make_unique<table_chunk>(
                     std::make_unique<cudf::table>(
                       cudf::slice(full_input_table,
                                   {static_cast<cudf::size_type>(i * chunk_size),
                                    static_cast<cudf::size_type>((i + 1) * chunk_size)},
                                   stream)
                         .at(0),
                       stream,
                       ctx->br()->device_mr()),
                     stream)));
    }

    // Create and run the streaming pipeline.
    std::vector<Message> output_chunks;
    {
      std::vector<Actor> actors;
      auto ch1 = ctx->create_channel();
      actors.push_back(actor::push_to_channel(ctx, ch1, std::move(input_chunks)));

      auto ch2 = ctx->create_channel();
      actors.push_back(
        cs_actor::partition_and_pack(ctx, ch1, ch2, {1}, num_partitions, hash_function, seed));

      auto ch3 = ctx->create_channel();
      actors.emplace_back(make_shuffler_actor_fn(ch2, ch3));

      auto ch4 = ctx->create_channel();
      actors.push_back(cs_actor::unpack_and_concat(ctx, ch3, ch4));

      actors.push_back(actor::pull_from_channel(ctx, ch4, output_chunks));

      run_actor_network(std::move(actors));
    }

    auto comm = GlobalEnvironment->comm_;
    std::unique_ptr<cudf::table> expected_table;
    if (comm->nranks() == 1) {  // full_input table is expected
      expected_table = std::make_unique<cudf::table>(std::move(full_input_table));
    } else {  // full_input table is replicated on all ranks
      // local partitions
      auto [table, offsets] =
        cudf::hash_partition(full_input_table.view(), {1}, num_partitions, hash_function, seed);

      auto local_pids =
        shuffler::Shuffler::local_partitions(comm, num_partitions, shuffler::Shuffler::round_robin);

      // every partition is replicated on all ranks
      std::vector<cudf::table_view> expected_tables;
      for (auto pid : local_pids) {
        auto t_view = cudf::slice(table->view(), {offsets[pid], offsets[pid + 1]}).at(0);
        // this will be replicated on all ranks
        for (rapidsmpf::Rank rank = 0; rank < comm->nranks(); ++rank) {
          expected_tables.push_back(t_view);
        }
      }
      expected_table = cudf::concatenate(expected_tables);
    }

    // Concat all output chunks to a single table.
    std::vector<cudf::table_view> output_chunks_as_views;
    for (auto& chunk : output_chunks) {
      output_chunks_as_views.push_back(chunk.get<table_chunk>().table_view());
    }
    auto result_table = cudf::concatenate(output_chunks_as_views);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(result_table->view()),
                                       sort_table(expected_table->view()));
  }
};

INSTANTIATE_TEST_SUITE_P(StreamingShuffler,
                         StreamingShuffler,
                         ::testing::Values(1, 2, 4),
                         [](testing::TestParamInfo<StreamingShuffler::ParamType> const& info) {
                           return "nthreads_" + std::to_string(info.param);
                         });

TEST_P(StreamingShuffler, basic_shuffler)
{
  EXPECT_NO_FATAL_FAILURE(run_test([&](auto ch_in, auto ch_out) -> Actor {
    return actor::shuffler(ctx, GlobalEnvironment->comm_, ch_in, ch_out, op_id, num_partitions);
  }));
}

class ShufflerAsyncTest
  : public BaseStreamingShuffle,
    public ::testing::WithParamInterface<std::tuple<std::size_t, std::uint32_t>> {
 protected:
  std::size_t n_inserts;
  std::uint32_t n_partitions;

  static constexpr OpID op_id             = 0;
  static constexpr std::size_t n_elements = 100;

  void SetUp() override
  {
    std::tie(n_inserts, n_partitions) = GetParam();

    BaseStreamingShuffle::SetUpWithThreads(4);
  }

  void TearDown() override { BaseStreamingShuffle::TearDown(); }
};

INSTANTIATE_TEST_SUITE_P(StreamingShuffler,
                         ShufflerAsyncTest,
                         ::testing::Combine(::testing::Values(1, 10),      // number of inserts
                                            ::testing::Values(1, 10, 100)  // number of partitions
                                            ),
                         [](const testing::TestParamInfo<ShufflerAsyncTest::ParamType>& info) {
                           return "ninserts_" + std::to_string(std::get<0>(info.param)) +
                                  "_nparts_" + std::to_string(std::get<1>(info.param));
                         });

TEST_P(ShufflerAsyncTest, insert_wait_extract)
{
  auto comm     = GlobalEnvironment->comm_;
  auto shuffler = std::make_unique<ShufflerAsync>(ctx, comm, op_id, n_partitions);

  for (std::size_t i = 0; i < n_inserts; ++i) {
    std::unordered_map<shuffler::PartID, PackedData> data;
    data.reserve(n_partitions);
    for (shuffler::PartID pid = 0; pid < n_partitions; ++pid) {
      data.emplace(pid, generate_packed_data(n_elements, 0, stream, *br));
    }
    shuffler->insert(std::move(data));
  }

  coro::sync_wait(shuffler->insert_finished());

  auto local_pids =
    shuffler::Shuffler::local_partitions(comm, n_partitions, &shuffler::Shuffler::round_robin);

  std::vector<shuffler::PartID> finished_pids;
  std::size_t n_chunks_received = 0;
  for (auto pid : local_pids) {
    auto chunks = shuffler->extract(pid);
    n_chunks_received += chunks.size();
    finished_pids.push_back(pid);
  }

  EXPECT_EQ(n_inserts * local_pids.size() * comm->nranks(), n_chunks_received);
  EXPECT_EQ(local_pids, finished_pids);
}
