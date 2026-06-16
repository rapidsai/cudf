/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "environment.hpp"
#include "utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf_streaming/partition_utils.hpp>

#include <gtest/gtest.h>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/memory/spill.hpp>
#include <rapidsmpf/shuffler/finish_counter.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>

extern Environment* GlobalEnvironment;
using namespace cudf_streaming;

TEST(ReceivedChunks, spill_skips_control_messages)
{
  auto mr = cudf::get_current_device_resource_ref();
  auto br = rapidsmpf::BufferResource::create(mr);

  rapidsmpf::shuffler::detail::ReceivedChunks received;

  // Control messages have no data buffer (data_ == nullptr); spill must skip them
  // rather than calling data_memory_type(), which throws if data_ is null.
  received.insert(rapidsmpf::shuffler::detail::Chunk::from_finished_partition(
    /*chunk_id=*/0, /*part_id=*/0, /*expected_num_chunks=*/1));

  EXPECT_EQ(received.spill(br.get(), /*amount=*/1024), 0UL);
}

TEST(ReceivedChunks, spill_respects_amount)
{
  auto mr     = cudf::get_current_device_resource_ref();
  auto br     = rapidsmpf::BufferResource::create(mr);
  auto stream = cudf::get_default_stream();

  rapidsmpf::shuffler::detail::ReceivedChunks received;
  constexpr std::size_t chunk_size = 100;

  for (rapidsmpf::shuffler::PartID pid = 0; pid < 2; ++pid) {
    auto metadata = std::make_unique<std::vector<std::uint8_t>>(std::size_t{1}, std::uint8_t{0});
    auto res      = br->reserve_or_fail(chunk_size, rapidsmpf::MemoryType::DEVICE);
    auto data     = br->make_buffer(chunk_size, stream, res);
    received.insert(rapidsmpf::shuffler::detail::Chunk::from_packed_data(
      0, pid, rapidsmpf::PackedData{std::move(metadata), std::move(data)}));
  }

  // Two partitions, one 100-byte chunk each. spill() must stop after the first
  // partition satisfies the request; the outer loop must not continue into partition 1.
  EXPECT_EQ(received.spill(br.get(), chunk_size), chunk_size);
}

TEST(MetadataMessage, round_trip)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  auto br     = rapidsmpf::BufferResource::create(mr);

  auto metadata = iota_vector<std::uint8_t>(100);

  auto expect = rapidsmpf::shuffler::detail::Chunk::from_packed_data(
    1,  // chunk_id
    2,  // part_id
    rapidsmpf::PackedData{
      std::make_unique<std::vector<std::uint8_t>>(metadata),    // non-empty metadata
      br->move(std::make_unique<rmm::device_buffer>(), stream)  // empty data
    });

  // Extract the metadata from then chunk.
  auto msg = expect.serialize();
  EXPECT_FALSE(expect.is_metadata_buffer_set());

  // Create a new chunk by deserializing the message.
  auto result = rapidsmpf::shuffler::detail::Chunk::deserialize(*msg, br.get());

  EXPECT_TRUE(expect.data_size() == 0 || result.is_data_buffer_set());
  // They should be identical.
  EXPECT_EQ(expect.part_id(), result.part_id());
  EXPECT_EQ(expect.chunk_id(), result.chunk_id());
  EXPECT_EQ(expect.expected_num_chunks(), result.expected_num_chunks());
  EXPECT_EQ(expect.data_size(), result.data_size());
  EXPECT_EQ(expect.metadata_size(), result.metadata_size());

  // The metadata should be identical to the original.
  EXPECT_EQ(metadata, *result.release_metadata_buffer());
}

using MemoryLimitsMap = std::unordered_map<rapidsmpf::MemoryType, std::int64_t>;

// Help function to get the `memory_limits` argument for a `BufferResource`
// that prioritizes the specified memory type.
MemoryLimitsMap get_memory_limits_map(rapidsmpf::MemoryType priorities)
{
  using namespace rapidsmpf;

  // We set all memory types to be unlimited.
  MemoryLimitsMap ret = {{MemoryType::DEVICE, std::numeric_limits<std::int64_t>::max()},
                         {MemoryType::HOST, std::numeric_limits<std::int64_t>::max()}};

  // And then set device memory to zero if it isn't prioritized.
  if (priorities != MemoryType::DEVICE) { ret.at(MemoryType::DEVICE) = 0; }
  // Note, we never set host memory to zero because it is used to allocate
  // stuff like metadata and control messages.
  return ret;
}

/// @tparam InsertFn: lambda that inserts the packed chunks into the shuffler.
/// Signature: void(std::vector<rapidsmpf::PackedData>&& packed_chunks)
/// @tparam InsertFinishedFn: lambda that inserts the finished flag into the shuffler.
/// Signature: void()
template <typename InsertFn, typename InsertFinishedFn>
void test_shuffler(std::shared_ptr<rapidsmpf::Communicator> const& comm,
                   rapidsmpf::shuffler::Shuffler& shuffler,
                   rapidsmpf::shuffler::PartID total_num_partitions,
                   InsertFn&& insert_fn,
                   InsertFinishedFn&& insert_finished_fn,
                   std::size_t total_num_rows,
                   std::int64_t seed,
                   cudf::hash_id hash_fn,
                   rmm::cuda_stream_view stream,
                   rapidsmpf::BufferResource* br)
{
  // To expose unexpected deadlocks, we use a 30s timeout. In a normal run, the
  // shuffle shouldn't get near 30s.
  std::chrono::seconds const wait_timeout(30);

  // Every rank creates the full input table and all the expected partitions (also
  // partitions this rank might not get after the shuffle).
  cudf::table full_input_table = random_table_with_index(seed, total_num_rows, 0, 10);
  auto [expect_partitions, owner] =
    cudf_streaming::partition_and_split(full_input_table,
                                        {1},
                                        static_cast<std::int32_t>(total_num_partitions),
                                        hash_fn,
                                        seed,
                                        stream,
                                        br,
                                        rapidsmpf::AllowOverbooking::YES);

  cudf::size_type row_offset = 0;
  cudf::size_type partiton_size =
    full_input_table.num_rows() / static_cast<cudf::size_type>(total_num_partitions);
  for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
    // To simulate that `full_input_table` is distributed between multiple ranks,
    // we divided them into `total_num_partitions` number of partitions and pick
    // the partitions this rank should use as input. We pick using round robin but
    // any distribution would work (as long as no rows are picked by multiple
    // ranks).
    // TODO: we should test different distributions of the input partitions.
    if (rapidsmpf::shuffler::Shuffler::round_robin(comm, i, total_num_partitions) == comm->rank()) {
      cudf::size_type row_end = row_offset + partiton_size;
      if (i == total_num_partitions - 1) {
        // Include the reminder of rows in the very last partition.
        row_end = full_input_table.num_rows();
      }
      // Select the partition from the full input table.
      auto slice = cudf::slice(full_input_table, {row_offset, row_end}).at(0);
      // Hash the `slice` into chunks and pack (serialize) them.
      auto packed_chunks =
        cudf_streaming::partition_and_pack(slice,
                                           {1},
                                           static_cast<std::int32_t>(total_num_partitions),
                                           hash_fn,
                                           seed,
                                           stream,
                                           br,
                                           rapidsmpf::AllowOverbooking::YES);
      // Add the chunks to the shuffle
      insert_fn(std::move(packed_chunks));
    }
    row_offset += partiton_size;
  }
  // Tell the shuffler that we have no more input partitions.
  insert_finished_fn();

  EXPECT_NO_THROW(shuffler.wait(wait_timeout));
  for (auto finished_partition : shuffler.local_partitions()) {
    auto packed_chunks = shuffler.extract(finished_partition);
    auto result        = cudf_streaming::unpack_and_concat(
      rapidsmpf::unspill_partitions(std::move(packed_chunks), br, rapidsmpf::AllowOverbooking::YES),
      stream,
      br,
      rapidsmpf::AllowOverbooking::YES);

    // We should only receive the partitions assigned to this rank.
    EXPECT_EQ(shuffler.partition_owner(comm, finished_partition, total_num_partitions),
              comm->rank());

    // Check the result while ignoring the row order.
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(result),
                                       sort_table(expect_partitions[finished_partition]));
  }
}

class MemoryLimits_NumPartition
  : public cudf::test::BaseFixtureWithParam<
      std::tuple<MemoryLimitsMap, rapidsmpf::shuffler::PartID, std::size_t>> {
 public:
  void SetUp() override
  {
    stream               = cudf::get_default_stream();
    memory_limits        = std::get<0>(GetParam());
    total_num_partitions = std::get<1>(GetParam());
    total_num_rows       = std::get<2>(GetParam());
    br                   = rapidsmpf::BufferResource::create(
      mr(), rapidsmpf::PinnedMemoryResource::Disabled, memory_limits);

    shuffler = std::make_unique<rapidsmpf::shuffler::Shuffler>(GlobalEnvironment->comm_,
                                                               0,  // op_id
                                                               total_num_partitions,
                                                               br.get());
  }

  void TearDown() override { shuffler.reset(); }

 protected:
  MemoryLimitsMap memory_limits;
  rapidsmpf::shuffler::PartID total_num_partitions;
  std::size_t total_num_rows;
  std::int64_t seed     = 42;
  cudf::hash_id hash_fn = cudf::hash_id::HASH_MURMUR3;
  rmm::cuda_stream_view stream;
  std::shared_ptr<rapidsmpf::BufferResource> br;
  std::unique_ptr<rapidsmpf::shuffler::Shuffler> shuffler;
};

// test different `memory_available` and `total_num_partitions`.
INSTANTIATE_TEST_SUITE_P(
  Shuffler,
  MemoryLimits_NumPartition,
  testing::Combine(testing::ValuesIn({get_memory_limits_map(rapidsmpf::MemoryType::HOST),
                                      get_memory_limits_map(rapidsmpf::MemoryType::DEVICE)}),
                   testing::Values(1, 2, 5, 10),        // total_num_partitions
                   testing::Values(1, 9, 100, 100'000)  // total_num_rows
                   ),
  [](const testing::TestParamInfo<MemoryLimits_NumPartition::ParamType>& info) {
    return std::to_string(info.index) + "__nparts_" + std::to_string(std::get<1>(info.param)) +
           "__nrows_" + std::to_string(std::get<2>(info.param));
  });

TEST_P(MemoryLimits_NumPartition, round_trip)
{
  EXPECT_NO_FATAL_FAILURE(test_shuffler(
    GlobalEnvironment->comm_,
    *shuffler,
    total_num_partitions,
    [&](auto&& packed_chunks) { shuffler->insert(std::move(packed_chunks)); },
    [&]() { shuffler->insert_finished(); },
    total_num_rows,
    seed,
    hash_fn,
    stream,
    br.get()));
}

// Test that the same communicator can be used concurrently by multiple shufflers in
// separate threads
class ConcurrentShuffleTest : public cudf::test::BaseFixtureWithParam<std::tuple<int, int>> {
 public:
  void SetUp() override
  {
    num_shufflers        = std::get<0>(GetParam());
    total_num_partitions = static_cast<rapidsmpf::shuffler::PartID>(std::get<1>(GetParam()));

    // these resources will be used by multiple threads to instantiate shufflers
    br     = rapidsmpf::BufferResource::create(mr());
    stream = cudf::get_default_stream();
  }

  void TearDown() override {}

  // test run for each thread. The test follows the same logic as
  // `MemoryLimits_NumPartition` test, but without any memory limitations
  template <typename InsertFn, typename InsertFinishedFn>
  void RunTest(int t_id, InsertFn&& insert_fn, InsertFinishedFn&& insert_finished_fn)
  {
    rapidsmpf::shuffler::Shuffler shuffler(GlobalEnvironment->comm_,
                                           t_id,  // op_id, use t_id as a proxy
                                           total_num_partitions,
                                           br.get());

    EXPECT_NO_FATAL_FAILURE(test_shuffler(
      GlobalEnvironment->comm_,
      shuffler,
      total_num_partitions,
      [&](auto&& packed_chunks) { insert_fn(shuffler, std::move(packed_chunks)); },
      [&]() { insert_finished_fn(shuffler); },
      100'000,  // total_num_rows
      t_id,     // seed
      cudf::hash_id::HASH_MURMUR3,
      stream,
      br.get()));
  }

  template <typename InsertFn, typename InsertFinishedFn>
  void RunTestTemplate(InsertFn insert_fn, InsertFinishedFn insert_finished_fn)
  {
    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<std::size_t>(num_shufflers));

    for (int t_id = 0; t_id < num_shufflers; t_id++) {
      // pass a copy of the insert_fn and insert_finished_fn to each thread
      futures.push_back(
        std::async(std::launch::async,
                   [this, t_id, insert_fn1 = insert_fn, insert_finished_fn1 = insert_finished_fn] {
                     ASSERT_NO_FATAL_FAILURE(
                       this->RunTest(t_id, std::move(insert_fn1), std::move(insert_finished_fn1)));
                   }));
    }

    for (auto& f : futures) {
      ASSERT_NO_THROW(f.wait());
    }
  }

  int num_shufflers;
  rapidsmpf::shuffler::PartID total_num_partitions;

  rmm::cuda_stream_view stream;
  std::shared_ptr<rapidsmpf::BufferResource> br;
};

TEST_P(ConcurrentShuffleTest, round_trip)
{
  ASSERT_NO_FATAL_FAILURE(RunTestTemplate(
    [&](auto& shuffler, auto&& packed_chunks) { shuffler.insert(std::move(packed_chunks)); },
    [&](auto& shuffler) { shuffler.insert_finished(); }));
}

// test different `num_shufflers` and `total_num_partitions`.
INSTANTIATE_TEST_SUITE_P(ConcurrentShuffle,
                         ConcurrentShuffleTest,
                         testing::Combine(testing::ValuesIn({1, 2, 4}),    // num_shufflers
                                          testing::ValuesIn({1, 10, 100})  // total_num_partitions
                                          ),
                         [](const testing::TestParamInfo<ConcurrentShuffleTest::ParamType>& info) {
                           return "num_shufflers_" + std::to_string(std::get<0>(info.param)) +
                                  "__total_num_partitions_" +
                                  std::to_string(std::get<1>(info.param));
                         });

TEST(Shuffler, SpillOnInsertAndExtraction)
{
  rapidsmpf::shuffler::PartID const total_num_partitions = 2;
  std::int64_t const seed                                = 42;
  cudf::hash_id const hash_fn                            = cudf::hash_id::HASH_MURMUR3;
  auto stream                                            = cudf::get_default_stream();

  // Use RapidsMPF's memory resource adaptor so the test can observe per-rank
  // allocation counts via `get_main_record().num_current_allocs()`.
  rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};

  // Control spilling by adjusting the DEVICE memory limit at runtime.
  // `memory_available(DEVICE)` is computed as `limit - current_allocated()`, so a
  // sufficiently large positive limit reliably keeps available memory > 0 (no spill),
  // while a sufficiently large negative limit reliably keeps available memory < 0
  // (force spill), regardless of how many bytes are currently allocated from `mr`.
  constexpr std::int64_t k_no_spill_limit    = (1LL << 40);
  constexpr std::int64_t k_force_spill_limit = -(1LL << 40);
  auto br                                    = rapidsmpf::BufferResource::create(mr,
                                              rapidsmpf::PinnedMemoryResource::Disabled,
                                                                                 {{rapidsmpf::MemoryType::DEVICE, k_no_spill_limit}},
                                              std::nullopt  // disable periodic spill check
  );

  // Create a communicator of size 1, such that each shuffler will run locally.
  auto comm = GlobalEnvironment->split_comm();
  EXPECT_EQ(comm->nranks(), 1);

  // Create a shuffler and input chunks.
  rapidsmpf::shuffler::Shuffler shuffler(comm,
                                         0,  // op_id
                                         total_num_partitions,
                                         br.get());
  cudf::table input_table = random_table_with_index(seed, 1000, 0, 10);
  auto input_chunks =
    cudf_streaming::partition_and_pack(input_table,
                                       {1},
                                       total_num_partitions,
                                       hash_fn,
                                       seed,
                                       stream,
                                       br.get(),
                                       rapidsmpf::AllowOverbooking::YES);  // with overbooking

  // Insert spills does nothing when device memory is available, we start
  // with 2 device allocations.
  EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);
  shuffler.insert(std::move(input_chunks));
  // And we end with two 2 device allocations.
  EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);

  // Let's force spilling.
  br->set_memory_limit(rapidsmpf::MemoryType::DEVICE, k_force_spill_limit);

  {
    // Now extract triggers spilling of the partition not being extracted.
    std::vector<rapidsmpf::PackedData> output_chunks = rapidsmpf::unspill_partitions(
      shuffler.extract(0), br.get(), rapidsmpf::AllowOverbooking::YES);
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 1);

    // And insert also triggers spilling. We end up with zero device allocations.
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
    chunk.emplace(0, std::move(output_chunks.at(0)));
    shuffler.insert(std::move(chunk));
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 0);
  }

  // Extract and unspill both partitions.
  std::vector<rapidsmpf::PackedData> out0 =
    rapidsmpf::unspill_partitions(shuffler.extract(0), br.get(), rapidsmpf::AllowOverbooking::YES);
  EXPECT_EQ(mr.get_main_record().num_current_allocs(), 1);
  std::vector<rapidsmpf::PackedData> out1 =
    rapidsmpf::unspill_partitions(shuffler.extract(1), br.get(), rapidsmpf::AllowOverbooking::YES);
  EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);

  // Disable spilling and insert the first partition.
  br->set_memory_limit(rapidsmpf::MemoryType::DEVICE, k_no_spill_limit);
  {
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
    chunk.emplace(0, std::move(out0.at(0)));
    shuffler.insert(std::move(chunk));
  }
  EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);

  // Enable spilling and insert the second partition, which should trigger spilling
  // of both the first partition already in the shuffler and the second partition
  // that are being inserted.
  br->set_memory_limit(rapidsmpf::MemoryType::DEVICE, k_force_spill_limit);
  {
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
    chunk.emplace(1, std::move(out1.at(0)));
    shuffler.insert(std::move(chunk));
  }
  EXPECT_EQ(mr.get_main_record().num_current_allocs(), 0);
  shuffler.insert_finished();
}

TEST(FinishCounterTests, zero_local_partitions_immediately_finished)
{
  rapidsmpf::shuffler::detail::FinishCounter finish_counter(
    /*nranks=*/2, /*n_local_partitions=*/0);

  EXPECT_TRUE(finish_counter.all_finished());
}

TEST(FinishCounterTests, nonzero_local_partitions_finishes_after_all_chunks)
{
  rapidsmpf::shuffler::detail::FinishCounter finish_counter(
    /*nranks=*/1, /*n_local_partitions=*/2);

  EXPECT_FALSE(finish_counter.all_finished());

  // One rank sends 3 chunks total.
  finish_counter.move_goalpost(0, 3);
  finish_counter.add_finished_chunk();
  finish_counter.add_finished_chunk();
  EXPECT_FALSE(finish_counter.all_finished());

  finish_counter.add_finished_chunk();
  EXPECT_TRUE(finish_counter.all_finished());
}

TEST(FinishCounterTests, multi_rank_completion)
{
  auto comm = GlobalEnvironment->comm_;

  if (comm->rank() != 0) { GTEST_SKIP() << "Test only runs on rank 0"; }

  // Use nranks partitions so each rank owns exactly 1 partition (round robin).
  auto out_nparts = rapidsmpf::safe_cast<rapidsmpf::shuffler::PartID>(comm->nranks());

  auto local_partitions = rapidsmpf::shuffler::Shuffler::local_partitions(
    comm, out_nparts, &rapidsmpf::shuffler::Shuffler::round_robin);
  ASSERT_EQ(local_partitions.size(), 1);

  rapidsmpf::shuffler::detail::FinishCounter finish_counter(comm->nranks(),
                                                            local_partitions.size());

  // Not finished yet.
  EXPECT_FALSE(finish_counter.all_finished());

  // For nranks ranks, each rank sends 1 data chunk + 1 control, so
  // move_goalpost(rank, 2) per rank.
  for (rapidsmpf::Rank r = 0; r < comm->nranks(); r++) {
    finish_counter.move_goalpost(r, 2);
  }

  // Add finished chunks: 1 data chunk per rank + 1 control per rank = 2 * nranks
  for (rapidsmpf::Rank r = 0; r < comm->nranks(); r++) {
    finish_counter.add_finished_chunk();  // data chunk
    finish_counter.add_finished_chunk();  // control chunk
  }

  EXPECT_TRUE(finish_counter.all_finished());
}

class FinishCounterMultithreadingTest
  : public ::testing::TestWithParam<std::tuple<rapidsmpf::shuffler::PartID, std::uint32_t>> {
 protected:
  rapidsmpf::Rank const nranks{1};  // simulate a single rank

  std::unique_ptr<rapidsmpf::shuffler::detail::FinishCounter> finish_counter;
  rapidsmpf::shuffler::PartID npartitions;
  std::uint32_t nthreads;

  void SetUp() override
  {
    std::tie(npartitions, nthreads) = GetParam();

    finish_counter =
      std::make_unique<rapidsmpf::shuffler::detail::FinishCounter>(nranks, npartitions);
  }

  void produce_data()
  {
    // Simulate nranks=1: one rank reports chunk count = npartitions + 1
    // (one data chunk per partition + 1 control message)
    finish_counter->move_goalpost(rapidsmpf::Rank{0}, npartitions + 1);
    for (rapidsmpf::shuffler::PartID i = 0; i <= npartitions; i++) {
      finish_counter->add_finished_chunk();
    }
  }
};

// Parametrize on number of partitions and number of consumer threads
INSTANTIATE_TEST_SUITE_P(FinishCounterMultithreadingTestP,
                         FinishCounterMultithreadingTest,
                         testing::Combine(testing::Values(1, 2, 100, 101),
                                          testing::Values(1, 2, 3)),
                         [](const auto& info) {
                           return "npartitions_" + std::to_string(std::get<0>(info.param)) +
                                  "__nthreads_" + std::to_string(std::get<1>(info.param));
                         });

TEST_P(FinishCounterMultithreadingTest, concurrent_all_finished_check)
{
  produce_data();

  std::atomic<std::uint32_t> n_checks{0};
  std::vector<std::future<void>> futures;
  for (std::uint32_t tid = 0; tid < nthreads; tid++) {
    futures.emplace_back(std::async(std::launch::async, [&, tid] {
      for (std::uint32_t i = tid; i < npartitions; i += nthreads) {
        EXPECT_TRUE(finish_counter->all_finished());
        n_checks.fetch_add(1, std::memory_order_relaxed);
      }
    }));
  }

  EXPECT_NO_THROW(std::ranges::for_each(futures, [](auto& f) { f.get(); }));

  EXPECT_EQ(npartitions, n_checks);
  EXPECT_TRUE(finish_counter->all_finished());
}

class ContiguousPartitionAssignmentTest
  : public ::testing::TestWithParam<rapidsmpf::shuffler::PartID> {
 protected:
  void SetUp() override
  {
    comm                 = GlobalEnvironment->comm_;
    nranks               = comm->nranks();
    rank                 = comm->rank();
    total_num_partitions = GetParam();
  }

  std::shared_ptr<rapidsmpf::Communicator> comm;
  rapidsmpf::Rank nranks;
  rapidsmpf::Rank rank;
  rapidsmpf::shuffler::PartID total_num_partitions;
};

INSTANTIATE_TEST_SUITE_P(PartitionAssignment,
                         ContiguousPartitionAssignmentTest,
                         testing::Values(1, 2, 3, 5, 7, 10, 16, 100),
                         [](const testing::TestParamInfo<rapidsmpf::shuffler::PartID>& info) {
                           return "nparts_" + std::to_string(info.param);
                         });

TEST_P(ContiguousPartitionAssignmentTest, contiguous)
{
  std::vector<std::vector<rapidsmpf::shuffler::PartID>> rank_partitions(nranks);
  for (rapidsmpf::shuffler::PartID pid = 0; pid < total_num_partitions; ++pid) {
    auto owner = rapidsmpf::shuffler::Shuffler::contiguous(comm, pid, total_num_partitions);
    EXPECT_GE(owner, 0);
    EXPECT_LT(owner, nranks);
    rank_partitions[owner].push_back(pid);
  }

  // Each rank's partitions must be contiguous.
  for (rapidsmpf::Rank r = 0; r < nranks; ++r) {
    auto const& pids = rank_partitions[r];
    for (std::size_t i = 1; i < pids.size(); ++i) {
      EXPECT_EQ(pids[i], pids[i - 1] + 1);
    }
  }

  // Concatenating all rank partitions should cover [0, total_num_partitions).
  std::vector<rapidsmpf::shuffler::PartID> all_pids;
  for (auto const& pids : rank_partitions) {
    all_pids.insert(all_pids.end(), pids.begin(), pids.end());
  }
  EXPECT_EQ(all_pids, iota_vector<rapidsmpf::shuffler::PartID>(total_num_partitions));
}

TEST(Shuffler, ShutdownWhilePaused)
{
  auto progress_thread = GlobalEnvironment->comm_->progress_thread();
  auto mr              = cudf::get_current_device_resource_ref();

  auto br = rapidsmpf::BufferResource::create(mr);

  auto shuffler = rapidsmpf::shuffler::Shuffler(GlobalEnvironment->comm_, 0, 1, br.get());

  progress_thread->pause();
  EXPECT_FALSE(progress_thread->is_running());
  shuffler.insert_finished();
  // Progress thread must be running before shuffle shutdown, otherwise we have some
  // orphan messages in the shuffle that are never sent/received.
  progress_thread->resume();
  EXPECT_TRUE(progress_thread->is_running());
  EXPECT_NO_THROW(shuffler.shutdown());
}

// check cudf pack conditions for empty table
TEST(EmptyPartitions, cudf_pack)
{
  auto stream     = cudf::get_default_stream();
  cudf::table tbl = random_table_with_index(0, 0, 0, 0);
  EXPECT_EQ(0, tbl.num_rows());

  // following conditions should be met for an empty cudf table
  auto packed = cudf::pack(tbl, stream);
  EXPECT_TRUE(packed.metadata);
  EXPECT_TRUE(packed.gpu_data);
  EXPECT_EQ(0, packed.gpu_data->size());
}

class ExtractEmptyPartitionsTest : public cudf::test::BaseFixture {
 public:
  static constexpr rapidsmpf::shuffler::PartID nparts = 10;
  static constexpr auto wait_timeout                  = std::chrono::seconds(30);

  void SetUp() override
  {
    stream = cudf::get_default_stream();
    br     = rapidsmpf::BufferResource::create(mr());

    shuffler = std::make_unique<rapidsmpf::shuffler::Shuffler>(
      GlobalEnvironment->comm_, 0, nparts, br.get());
  }

  void TearDown() override { shuffler.reset(); }

  void insert_chunks(
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData>&& chunks)
  {
    if (!chunks.empty()) { shuffler->insert(std::move(chunks)); }
    shuffler->insert_finished();
  }

  void verify_extracted_chunks(auto expected_empty_fn)
  {
    EXPECT_NO_THROW(shuffler->wait(wait_timeout));
    for (auto pid : shuffler->local_partitions()) {
      SCOPED_TRACE("pid: " + std::to_string(pid));
      std::vector<rapidsmpf::PackedData> chunks;
      EXPECT_NO_THROW({ chunks = shuffler->extract(pid); });

      if (expected_empty_fn(pid)) {
        EXPECT_TRUE(chunks.empty());
      } else {
        EXPECT_EQ(GlobalEnvironment->comm_->nranks(), chunks.size());
      }
    }
  }

  auto empty_packed_data()
  {
    return rapidsmpf::PackedData{std::make_unique<std::vector<std::uint8_t>>(),
                                 br->move(std::make_unique<rmm::device_buffer>(), stream)};
  }

  auto non_empty_packed_data()
  {
    return rapidsmpf::PackedData{
      std::make_unique<std::vector<std::uint8_t>>(10),
      br->move(std::make_unique<rmm::device_buffer>(10, stream), stream)};
  }

  rmm::cuda_stream_view stream;
  std::shared_ptr<rapidsmpf::BufferResource> br;
  std::unique_ptr<rapidsmpf::shuffler::Shuffler> shuffler;
};

TEST_F(ExtractEmptyPartitionsTest, NoInsertions)
{
  insert_chunks({});
  EXPECT_NO_FATAL_FAILURE(verify_extracted_chunks([](auto) { return true; }));
}

TEST_F(ExtractEmptyPartitionsTest, AllEmptyInsertions)
{
  std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunks;
  for (rapidsmpf::shuffler::PartID pid = 0; pid < nparts; ++pid) {
    chunks.emplace(pid, empty_packed_data());
  }

  insert_chunks(std::move(chunks));
  EXPECT_NO_FATAL_FAILURE(verify_extracted_chunks([](auto) { return true; }));
}

TEST_F(ExtractEmptyPartitionsTest, SomeEmptyInsertions)
{
  std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunks;
  for (rapidsmpf::shuffler::PartID pid = 0; pid < nparts; ++pid) {
    if (pid % 3 == 0) { chunks.emplace(pid, empty_packed_data()); }
  }

  insert_chunks(std::move(chunks));
  EXPECT_NO_FATAL_FAILURE(verify_extracted_chunks([](auto) { return true; }));
}

TEST_F(ExtractEmptyPartitionsTest, SomeEmptyAndNonEmptyInsertions)
{
  std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunks;
  for (rapidsmpf::shuffler::PartID pid = 0; pid < nparts; ++pid) {
    if (pid % 3 == 0) {
      chunks.emplace(pid, empty_packed_data());
    } else {
      chunks.emplace(pid, non_empty_packed_data());
    }
  }

  insert_chunks(std::move(chunks));
  EXPECT_NO_FATAL_FAILURE(verify_extracted_chunks([](auto pid) { return pid % 3 == 0; }));
}

TEST(ShufflerTest, multiple_shutdowns)
{
  auto& comm = GlobalEnvironment->comm_;
  auto br    = rapidsmpf::BufferResource::create(cudf::get_current_device_resource_ref());
  auto shuffler =
    std::make_unique<rapidsmpf::shuffler::Shuffler>(comm, 0, comm->nranks(), br.get());

  shuffler->insert_finished();
  EXPECT_NO_THROW(shuffler->wait(std::chrono::seconds(30)));
  for (auto pid : shuffler->local_partitions()) {
    std::ignore = shuffler->extract(pid);
  }

  constexpr int n_threads = 10;
  std::vector<std::future<void>> futures;
  for (int i = 0; i < n_threads; ++i) {
    futures.emplace_back(std::async(std::launch::async, [&] { shuffler->shutdown(); }));
  }
  std::ranges::for_each(futures, [](auto& future) { future.get(); });
}

// Test that multiple threads can call wait() concurrently.
TEST(Shuffler, concurrent_wait)
{
  auto const& comm = GlobalEnvironment->comm_;
  auto stream      = cudf::get_default_stream();
  auto br          = rapidsmpf::BufferResource::create(cudf::get_current_device_resource_ref());

  // Use more partitions than ranks so each rank owns multiple partitions, ensuring
  // multiple threads call wait() concurrently on the same shuffler.
  auto const total_num_partitions =
    rapidsmpf::safe_cast<rapidsmpf::shuffler::PartID>(comm->nranks()) * 8;
  constexpr std::size_t total_num_rows = 1000;
  constexpr cudf::hash_id hash_fn      = cudf::hash_id::HASH_MURMUR3;
  constexpr std::int64_t seed          = 42;
  constexpr auto wait_timeout          = std::chrono::seconds{30};

  rapidsmpf::shuffler::Shuffler shuffler(comm, 0, total_num_partitions, br.get());

  cudf::table full_input = random_table_with_index(seed, total_num_rows, 0, 10);
  auto [expected, owner] =
    cudf_streaming::partition_and_split(full_input,
                                        {1},
                                        static_cast<std::int32_t>(total_num_partitions),
                                        hash_fn,
                                        seed,
                                        stream,
                                        br.get(),
                                        rapidsmpf::AllowOverbooking::YES);

  {
    std::vector<std::future<void>> insert_futures;
    cudf::size_type row_offset = 0;
    cudf::size_type part_size =
      full_input.num_rows() / static_cast<cudf::size_type>(total_num_partitions);
    for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
      if (rapidsmpf::shuffler::Shuffler::round_robin(comm, i, total_num_partitions) ==
          comm->rank()) {
        cudf::size_type row_end = row_offset + part_size;
        if (i == total_num_partitions - 1) { row_end = full_input.num_rows(); }
        auto slice = cudf::slice(full_input, {row_offset, row_end}).at(0);
        insert_futures.push_back(std::async(std::launch::async, [&, slice] {
          shuffler.insert(
            cudf_streaming::partition_and_pack(slice,
                                               {1},
                                               static_cast<std::int32_t>(total_num_partitions),
                                               hash_fn,
                                               seed,
                                               br->stream_pool().get_stream(),
                                               br.get(),
                                               rapidsmpf::AllowOverbooking::YES));
        }));
      }
      row_offset += part_size;
    }
    std::ranges::for_each(insert_futures, [](auto& f) { f.get(); });
    shuffler.insert_finished();
  }

  auto local_pids = shuffler.local_partitions();
  std::vector<std::future<void>> futures;
  for (auto pid : local_pids) {
    futures.push_back(std::async(std::launch::async, [&, pid] {
      EXPECT_NO_THROW(shuffler.wait(wait_timeout));
      auto chunks = shuffler.extract(pid);
      auto result = cudf_streaming::unpack_and_concat(
        rapidsmpf::unspill_partitions(
          std::move(chunks), br.get(), rapidsmpf::AllowOverbooking::YES),
        stream,
        br.get(),
        rapidsmpf::AllowOverbooking::YES);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(result), sort_table(expected[pid]));
    }));
  }
  std::ranges::for_each(futures, [](auto& f) { f.get(); });
}

// Test that reusing an OpID after a completed shuffle doesn't cause cross-matching of
// messages between the old and new shuffle.
//
// On rank 0 we inject a stream-ordered delay into device allocations so that received
// chunks stay "not ready" in the event loop. With small messages, other ranks can finish
// and move on to the next shuffle. Its messages will then be matched on rank 0 by the
// blocked previous shuffle, unless recv gating correctly prevents cross-talk.
TEST(Shuffler, opid_reuse)
{
  auto const& comm = GlobalEnvironment->comm_;
  if (comm->nranks() == 1) { GTEST_SKIP() << "OpID reuse test requires multiple ranks"; }

  auto stream = cudf::get_default_stream();
  auto const total_num_partitions =
    rapidsmpf::safe_cast<rapidsmpf::shuffler::PartID>(comm->nranks());
  constexpr std::size_t total_num_rows = 1000;
  constexpr cudf::hash_id hash_fn      = cudf::hash_id::HASH_MURMUR3;
  constexpr rapidsmpf::OpID op_id      = 0;
  constexpr auto wait_timeout          = std::chrono::seconds{30};

  rmm::mr::cuda_memory_resource mr;
  auto br = rapidsmpf::BufferResource::create(mr);

  // On rank 0, wrap the device MR with a delayed version for the shuffler.
  std::unique_ptr<DelayedMemoryResource> delayed_mr;
  std::shared_ptr<rapidsmpf::BufferResource> delayed_br;
  rapidsmpf::BufferResource* shuffler_br = br.get();
  if (comm->rank() == 0) {
    delayed_mr  = std::make_unique<DelayedMemoryResource>(mr, std::chrono::milliseconds(500));
    delayed_br  = rapidsmpf::BufferResource::create(*delayed_mr);
    shuffler_br = delayed_br.get();
  }

  auto insert_data = [&](rapidsmpf::shuffler::Shuffler& shuffler, std::int64_t seed) {
    cudf::table full_input     = random_table_with_index(seed, total_num_rows, 0, 10);
    cudf::size_type row_offset = 0;
    cudf::size_type part_size =
      full_input.num_rows() / static_cast<cudf::size_type>(total_num_partitions);
    for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
      if (rapidsmpf::shuffler::Shuffler::round_robin(comm, i, total_num_partitions) ==
          comm->rank()) {
        cudf::size_type row_end = row_offset + part_size;
        if (i == total_num_partitions - 1) { row_end = full_input.num_rows(); }
        auto slice = cudf::slice(full_input, {row_offset, row_end}).at(0);
        auto packed =
          cudf_streaming::partition_and_pack(slice,
                                             {1},
                                             static_cast<std::int32_t>(total_num_partitions),
                                             hash_fn,
                                             seed,
                                             stream,
                                             br.get(),
                                             rapidsmpf::AllowOverbooking::YES);
        shuffler.insert(std::move(packed));
      }
      row_offset += part_size;
    }
  };

  auto validate_results = [&](rapidsmpf::shuffler::Shuffler& shuffler, std::int64_t seed) {
    cudf::table full_input = random_table_with_index(seed, total_num_rows, 0, 10);
    auto [expected, owner] =
      cudf_streaming::partition_and_split(full_input,
                                          {1},
                                          static_cast<std::int32_t>(total_num_partitions),
                                          hash_fn,
                                          seed,
                                          stream,
                                          br.get(),
                                          rapidsmpf::AllowOverbooking::YES);
    for (auto pid : shuffler.local_partitions()) {
      auto chunks = shuffler.extract(pid);
      auto result = cudf_streaming::unpack_and_concat(
        rapidsmpf::unspill_partitions(
          std::move(chunks), br.get(), rapidsmpf::AllowOverbooking::YES),
        stream,
        br.get(),
        rapidsmpf::AllowOverbooking::YES);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(result), sort_table(expected[pid]));
    }
  };

  rapidsmpf::shuffler::Shuffler shuffle1(comm, op_id, total_num_partitions, shuffler_br);
  insert_data(shuffle1, 42);
  shuffle1.insert_finished();
  EXPECT_NO_THROW(shuffle1.wait(wait_timeout));

  rapidsmpf::shuffler::Shuffler shuffle2(comm, op_id, total_num_partitions, shuffler_br);
  insert_data(shuffle2, 123);
  shuffle2.insert_finished();
  EXPECT_NO_THROW(shuffle2.wait(wait_timeout));

  validate_results(shuffle1, 42);
  validate_results(shuffle2, 123);
}

// Same as opid_reuse but with total_num_partitions=1, so only rank 0 owns a partition.
// All other ranks have empty local_partitions and empty recv loops. This exercises the
// edge case where non-partition-owning ranks must still correctly handle op_id reuse.
TEST(Shuffler, opid_reuse_with_empty_partitions)
{
  auto const& comm = GlobalEnvironment->comm_;
  if (comm->nranks() == 1) { GTEST_SKIP() << "OpID reuse test requires multiple ranks"; }

  auto stream                                                = cudf::get_default_stream();
  constexpr rapidsmpf::shuffler::PartID total_num_partitions = 1;
  constexpr std::size_t total_num_rows                       = 1000;
  constexpr cudf::hash_id hash_fn                            = cudf::hash_id::HASH_MURMUR3;
  constexpr rapidsmpf::OpID op_id                            = 0;
  constexpr auto wait_timeout                                = std::chrono::seconds{30};

  rmm::mr::cuda_memory_resource mr;
  auto br = rapidsmpf::BufferResource::create(mr);

  // On rank 0, wrap the device MR with a delayed version for the shuffler.
  std::unique_ptr<DelayedMemoryResource> delayed_mr;
  std::shared_ptr<rapidsmpf::BufferResource> delayed_br;
  rapidsmpf::BufferResource* shuffler_br = br.get();
  if (comm->rank() == 0) {
    delayed_mr  = std::make_unique<DelayedMemoryResource>(mr, std::chrono::milliseconds(500));
    delayed_br  = rapidsmpf::BufferResource::create(*delayed_mr);
    shuffler_br = delayed_br.get();
  }

  auto insert_data = [&](rapidsmpf::shuffler::Shuffler& shuffler, std::int64_t seed) {
    cudf::table full_input = random_table_with_index(seed, total_num_rows, 0, 10);
    // With total_num_partitions=1, only rank 0 owns the single partition.
    if (rapidsmpf::shuffler::Shuffler::round_robin(comm, 0, total_num_partitions) == comm->rank()) {
      auto packed =
        cudf_streaming::partition_and_pack(full_input,
                                           {1},
                                           static_cast<std::int32_t>(total_num_partitions),
                                           hash_fn,
                                           seed,
                                           stream,
                                           br.get(),
                                           rapidsmpf::AllowOverbooking::YES);
      shuffler.insert(std::move(packed));
    }
  };

  auto validate_results = [&](rapidsmpf::shuffler::Shuffler& shuffler, std::int64_t seed) {
    cudf::table full_input = random_table_with_index(seed, total_num_rows, 0, 10);
    auto [expected, owner] =
      cudf_streaming::partition_and_split(full_input,
                                          {1},
                                          static_cast<std::int32_t>(total_num_partitions),
                                          hash_fn,
                                          seed,
                                          stream,
                                          br.get(),
                                          rapidsmpf::AllowOverbooking::YES);
    for (auto pid : shuffler.local_partitions()) {
      auto chunks = shuffler.extract(pid);
      auto result = cudf_streaming::unpack_and_concat(
        rapidsmpf::unspill_partitions(
          std::move(chunks), br.get(), rapidsmpf::AllowOverbooking::YES),
        stream,
        br.get(),
        rapidsmpf::AllowOverbooking::YES);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(result), sort_table(expected[pid]));
    }
  };

  rapidsmpf::shuffler::Shuffler shuffle1(comm, op_id, total_num_partitions, shuffler_br);
  insert_data(shuffle1, 42);
  shuffle1.insert_finished();
  EXPECT_NO_THROW(shuffle1.wait(wait_timeout));

  rapidsmpf::shuffler::Shuffler shuffle2(comm, op_id, total_num_partitions, shuffler_br);
  insert_data(shuffle2, 123);
  shuffle2.insert_finished();
  EXPECT_NO_THROW(shuffle2.wait(wait_timeout));

  validate_results(shuffle1, 42);
  validate_results(shuffle2, 123);
}
