/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "base_streaming_fixture.hpp"

#include <cuda_runtime_api.h>

#include <coro/latch.hpp>
#include <cudf_streaming/integrations/partition.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/streaming/chunks/packed_data.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_actor.hpp>

#include <cstring>
#include <numeric>

using namespace rapidsmpf;

class StreamingAllGather : public BaseStreamingFixture,
                           public ::testing::WithParamInterface<std::tuple<int, MemoryType>> {
 public:
  void SetUp() override { BaseStreamingFixture::SetUpWithThreads(std::get<0>(GetParam())); }

  void TearDown() override { BaseStreamingFixture::TearDown(); }
};

INSTANTIATE_TEST_SUITE_P(StreamingAllGather,
                         StreamingAllGather,
                         ::testing::Combine(::testing::Values(1, 4),
                                            ::testing::Values(MemoryType::HOST,
                                                              MemoryType::DEVICE)),
                         [](testing::TestParamInfo<StreamingAllGather::ParamType> const& info) {
                           return "nthreads_" + std::to_string(std::get<0>(info.param)) +
                                  "_mem_type_" +
                                  (std::get<1>(info.param) == MemoryType::HOST ? "HOST" : "DEVICE");
                         });

TEST_P(StreamingAllGather, basic)
{
  auto mem_type  = std::get<1>(GetParam());
  auto comm      = GlobalEnvironment->comm_;
  auto allgather = streaming::AllGather(ctx, comm, OpID{0});

  int size = comm->nranks();
  int rank = comm->rank();

  constexpr int n_inserts{100};

  coro::latch latch(n_inserts);
  auto insert = [&](int insertion_id, std::uint64_t sequence) -> coro::task<void> {
    std::vector<int> data(size);
    for (int i = 0; i < size; i++) {
      data[i] = (n_inserts * rank + insertion_id) * size + i;
    }

    auto br  = ctx->br();
    auto buf = br->make_buffer(br->stream_pool().get_stream(),
                               br->reserve_or_fail(data.size() * sizeof(int), mem_type));
    buf->write_access([&](std::byte* buf_data, rmm::cuda_stream_view stream) {
      RAPIDSMPF_CUDA_TRY(
        cuda_memcpy_async(buf_data, data.data(), data.size() * sizeof(int), stream));
    });
    auto meta = std::make_unique<std::vector<std::uint8_t>>(sizeof(int));
    std::memcpy(meta->data(), &size, sizeof(int));
    allgather.insert(sequence, PackedData{std::move(meta), std::move(buf)});
    latch.count_down();
    co_return;
  };
  auto insert_finished = [&]() -> coro::task<void> {
    co_await latch;
    allgather.insert_finished();
  };
  std::vector<int> result(size * size * n_inserts);

  auto extract = [&]() -> coro::task<void> {
    auto data = co_await allgather.extract_all(streaming::AllGather::Ordered::YES);
    std::size_t offset{0};
    for (auto& pd : data) {
      RAPIDSMPF_EXPECTS(pd.metadata->size() == sizeof(int), "Invalid metadata buffer size");
      int msize;
      std::memcpy(&msize, pd.metadata->data(), sizeof(int));
      RAPIDSMPF_EXPECTS(msize == size, "Corrupted metadata value");
      RAPIDSMPF_CUDA_TRY(cuda_memcpy_async(
        result.data() + offset, pd.data->data(), pd.data->size, pd.data->stream()));
      offset += msize;
      pd.data->stream().synchronize();
    }
  };

  std::vector<coro::task<void>> pipeline{};
  // Insertions can all run concurrently.
  for (int i = 0; i < n_inserts; i++) {
    pipeline.push_back(ctx->executor()->schedule(insert(i, i)));
  }
  pipeline.push_back(ctx->executor()->schedule(insert_finished()));
  pipeline.push_back(ctx->executor()->schedule(extract()));
  streaming::run_actor_network(std::move(pipeline));
  std::vector<int> expected(size * size * n_inserts);
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_EQ(expected, result);
}

TEST_P(StreamingAllGather, streaming_actor)
{
  auto mem_type = std::get<1>(GetParam());

  auto ch_in  = ctx->create_channel();
  auto ch_out = ctx->create_channel();

  auto comm = GlobalEnvironment->comm_;
  int size  = comm->nranks();
  int rank  = comm->rank();

  constexpr int n_inserts{100};
  std::vector<streaming::Message> input_messages;
  for (int insertion_id = 0; insertion_id < n_inserts; insertion_id++) {
    std::vector<int> data(size);
    for (int i = 0; i < size; i++) {
      data[i] = (n_inserts * rank + insertion_id) * size + i;
    }

    auto br  = ctx->br();
    auto buf = br->make_buffer(br->stream_pool().get_stream(),
                               br->reserve_or_fail(data.size() * sizeof(int), mem_type));
    buf->write_access([&](std::byte* buf_data, rmm::cuda_stream_view stream) {
      RAPIDSMPF_CUDA_TRY(
        cuda_memcpy_async(buf_data, data.data(), data.size() * sizeof(int), stream));
    });
    auto meta = std::make_unique<std::vector<std::uint8_t>>(sizeof(int));
    std::memcpy(meta->data(), &size, sizeof(int));
    input_messages.emplace_back(streaming::to_message(
      insertion_id, std::make_unique<PackedData>(std::move(meta), std::move(buf))));
  }
  std::vector<streaming::Message> output_messages;
  std::vector<coro::task<void>> pipeline{};
  pipeline.push_back(ctx->executor()->schedule(
    streaming::actor::push_to_channel(ctx, ch_in, std::move(input_messages))));
  pipeline.push_back(
    ctx->executor()->schedule(streaming::actor::allgather(ctx, comm, ch_in, ch_out, OpID{0})));
  pipeline.push_back(
    ctx->executor()->schedule(streaming::actor::pull_from_channel(ctx, ch_out, output_messages)));
  streaming::run_actor_network(std::move(pipeline));
  std::vector<int> actual(size * size * n_inserts);
  std::size_t offset{0};
  for (auto& msg : output_messages) {
    auto pd = msg.release<PackedData>();
    RAPIDSMPF_EXPECTS(pd.metadata->size() == sizeof(int), "Invalid metadata buffer size");
    int msize;
    std::memcpy(&msize, pd.metadata->data(), sizeof(int));
    RAPIDSMPF_EXPECTS(msize == size, "Corrupted metadata value");
    RAPIDSMPF_CUDA_TRY(
      cuda_memcpy_async(actual.data() + offset, pd.data->data(), pd.data->size, pd.stream()));
    offset += msize;
    pd.stream().synchronize();
  }
  std::vector<int> expected(size * size * n_inserts);
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_EQ(expected, actual);
}
