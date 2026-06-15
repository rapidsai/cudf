/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

#include <cudf_test/table_utilities.hpp>

#include <cudf_streaming/streaming/table_chunk.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/content_description.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/leaf_actor.hpp>
#include <rapidsmpf/streaming/core/queue.hpp>

#include <atomic>
#include <memory>
#include <vector>

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace actor = rapidsmpf::streaming::actor;
using namespace cudf_streaming::streaming;

using StreamingLeafTasks = BaseStreamingFixture;

TEST_F(StreamingLeafTasks, PushAndPullChunks)
{
  constexpr int num_rows   = 100;
  constexpr int num_chunks = 10;

  std::vector<cudf::table> expects;
  for (int i = 0; i < num_chunks; ++i) {
    expects.emplace_back(random_table_with_index(i, num_rows, 0, 10));
  }

  std::vector<Actor> actors;
  auto ch1 = ctx->create_channel();

  // Note, we use a scope to check that coroutines keeps the input alive.
  {
    std::vector<Message> inputs;
    for (int i = 0; i < num_chunks; ++i) {
      inputs.emplace_back(to_message(
        i,
        std::make_unique<TableChunk>(
          std::make_unique<cudf::table>(expects[i], stream, ctx->br()->device_mr()), stream)));
    }

    actors.push_back(actor::push_to_channel(ctx, ch1, std::move(inputs)));
  }

  std::vector<Message> outputs;
  actors.push_back(actor::pull_from_channel(ctx, ch1, outputs));

  run_actor_network(std::move(actors));

  EXPECT_EQ(expects.size(), outputs.size());
  for (std::size_t i = 0; i < expects.size(); ++i) {
    EXPECT_EQ(outputs[i].sequence_number(), i);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(outputs[i].get<TableChunk>().table_view(),
                                       expects[i].view());
  }
}

namespace {
Actor shutdown(std::shared_ptr<Context> ctx,
               std::shared_ptr<Channel> ch,
               std::vector<Actor>&& tasks)
{
  ShutdownAtExit c{ch};
  coro_results(co_await coro::when_all(std::move(tasks)));
  co_await ch->drain(ctx->executor());
}

Actor producer(std::shared_ptr<Context> ctx,
               std::shared_ptr<ThrottlingAdaptor> ch,
               int val,
               bool should_throw = false)
{
  co_await ctx->executor()->schedule();
  auto ticket = co_await ch->acquire();
  auto [_, receipt] =
    co_await ticket.send(Message{0, std::make_unique<int>(val), ContentDescription{}});
  if (should_throw) { throw std::runtime_error("Producer throws"); }
  EXPECT_THROW(co_await ticket.send(Message{0, std::make_unique<int>(val), ContentDescription{}}),
               std::logic_error);
  co_await receipt;
  EXPECT_TRUE(receipt.is_ready());
}

Actor consumer(std::shared_ptr<Context> ctx,
               std::shared_ptr<Channel> ch,
               std::atomic<int>& result,
               bool should_throw = false)
{
  ShutdownAtExit c{ch};
  co_await ctx->executor()->schedule();
  while (true) {
    auto msg = co_await ch->receive();
    if (should_throw) { throw std::runtime_error("Consumer throws"); }
    if (msg.empty()) { break; }
    auto val = msg.release<int>();
    result.fetch_add(val, std::memory_order_relaxed);
  }
}
}  // namespace

TEST_F(StreamingLeafTasks, ThrottledAdaptor)
{
  auto ch       = ctx->create_channel();
  auto throttle = std::make_shared<ThrottlingAdaptor>(ch, 4);
  std::vector<Actor> producers;
  std::vector<Actor> consumers;
  constexpr int n_producer{100};
  constexpr int n_consumer{3};
  for (int i = 0; i < n_producer; i++) {
    producers.push_back(producer(ctx, throttle, i));
  }
  consumers.push_back(shutdown(ctx, ch, std::move(producers)));
  std::atomic<int> result{0};
  for (int i = 0; i < n_consumer; i++) {
    consumers.push_back(consumer(ctx, ch, result));
  }
  run_actor_network(std::move(consumers));
  EXPECT_EQ(result, ((n_producer - 1) * n_producer) / 2);
}

TEST_F(StreamingLeafTasks, ThrottledAdaptorThrowInProduce)
{
  auto ch       = ctx->create_channel();
  auto throttle = std::make_shared<ThrottlingAdaptor>(ch, 4);
  std::vector<Actor> producers;
  std::vector<Actor> consumers;
  constexpr int n_producer{10};
  for (int i = 0; i < n_producer; i++) {
    producers.push_back(producer(ctx, throttle, i, i == 2));
  }
  consumers.push_back(shutdown(ctx, ch, std::move(producers)));
  std::atomic<int> result;
  consumers.push_back(consumer(ctx, ch, result));
  EXPECT_THROW(run_actor_network(std::move(consumers)), std::runtime_error);
}

TEST_F(StreamingLeafTasks, ThrottledAdaptorThrowInConsume)
{
  auto ch       = ctx->create_channel();
  auto throttle = std::make_shared<ThrottlingAdaptor>(ch, 4);
  std::vector<Actor> producers;
  std::vector<Actor> consumers;
  constexpr int n_producer{100};
  constexpr int n_consumer{3};
  for (int i = 0; i < n_producer; i++) {
    producers.push_back(producer(ctx, throttle, i));
  }
  consumers.push_back(shutdown(ctx, ch, std::move(producers)));
  std::atomic<int> result;
  for (int i = 0; i < n_consumer; i++) {
    consumers.push_back(consumer(ctx, ch, result, i == 1));
  }
  EXPECT_THROW(run_actor_network(std::move(consumers)), std::runtime_error);
}

class StreamingThrottledAdaptor : public StreamingLeafTasks,
                                  public ::testing::WithParamInterface<int> {};

INSTANTIATE_TEST_SUITE_P(InvalidMaxTickets, StreamingThrottledAdaptor, ::testing::Values(-1, 0));

TEST_P(StreamingThrottledAdaptor, NonPositiveThrottleThrows)
{
  if (GlobalEnvironment->comm_->rank() != 0) {
    // Test is independent of size of communicator.
    GTEST_SKIP() << "Test only runs on rank zero";
  }
  int max_tickets = GetParam();
  auto ch         = ctx->create_channel();
  EXPECT_THROW(ThrottlingAdaptor(ch, max_tickets), std::logic_error);
}

using StreamingBoundedQueue = StreamingLeafTasks;

TEST_F(StreamingBoundedQueue, TicketUseOnce)
{
  if (GlobalEnvironment->comm_->rank() != 0) {
    // Test is independent of size of communicator.
    GTEST_SKIP() << "Test only runs on rank zero";
  }

  auto q = ctx->create_bounded_queue(2);

  auto producer = [](std::shared_ptr<BoundedQueue> q) -> coro::task<void> {
    auto shutdown = q->raii_shutdown();
    auto ticket   = co_await q->acquire();
    EXPECT_TRUE(ticket.has_value());
    auto sent = co_await ticket->send(Message{0, std::make_unique<int>(0), ContentDescription{}});
    EXPECT_TRUE(sent);
    EXPECT_THROW(co_await ticket->send(Message{1, std::make_unique<int>(1), ContentDescription{}}),
                 std::logic_error);
  };

  coro::sync_wait(producer(q));
}

TEST_F(StreamingBoundedQueue, ShutdownStopsProducer)
{
  if (GlobalEnvironment->comm_->rank() != 0) {
    // Test is independent of size of communicator.
    GTEST_SKIP() << "Test only runs on rank zero";
  }

  auto q = ctx->create_bounded_queue(2);

  auto producer = [](std::shared_ptr<BoundedQueue> q) -> coro::task<void> {
    auto shutdown = q->raii_shutdown();
    while (true) {
      auto ticket = co_await q->acquire();
      // We're going to shutdown before the producer gets to go.
      EXPECT_TRUE(!ticket.has_value());
      if (!ticket.has_value()) { break; }
    }
  };
  coro::sync_wait(q->shutdown());
  coro::sync_wait(producer(q));
}

TEST_F(StreamingBoundedQueue, ProducerThrows)
{
  if (GlobalEnvironment->comm_->rank() != 0) {
    // Test is independent of size of communicator.
    GTEST_SKIP() << "Test only runs on rank zero";
  }

  auto q = ctx->create_bounded_queue(2);

  auto producer = [](std::shared_ptr<BoundedQueue> q) -> coro::task<void> {
    auto shutdown = q->raii_shutdown();
    throw std::runtime_error("Producer throws");
  };
  auto consumer = [](std::shared_ptr<BoundedQueue> q) -> coro::task<void> {
    auto shutdown       = q->raii_shutdown();
    auto [receipt, msg] = co_await q->receive();
    EXPECT_TRUE(msg.empty());
  };
  EXPECT_THROW(coro_results(coro::sync_wait(coro::when_all(consumer(q), producer(q)))),
               std::runtime_error);
}

TEST_F(StreamingBoundedQueue, ConsumerThrows)
{
  if (GlobalEnvironment->comm_->rank() != 0) {
    // Test is independent of size of communicator.
    GTEST_SKIP() << "Test only runs on rank zero";
  }

  auto q = ctx->create_bounded_queue(2);

  auto producer = [](std::shared_ptr<BoundedQueue> q) -> coro::task<void> {
    auto shutdown = q->raii_shutdown();
    auto ticket   = co_await q->acquire();
    EXPECT_TRUE(ticket.has_value());
    EXPECT_TRUE(co_await ticket->send(Message{0, std::make_unique<int>(1), ContentDescription{}}));
    while (true) {
      auto ticket = co_await q->acquire();
      if (!ticket.has_value()) { break; }
    }
  };
  auto consumer = [](std::shared_ptr<BoundedQueue> q) -> coro::task<void> {
    auto shutdown       = q->raii_shutdown();
    auto [receipt, msg] = co_await q->receive();
    EXPECT_FALSE(msg.empty());
    EXPECT_EQ(msg.sequence_number(), 0);
    EXPECT_EQ(msg.release<int>(), 1);
    co_await receipt;
    throw std::runtime_error("Consumer throws");
  };
  EXPECT_THROW(coro_results(coro::sync_wait(coro::when_all(consumer(q), producer(q)))),
               std::runtime_error);
}

TEST_F(StreamingBoundedQueue, MultipleAcquire)
{
  if (GlobalEnvironment->comm_->rank() != 0) {
    // Test is independent of size of communicator.
    GTEST_SKIP() << "Test only runs on rank zero";
  }

  auto q = ctx->create_bounded_queue(2);

  auto producer = [](std::shared_ptr<BoundedQueue> q) -> coro::task<void> {
    auto shutdown = q->raii_shutdown();
    int i         = 0;
    while (true) {
      auto ticket = co_await q->acquire();
      if (!ticket.has_value()) {
        EXPECT_EQ(i, 2);
        break;
      }
      EXPECT_TRUE(ticket.has_value());
      EXPECT_TRUE(co_await ticket->send(
        Message{static_cast<std::uint64_t>(i), std::make_unique<int>(i), ContentDescription{}}));
      i++;
    }
  };
  auto consumer = [](std::shared_ptr<BoundedQueue> q) -> coro::task<void> {
    auto shutdown = q->raii_shutdown();
    // Receiving, but not releasing the tickets
    auto [_, msg] = co_await q->receive();
    EXPECT_FALSE(msg.empty());
    EXPECT_EQ(msg.sequence_number(), 0);
    EXPECT_EQ(msg.release<int>(), 0);
    std::tie(std::ignore, msg) = co_await q->receive();
    EXPECT_FALSE(msg.empty());
    EXPECT_EQ(msg.sequence_number(), 1);
    EXPECT_EQ(msg.release<int>(), 1);
  };
  coro_results(coro::sync_wait(coro::when_all(consumer(q), producer(q))));
}
