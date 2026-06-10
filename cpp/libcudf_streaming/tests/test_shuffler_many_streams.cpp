/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "environment.hpp"
#include "utils.hpp"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cudf_streaming/integrations/partition.hpp>
#include <gtest/gtest.h>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>

#include <random>

extern Environment* GlobalEnvironment;

namespace {

/**
 * @brief Generate a random CUDA stream priority.
 *
 * @param random_generator A random number generator used to produce the priority.
 * @return A valid CUDA stream priority in the device range.
 */
int gen_stream_priority(std::mt19937& random_generator)
{
  int least_priority    = 0;  // numerically larger (often 0) => lower priority
  int greatest_priority = 0;  // numerically smaller (often negative) => higher priority
  RAPIDSMPF_CUDA_TRY(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  int const num_priorities = least_priority - greatest_priority + 1;
  std::uniform_int_distribution<int> dist(0, num_priorities - 1);
  return greatest_priority + dist(random_generator);
}

// To expose unexpected deadlocks, we use a 30s timeout. In a normal run, the
// shuffle shouldn't get near 30s.
constexpr auto wait_timeout = std::chrono::seconds{30};

}  // namespace

TEST(ShufflerManyStreams, Test)
{
  std::mt19937 random_generator{42};
  constexpr std::size_t chunksize = 1 << 20;
  constexpr int num_partitions    = 100;
  auto br = rapidsmpf::BufferResource::create(cudf::get_current_device_resource_ref());

  // Create a CUDA stream for each partition.
  // To stress-test stream handling, assign random priorities so streams are more
  // likely to execute in mixed order.
  std::array<cudaStream_t, num_partitions> partition_streams{};
  for (rapidsmpf::shuffler::PartID pid = 0; pid < num_partitions; ++pid) {
    RAPIDSMPF_CUDA_TRY(cudaStreamCreateWithPriority(
      &partition_streams[pid], cudaStreamNonBlocking, gen_stream_priority(random_generator)));
  }

  rapidsmpf::shuffler::Shuffler shuffler(GlobalEnvironment->comm_, 0, num_partitions, br.get());

  std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> partitions;
  for (rapidsmpf::shuffler::PartID pid = 0; pid < num_partitions; ++pid) {
    partitions.insert({pid, generate_packed_data(chunksize, pid, partition_streams[pid], *br)});
  }

  shuffler.insert(std::move(partitions));
  shuffler.insert_finished();

  EXPECT_NO_THROW(shuffler.wait(wait_timeout));
  for (auto const pid : shuffler.local_partitions()) {
    auto partition_chunks = shuffler.extract(pid);
    for (auto& chunk : partition_chunks) {
      auto const stream = chunk.data->stream();
      EXPECT_NO_FATAL_FAILURE(validate_packed_data(std::move(chunk), chunksize, pid, stream, *br));
    }
  }

  for (auto& stream : partition_streams) {
    RAPIDSMPF_CUDA_TRY(cudaStreamDestroy(stream));
  }
}
