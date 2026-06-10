/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/types.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <cstdint>
#include <memory>

namespace cudf_streaming::streaming {

/**
 * @brief Utility managing construction and use of a bloom filter.
 *
 * This class provides methods to build a bloom filter from a stream of `TableChunk`s and
 * then apply that filter to a different stream.
 *
 * A bloom filter is a fixed size probabilistic data structure that provides approximate
 * set membership queries with no false negatives. That is, let `A` be some set and `f(A)`
 * be the bloom filter representation of that set. Then, for all `a ∈ A` it holds that `a
 * ∈ f(A)`. Conversely, there is a false positive rate that increases with the number of
 * distinct values inserted into the bloom filter, and decreases with the number of filter
 * blocks. That is, for any given bloom filter, there exists `a ∉ A` such that `a ∈ f(A)`.
 *
 * See https://arxiv.org/pdf/2512.15595 for details on the GPU implementation used.
 *
 * We use bloom filters to provide runtime pre-filtering of tables during shuffle-based
 * joins. We gather the keys that will match from the build side and use those to
 * pre-filter the probe side before shuffling.
 */
struct BloomFilter {
  /**
   * @brief Construct storage for a bloom filter.
   *
   * @param ctx Streaming context.
   * @param comm Communicator for the collective operation.
   * @param seed Hash seed used when hashing values into the filter.
   * @param num_filter_blocks Number of blocks in the filter.
   */
  explicit BloomFilter(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                       std::shared_ptr<rapidsmpf::Communicator> comm,
                       std::uint64_t seed,
                       std::size_t num_filter_blocks) noexcept
    : ctx_{std::move(ctx)},
      comm_{std::move(comm)},
      seed_{seed},
      num_filter_blocks_{num_filter_blocks}
  {
  }

  /**
   * @brief Gets the communicator associated with this BloomFilter.
   *
   * @return Shared pointer to communicator.
   */
  [[nodiscard]] std::shared_ptr<rapidsmpf::Communicator> const& comm() const noexcept
  {
    return comm_;
  }

  /**
   * @brief Build a bloom filter from the input channel.
   *
   * @param ch_in Input channel of `TableChunk`s to build bloom filter for.
   * @param ch_out Output channel receiving a single message containing the bloom
   * filter.
   * @param tag Disambiguating tag to combine filters across ranks.
   * @return Coroutine representing the construction of the bloom filter.
   */
  [[nodiscard]] rapidsmpf::streaming::Actor build(
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID tag);

  /**
   * @brief Apply a bloom filter to an input channel.
   *
   * @param bloom_filter Channel containing the bloom filter (a single message).
   * @param ch_in Input channel of `TableChunk`s to apply bloom filter to.
   * @param ch_out Output channel receiving filtered `TableChunk`s.
   * @param keys Indices selecting the key columns for the hash fingerprint
   *
   * @note The application of the bloom filter expects _exactly one_ message to come
   * through the `bloom_filter` channel, which must be drained after that message is
   * sent.
   *
   * @return Coroutine representing the application of the bloom filter.
   */
  [[nodiscard]] rapidsmpf::streaming::Actor apply(
    std::shared_ptr<rapidsmpf::streaming::Channel> bloom_filter,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::vector<cudf::size_type> keys);

  /**
   * @brief Compute number of filter blocks that fit in the given L2 cache size.
   *
   * @param l2size L2 cache size in bytes.
   * @return Number of filter blocks that fit.
   */
  [[nodiscard]] static std::size_t fitting_num_blocks(std::size_t l2size) noexcept
  {
    using StorageType = std::uint32_t;
    return (l2size * 2) / (3 * sizeof(StorageType));
  }

 private:
  std::shared_ptr<rapidsmpf::streaming::Context> ctx_{};
  std::shared_ptr<rapidsmpf::Communicator> comm_{};
  std::uint64_t seed_{};
  std::size_t num_filter_blocks_{};
};
}  // namespace cudf_streaming::streaming
