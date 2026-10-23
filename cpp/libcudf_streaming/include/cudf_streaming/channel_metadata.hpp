/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <cudf_streaming/table_chunk.hpp>

#include <cuda_runtime_api.h>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/content_description.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace cudf_streaming {

/**
 * @brief Hash partitioning scheme.
 *
 * Rows are distributed by `hash(columns[column_indices]) % modulus`.
 */
struct hash_scheme {
  std::vector<cudf::size_type> column_indices;  ///< Column indices to hash on.
  int modulus;                                  ///< Hash modulus (number of partitions).

  /// @brief Default constructor. Produces an invalid (empty) hash scheme.
  hash_scheme() = default;

  /**
   * @brief Construct a validated hash scheme.
   *
   * @param column_indices Non-empty, unique, non-negative column indices.
   * @param modulus Positive hash modulus.
   */
  hash_scheme(std::vector<cudf::size_type> column_indices, int modulus);

  /**
   * @brief Equality comparison.
   * @return True if both schemes are equal.
   */
  bool operator==(hash_scheme const&) const = default;
};

/**
 * @brief A single sort key: column index, sort direction, and null placement.
 */
struct order_key {
  cudf::size_type column_index;  ///< Column to sort on.
  cudf::order order;             ///< ASCENDING or DESCENDING.
  cudf::null_order null_order;   ///< BEFORE or AFTER.

  /**
   * @brief Equality comparison.
   * @return True if all fields are equal.
   */
  bool operator==(order_key const&) const = default;

  /**
   * @brief Inequality comparison.
   * @return True if any field is different.
   */
  bool operator!=(order_key const&) const = default;
};

/**
 * @brief A partition-ordering description for range-partitioned data.
 *
 * Data is partitioned by value ranges based on predetermined boundaries.
 * For N partitions, there are N-1 boundary rows:
 * - Partition 0: values < boundaries[0]
 * - Partition i (0 < i < N-1): boundaries[i-1] <= values < boundaries[i]
 * - Partition N-1: values >= boundaries[N-2]
 *
 * `keys[i]` is the i-th sort column; ordering is lexicographic by `keys[0]`,
 * then `keys[1]`, and so on.
 *
 * When `boundaries` is set, its columns must align with `keys`
 * (same count and compatible dtypes). Mismatched dtypes are a usage error.
 *
 * `strict_boundaries`: when true, every row in a chunk belongs to a single partition's
 * half-open key range (partition keys do not straddle chunk interiors). When false,
 * a chunk may contain keys spanning multiple partitions.
 */
struct partition_ordering {
  std::vector<order_key> keys;              ///< Sort keys (column, order, null_order per entry).
  std::shared_ptr<table_chunk> boundaries;  ///< N-1 boundary rows for N partitions.
  /// See struct-level note on `strict_boundaries` semantics.
  bool strict_boundaries{false};

  /// @brief Default constructor. Produces an invalid (empty) partition ordering.
  partition_ordering() = default;

  /**
   * @brief Construct a validated partition ordering.
   *
   * @param keys Non-empty sort keys; size must equal `boundaries->shape().second`.
   * @param boundaries Non-null, device-resident boundary table (N-1 rows for N
   * partitions). Accepts a `unique_ptr<table_chunk>` via implicit conversion.
   * @param strict_boundaries See struct-level doc. Defaults to false.
   * @throws std::invalid_argument if `keys` is empty, `boundaries` is null or not
   * device-resident, or `keys.size() != boundaries->shape().second`.
   */
  partition_ordering(std::vector<order_key> keys,
                     std::shared_ptr<table_chunk> boundaries,
                     bool strict_boundaries = false);

  /**
   * @brief Return a new partition ordering with updated key column indices,
   * sharing boundary rows.
   *
   * @param new_keys Replacement sort keys; size must equal
   * `boundaries->shape().second`.
   * @return A new partition ordering with `new_keys` and the same boundaries and strictness.
   * @throws std::invalid_argument if `new_keys` changes the number, direction,
   * or null order of the keys.
   */
  [[nodiscard]] partition_ordering with_keys(std::vector<order_key> new_keys) const;

  /**
   * @brief Check whether boundary values are aligned with another partition ordering.
   *
   * @param other The partition ordering to compare against.
   * @param br Buffer resource used for temporary allocations during comparison.
   * @return True when both orderings have matching boundary values and
   * strict_boundaries attributes, and are otherwise compatible (same order and
   * null_order).
   */
  [[nodiscard]] bool boundaries_aligned_with(partition_ordering const& other,
                                             rapidsmpf::BufferResource& br) const;
};

/**
 * @brief Range-based partitioning scheme.
 *
 * A range_scheme advertises that the same stream is range-partitioned with
 * respect to any individual partition ordering it contains. Consumers select
 * the partition ordering relevant to an operation.
 */
struct range_scheme {
  /// Partition-ordering descriptions valid for the stream.
  std::vector<partition_ordering> partition_orderings;

  /// @brief Default constructor. Produces an invalid (empty) scheme.
  range_scheme() = default;

  /**
   * @brief Construct a validated multi-partition-ordering range_scheme.
   *
   * @param partition_orderings Non-empty sequence of partition orderings valid for the
   * stream.
   */
  explicit range_scheme(std::vector<partition_ordering> partition_orderings);
};

/**
 * @brief Scope of a row-ordering guarantee.
 */
enum class row_ordering_scope : std::uint8_t {
  GLOBAL,      ///< The logical stream is ordered across ranks within each equality group.
  RANK_LOCAL,  ///< Each rank's FIFO stream is ordered independently.
};

/**
 * @brief A row-ordering guarantee for a data stream.
 *
 * With no group keys, rows are lexicographically ordered by `order_keys`.
 * With group keys, the rows in each equality group are ordered by
 * `order_keys`. Groups need not be contiguous, and group-key order is not
 * significant. Equality follows the grouping semantics of the operation that
 * consumes the guarantee. A producer may combine this fact with hash
 * partitioning only when equal rows are guaranteed to hash to the same rank.
 *
 * A GLOBAL guarantee applies to the rank-ordered concatenation of FIFO
 * streams. When channel metadata marks the data as duplicated, each rank is
 * instead a complete replica of that logical stream.
 */
struct row_ordering {
  std::vector<order_key> order_keys;                     ///< Keys that order rows.
  row_ordering_scope scope{row_ordering_scope::GLOBAL};  ///< Scope of the guarantee.
  std::vector<cudf::size_type> group_keys;  ///< Sorted columns defining equality groups.

  /// @brief Default constructor. Produces an invalid (empty) row ordering.
  row_ordering() = default;

  /**
   * @brief Construct a validated row ordering.
   *
   * @param order_keys Non-empty keys that order rows.
   * @param scope Scope of the ordering guarantee.
   * @param group_keys Columns defining equality groups.
   * @throws std::invalid_argument if a column index is negative or repeated,
   * `order_keys` is empty, a column is both a group and order key, or `scope`
   * is invalid.
   */
  row_ordering(std::vector<order_key> order_keys,
               row_ordering_scope scope,
               std::vector<cudf::size_type> group_keys = {});

  /**
   * @brief Equality comparison.
   * @return True if all fields are equal.
   */
  bool operator==(row_ordering const&) const = default;
};

/**
 * @brief Partitioning specification for a single hierarchical level.
 *
 * Represents how data is partitioned at one level of the hierarchy
 * (e.g., inter-rank or local). Use the static factory methods to construct.
 *
 * - `none()`: No partitioning information at this level.
 * - `inherit()`: Partitioning is inherited from the parent level unchanged.
 * - `from_hash(h)`: Explicit hash partitioning with the given scheme.
 * - `from_range(r)`: Explicit range partitioning with the given scheme.
 */
struct partitioning_spec {
  /**
   * @brief Type tag for partitioning_spec.
   */
  enum class type : std::uint8_t {
    NONE,     ///< No partitioning information at this level.
    INHERIT,  ///< Partitioning is inherited from parent level unchanged.
    HASH,     ///< Hash partitioning.
    RANGE,    ///< Range partitioning.
  };

  type type = type::NONE;             ///< The type of partitioning.
  std::optional<hash_scheme> hash;    ///< Valid only when type == HASH.
  std::optional<range_scheme> range;  ///< Valid only when type == RANGE.

  /**
   * @brief Create a spec indicating no partitioning information.
   * @return A partitioning_spec with type NONE.
   */
  static partitioning_spec none() { return {}; }

  /**
   * @brief Create a spec indicating partitioning passes through from parent.
   * @return A partitioning_spec with type INHERIT.
   */
  static partitioning_spec inherit()
  {
    return {.type = type::INHERIT, .hash = std::nullopt, .range = std::nullopt};
  }

  /**
   * @brief Create a spec for hash partitioning.
   * @param h The hash scheme to use.
   * @return A partitioning_spec with type HASH.
   */
  static partitioning_spec from_hash(hash_scheme h);

  /**
   * @brief Create a spec for range partitioning.
   * @param r The range scheme to use. `r.partition_orderings` must be non-empty.
   * @return A partitioning_spec with type RANGE.
   */
  static partitioning_spec from_range(range_scheme r);
};

/**
 * @brief Hierarchical partitioning metadata for a data stream.
 *
 * Describes how data flowing through a channel is partitioned at multiple
 * levels of the system hierarchy. Each level corresponds to a communicator
 * used to shuffle data at that level:
 *
 * - `inter_rank`: Distribution across ranks, corresponding to the primary
 *   communicator (e.g., `Context::comm()`). Shuffle operations at this level
 *   move data between ranks.
 * - `local`: Distribution within a rank, corresponding to a single-rank
 *   communicator. Operations at this level repartition data locally without
 *   network communication.
 */
struct partitioning {
  /// Distribution across ranks (corresponds to primary communicator).
  partitioning_spec inter_rank;
  /// Distribution within a rank (corresponds to local/single communicator).
  partitioning_spec local;
};

/**
 * @brief Channel-level metadata describing the data stream.
 *
 * Contains information about chunk counts, partitioning, duplication, and row
 * ordering for the data flowing through a channel.
 */
struct channel_metadata {
  std::uint64_t local_count{};              ///< Local chunk-count estimate for this rank.
  struct partitioning partitioning;         ///< How the data is partitioned.
  bool duplicated{};                        ///< Whether data is duplicated on all workers.
  std::vector<row_ordering> row_orderings;  ///< Row-ordering guarantees for the stream.

  /// @brief Default constructor.
  channel_metadata() = default;

  /**
   * @brief Construct metadata with specified values.
   *
   * @param local_count Local chunk count.
   * @param partitioning Partitioning metadata (default: no partitioning).
   * @param duplicated Whether data is duplicated (default: false).
   * @param row_orderings Row-ordering guarantees (default: none).
   */
  channel_metadata(std::uint64_t local_count,
                   struct partitioning partitioning        = {},
                   bool duplicated                         = false,
                   std::vector<row_ordering> row_orderings = {});
};

/**
 * @brief Wrap a `channel_metadata` into a `Message`.
 *
 * @param sequence_number Ordering identifier for the message.
 * @param m The metadata to wrap.
 * @return A `Message` encapsulating the metadata as its payload.
 */
rapidsmpf::streaming::Message to_message(std::uint64_t sequence_number,
                                         std::unique_ptr<channel_metadata> m);

}  // namespace cudf_streaming
