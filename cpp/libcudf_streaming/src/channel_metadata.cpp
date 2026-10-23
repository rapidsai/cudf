/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <cudf_streaming/channel_metadata.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace cudf_streaming {
namespace {

void validate_order_key(order_key const& key)
{
  RAPIDSMPF_EXPECTS(
    key.column_index >= 0, "order_key: column index must be non-negative", std::invalid_argument);
  RAPIDSMPF_EXPECTS(key.order == cudf::order::ASCENDING || key.order == cudf::order::DESCENDING,
                    "order_key: invalid sort order",
                    std::invalid_argument);
  RAPIDSMPF_EXPECTS(
    key.null_order == cudf::null_order::BEFORE || key.null_order == cudf::null_order::AFTER,
    "order_key: invalid null order",
    std::invalid_argument);
}

void validate_hash_scheme(hash_scheme const& scheme)
{
  RAPIDSMPF_EXPECTS(!scheme.column_indices.empty(),
                    "hash_scheme: columns must not be empty",
                    std::invalid_argument);
  RAPIDSMPF_EXPECTS(
    scheme.modulus > 0, "hash_scheme: modulus must be positive", std::invalid_argument);
  std::unordered_set<cudf::size_type> columns;
  for (auto const index : scheme.column_indices) {
    RAPIDSMPF_EXPECTS(
      index >= 0, "hash_scheme: column indices must be non-negative", std::invalid_argument);
    RAPIDSMPF_EXPECTS(columns.insert(index).second,
                      "hash_scheme: column indices must be unique",
                      std::invalid_argument);
  }
}

void validate_partition_ordering(partition_ordering const& ordering)
{
  RAPIDSMPF_EXPECTS(
    !ordering.keys.empty(), "partition_ordering: keys must not be empty", std::invalid_argument);
  RAPIDSMPF_EXPECTS(ordering.boundaries != nullptr,
                    "partition_ordering: boundaries must not be null",
                    std::invalid_argument);
  RAPIDSMPF_EXPECTS(ordering.boundaries->is_available(),
                    "partition_ordering: boundaries must be device-resident",
                    std::invalid_argument);
  RAPIDSMPF_EXPECTS(
    ordering.keys.size() == static_cast<std::size_t>(ordering.boundaries->shape().second),
    "partition_ordering: number of keys must match number of boundary columns",
    std::invalid_argument);
  std::unordered_set<cudf::size_type> columns;
  for (auto const& key : ordering.keys) {
    validate_order_key(key);
    RAPIDSMPF_EXPECTS(columns.insert(key.column_index).second,
                      "partition_ordering: key column indices must be unique",
                      std::invalid_argument);
  }
}

void validate_range_scheme(range_scheme const& scheme)
{
  RAPIDSMPF_EXPECTS(!scheme.partition_orderings.empty(),
                    "range_scheme: partition_orderings must not be empty",
                    std::invalid_argument);
  for (auto const& ordering : scheme.partition_orderings) {
    validate_partition_ordering(ordering);
  }
}

void validate_row_ordering(row_ordering const& ordering)
{
  RAPIDSMPF_EXPECTS(!ordering.order_keys.empty(),
                    "row_ordering: order_keys must not be empty",
                    std::invalid_argument);
  RAPIDSMPF_EXPECTS(ordering.scope == row_ordering_scope::GLOBAL ||
                      ordering.scope == row_ordering_scope::RANK_LOCAL,
                    "row_ordering: invalid scope",
                    std::invalid_argument);

  RAPIDSMPF_EXPECTS(std::is_sorted(ordering.group_keys.begin(), ordering.group_keys.end()),
                    "row_ordering: group key indices must be sorted",
                    std::invalid_argument);
  std::unordered_set<cudf::size_type> columns;
  for (auto const index : ordering.group_keys) {
    RAPIDSMPF_EXPECTS(
      index >= 0, "row_ordering: group key indices must be non-negative", std::invalid_argument);
    RAPIDSMPF_EXPECTS(columns.insert(index).second,
                      "row_ordering: group key indices must be unique",
                      std::invalid_argument);
  }
  for (auto const& key : ordering.order_keys) {
    validate_order_key(key);
    RAPIDSMPF_EXPECTS(columns.insert(key.column_index).second,
                      "row_ordering: group and order key indices must be unique",
                      std::invalid_argument);
  }
}

void validate_channel_metadata(channel_metadata const& metadata)
{
  auto validate_spec = [](partitioning_spec const& spec) {
    switch (spec.type) {
      case partitioning_spec::type::NONE:
      case partitioning_spec::type::INHERIT:
        RAPIDSMPF_EXPECTS(!spec.hash.has_value() && !spec.range.has_value(),
                          "partitioning_spec: NONE and INHERIT must not carry a scheme",
                          std::invalid_argument);
        break;
      case partitioning_spec::type::HASH:
        RAPIDSMPF_EXPECTS(spec.hash.has_value() && !spec.range.has_value(),
                          "partitioning_spec: HASH requires only a hash scheme",
                          std::invalid_argument);
        validate_hash_scheme(*spec.hash);
        break;
      case partitioning_spec::type::RANGE:
        RAPIDSMPF_EXPECTS(!spec.hash.has_value() && spec.range.has_value(),
                          "partitioning_spec: RANGE requires only a range scheme",
                          std::invalid_argument);
        validate_range_scheme(*spec.range);
        break;
      default: throw std::invalid_argument("partitioning_spec: invalid type");
    }
  };
  validate_spec(metadata.partitioning.inter_rank);
  validate_spec(metadata.partitioning.local);
  for (auto const& ordering : metadata.row_orderings) {
    validate_row_ordering(ordering);
  }
}

}  // namespace

hash_scheme::hash_scheme(std::vector<cudf::size_type> column_indices, int modulus)
  : column_indices{std::move(column_indices)}, modulus{modulus}
{
  validate_hash_scheme(*this);
}

partition_ordering::partition_ordering(std::vector<order_key> keys,
                                       std::shared_ptr<table_chunk> boundaries,
                                       bool strict_boundaries)
  : keys{std::move(keys)}, boundaries{std::move(boundaries)}, strict_boundaries{strict_boundaries}
{
  validate_partition_ordering(*this);
}

partition_ordering partition_ordering::with_keys(std::vector<order_key> new_keys) const
{
  RAPIDSMPF_EXPECTS(new_keys.size() == keys.size(),
                    "partition_ordering: replacement keys must preserve key count",
                    std::invalid_argument);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    RAPIDSMPF_EXPECTS(
      new_keys[i].order == keys[i].order && new_keys[i].null_order == keys[i].null_order,
      "partition_ordering: replacement keys must preserve ordering semantics",
      std::invalid_argument);
  }
  return partition_ordering{std::move(new_keys), boundaries, strict_boundaries};
}

bool partition_ordering::boundaries_aligned_with(partition_ordering const& other,
                                                 rapidsmpf::BufferResource& br) const
{
  if (strict_boundaries != other.strict_boundaries ||
      boundaries->shape() != other.boundaries->shape()) {
    return false;
  }
  if (!std::equal(
        keys.begin(), keys.end(), other.keys.begin(), [](order_key const& a, order_key const& b) {
          return a.order == b.order && a.null_order == b.null_order;
        })) {
    return false;
  }
  if (boundaries->shape().first == 0) { return true; }
  auto const lhs    = boundaries->table_view();
  auto const rhs    = other.boundaries->table_view();
  auto const stream = boundaries->stream();
  rapidsmpf::cuda_stream_join(stream, other.boundaries->stream());
  for (cudf::size_type i = 0; i < lhs.num_columns(); ++i) {
    auto eq      = cudf::binary_operation(lhs.column(i),
                                     rhs.column(i),
                                     cudf::binary_operator::NULL_EQUALS,
                                     cudf::data_type{cudf::type_id::BOOL8},
                                     stream,
                                     br.device_mr());
    auto result  = cudf::reduce(eq->view(),
                               *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::BOOL8},
                               stream,
                               br.device_mr());
    auto& scalar = static_cast<cudf::numeric_scalar<bool>&>(*result);
    if (!scalar.value(stream)) { return false; }
  }
  return true;
}

range_scheme::range_scheme(std::vector<partition_ordering> partition_orderings)
  : partition_orderings{std::move(partition_orderings)}
{
  validate_range_scheme(*this);
}

row_ordering::row_ordering(std::vector<order_key> order_keys,
                           row_ordering_scope scope,
                           std::vector<cudf::size_type> group_keys)
  : order_keys{std::move(order_keys)}, scope{scope}, group_keys{std::move(group_keys)}
{
  std::sort(this->group_keys.begin(), this->group_keys.end());
  validate_row_ordering(*this);
}

partitioning_spec partitioning_spec::from_range(range_scheme r)
{
  validate_range_scheme(r);
  return {.type = type::RANGE, .hash = std::nullopt, .range = std::move(r)};
}

partitioning_spec partitioning_spec::from_hash(hash_scheme h)
{
  validate_hash_scheme(h);
  return {.type = type::HASH, .hash = std::move(h), .range = std::nullopt};
}

channel_metadata::channel_metadata(std::uint64_t local_count,
                                   struct partitioning partitioning,
                                   bool duplicated,
                                   std::vector<row_ordering> row_orderings)
  : local_count{local_count},
    partitioning{std::move(partitioning)},
    duplicated{duplicated},
    row_orderings{std::move(row_orderings)}
{
  validate_channel_metadata(*this);
}

rapidsmpf::streaming::Message to_message(std::uint64_t sequence_number,
                                         std::unique_ptr<channel_metadata> m)
{
  RAPIDSMPF_EXPECTS(
    m != nullptr, "channel_metadata: message payload must not be null", std::invalid_argument);
  validate_channel_metadata(*m);
  return rapidsmpf::streaming::Message{
    sequence_number,
    std::move(m),
    rapidsmpf::ContentDescription{},
    rapidsmpf::streaming::Message::Callbacks{
      .copy = [](rapidsmpf::streaming::Message const& msg,
                 rapidsmpf::MemoryReservation& /* reservation */) -> rapidsmpf::streaming::Message {
        validate_channel_metadata(msg.get<channel_metadata>());
        auto copy = std::make_unique<channel_metadata>(msg.get<channel_metadata>());
        return rapidsmpf::streaming::Message{
          msg.sequence_number(), std::move(copy), rapidsmpf::ContentDescription{}, msg.callbacks()};
      },
      .move = nullptr  // Metadata is not spillable, so it is only copied.
    }};
}

}  // namespace cudf_streaming
