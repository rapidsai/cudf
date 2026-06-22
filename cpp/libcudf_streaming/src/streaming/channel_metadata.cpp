/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <cudf_streaming/streaming/channel_metadata.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace cudf_streaming::streaming {

OrderScheme::OrderScheme(std::vector<OrderKey> keys,
                         std::shared_ptr<TableChunk> boundaries,
                         bool strict_boundaries)
  : keys{std::move(keys)}, boundaries{std::move(boundaries)}, strict_boundaries{strict_boundaries}
{
  RAPIDSMPF_EXPECTS(
    !this->keys.empty(), "OrderScheme: keys must not be empty", std::invalid_argument);
  RAPIDSMPF_EXPECTS(
    this->boundaries != nullptr, "OrderScheme: boundaries must not be null", std::invalid_argument);
  RAPIDSMPF_EXPECTS(this->boundaries->is_available(),
                    "OrderScheme: boundaries must be device-resident",
                    std::invalid_argument);
  RAPIDSMPF_EXPECTS(this->keys.size() == static_cast<std::size_t>(this->boundaries->shape().second),
                    "OrderScheme: number of keys must match number of boundary columns",
                    std::invalid_argument);
}

PartitioningSpec PartitioningSpec::from_order(OrderScheme o)
{
  return {.type = Type::ORDER, .hash = std::nullopt, .order = std::move(o)};
}

OrderScheme OrderScheme::with_keys(std::vector<OrderKey> new_keys) const
{
  return OrderScheme(std::move(new_keys), boundaries, strict_boundaries);
}

bool OrderScheme::boundaries_aligned_with(OrderScheme const& other,
                                          rapidsmpf::BufferResource& br) const
{
  if (strict_boundaries != other.strict_boundaries ||
      boundaries->shape() != other.boundaries->shape()) {
    return false;
  }
  if (!std::equal(
        keys.begin(), keys.end(), other.keys.begin(), [](OrderKey const& a, OrderKey const& b) {
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

rapidsmpf::streaming::Message to_message(std::uint64_t sequence_number,
                                         std::unique_ptr<ChannelMetadata> m)
{
  return rapidsmpf::streaming::Message{
    sequence_number,
    std::move(m),
    {},
    [](rapidsmpf::streaming::Message const& msg,
       rapidsmpf::MemoryReservation& /* reservation */) -> rapidsmpf::streaming::Message {
      auto copy = std::make_unique<ChannelMetadata>(msg.get<ChannelMetadata>());
      return rapidsmpf::streaming::Message{
        msg.sequence_number(), std::move(copy), {}, msg.copy_cb()};
    }};
}

}  // namespace cudf_streaming::streaming
