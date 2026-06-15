/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/cudf_gtest.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cudf_streaming/streaming/channel_metadata.hpp>
#include <cudf_streaming/streaming/table_chunk.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/memory/buffer_resource.hpp>

#include <memory>
#include <vector>

using namespace cudf_streaming::streaming;

class StreamingChannelMetadata : public ::testing::Test {};

TEST_F(StreamingChannelMetadata, HashScheme)
{
  HashScheme h{{0, 1}, 16};
  EXPECT_EQ(h.column_indices.size(), 2);
  EXPECT_EQ(h.column_indices[0], 0);
  EXPECT_EQ(h.column_indices[1], 1);
  EXPECT_EQ(h.modulus, 16);

  // Equality
  EXPECT_EQ(h, (HashScheme{{0, 1}, 16}));
  EXPECT_NE(h, (HashScheme{{0, 1}, 32}));
  EXPECT_NE(h, (HashScheme{{2}, 16}));
}

TEST_F(StreamingChannelMetadata, OrderSchemeCtorRejectsEmptyKeys)
{
  EXPECT_THROW(static_cast<void>(OrderScheme({}, nullptr)), std::invalid_argument);
}

TEST_F(StreamingChannelMetadata, OrderSchemeCtorRejectsNullBoundaries)
{
  EXPECT_THROW(static_cast<void>(
                 OrderScheme({{0, cudf::order::ASCENDING, cudf::null_order::BEFORE}}, nullptr)),
               std::invalid_argument);
}

TEST_F(StreamingChannelMetadata, OrderSchemeCtorRejectsEmptyOrderings)
{
  EXPECT_THROW(static_cast<void>(OrderScheme(std::vector<Ordering>{})), std::invalid_argument);
}

TEST_F(StreamingChannelMetadata, PartitioningSpec)
{
  // None
  auto spec_none = PartitioningSpec::none();
  EXPECT_EQ(spec_none.type, PartitioningSpec::Type::NONE);

  // Inherit
  auto spec_inherit = PartitioningSpec::inherit();
  EXPECT_EQ(spec_inherit.type, PartitioningSpec::Type::INHERIT);

  // Hash
  auto spec_hash = PartitioningSpec::from_hash(HashScheme{{0}, 16});
  EXPECT_EQ(spec_hash.type, PartitioningSpec::Type::HASH);
  EXPECT_EQ(spec_hash.hash->column_indices[0], 0);
  EXPECT_EQ(spec_hash.hash->modulus, 16);

  // Type checks (operator== removed; use field comparisons)
  EXPECT_EQ(spec_none.type, PartitioningSpec::Type::NONE);
  EXPECT_EQ(spec_inherit.type, PartitioningSpec::Type::INHERIT);
  EXPECT_EQ(spec_hash.type, PartitioningSpec::Type::HASH);
  EXPECT_NE(spec_none.type, spec_inherit.type);
  EXPECT_EQ(spec_hash.hash->modulus, 16);
  EXPECT_NE((PartitioningSpec::from_hash(HashScheme{{0}, 32}).hash->modulus), 16);
}

TEST_F(StreamingChannelMetadata, PartitioningScenarios)
{
  // Default construction
  Partitioning p_default{};
  EXPECT_EQ(p_default.inter_rank.type, PartitioningSpec::Type::NONE);
  EXPECT_EQ(p_default.local.type, PartitioningSpec::Type::NONE);

  // Direct global shuffle: inter_rank=Hash, local=Inherit
  Partitioning p_global{PartitioningSpec::from_hash(HashScheme{{0}, 16}),
                        PartitioningSpec::inherit()};
  EXPECT_EQ(p_global.inter_rank.type, PartitioningSpec::Type::HASH);
  EXPECT_EQ(p_global.local.type, PartitioningSpec::Type::INHERIT);
  EXPECT_EQ(p_global.inter_rank.hash->modulus, 16);

  // Two-stage shuffle: inter_rank=Hash(nranks), local=Hash(N_l)
  Partitioning p_twostage{PartitioningSpec::from_hash(HashScheme{{0}, 4}),
                          PartitioningSpec::from_hash(HashScheme{{0}, 8})};
  EXPECT_EQ(p_twostage.inter_rank.hash->modulus, 4);
  EXPECT_EQ(p_twostage.local.hash->modulus, 8);

  // Field comparisons (Partitioning::operator== removed)
  {
    Partitioning p_same{PartitioningSpec::from_hash(HashScheme{{0}, 16}),
                        PartitioningSpec::inherit()};
    EXPECT_EQ(p_global.inter_rank.type, p_same.inter_rank.type);
    EXPECT_EQ(p_global.inter_rank.hash->modulus, p_same.inter_rank.hash->modulus);
    EXPECT_EQ(p_global.local.type, p_same.local.type);
  }
  EXPECT_NE(p_global.inter_rank.hash->modulus, p_twostage.inter_rank.hash->modulus);
}

TEST_F(StreamingChannelMetadata, ChannelMetadata)
{
  // Full construction - use std::move to avoid GCC false positive on vector copy
  Partitioning p{PartitioningSpec::from_hash(HashScheme{{0}, 16}), PartitioningSpec::inherit()};
  ChannelMetadata m{4, std::move(p), true};
  EXPECT_EQ(m.local_count, 4);
  EXPECT_EQ(m.partitioning.inter_rank.type, PartitioningSpec::Type::HASH);
  EXPECT_EQ(m.partitioning.local.type, PartitioningSpec::Type::INHERIT);
  EXPECT_TRUE(m.duplicated);

  // Minimal construction
  ChannelMetadata m_minimal{4};
  EXPECT_EQ(m_minimal.local_count, 4);
  EXPECT_FALSE(m_minimal.duplicated);

  // Equality - create fresh partitionings and move them
  ChannelMetadata m_same{
    4,
    Partitioning{PartitioningSpec::from_hash(HashScheme{{0}, 16}), PartitioningSpec::inherit()},
    true};
  ChannelMetadata m_diff{
    8,
    Partitioning{PartitioningSpec::from_hash(HashScheme{{0}, 16}), PartitioningSpec::inherit()},
    true};
  // Field comparisons (ChannelMetadata::operator== removed)
  EXPECT_EQ(m.local_count, m_same.local_count);
  EXPECT_EQ(m.duplicated, m_same.duplicated);
  EXPECT_EQ(m.partitioning.inter_rank.hash->modulus, m_same.partitioning.inter_rank.hash->modulus);
  EXPECT_NE(m.local_count, m_diff.local_count);
}

TEST_F(StreamingChannelMetadata, MessageRoundTrip)
{
  // ChannelMetadata round-trip
  Partitioning part{PartitioningSpec::from_hash(HashScheme{{0}, 16}), PartitioningSpec::inherit()};
  auto m     = std::make_unique<ChannelMetadata>(4, std::move(part), false);
  auto msg_m = to_message(99, std::move(m));
  EXPECT_EQ(msg_m.sequence_number(), 99);
  EXPECT_TRUE(msg_m.holds<ChannelMetadata>());
  auto released = msg_m.release<ChannelMetadata>();
  EXPECT_EQ(released.local_count, 4);
  EXPECT_FALSE(released.duplicated);
  EXPECT_EQ(released.partitioning.inter_rank.hash->modulus, 16);
  EXPECT_TRUE(msg_m.empty());
}

class StreamingChannelMetadataGPU : public ::testing::Test {
 protected:
  rmm::cuda_stream_view stream{cudf::get_default_stream()};
  std::shared_ptr<rapidsmpf::BufferResource> br =
    rapidsmpf::BufferResource::create(cudf::get_current_device_resource_ref());

  std::shared_ptr<TableChunk> make_chunk(std::vector<int32_t> vals)
  {
    rmm::device_buffer buf(vals.data(), vals.size() * sizeof(int32_t), stream);
    auto col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                              static_cast<cudf::size_type>(vals.size()),
                                              std::move(buf),
                                              rmm::device_buffer{},
                                              0);
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(col));
    return std::make_shared<TableChunk>(std::make_unique<cudf::table>(std::move(cols)), stream);
  }
};

TEST_F(StreamingChannelMetadataGPU, OrderingReplaceKeys)
{
  OrderKey k0{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  OrderKey k5{5, cudf::order::DESCENDING, cudf::null_order::AFTER};

  auto b = make_chunk({100, 200});
  Ordering o1({k0}, b);
  auto o2 = o1.with_keys({k5});

  EXPECT_EQ(o2.keys[0].column_index, 5);
  EXPECT_EQ(o2.keys[0].order, cudf::order::DESCENDING);
  EXPECT_EQ(o2.strict_boundaries, o1.strict_boundaries);
  EXPECT_EQ(o2.boundaries->shape(), o1.boundaries->shape());
  EXPECT_EQ(o2.boundaries.get(), b.get());
  EXPECT_NE(o1.keys[0].column_index, o2.keys[0].column_index);

  EXPECT_THROW(static_cast<void>(o1.with_keys({k0, k5})), std::invalid_argument);
}

TEST_F(StreamingChannelMetadataGPU, OrderSchemeMultipleOrderings)
{
  OrderKey k0{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  OrderKey k2{2, cudf::order::DESCENDING, cudf::null_order::AFTER};

  auto b0 = make_chunk({100, 200});
  auto b1 = make_chunk({300, 400});
  OrderScheme o({Ordering{{k0}, b0, true}, Ordering{{k2}, b1, false}});

  ASSERT_EQ(o.orderings.size(), 2);
  EXPECT_EQ(o.orderings[0].keys[0], k0);
  EXPECT_EQ(o.orderings[0].boundaries.get(), b0.get());
  EXPECT_TRUE(o.orderings[0].strict_boundaries);
  EXPECT_EQ(o.orderings[1].keys[0], k2);
  EXPECT_EQ(o.orderings[1].boundaries.get(), b1.get());
  EXPECT_FALSE(o.orderings[1].strict_boundaries);
}

TEST_F(StreamingChannelMetadataGPU, OrderingBoundariesAlignedWith)
{
  OrderKey k0{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  OrderKey k3{3, cudf::order::ASCENDING, cudf::null_order::BEFORE};

  Ordering o1({k0}, make_chunk({100, 200}));
  Ordering o2({k0}, make_chunk({100, 200}));
  EXPECT_TRUE(o1.boundaries_aligned_with(o2, *br));

  Ordering o_shifted({k3}, make_chunk({100, 200}));
  EXPECT_TRUE(o1.boundaries_aligned_with(o_shifted, *br));

  Ordering o_strict({k0}, make_chunk({100, 200}), /*strict=*/true);
  EXPECT_FALSE(o1.boundaries_aligned_with(o_strict, *br));

  Ordering o_diff({k0}, make_chunk({100, 300}));
  EXPECT_FALSE(o1.boundaries_aligned_with(o_diff, *br));
}

TEST_F(StreamingChannelMetadataGPU, PartitioningSpecOrder)
{
  OrderKey k0{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  OrderScheme o({k0}, make_chunk({100, 200}));

  auto spec = PartitioningSpec::from_order(o);
  EXPECT_EQ(spec.type, PartitioningSpec::Type::ORDER);
  EXPECT_TRUE(spec.order.has_value());
  EXPECT_EQ(spec.order->orderings[0].keys[0].column_index, 0);

  // Type checks only (PartitioningSpec::operator== removed).
  EXPECT_EQ(spec.type, PartitioningSpec::Type::ORDER);
  EXPECT_NE(spec.type, PartitioningSpec::from_hash(HashScheme{{0}, 16}).type);
  EXPECT_NE(spec.type, PartitioningSpec::none().type);
}
