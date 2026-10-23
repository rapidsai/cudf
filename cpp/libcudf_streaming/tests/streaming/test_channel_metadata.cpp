/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/cudf_gtest.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cudf_streaming/channel_metadata.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <rapidsmpf/memory/buffer_resource.hpp>

#include <memory>
#include <vector>

using namespace cudf_streaming;

class StreamingChannelMetadata : public ::testing::Test {};

TEST_F(StreamingChannelMetadata, HashScheme)
{
  hash_scheme h{{0, 1}, 16};
  EXPECT_EQ(h.column_indices.size(), 2);
  EXPECT_EQ(h.column_indices[0], 0);
  EXPECT_EQ(h.column_indices[1], 1);
  EXPECT_EQ(h.modulus, 16);

  // Equality
  EXPECT_EQ(h, (hash_scheme{{0, 1}, 16}));
  EXPECT_NE(h, (hash_scheme{{0, 1}, 32}));
  EXPECT_NE(h, (hash_scheme{{2}, 16}));
}

TEST_F(StreamingChannelMetadata, RejectsMalformedPartitioningSpec)
{
  auto hash_without_scheme = partitioning_spec{};
  hash_without_scheme.type = partitioning_spec::type::HASH;
  EXPECT_THROW(static_cast<void>(
                 channel_metadata{1, partitioning{hash_without_scheme, partitioning_spec::none()}}),
               std::invalid_argument);

  auto hash_with_range = partitioning_spec::from_hash(hash_scheme{{0}, 2});
  hash_with_range.range.emplace();
  EXPECT_THROW(static_cast<void>(
                 channel_metadata{1, partitioning{hash_with_range, partitioning_spec::none()}}),
               std::invalid_argument);
}

TEST_F(StreamingChannelMetadata, PartitionOrderingRejectsEmptyKeys)
{
  EXPECT_THROW(static_cast<void>(partition_ordering({}, nullptr)), std::invalid_argument);
}

TEST_F(StreamingChannelMetadata, PartitionOrderingRejectsNullBoundaries)
{
  EXPECT_THROW(static_cast<void>(partition_ordering(
                 {{0, cudf::order::ASCENDING, cudf::null_order::BEFORE}}, nullptr)),
               std::invalid_argument);
}

TEST_F(StreamingChannelMetadata, RangeSchemeCtorRejectsEmptyPartitionOrderings)
{
  EXPECT_THROW(static_cast<void>(range_scheme(std::vector<partition_ordering>{})),
               std::invalid_argument);
}

TEST_F(StreamingChannelMetadata, RowOrdering)
{
  order_key time{2, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  row_ordering global{{time}, row_ordering_scope::GLOBAL};
  row_ordering grouped{{time}, row_ordering_scope::RANK_LOCAL, {0, 1}};
  row_ordering canonical{{time}, row_ordering_scope::RANK_LOCAL, {1, 0}};

  EXPECT_EQ(global.order_keys, std::vector<order_key>{time});
  EXPECT_EQ(global.scope, row_ordering_scope::GLOBAL);
  EXPECT_TRUE(global.group_keys.empty());
  EXPECT_EQ(grouped.scope, row_ordering_scope::RANK_LOCAL);
  EXPECT_EQ(grouped.group_keys, (std::vector<cudf::size_type>{0, 1}));
  EXPECT_EQ(grouped, (row_ordering{{time}, row_ordering_scope::RANK_LOCAL, {0, 1}}));
  EXPECT_EQ(grouped, canonical);
  EXPECT_NE(grouped, global);
}

TEST_F(StreamingChannelMetadata, RowOrderingValidation)
{
  order_key k0{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  order_key k1{1, cudf::order::DESCENDING, cudf::null_order::AFTER};
  order_key negative{-1, cudf::order::ASCENDING, cudf::null_order::BEFORE};

  EXPECT_THROW(static_cast<void>(row_ordering({}, row_ordering_scope::GLOBAL)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(row_ordering({negative}, row_ordering_scope::GLOBAL)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(row_ordering({k0, k0}, row_ordering_scope::GLOBAL)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(row_ordering({k1}, row_ordering_scope::GLOBAL, {-1})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(row_ordering({k1}, row_ordering_scope::GLOBAL, {0, 0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(row_ordering({k0}, row_ordering_scope::GLOBAL, {0})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(row_ordering({k0}, static_cast<row_ordering_scope>(255))),
               std::invalid_argument);

  row_ordering mutated{{k0}, row_ordering_scope::GLOBAL};
  mutated.order_keys.clear();
  EXPECT_THROW(static_cast<void>(channel_metadata(0, {}, false, {mutated})), std::invalid_argument);
}

TEST_F(StreamingChannelMetadata, PartitioningSpec)
{
  // None
  auto spec_none = partitioning_spec::none();
  EXPECT_EQ(spec_none.type, partitioning_spec::type::NONE);

  // Inherit
  auto spec_inherit = partitioning_spec::inherit();
  EXPECT_EQ(spec_inherit.type, partitioning_spec::type::INHERIT);

  // Hash
  auto spec_hash = partitioning_spec::from_hash(hash_scheme{{0}, 16});
  EXPECT_EQ(spec_hash.type, partitioning_spec::type::HASH);
  EXPECT_EQ(spec_hash.hash->column_indices[0], 0);
  EXPECT_EQ(spec_hash.hash->modulus, 16);

  // Type checks (operator== removed; use field comparisons)
  EXPECT_EQ(spec_none.type, partitioning_spec::type::NONE);
  EXPECT_EQ(spec_inherit.type, partitioning_spec::type::INHERIT);
  EXPECT_EQ(spec_hash.type, partitioning_spec::type::HASH);
  EXPECT_NE(spec_none.type, spec_inherit.type);
  EXPECT_EQ(spec_hash.hash->modulus, 16);
  EXPECT_NE((partitioning_spec::from_hash(hash_scheme{{0}, 32}).hash->modulus), 16);
}

TEST_F(StreamingChannelMetadata, PartitioningScenarios)
{
  // Default construction
  partitioning p_default{};
  EXPECT_EQ(p_default.inter_rank.type, partitioning_spec::type::NONE);
  EXPECT_EQ(p_default.local.type, partitioning_spec::type::NONE);

  // Direct global shuffle: inter_rank=Hash, local=Inherit
  partitioning p_global{partitioning_spec::from_hash(hash_scheme{{0}, 16}),
                        partitioning_spec::inherit()};
  EXPECT_EQ(p_global.inter_rank.type, partitioning_spec::type::HASH);
  EXPECT_EQ(p_global.local.type, partitioning_spec::type::INHERIT);
  EXPECT_EQ(p_global.inter_rank.hash->modulus, 16);

  // Two-stage shuffle: inter_rank=Hash(nranks), local=Hash(N_l)
  partitioning p_twostage{partitioning_spec::from_hash(hash_scheme{{0}, 4}),
                          partitioning_spec::from_hash(hash_scheme{{0}, 8})};
  EXPECT_EQ(p_twostage.inter_rank.hash->modulus, 4);
  EXPECT_EQ(p_twostage.local.hash->modulus, 8);

  // Field comparisons (partitioning::operator== removed)
  {
    partitioning p_same{partitioning_spec::from_hash(hash_scheme{{0}, 16}),
                        partitioning_spec::inherit()};
    EXPECT_EQ(p_global.inter_rank.type, p_same.inter_rank.type);
    EXPECT_EQ(p_global.inter_rank.hash->modulus, p_same.inter_rank.hash->modulus);
    EXPECT_EQ(p_global.local.type, p_same.local.type);
  }
  EXPECT_NE(p_global.inter_rank.hash->modulus, p_twostage.inter_rank.hash->modulus);
}

TEST_F(StreamingChannelMetadata, ChannelMetadata)
{
  // Full construction - use std::move to avoid GCC false positive on vector copy
  partitioning p{partitioning_spec::from_hash(hash_scheme{{0}, 16}), partitioning_spec::inherit()};
  channel_metadata m{4, std::move(p), true};
  EXPECT_EQ(m.local_count, 4);
  EXPECT_EQ(m.partitioning.inter_rank.type, partitioning_spec::type::HASH);
  EXPECT_EQ(m.partitioning.local.type, partitioning_spec::type::INHERIT);
  EXPECT_TRUE(m.duplicated);
  EXPECT_TRUE(m.row_orderings.empty());

  // Minimal construction
  channel_metadata m_minimal{4};
  EXPECT_EQ(m_minimal.local_count, 4);
  EXPECT_FALSE(m_minimal.duplicated);
  EXPECT_TRUE(m_minimal.row_orderings.empty());

  order_key key{1, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  channel_metadata m_ordered{
    4, {}, false, {row_ordering{{key}, row_ordering_scope::RANK_LOCAL, {0}}}};
  ASSERT_EQ(m_ordered.row_orderings.size(), 1);
  EXPECT_EQ(m_ordered.row_orderings.front().order_keys.front(), key);
  EXPECT_EQ(m_ordered.row_orderings.front().scope, row_ordering_scope::RANK_LOCAL);
  EXPECT_EQ(m_ordered.row_orderings.front().group_keys, (std::vector<cudf::size_type>{0}));

  // Equality - create fresh partitionings and move them
  channel_metadata m_same{
    4,
    partitioning{partitioning_spec::from_hash(hash_scheme{{0}, 16}), partitioning_spec::inherit()},
    true};
  channel_metadata m_diff{
    8,
    partitioning{partitioning_spec::from_hash(hash_scheme{{0}, 16}), partitioning_spec::inherit()},
    true};
  // Field comparisons (channel_metadata::operator== removed)
  EXPECT_EQ(m.local_count, m_same.local_count);
  EXPECT_EQ(m.duplicated, m_same.duplicated);
  EXPECT_EQ(m.partitioning.inter_rank.hash->modulus, m_same.partitioning.inter_rank.hash->modulus);
  EXPECT_NE(m.local_count, m_diff.local_count);
}

TEST_F(StreamingChannelMetadata, MessageRoundTrip)
{
  // channel_metadata round-trip
  partitioning part{partitioning_spec::from_hash(hash_scheme{{0}, 16}),
                    partitioning_spec::inherit()};
  order_key key{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  auto m = std::make_unique<channel_metadata>(
    4, std::move(part), false, std::vector{row_ordering{{key}, row_ordering_scope::GLOBAL}});
  auto msg_m = to_message(99, std::move(m));
  EXPECT_EQ(msg_m.sequence_number(), 99);
  EXPECT_TRUE(msg_m.holds<channel_metadata>());
  auto released = msg_m.release<channel_metadata>();
  EXPECT_EQ(released.local_count, 4);
  EXPECT_FALSE(released.duplicated);
  EXPECT_EQ(released.partitioning.inter_rank.hash->modulus, 16);
  ASSERT_EQ(released.row_orderings.size(), 1);
  EXPECT_EQ(released.row_orderings.front().order_keys.front(), key);
  EXPECT_EQ(released.row_orderings.front().scope, row_ordering_scope::GLOBAL);
  EXPECT_TRUE(msg_m.empty());
}

TEST_F(StreamingChannelMetadata, MessageRejectsMalformedRowOrdering)
{
  order_key key{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  auto metadata = std::make_unique<channel_metadata>(
    1, partitioning{}, false, std::vector{row_ordering{{key}, row_ordering_scope::GLOBAL}});
  metadata->row_orderings.front().order_keys.clear();

  EXPECT_THROW(static_cast<void>(to_message(0, std::move(metadata))), std::invalid_argument);
}

TEST_F(StreamingChannelMetadata, PartitioningSpecRejectsEmptyRangeScheme)
{
  EXPECT_THROW(static_cast<void>(partitioning_spec::from_range(range_scheme{})),
               std::invalid_argument);
}

class StreamingChannelMetadataGPU : public ::testing::Test {
 protected:
  cuda::stream_ref stream{cudf::get_default_stream()};
  std::shared_ptr<rapidsmpf::BufferResource> br =
    rapidsmpf::BufferResource::create(cudf::get_current_device_resource_ref());

  std::shared_ptr<table_chunk> make_chunk(std::vector<int32_t> vals)
  {
    rmm::device_buffer buf(vals.data(), vals.size() * sizeof(int32_t), stream);
    stream.sync();
    auto col =
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                     static_cast<cudf::size_type>(vals.size()),
                                     std::move(buf),
                                     cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                     0);
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(col));
    return std::make_shared<table_chunk>(std::make_unique<cudf::table>(std::move(cols)), stream);
  }
};

TEST_F(StreamingChannelMetadataGPU, PartitionOrderingReplaceKeys)
{
  order_key k0{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  order_key k5{5, cudf::order::ASCENDING, cudf::null_order::BEFORE};

  auto b                = make_chunk({100, 200});
  partition_ordering o1 = partition_ordering({k0}, b);
  auto o2               = o1.with_keys({k5});

  EXPECT_EQ(o2.keys[0].column_index, 5);
  EXPECT_EQ(o2.keys[0].order, cudf::order::ASCENDING);
  EXPECT_EQ(o2.strict_boundaries, o1.strict_boundaries);
  EXPECT_EQ(o2.boundaries->shape(), o1.boundaries->shape());
  EXPECT_EQ(o2.boundaries.get(), b.get());
  EXPECT_NE(o1.keys[0].column_index, o2.keys[0].column_index);

  EXPECT_THROW(static_cast<void>(o1.with_keys({k0, k5})), std::invalid_argument);
  EXPECT_THROW(
    static_cast<void>(o1.with_keys({{5, cudf::order::DESCENDING, cudf::null_order::BEFORE}})),
    std::invalid_argument);
}

TEST_F(StreamingChannelMetadataGPU, RangeSchemeMultipleOrderings)
{
  order_key k0{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  order_key k2{2, cudf::order::DESCENDING, cudf::null_order::AFTER};

  auto b0 = make_chunk({100, 200});
  auto b1 = make_chunk({300, 400});
  range_scheme o({partition_ordering{{k0}, b0, true}, partition_ordering{{k2}, b1, false}});

  ASSERT_EQ(o.partition_orderings.size(), 2);
  EXPECT_EQ(o.partition_orderings[0].keys[0], k0);
  EXPECT_EQ(o.partition_orderings[0].boundaries.get(), b0.get());
  EXPECT_TRUE(o.partition_orderings[0].strict_boundaries);
  EXPECT_EQ(o.partition_orderings[1].keys[0], k2);
  EXPECT_EQ(o.partition_orderings[1].boundaries.get(), b1.get());
  EXPECT_FALSE(o.partition_orderings[1].strict_boundaries);
}

TEST_F(StreamingChannelMetadataGPU, RangeSchemeRejectsMalformedOrdering)
{
  order_key key{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  partition_ordering ordering{{key}, make_chunk({100, 200})};
  ordering.boundaries.reset();

  EXPECT_THROW(static_cast<void>(range_scheme({ordering})), std::invalid_argument);
}

TEST_F(StreamingChannelMetadataGPU, PartitionOrderingBoundariesAlignedWith)
{
  order_key k0{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  order_key k3{3, cudf::order::ASCENDING, cudf::null_order::BEFORE};

  partition_ordering o1({k0}, make_chunk({100, 200}));
  partition_ordering o2({k0}, make_chunk({100, 200}));
  EXPECT_TRUE(o1.boundaries_aligned_with(o2, *br));

  partition_ordering o_shifted({k3}, make_chunk({100, 200}));
  EXPECT_TRUE(o1.boundaries_aligned_with(o_shifted, *br));

  partition_ordering o_strict({k0}, make_chunk({100, 200}), /*strict_boundaries=*/true);
  EXPECT_FALSE(o1.boundaries_aligned_with(o_strict, *br));

  partition_ordering o_diff({k0}, make_chunk({100, 300}));
  EXPECT_FALSE(o1.boundaries_aligned_with(o_diff, *br));
}

TEST_F(StreamingChannelMetadataGPU, PartitioningSpecRange)
{
  order_key k0{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  range_scheme r({partition_ordering{{k0}, make_chunk({100, 200})}});

  auto spec = partitioning_spec::from_range(r);
  EXPECT_EQ(spec.type, partitioning_spec::type::RANGE);
  EXPECT_TRUE(spec.range.has_value());
  EXPECT_EQ(spec.range->partition_orderings[0].keys[0].column_index, 0);

  // Type checks only. Range values use boundaries_aligned_with for comparison.
  EXPECT_EQ(spec.type, partitioning_spec::type::RANGE);
  EXPECT_NE(spec.type, partitioning_spec::from_hash(hash_scheme{{0}, 16}).type);
  EXPECT_NE(spec.type, partitioning_spec::none().type);
}

TEST_F(StreamingChannelMetadataGPU, MessageCopyPreservesMetadata)
{
  order_key key{0, cudf::order::ASCENDING, cudf::null_order::BEFORE};
  auto metadata = std::make_unique<channel_metadata>(
    2,
    partitioning{partitioning_spec::from_range(
                   range_scheme{{partition_ordering{{key}, make_chunk({100, 200})}}}),
                 partitioning_spec::inherit()},
    false,
    std::vector{row_ordering{{key}, row_ordering_scope::GLOBAL}});
  auto message     = to_message(5, std::move(metadata));
  auto reservation = br->reserve_or_fail(message.copy_cost(), rapidsmpf::MemoryType::DEVICE);
  auto copy        = message.copy(reservation);
  EXPECT_TRUE(message.holds<channel_metadata>());

  auto copied = copy.release<channel_metadata>();
  EXPECT_EQ(copied.local_count, 2);
  EXPECT_EQ(copied.partitioning.inter_rank.type, partitioning_spec::type::RANGE);
  ASSERT_EQ(copied.row_orderings.size(), 1);
  EXPECT_EQ(copied.row_orderings.front().scope, row_ordering_scope::GLOBAL);
  EXPECT_EQ(copied.row_orderings.front().order_keys.front(), key);
}
