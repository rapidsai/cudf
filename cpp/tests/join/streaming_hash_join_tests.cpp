/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/streaming_hash_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

using cudf::size_type;
template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

std::pair<std::vector<size_type>, std::vector<size_type>> to_sorted_host_pairs(
  rmm::device_uvector<size_type> const& left_indices,
  rmm::device_uvector<size_type> const& right_indices,
  rmm::cuda_stream_view stream)
{
  auto const h_left  = cudf::detail::make_host_vector(left_indices, stream);
  auto const h_right = cudf::detail::make_host_vector(right_indices, stream);
  std::vector<std::pair<size_type, size_type>> paired;
  paired.reserve(h_left.size());
  for (std::size_t i = 0; i < h_left.size(); ++i) {
    paired.emplace_back(h_left[i], h_right[i]);
  }
  std::sort(paired.begin(), paired.end());

  std::vector<size_type> sorted_left;
  std::vector<size_type> sorted_right;
  sorted_left.reserve(paired.size());
  sorted_right.reserve(paired.size());
  for (auto const& [l, r] : paired) {
    sorted_left.push_back(l);
    sorted_right.push_back(r);
  }
  return {std::move(sorted_left), std::move(sorted_right)};
}

void expect_all_batch_zero(rmm::device_uvector<size_type> const& batch_indices,
                           rmm::cuda_stream_view stream)
{
  auto const h_batch = cudf::detail::make_host_vector(batch_indices, stream);
  EXPECT_TRUE(std::all_of(h_batch.begin(), h_batch.end(), [](size_type b) { return b == 0; }));
}

using join_match = std::tuple<size_type, size_type, size_type>;

std::vector<join_match> to_sorted_host_matches(rmm::device_uvector<size_type> const& left_indices,
                                               rmm::device_uvector<size_type> const& batch_indices,
                                               rmm::device_uvector<size_type> const& row_indices,
                                               rmm::cuda_stream_view stream)
{
  auto const h_left  = cudf::detail::make_host_vector(left_indices, stream);
  auto const h_batch = cudf::detail::make_host_vector(batch_indices, stream);
  auto const h_row   = cudf::detail::make_host_vector(row_indices, stream);

  std::vector<join_match> matches;
  matches.reserve(h_left.size());
  for (std::size_t i = 0; i < h_left.size(); ++i) {
    matches.emplace_back(h_left[i], h_batch[i], h_row[i]);
  }
  std::sort(matches.begin(), matches.end());
  return matches;
}

std::vector<join_match> hash_join_reference_matches(std::span<cudf::table_view const> right_batches,
                                                    cudf::table_view const& left,
                                                    cudf::nullable_join has_nulls,
                                                    cudf::null_equality compare_nulls,
                                                    rmm::cuda_stream_view stream)
{
  auto concatenated_right = cudf::concatenate(right_batches, stream);
  cudf::hash_join reference_joiner{
    concatenated_right->view(), has_nulls, compare_nulls, 0.5, stream};
  auto [left_indices, global_right_indices] = reference_joiner.inner_join(left, {}, stream);
  auto const h_left                         = cudf::detail::make_host_vector(*left_indices, stream);
  auto const h_global_right = cudf::detail::make_host_vector(*global_right_indices, stream);

  std::vector<size_type> batch_offsets;
  batch_offsets.reserve(right_batches.size() + 1);
  batch_offsets.push_back(0);
  for (auto const& batch : right_batches) {
    batch_offsets.push_back(batch_offsets.back() + batch.num_rows());
  }

  std::vector<join_match> matches;
  matches.reserve(h_left.size());
  for (std::size_t i = 0; i < h_left.size(); ++i) {
    auto const batch_iter =
      std::upper_bound(batch_offsets.begin(), batch_offsets.end(), h_global_right[i]);
    auto const batch_id = static_cast<size_type>(batch_iter - batch_offsets.begin() - 1);
    matches.emplace_back(h_left[i], batch_id, h_global_right[i] - batch_offsets[batch_id]);
  }
  std::sort(matches.begin(), matches.end());
  return matches;
}

}  // namespace

struct StreamingHashJoinTest : public cudf::test::BaseFixture {};

TEST_F(StreamingHashJoinTest, InnerJoinSinglePartitionMatchesHashJoin)
{
  auto const stream = cudf::test::get_default_stream();

  column_wrapper<int32_t> right_keys{2, 4, 5, 7, 9, 1, 3, 5};
  cudf::table_view right_view{{right_keys}};

  column_wrapper<int32_t> left_keys{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  cudf::table_view left_view{{left_keys}};

  std::vector<cudf::data_type> const right_schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const right_key_indices{0};

  cudf::streaming_hash_join streaming_joiner{right_schema,
                                             right_key_indices,
                                             right_view.num_rows(),
                                             /*max_num_batches=*/1,
                                             cudf::nullable_join::NO,
                                             cudf::null_equality::EQUAL};
  streaming_joiner.insert(right_view, stream);
  auto [streaming_left, streaming_right] = streaming_joiner.inner_join(left_view, {}, stream);
  auto& [streaming_batch, streaming_row] = streaming_right;

  cudf::hash_join reference_joiner{
    right_view, cudf::nullable_join::NO, cudf::null_equality::EQUAL, 0.5, stream};
  auto [reference_left, reference_right] = reference_joiner.inner_join(left_view, {}, stream);

  expect_all_batch_zero(*streaming_batch, stream);

  auto const [streaming_l, streaming_r] =
    to_sorted_host_pairs(*streaming_left, *streaming_row, stream);
  auto const [reference_l, reference_r] =
    to_sorted_host_pairs(*reference_left, *reference_right, stream);
  EXPECT_EQ(streaming_l, reference_l);
  EXPECT_EQ(streaming_r, reference_r);
}

TEST_F(StreamingHashJoinTest, EmptyRightPartition)
{
  auto const stream = cudf::test::get_default_stream();

  column_wrapper<int32_t> empty_right{};
  cudf::table_view empty_right_view{{empty_right}};

  column_wrapper<int32_t> left_keys{1, 2, 3};
  cudf::table_view left_view{{left_keys}};

  std::vector<cudf::data_type> const right_schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const right_key_indices{0};

  cudf::streaming_hash_join streaming_joiner{right_schema,
                                             right_key_indices,
                                             /*total_right_rows=*/0,
                                             /*max_num_batches=*/1,
                                             cudf::nullable_join::NO,
                                             cudf::null_equality::EQUAL};
  streaming_joiner.insert(empty_right_view, stream);
  auto [l, right_pair]    = streaming_joiner.inner_join(left_view, {}, stream);
  auto& [batch_ids, rows] = right_pair;
  EXPECT_EQ(l->size(), 0u);
  EXPECT_EQ(batch_ids->size(), 0u);
  EXPECT_EQ(rows->size(), 0u);
}

TEST_F(StreamingHashJoinTest, MultiColumnKey)
{
  auto const stream = cudf::test::get_default_stream();

  column_wrapper<int32_t> right_k0{1, 2, 3, 1};
  column_wrapper<int32_t> right_k1{10, 20, 30, 11};
  cudf::table_view right_view{{right_k0, right_k1}};

  column_wrapper<int32_t> left_k0{1, 1, 2, 3, 4};
  column_wrapper<int32_t> left_k1{10, 11, 20, 30, 40};
  cudf::table_view left_view{{left_k0, left_k1}};

  std::vector<cudf::data_type> const right_schema(2, cudf::data_type{cudf::type_id::INT32});
  std::vector<size_type> const right_key_indices{0, 1};

  cudf::streaming_hash_join streaming_joiner{right_schema,
                                             right_key_indices,
                                             right_view.num_rows(),
                                             /*max_num_batches=*/1,
                                             cudf::nullable_join::NO,
                                             cudf::null_equality::EQUAL};
  streaming_joiner.insert(right_view, stream);
  auto [streaming_left, streaming_right] = streaming_joiner.inner_join(left_view, {}, stream);
  auto& [streaming_batch, streaming_row] = streaming_right;

  cudf::hash_join reference_joiner{
    right_view, cudf::nullable_join::NO, cudf::null_equality::EQUAL, 0.5, stream};
  auto [reference_left, reference_right] = reference_joiner.inner_join(left_view, {}, stream);

  expect_all_batch_zero(*streaming_batch, stream);

  auto const [streaming_l, streaming_r] =
    to_sorted_host_pairs(*streaming_left, *streaming_row, stream);
  auto const [reference_l, reference_r] =
    to_sorted_host_pairs(*reference_left, *reference_right, stream);
  EXPECT_EQ(streaming_l, reference_l);
  EXPECT_EQ(streaming_r, reference_r);
}

TEST_F(StreamingHashJoinTest, MultiplePartitionsReturnBatchLocalIndices)
{
  auto const stream = cudf::test::get_default_stream();
  column_wrapper<int32_t> right0{1, 2, 5};
  column_wrapper<int32_t> right1{2, 3, 5};
  column_wrapper<int32_t> right2{2, 5, 9};
  cudf::table_view right0_view{{right0}};
  cudf::table_view right1_view{{right1}};
  cudf::table_view right2_view{{right2}};
  std::array<cudf::table_view, 3> const right_batches{right0_view, right1_view, right2_view};
  column_wrapper<int32_t> left{2, 5, 3, 7};
  cudf::table_view left_view{{left}};

  std::vector<cudf::data_type> const schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const keys{0};
  cudf::streaming_hash_join joiner{
    schema,
    keys,
    right0_view.num_rows() + right1_view.num_rows() + right2_view.num_rows(),
    // A non-power-of-two maximum requires two batch-ID bits.
    /*max_num_batches=*/3,
    cudf::nullable_join::NO,
    cudf::null_equality::EQUAL};
  joiner.insert(right0_view, stream);
  joiner.insert(right1_view, stream);
  joiner.insert(right2_view, stream);

  auto [left_indices, right]         = joiner.inner_join(left_view, {}, stream);
  auto& [batch_indices, row_indices] = right;
  auto const actual = to_sorted_host_matches(*left_indices, *batch_indices, *row_indices, stream);
  std::vector<join_match> const expected{
    {0, 0, 1}, {0, 1, 0}, {0, 2, 0}, {1, 0, 2}, {1, 1, 2}, {1, 2, 1}, {2, 1, 1}};
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(
    actual,
    hash_join_reference_matches(
      right_batches, left_view, cudf::nullable_join::NO, cudf::null_equality::EQUAL, stream));
}

TEST_F(StreamingHashJoinTest, MaxNumBatchesCountsEmptyBatches)
{
  auto const stream = cudf::test::get_default_stream();

  std::vector<cudf::data_type> const schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const keys{0};
  cudf::streaming_hash_join joiner{schema,
                                   keys,
                                   /*total_right_rows=*/1,
                                   /*max_num_batches=*/1,
                                   cudf::nullable_join::NO,
                                   cudf::null_equality::EQUAL};

  column_wrapper<int32_t> empty{};
  joiner.insert(cudf::table_view{{empty}}, stream);

  column_wrapper<int32_t> nonempty{1};
  EXPECT_THROW(joiner.insert(cudf::table_view{{nonempty}}, stream), std::invalid_argument);
}

TEST_F(StreamingHashJoinTest, InnerJoinBeforeInsertThrows)
{
  auto const stream = cudf::test::get_default_stream();

  std::vector<cudf::data_type> const right_schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const right_key_indices{0};

  cudf::streaming_hash_join streaming_joiner{right_schema,
                                             right_key_indices,
                                             /*total_right_rows=*/4,
                                             /*max_num_batches=*/1,
                                             cudf::nullable_join::NO,
                                             cudf::null_equality::EQUAL};
  column_wrapper<int32_t> left_keys{1, 2, 3};
  cudf::table_view left_view{{left_keys}};
  EXPECT_THROW(static_cast<void>(streaming_joiner.inner_join(left_view, {}, stream)),
               std::logic_error);
}

TEST_F(StreamingHashJoinTest, SchemaMismatchThrows)
{
  auto const stream = cudf::test::get_default_stream();

  std::vector<cudf::data_type> const right_schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const right_key_indices{0};
  cudf::streaming_hash_join streaming_joiner{right_schema,
                                             right_key_indices,
                                             /*total_right_rows=*/4,
                                             /*max_num_batches=*/2,
                                             cudf::nullable_join::NO,
                                             cudf::null_equality::EQUAL};

  column_wrapper<int64_t> wrong_type_keys{1L, 2L, 3L};
  cudf::table_view wrong_type_view{{wrong_type_keys}};
  EXPECT_THROW(streaming_joiner.insert(wrong_type_view, stream), std::invalid_argument);

  column_wrapper<int32_t> col_a{1, 2, 3};
  column_wrapper<int32_t> col_b{4, 5, 6};
  cudf::table_view wrong_count_view{{col_a, col_b}};
  EXPECT_THROW(streaming_joiner.insert(wrong_count_view, stream), std::invalid_argument);

  streaming_joiner.insert(cudf::table_view{{col_a}}, stream);
  EXPECT_THROW(streaming_joiner.insert(cudf::table_view{{col_a}}, stream), std::invalid_argument);
}

TEST_F(StreamingHashJoinTest, UsesSelectedKeysAndPreservesEmptyBatchIds)
{
  auto const stream = cudf::test::get_default_stream();
  column_wrapper<int32_t> empty_payload{};
  column_wrapper<int64_t> empty_keys{};
  cudf::table_view empty{{empty_payload, empty_keys}};
  column_wrapper<int32_t> payload{100, 200, 300};
  column_wrapper<int64_t> keys{7, 8, 7};
  cudf::table_view right{{payload, keys}};
  column_wrapper<int64_t> left_keys{7, 9};
  cudf::table_view left{{left_keys}};
  cudf::table_view empty_key_view{{empty_keys}};
  cudf::table_view right_key_view{{keys}};
  std::array<cudf::table_view, 2> const right_key_batches{empty_key_view, right_key_view};

  std::vector<cudf::data_type> const schema{cudf::data_type{cudf::type_id::INT32},
                                            cudf::data_type{cudf::type_id::INT64}};
  std::vector<size_type> const key_indices{1};
  cudf::streaming_hash_join joiner{schema,
                                   key_indices,
                                   right.num_rows(),
                                   /*max_num_batches=*/2,
                                   cudf::nullable_join::NO,
                                   cudf::null_equality::EQUAL};
  joiner.insert(empty, stream);
  joiner.insert(right, stream);

  auto [left_indices, right_indices] = joiner.inner_join(left, /*output_size=*/2, stream);
  auto& [batch_indices, row_indices] = right_indices;
  auto const actual = to_sorted_host_matches(*left_indices, *batch_indices, *row_indices, stream);
  std::vector<join_match> const expected{{0, 1, 0}, {0, 1, 2}};
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(
    actual,
    hash_join_reference_matches(
      right_key_batches, left, cudf::nullable_join::NO, cudf::null_equality::EQUAL, stream));
}

TEST_F(StreamingHashJoinTest, NullEqualityAcrossPartitions)
{
  auto const stream = cudf::test::get_default_stream();
  column_wrapper<int32_t> right0{{1, 0}, {1, 0}};
  column_wrapper<int32_t> right1{{0, 2}, {0, 1}};
  cudf::table_view right0_view{{right0}};
  cudf::table_view right1_view{{right1}};
  std::array<cudf::table_view, 2> const right_batches{right0_view, right1_view};
  column_wrapper<int32_t> left_keys{{0, 1, 2}, {0, 1, 1}};
  cudf::table_view left{{left_keys}};
  std::vector<cudf::data_type> const schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const key_indices{0};

  auto run = [&](cudf::null_equality nulls_equal) {
    cudf::streaming_hash_join joiner{schema,
                                     key_indices,
                                     right0_view.num_rows() + right1_view.num_rows(),
                                     /*max_num_batches=*/2,
                                     cudf::nullable_join::YES,
                                     nulls_equal};
    joiner.insert(right0_view, stream);
    joiner.insert(right1_view, stream);
    auto [left_indices, right_indices] = joiner.inner_join(left, {}, stream);
    auto& [batch_indices, row_indices] = right_indices;
    auto const result = to_sorted_host_matches(*left_indices, *batch_indices, *row_indices, stream);
    EXPECT_EQ(result,
              hash_join_reference_matches(
                right_batches, left, cudf::nullable_join::YES, nulls_equal, stream));
    return result;
  };

  std::vector<join_match> const equal_expected{{0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {2, 1, 1}};
  std::vector<join_match> const unequal_expected{{1, 0, 0}, {2, 1, 1}};
  EXPECT_EQ(run(cudf::null_equality::EQUAL), equal_expected);
  EXPECT_EQ(run(cudf::null_equality::UNEQUAL), unequal_expected);
}

TEST_F(StreamingHashJoinTest, NestedKeysAcrossPartitions)
{
  auto const stream = cudf::test::get_default_stream();
  column_wrapper<int32_t> right0_child{1, 2};
  column_wrapper<int32_t> right1_child{2, 3};
  column_wrapper<int32_t> left_child{2, 3};
  auto right0_struct = cudf::test::structs_column_wrapper{{right0_child}};
  auto right1_struct = cudf::test::structs_column_wrapper{{right1_child}};
  auto left_struct   = cudf::test::structs_column_wrapper{{left_child}};
  cudf::table_view right0{{right0_struct}};
  cudf::table_view right1{{right1_struct}};
  cudf::table_view left{{left_struct}};
  std::array<cudf::table_view, 2> const right_batches{right0, right1};

  std::vector<cudf::data_type> const schema{right0.column(0).type()};
  std::vector<size_type> const key_indices{0};
  cudf::streaming_hash_join joiner{schema,
                                   key_indices,
                                   right0.num_rows() + right1.num_rows(),
                                   /*max_num_batches=*/2,
                                   cudf::nullable_join::NO,
                                   cudf::null_equality::EQUAL};
  joiner.insert(right0, stream);
  joiner.insert(right1, stream);

  auto [left_indices, right_indices] = joiner.inner_join(left, {}, stream);
  auto& [batch_indices, row_indices] = right_indices;
  auto const actual = to_sorted_host_matches(*left_indices, *batch_indices, *row_indices, stream);
  std::vector<join_match> const expected{{0, 0, 1}, {0, 1, 0}, {1, 1, 1}};
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(actual,
            hash_join_reference_matches(
              right_batches, left, cudf::nullable_join::NO, cudf::null_equality::EQUAL, stream));
}

TEST_F(StreamingHashJoinTest, SlicedPartitionReturnsSliceLocalRows)
{
  auto const stream = cudf::test::get_default_stream();
  column_wrapper<int32_t> right0_owner{99, 4, 5, 88};
  cudf::table_view right0_owner_view{{right0_owner}};
  auto const right0 = cudf::slice(right0_owner_view, {1, 3}).front();
  column_wrapper<int32_t> right1_keys{5, 6};
  cudf::table_view right1{{right1_keys}};
  column_wrapper<int32_t> left_keys{5};
  cudf::table_view left{{left_keys}};
  std::array<cudf::table_view, 2> const right_batches{right0, right1};

  std::vector<cudf::data_type> const schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const key_indices{0};
  cudf::streaming_hash_join joiner{schema,
                                   key_indices,
                                   right0.num_rows() + right1.num_rows(),
                                   /*max_num_batches=*/2,
                                   cudf::nullable_join::NO,
                                   cudf::null_equality::EQUAL};
  joiner.insert(right0, stream);
  joiner.insert(right1, stream);

  auto [left_indices, right_indices] = joiner.inner_join(left, {}, stream);
  auto& [batch_indices, row_indices] = right_indices;
  auto const actual = to_sorted_host_matches(*left_indices, *batch_indices, *row_indices, stream);
  std::vector<join_match> const expected{{0, 0, 1}, {0, 1, 0}};
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(actual,
            hash_join_reference_matches(
              right_batches, left, cudf::nullable_join::NO, cudf::null_equality::EQUAL, stream));
  EXPECT_EQ(left_indices->size(), 2);
}

TEST_F(StreamingHashJoinTest, MemoryResources)
{
  auto const stream = cudf::test::get_default_stream();

  column_wrapper<int32_t> right_keys{1, 2};
  cudf::table_view right{{right_keys}};
  column_wrapper<int32_t> left_keys{2};
  cudf::table_view left{{left_keys}};
  std::vector<cudf::data_type> const schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const key_indices{0};

  auto persistent_mr =
    rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());
  cudf::streaming_hash_join joiner{schema,
                                   key_indices,
                                   right.num_rows(),
                                   /*max_num_batches=*/1,
                                   cudf::nullable_join::NO,
                                   cudf::null_equality::EQUAL,
                                   /*load_factor=*/0.5,
                                   stream,
                                   persistent_mr};
  EXPECT_GT(persistent_mr.get_bytes_counter().peak, 0);
  joiner.insert(right, stream);

  auto output_mr = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());
  auto [left_indices, right_indices] = joiner.inner_join(left, {}, stream, output_mr);
  EXPECT_GT(output_mr.get_bytes_counter().peak, 0);
  EXPECT_EQ(left_indices->size(), 1);
  EXPECT_EQ(right_indices.first->size(), 1);
  EXPECT_EQ(right_indices.second->size(), 1);
}

TEST_F(StreamingHashJoinTest, ConstructorValidatesArguments)
{
  std::vector<cudf::data_type> const single_schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const single_key{0};

  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         single_key,
                                         4,
                                         /*max_num_batches=*/0,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL),
               std::invalid_argument);

  EXPECT_THROW(cudf::streaming_hash_join(std::vector<cudf::data_type>{},
                                         single_key,
                                         4,
                                         /*max_num_batches=*/1,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL),
               std::invalid_argument);

  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         std::vector<size_type>{},
                                         4,
                                         /*max_num_batches=*/1,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL),
               std::invalid_argument);

  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         std::vector<size_type>{5},
                                         4,
                                         /*max_num_batches=*/1,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL),
               std::invalid_argument);

  EXPECT_THROW(
    cudf::streaming_hash_join(
      single_schema, single_key, -1, 1, cudf::nullable_join::NO, cudf::null_equality::EQUAL),
    std::invalid_argument);

  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         single_key,
                                         4,
                                         /*max_num_batches=*/1,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL,
                                         /*load_factor=*/0.0),
               std::invalid_argument);
  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         single_key,
                                         4,
                                         /*max_num_batches=*/1,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL,
                                         /*load_factor=*/1.5),
               std::invalid_argument);
}
