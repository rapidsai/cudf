/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/streaming_hash_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <stdexcept>
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

TEST_F(StreamingHashJoinTest, SecondInsertThrows)
{
  auto const stream = cudf::test::get_default_stream();
  column_wrapper<int32_t> right_keys{1, 2, 3};
  cudf::table_view right_view{{right_keys}};

  std::vector<cudf::data_type> const right_schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const right_key_indices{0};

  cudf::streaming_hash_join streaming_joiner{right_schema,
                                             right_key_indices,
                                             /*total_right_rows=*/8,
                                             cudf::nullable_join::NO,
                                             cudf::null_equality::EQUAL};
  streaming_joiner.insert(right_view, stream);
  EXPECT_THROW(streaming_joiner.insert(right_view, stream), std::invalid_argument);
}

TEST_F(StreamingHashJoinTest, InnerJoinBeforeInsertThrows)
{
  auto const stream = cudf::test::get_default_stream();

  std::vector<cudf::data_type> const right_schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const right_key_indices{0};

  cudf::streaming_hash_join streaming_joiner{right_schema,
                                             right_key_indices,
                                             /*total_right_rows=*/4,
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
                                             cudf::nullable_join::NO,
                                             cudf::null_equality::EQUAL};

  column_wrapper<int64_t> wrong_type_keys{1L, 2L, 3L};
  cudf::table_view wrong_type_view{{wrong_type_keys}};
  EXPECT_THROW(streaming_joiner.insert(wrong_type_view, stream), std::invalid_argument);

  column_wrapper<int32_t> col_a{1, 2, 3};
  column_wrapper<int32_t> col_b{4, 5, 6};
  cudf::table_view wrong_count_view{{col_a, col_b}};
  EXPECT_THROW(streaming_joiner.insert(wrong_count_view, stream), std::invalid_argument);
}

TEST_F(StreamingHashJoinTest, ConstructorValidatesArguments)
{
  std::vector<cudf::data_type> const single_schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const single_key{0};

  EXPECT_THROW(cudf::streaming_hash_join(std::vector<cudf::data_type>{},
                                         single_key,
                                         4,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL),
               std::invalid_argument);

  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         std::vector<size_type>{},
                                         4,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL),
               std::invalid_argument);

  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         std::vector<size_type>{5},
                                         4,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL),
               std::invalid_argument);

  EXPECT_THROW(
    cudf::streaming_hash_join(
      single_schema, single_key, -1, cudf::nullable_join::NO, cudf::null_equality::EQUAL),
    std::invalid_argument);

  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         single_key,
                                         4,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL,
                                         /*load_factor=*/0.0),
               std::invalid_argument);
  EXPECT_THROW(cudf::streaming_hash_join(single_schema,
                                         single_key,
                                         4,
                                         cudf::nullable_join::NO,
                                         cudf::null_equality::EQUAL,
                                         /*load_factor=*/1.5),
               std::invalid_argument);
}
