/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/streaming_hash_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

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

}  // namespace

struct StreamingHashJoinTest : public cudf::test::BaseFixture {};

TEST_F(StreamingHashJoinTest, InnerJoinSinglePartitionMatchesHashJoin)
{
  auto const stream = cudf::get_default_stream();

  // Right (build) side
  column_wrapper<int32_t> right_keys{2, 4, 5, 7, 9, 1, 3, 5};
  cudf::table_view right_view{{right_keys}};

  // Left (probe) side
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

  cudf::hash_join reference_joiner{
    right_view, cudf::nullable_join::NO, cudf::null_equality::EQUAL, 0.5, stream};
  auto [reference_left, reference_right] = reference_joiner.inner_join(left_view, {}, stream);

  auto const [streaming_l, streaming_r] = to_sorted_host_pairs(*streaming_left, *streaming_right, stream);
  auto const [reference_l, reference_r] = to_sorted_host_pairs(*reference_left, *reference_right, stream);
  EXPECT_EQ(streaming_l, reference_l);
  EXPECT_EQ(streaming_r, reference_r);
}

TEST_F(StreamingHashJoinTest, SecondInsertThrows)
{
  auto const stream = cudf::get_default_stream();
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
  EXPECT_ANY_THROW(streaming_joiner.insert(right_view, stream));
}

TEST_F(StreamingHashJoinTest, InnerJoinBeforeInsertThrows)
{
  auto const stream = cudf::get_default_stream();

  std::vector<cudf::data_type> const right_schema{cudf::data_type{cudf::type_id::INT32}};
  std::vector<size_type> const right_key_indices{0};

  cudf::streaming_hash_join streaming_joiner{right_schema,
                                              right_key_indices,
                                              /*total_right_rows=*/4,
                                              cudf::nullable_join::NO,
                                              cudf::null_equality::EQUAL};
  column_wrapper<int32_t> left_keys{1, 2, 3};
  cudf::table_view left_view{{left_keys}};
  EXPECT_THROW(
    static_cast<void>(streaming_joiner.inner_join(left_view, {}, stream)), std::logic_error);
}

