/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streaming_hash_join.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/xxhash_32.cuh>
#include <cudf/join/streaming_hash_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/pair.cuh>
#include <cuco/static_multiset.cuh>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <thrust/transform.h>

#include <bit>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudf::detail {
namespace {

using slot_type = cuco::pair<hash_value_type, size_type>;

std::size_t checked_row_count(size_type rows)
{
  CUDF_EXPECTS(
    rows >= 0, "streaming_hash_join requires total_right_rows >= 0.", std::invalid_argument);
  return static_cast<std::size_t>(rows);
}

size_type checked_batch_count(size_type batches)
{
  CUDF_EXPECTS(
    batches > 0, "streaming_hash_join requires max_num_batches > 0.", std::invalid_argument);
  return batches;
}

double checked_load_factor(double load_factor)
{
  CUDF_EXPECTS(load_factor > 0.0 && load_factor <= 1.0,
               "streaming_hash_join requires load_factor in (0, 1].",
               std::invalid_argument);
  return load_factor;
}

/**
 * @brief Describes how a slot's first 32-bit word is divided between hash and batch ID.
 *
 * The high `batch_bits` store the batch ID and the remaining low bits store the row hash. CUCO's
 * probing hash functions use only the low hash bits, ensuring that equal rows from different
 * batches share the same probe sequence.
 */
struct batch_hash_layout {
  static constexpr int32_t num_hash_bits = std::numeric_limits<hash_value_type>::digits;
  static_assert(num_hash_bits == 32, "streaming_hash_join requires a 32-bit row hash");

  explicit batch_hash_layout(size_type max_num_batches)
    : batch_bits{std::bit_width(
        static_cast<hash_value_type>(checked_batch_count(max_num_batches) - 1))},
      batch_shift{num_hash_bits - batch_bits},
      hash_mask{std::numeric_limits<hash_value_type>::max() >> batch_bits}
  {
  }

  [[nodiscard]] CUDF_HOST_DEVICE constexpr hash_value_type masked_hash(
    hash_value_type hash) const noexcept
  {
    return hash & hash_mask;
  }

  [[nodiscard]] CUDF_HOST_DEVICE constexpr hash_value_type pack(hash_value_type hash,
                                                                size_type batch_id) const noexcept
  {
    if (batch_bits == 0) { return hash; }
    return masked_hash(hash) | (static_cast<hash_value_type>(batch_id) << batch_shift);
  }

  [[nodiscard]] CUDF_HOST_DEVICE constexpr size_type batch_id(
    hash_value_type packed_hash) const noexcept
  {
    if (batch_bits == 0) { return 0; }
    return static_cast<size_type>(packed_hash >> batch_shift);
  }

  int32_t batch_bits;
  int32_t batch_shift;
  hash_value_type hash_mask;
};

struct always_not_equal {
  __device__ constexpr bool operator()(slot_type const&, slot_type const&) const noexcept
  {
    return false;
  }
};

struct masked_hasher1 {
  hash_value_type hash_mask{std::numeric_limits<hash_value_type>::max()};

  __device__ constexpr hash_value_type operator()(slot_type const& key) const noexcept
  {
    return key.first & hash_mask;
  }
};

struct masked_hasher2 {
  masked_hasher2(hash_value_type mask = std::numeric_limits<hash_value_type>::max(),
                 hash_value_type seed = cudf::DEFAULT_HASH_SEED)
    : hash_mask{mask}, hash{seed}
  {
  }

  __device__ constexpr hash_value_type operator()(slot_type const& key) const noexcept
  {
    return hash(key.first & hash_mask);
  }

  hash_value_type hash_mask;
  cudf::hashing::detail::XXHash_32<hash_value_type> hash;
};

using probing_scheme  = cuco::double_hashing<DEFAULT_JOIN_CG_SIZE, masked_hasher1, masked_hasher2>;
using hash_table_type = cuco::static_multiset<slot_type,
                                              cuco::extent<std::size_t>,
                                              cuda::thread_scope_device,
                                              always_not_equal,
                                              probing_scheme,
                                              rmm::mr::polymorphic_allocator<char>,
                                              cuco::storage<2>>;

template <typename Hasher>
struct build_pair_fn {
  Hasher hash;
  batch_hash_layout layout;
  size_type batch_id;

  __device__ slot_type operator()(size_type row_index) const noexcept
  {
    return slot_type{layout.pack(hash(row_index), batch_id), row_index};
  }
};

template <typename Hasher>
struct probe_pair_fn {
  Hasher hash;
  batch_hash_layout layout;

  __device__ slot_type operator()(size_type row_index) const noexcept
  {
    return slot_type{layout.masked_hash(hash(row_index)), row_index};
  }
};

struct row_is_valid {
  bitmask_type const* row_bitmask;

  __device__ bool operator()(size_type row_index) const noexcept
  {
    return cudf::bit_is_set(row_bitmask, row_index);
  }
};

template <typename RowEqual>
struct n_table_pair_equal {
  RowEqual const* comparators;
  batch_hash_layout layout;

  __device__ bool operator()(slot_type const& probe, slot_type const& build) const noexcept
  {
    auto const batch_id = layout.batch_id(build.first);
    return layout.masked_hash(probe.first) == layout.masked_hash(build.first) &&
           comparators[batch_id](probe.second, build.second);
  }
};

struct extract_index_fn {
  __device__ constexpr size_type operator()(slot_type const& value) const noexcept
  {
    return value.second;
  }
};

struct decode_slot_fn {
  batch_hash_layout layout;

  __device__ cuda::std::tuple<size_type, size_type> operator()(slot_type const& slot) const noexcept
  {
    return {layout.batch_id(slot.first), slot.second};
  }
};

template <bool has_nested>
auto build_probe_comparators(
  std::shared_ptr<row::equality::preprocessed_table> const& preprocessed_left,
  std::vector<std::shared_ptr<row::equality::preprocessed_table>> const& preprocessed_right,
  nullate::DYNAMIC has_nulls,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream)
{
  using equality_type =
    row::equality::device_row_comparator<has_nested,
                                         nullate::DYNAMIC,
                                         row::equality::nan_equal_physical_equality_comparator>;

  std::vector<equality_type> host_comparators;
  host_comparators.reserve(preprocessed_right.size());
  for (auto const& right : preprocessed_right) {
    auto const comparator = row::equality::two_table_comparator{preprocessed_left, right};
    host_comparators.push_back(
      comparator.equal_to<has_nested>(has_nulls, compare_nulls).comparator);
  }
  return cudf::detail::make_device_uvector_async(
    host_comparators, stream, cudf::get_current_device_resource_ref());
}

std::vector<column_view> select_columns(table_view const& table, std::span<size_type const> indices)
{
  std::vector<column_view> columns;
  columns.reserve(indices.size());
  for (auto const index : indices) {
    columns.push_back(table.column(index));
  }
  return columns;
}

}  // namespace

struct streaming_hash_join::impl {
  std::vector<data_type> right_schema;
  std::vector<size_type> right_key_indices;
  size_type total_right_rows;
  size_type max_num_batches;
  bool has_nulls;
  null_equality compare_nulls;
  size_type inserted_rows{0};
  bool has_nested_keys{false};
  batch_hash_layout layout;

  // Declared before the hash table so the resource outlives allocations that refer to it.
  cuda::mr::any_resource<cuda::mr::device_accessible> mr;
  hash_table_type hash_table;

  std::vector<table_view> right_key_tables;
  std::vector<std::shared_ptr<row::equality::preprocessed_table>> preprocessed_right;

  impl(std::span<data_type const> schema,
       std::span<size_type const> key_indices,
       size_type total_rows,
       size_type maximum_batches,
       nullable_join nullable,
       null_equality nulls_equal,
       double load_factor,
       rmm::cuda_stream_view stream,
       cuda::mr::any_resource<cuda::mr::device_accessible> resource)
    : right_schema{schema.begin(), schema.end()},
      right_key_indices{key_indices.begin(), key_indices.end()},
      total_right_rows{total_rows},
      max_num_batches{checked_batch_count(maximum_batches)},
      has_nulls{nullable == nullable_join::YES},
      compare_nulls{nulls_equal},
      layout{max_num_batches},
      mr{std::move(resource)},
      hash_table{
        cuco::extent{checked_row_count(total_rows)},
        checked_load_factor(load_factor),
        cuco::empty_key{slot_type{std::numeric_limits<hash_value_type>::max(), cudf::JoinNoMatch}},
        always_not_equal{},
        probing_scheme{masked_hasher1{layout.hash_mask}, masked_hasher2{layout.hash_mask}},
        {},
        {},
        rmm::mr::polymorphic_allocator<char>{mr},
        stream.value()}
  {
    CUDF_EXPECTS(!right_schema.empty(),
                 "streaming_hash_join requires at least one right-side column.",
                 std::invalid_argument);
    CUDF_EXPECTS(!right_key_indices.empty(),
                 "streaming_hash_join requires at least one right-side key column.",
                 std::invalid_argument);
    auto const schema_size = static_cast<size_type>(right_schema.size());
    for (auto const index : right_key_indices) {
      CUDF_EXPECTS(index >= 0 && index < schema_size,
                   "streaming_hash_join key index is out of range for the provided schema.",
                   std::invalid_argument);
    }
  }

  void insert(table_view const& right_partition, rmm::cuda_stream_view stream)
  {
    CUDF_EXPECTS(static_cast<size_type>(right_key_tables.size()) < max_num_batches,
                 "streaming_hash_join: number of inserted batches would exceed max_num_batches.",
                 std::invalid_argument);
    CUDF_EXPECTS(right_partition.num_columns() == static_cast<size_type>(right_schema.size()),
                 "streaming_hash_join: inserted partition column count does not match schema.",
                 std::invalid_argument);
    for (size_type i = 0; i < right_partition.num_columns(); ++i) {
      CUDF_EXPECTS(right_partition.column(i).type() == right_schema[i],
                   "streaming_hash_join: inserted partition column type does not match schema.",
                   std::invalid_argument);
    }
    CUDF_EXPECTS(right_partition.num_rows() <= total_right_rows - inserted_rows,
                 "streaming_hash_join: cumulative inserted rows would exceed total_right_rows.",
                 std::invalid_argument);

    auto key_columns = select_columns(right_partition, right_key_indices);
    table_view keys{key_columns};
    if (!right_key_tables.empty()) {
      CUDF_EXPECTS(cudf::have_same_types(right_key_tables.front(), keys),
                   "streaming_hash_join: inserted key schema does not match prior partitions.",
                   cudf::data_type_error);
    } else {
      has_nested_keys = cudf::detail::has_nested_columns(keys);
    }

    auto preprocessed     = row::equality::preprocessed_table::create(keys, stream);
    auto const batch_id   = static_cast<size_type>(right_key_tables.size());
    auto const batch_rows = keys.num_rows();

    if (batch_rows > 0) {
      auto const nulls       = nullate::DYNAMIC{has_nulls};
      auto const row_hasher  = row::hash::row_hasher{preprocessed}.device_hasher(nulls);
      auto const input_begin = cudf::detail::make_counting_transform_iterator(
        0, build_pair_fn{row_hasher, layout, batch_id});

      if (compare_nulls == null_equality::EQUAL || !nullable(keys)) {
        hash_table.insert(input_begin, input_begin + batch_rows, stream.value());
      } else {
        auto const row_bitmask =
          cudf::detail::bitmask_and(keys, stream, cudf::get_current_device_resource_ref()).first;
        hash_table.insert_if(
          input_begin,
          input_begin + batch_rows,
          cuda::counting_iterator<size_type>{0},
          row_is_valid{reinterpret_cast<bitmask_type const*>(row_bitmask.data())},
          stream.value());
      }
    }

    right_key_tables.push_back(keys);
    preprocessed_right.push_back(std::move(preprocessed));
    inserted_rows += batch_rows;
  }

  template <bool has_nested>
  auto probe(table_view const& left,
             std::optional<std::size_t> output_size,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref output_mr) const
  {
    auto const temp_mr     = cudf::get_current_device_resource_ref();
    auto preprocessed_left = row::equality::preprocessed_table::create(left, stream);
    auto comparators       = build_probe_comparators<has_nested>(
      preprocessed_left, preprocessed_right, nullate::DYNAMIC{has_nulls}, compare_nulls, stream);
    auto const equality = n_table_pair_equal{comparators.data(), layout};
    auto const row_hasher =
      row::hash::row_hasher{preprocessed_left}.device_hasher(nullate::DYNAMIC{has_nulls});
    auto const input_begin =
      cudf::detail::make_counting_transform_iterator(0, probe_pair_fn{row_hasher, layout});

    auto const join_size = output_size ? *output_size
                                       : hash_table.count(input_begin,
                                                          input_begin + left.num_rows(),
                                                          equality,
                                                          hash_table.hash_function(),
                                                          stream.value());

    auto left_indices =
      std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, output_mr);
    auto batch_indices =
      std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, output_mr);
    auto row_indices =
      std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, output_mr);
    rmm::device_uvector<slot_type> build_slots(join_size, stream, temp_mr);

    if (join_size > 0) {
      cudf::prefetch::detail::prefetch(*left_indices, stream);
      cudf::prefetch::detail::prefetch(*batch_indices, stream);
      cudf::prefetch::detail::prefetch(*row_indices, stream);
      auto const probe_output =
        cuda::transform_output_iterator{left_indices->begin(), extract_index_fn{}};
      hash_table.retrieve(input_begin,
                          input_begin + left.num_rows(),
                          equality,
                          hash_table.hash_function(),
                          probe_output,
                          build_slots.begin(),
                          stream.value());

      thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                        build_slots.begin(),
                        build_slots.end(),
                        cuda::zip_iterator(batch_indices->begin(), row_indices->begin()),
                        decode_slot_fn{layout});
    }

    return std::pair{std::move(left_indices),
                     std::pair{std::move(batch_indices), std::move(row_indices)}};
  }

  auto inner_join(table_view const& left,
                  std::optional<std::size_t> output_size,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref output_mr) const
  {
    CUDF_EXPECTS(!right_key_tables.empty(),
                 "streaming_hash_join: inner_join called before any insert().",
                 std::logic_error);
    CUDF_EXPECTS(left.num_columns() == right_key_tables.front().num_columns(),
                 "Mismatch in number of columns to be joined on",
                 std::invalid_argument);
    CUDF_EXPECTS(cudf::have_same_types(left, right_key_tables.front()),
                 "Mismatch in joining column data types",
                 cudf::data_type_error);
    CUDF_EXPECTS(has_nulls || !cudf::has_nested_nulls(left),
                 "Left table has nulls while right table was not hashed with null check.",
                 std::invalid_argument);

    if (left.num_rows() == 0 || inserted_rows == 0) {
      return std::pair{
        std::make_unique<rmm::device_uvector<size_type>>(0, stream, output_mr),
        std::pair{std::make_unique<rmm::device_uvector<size_type>>(0, stream, output_mr),
                  std::make_unique<rmm::device_uvector<size_type>>(0, stream, output_mr)}};
    }
    return has_nested_keys ? probe<true>(left, output_size, stream, output_mr)
                           : probe<false>(left, output_size, stream, output_mr);
  }
};

streaming_hash_join::streaming_hash_join(std::span<data_type const> right_schema,
                                         std::span<size_type const> right_key_indices,
                                         size_type total_right_rows,
                                         size_type max_num_batches,
                                         nullable_join has_nulls,
                                         null_equality compare_nulls,
                                         double load_factor,
                                         rmm::cuda_stream_view stream,
                                         cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl{std::make_unique<impl>(right_schema,
                                 right_key_indices,
                                 total_right_rows,
                                 max_num_batches,
                                 has_nulls,
                                 compare_nulls,
                                 load_factor,
                                 stream,
                                 std::move(mr))}
{
}

streaming_hash_join::~streaming_hash_join()                                         = default;
streaming_hash_join::streaming_hash_join(streaming_hash_join&&) noexcept            = default;
streaming_hash_join& streaming_hash_join::operator=(streaming_hash_join&&) noexcept = default;

void streaming_hash_join::insert(table_view const& right_partition, rmm::cuda_stream_view stream)
{
  _impl->insert(right_partition, stream);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                    std::unique_ptr<rmm::device_uvector<size_type>>>>
streaming_hash_join::inner_join(table_view const& left,
                                std::optional<std::size_t> output_size,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(left, output_size, stream, mr);
}

}  // namespace cudf::detail

namespace cudf {

streaming_hash_join::streaming_hash_join(std::span<data_type const> right_schema,
                                         std::span<size_type const> right_key_indices,
                                         size_type total_right_rows,
                                         size_type max_num_batches,
                                         nullable_join has_nulls,
                                         null_equality compare_nulls,
                                         double load_factor,
                                         rmm::cuda_stream_view stream,
                                         cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl{std::make_unique<cudf::detail::streaming_hash_join>(right_schema,
                                                              right_key_indices,
                                                              total_right_rows,
                                                              max_num_batches,
                                                              has_nulls,
                                                              compare_nulls,
                                                              load_factor,
                                                              stream,
                                                              std::move(mr))}
{
}

streaming_hash_join::~streaming_hash_join()                                         = default;
streaming_hash_join::streaming_hash_join(streaming_hash_join&&) noexcept            = default;
streaming_hash_join& streaming_hash_join::operator=(streaming_hash_join&&) noexcept = default;

void streaming_hash_join::insert(table_view const& right_partition, rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  _impl->insert(right_partition, stream);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                    std::unique_ptr<rmm::device_uvector<size_type>>>>
streaming_hash_join::inner_join(table_view const& left,
                                std::optional<std::size_t> output_size,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->inner_join(left, output_size, stream, mr);
}

}  // namespace cudf
