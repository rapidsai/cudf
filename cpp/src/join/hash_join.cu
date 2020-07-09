/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>

#include "hash_join.cuh"

namespace cudf {
namespace detail {

/**
 * @brief Returns a vector with non-common indices which is set difference
 * between `[0, num_columns)` and index values in common_column_indices
 *
 * @param num_columns The number of columns, which represents column indices
 * from `[0, num_columns)` in a table
 * @param common_column_indices A vector of common indices which needs to be
 * excluded from `[0, num_columns)`
 *
 * @return vector A vector containing only the indices which are not present in
 * `common_column_indices`
 */
auto non_common_column_indices(size_type num_columns,
                               std::vector<size_type> const &common_column_indices)
{
  CUDF_EXPECTS(common_column_indices.size() <= static_cast<unsigned long>(num_columns),
               "Too many columns in common");
  std::vector<size_type> all_column_indices(num_columns);
  std::iota(std::begin(all_column_indices), std::end(all_column_indices), 0);
  std::vector<size_type> sorted_common_column_indices{common_column_indices};
  std::sort(std::begin(sorted_common_column_indices), std::end(sorted_common_column_indices));
  std::vector<size_type> non_common_column_indices(num_columns - common_column_indices.size());
  std::set_difference(std::cbegin(all_column_indices),
                      std::cend(all_column_indices),
                      std::cbegin(sorted_common_column_indices),
                      std::cend(sorted_common_column_indices),
                      std::begin(non_common_column_indices));
  return non_common_column_indices;
}

std::unique_ptr<table> get_empty_joined_table(
  table_view const &left,
  table_view const &right,
  std::vector<std::pair<size_type, size_type>> const &columns_in_common)
{
  std::vector<size_type> right_columns_in_common(columns_in_common.size());
  std::transform(columns_in_common.begin(),
                 columns_in_common.end(),
                 right_columns_in_common.begin(),
                 [](auto &col) { return col.second; });
  std::unique_ptr<table> empty_left  = empty_like(left);
  std::unique_ptr<table> empty_right = empty_like(right);
  std::vector<size_type> right_non_common_indices =
    non_common_column_indices(right.num_columns(), right_columns_in_common);
  table_view tmp_right_table = (*empty_right).select(right_non_common_indices);
  table_view tmp_table{{*empty_left, tmp_right_table}};
  return std::make_unique<table>(tmp_table);
}

VectorPair concatenate_vector_pairs(VectorPair &a, VectorPair &b)
{
  CUDF_EXPECTS((a.first.size() == a.second.size()),
               "Mismatch between sizes of vectors in vector pair");
  CUDF_EXPECTS((b.first.size() == b.second.size()),
               "Mismatch between sizes of vectors in vector pair");
  if (a.first.size() == 0) {
    return b;
  } else if (b.first.size() == 0) {
    return a;
  }
  auto original_size = a.first.size();
  a.first.resize(a.first.size() + b.first.size());
  a.second.resize(a.second.size() + b.second.size());
  thrust::copy(b.first.begin(), b.first.end(), a.first.begin() + original_size);
  thrust::copy(b.second.begin(), b.second.end(), a.second.begin() + original_size);
  return a;
}

template <typename T>
struct valid_range {
  T start, stop;
  __host__ __device__ valid_range(const T begin, const T end) : start(begin), stop(end) {}

  __host__ __device__ __forceinline__ bool operator()(const T index)
  {
    return ((index >= start) && (index < stop));
  }
};

/**
 * @brief  Creates a table containing the complement of left join indices.
 * This table has two columns. The first one is filled with JoinNoneValue(-1)
 * and the second one contains values from 0 to right_table_row_count - 1
 * excluding those found in the right_indices column.
 *
 * @param right_indices Vector of indices
 * @param left_table_row_count Number of rows of left table
 * @param right_table_row_count Number of rows of right table
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return Pair of vectors containing the left join indices complement
 */
std::pair<rmm::device_vector<size_type>, rmm::device_vector<size_type>>
get_left_join_indices_complement(rmm::device_vector<size_type> &right_indices,
                                 size_type left_table_row_count,
                                 size_type right_table_row_count,
                                 cudaStream_t stream)
{
  // Get array of indices that do not appear in right_indices

  // Vector allocated for unmatched result
  rmm::device_vector<size_type> right_indices_complement(right_table_row_count);

  // If left table is empty in a full join call then all rows of the right table
  // should be represented in the joined indices. This is an optimization since
  // if left table is empty and full join is called all the elements in
  // right_indices will be JoinNoneValue, i.e. -1. This if path should
  // produce exactly the same result as the else path but will be faster.
  if (left_table_row_count == 0) {
    thrust::sequence(rmm::exec_policy(stream)->on(stream),
                     right_indices_complement.begin(),
                     right_indices_complement.end(),
                     0);
  } else {
    // Assume all the indices in invalid_index_map are invalid
    rmm::device_vector<size_type> invalid_index_map(right_table_row_count, 1);
    // Functor to check for index validity since left joins can create invalid indices
    valid_range<size_type> valid(0, right_table_row_count);

    // invalid_index_map[index_ptr[i]] = 0 for i = 0 to right_table_row_count
    // Thus specifying that those locations are valid
    thrust::scatter_if(rmm::exec_policy(stream)->on(stream),
                       thrust::make_constant_iterator(0),
                       thrust::make_constant_iterator(0) + right_indices.size(),
                       right_indices.begin(),      // Index locations
                       right_indices.begin(),      // Stencil - Check if index location is valid
                       invalid_index_map.begin(),  // Output indices
                       valid);                     // Stencil Predicate
    size_type begin_counter = static_cast<size_type>(0);
    size_type end_counter   = static_cast<size_type>(right_table_row_count);

    // Create list of indices that have been marked as invalid
    size_type indices_count = thrust::copy_if(rmm::exec_policy(stream)->on(stream),
                                              thrust::make_counting_iterator(begin_counter),
                                              thrust::make_counting_iterator(end_counter),
                                              invalid_index_map.begin(),
                                              right_indices_complement.begin(),
                                              thrust::identity<size_type>()) -
                              right_indices_complement.begin();
    right_indices_complement.resize(indices_count);
  }

  rmm::device_vector<size_type> left_invalid_indices(right_indices_complement.size(),
                                                     JoinNoneValue);

  return std::make_pair(std::move(left_invalid_indices), std::move(right_indices_complement));
}

/**
 * @brief Builds the hash table based on the given `build_table`.
 *
 * @throw cudf::logic_error if the number of columns in `build` table is 0.
 * @throw cudf::logic_error if the number of rows in `build` table is 0.
 * @throw cudf::logic_error if insertion to the hash table fails.
 * @throw std::out_of_range if elements of `build_on` exceed the number of columns in the `build`
 * table.
 *
 * @param build_table Table of build side columns to join.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return Built hash table.
 */
std::unique_ptr<multimap_type, std::function<void(multimap_type *)>> build_join_hash_table(
  cudf::table_device_view build_table, cudaStream_t stream)
{
  CUDF_EXPECTS(0 != build_table.num_columns(), "Selected build dataset is empty");
  CUDF_EXPECTS(0 != build_table.num_rows(), "Build side table has no rows");

  const size_type build_table_num_rows{build_table.num_rows()};
  size_t const hash_table_size = compute_hash_table_size(build_table_num_rows);

  auto hash_table = multimap_type::create(hash_table_size,
                                          true,
                                          multimap_type::hasher(),
                                          multimap_type::key_equal(),
                                          multimap_type::allocator_type(),
                                          stream);

  row_hash hash_build{build_table};
  rmm::device_scalar<int> failure(0, 0);
  constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
  detail::grid_1d config(build_table_num_rows, block_size);
  build_hash_table<<<config.num_blocks, config.num_threads_per_block, 0, 0>>>(
    *hash_table, hash_build, build_table_num_rows, failure.data());
  // Check error code from the kernel
  if (failure.value() == 1) { CUDF_FAIL("Hash Table insert failure."); }

  return hash_table;
}

/**
 * @brief Probes the `hash_table` built from `build_table` for tuples in `probe_table`,
 * and returns the output indices of `build_table` and `probe_table` as a combined table.
 *
 * @tparam JoinKind The type of join to be performed.
 *
 * @param build_table Table of build side columns to join.
 * @param probe_table Table of probe side columns to join.
 * @param hash_table Hash table built from `build_table`.
 * @param flip_join_indices Flag that indicates whether the left (probe) and right (build)
 * tables have been flipped, meaning the output indices should also be flipped.
 * @param compare_nulls Controls whether null join-key values should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return Join output indices vector pair.
 */
template <join_kind JoinKind>
std::pair<rmm::device_vector<size_type>, rmm::device_vector<size_type>> probe_join_hash_table(
  cudf::table_device_view build_table,
  cudf::table_device_view probe_table,
  multimap_type const &hash_table,
  bool flip_join_indices,
  null_equality compare_nulls,
  cudaStream_t stream)
{
  size_type estimated_size = estimate_join_output_size<JoinKind, multimap_type>(
    build_table, probe_table, hash_table, compare_nulls, stream);

  // If the estimated output size is zero, return immediately
  if (estimated_size == 0) {
    return std::make_pair(rmm::device_vector<size_type>{}, rmm::device_vector<size_type>{});
  }

  // Because we are approximating the number of joined elements, our approximation
  // might be incorrect and we might have underestimated the number of joined elements.
  // As such we will need to de-allocate memory and re-allocate memory to ensure
  // that the final output is correct.
  rmm::device_scalar<size_type> write_index(0, stream);
  size_type join_size{0};

  rmm::device_vector<size_type> left_indices;
  rmm::device_vector<size_type> right_indices;
  auto current_estimated_size = estimated_size;
  do {
    left_indices.resize(estimated_size);
    right_indices.resize(estimated_size);

    constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
    detail::grid_1d config(probe_table.num_rows(), block_size);
    write_index.set_value(0);

    row_hash hash_probe{probe_table};
    row_equality equality{probe_table, build_table, compare_nulls == null_equality::EQUAL};
    const auto &join_output_l =
      flip_join_indices ? right_indices.data().get() : left_indices.data().get();
    const auto &join_output_r =
      flip_join_indices ? left_indices.data().get() : right_indices.data().get();
    probe_hash_table<JoinKind, multimap_type, block_size, DEFAULT_JOIN_CACHE_SIZE>
      <<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(hash_table,
                                                                       build_table,
                                                                       probe_table,
                                                                       hash_probe,
                                                                       equality,
                                                                       join_output_l,
                                                                       join_output_r,
                                                                       write_index.data(),
                                                                       estimated_size);

    CHECK_CUDA(stream);

    join_size              = write_index.value();
    current_estimated_size = estimated_size;
    estimated_size *= 2;
  } while ((current_estimated_size < join_size));

  left_indices.resize(join_size);
  right_indices.resize(join_size);
  return std::make_pair(std::move(left_indices), std::move(right_indices));
}

/**
 * @brief  Combines the non common left, common left and non common right
 * columns in the correct order to form the join output table.
 *
 * @param left_noncommon_cols Columns obtained by gathering non common left
 * columns.
 * @param left_noncommon_col_indices Output locations of non common left columns
 * in the final table output
 * @param left_common_cols Columns obtained by gathering common left
 * columns.
 * @param left_common_col_indices Output locations of common left columns in the
 * final table output
 * @param right_noncommon_cols Table obtained by gathering non common right
 * columns.
 *
 * @return Rearranged columns.
 */
std::vector<std::unique_ptr<column>> combine_join_columns(
  std::vector<std::unique_ptr<column>> &&left_noncommon_cols,
  std::vector<size_type> const &left_noncommon_col_indices,
  std::vector<std::unique_ptr<column>> &&left_common_cols,
  std::vector<size_type> const &left_common_col_indices,
  std::vector<std::unique_ptr<column>> &&right_noncommon_cols)
{
  std::vector<std::unique_ptr<column>> combined_cols(left_noncommon_cols.size() +
                                                     left_common_cols.size());
  for (size_t i = 0; i < left_noncommon_cols.size(); ++i) {
    combined_cols.at(left_noncommon_col_indices.at(i)) = std::move(left_noncommon_cols.at(i));
  }
  for (size_t i = 0; i < left_common_cols.size(); ++i) {
    combined_cols.at(left_common_col_indices.at(i)) = std::move(left_common_cols.at(i));
  }
  combined_cols.insert(combined_cols.end(),
                       std::make_move_iterator(right_noncommon_cols.begin()),
                       std::make_move_iterator(right_noncommon_cols.end()));
  return combined_cols;
}

/**
 * @brief  Gathers rows from `left` and `right` table and combines them into a
 * single table.
 *
 * @tparam JoinKind The type of join to be performed
 *
 * @param left Left input table
 * @param right Right input table
 * @param joined_indices Pair of vectors containing row indices from which
 * `left` and `right` tables are gathered. If any row index is out of bounds,
 * the contribution in the output `table` will be NULL.
 * @param columns_in_common is a vector of pairs of column indices
 * from tables `left` and `right` respectively, that are "in common".
 * For "common" columns, only a single output column will be produced.
 * For an inner or left join, the result will be gathered from the column in
 * `left`. For a full join, the result will be gathered from both common
 * columns in `left` and `right` and concatenated to form a single column.
 *
 * @return `table` containing the concatenation of rows from `left` and
 * `right` specified by `joined_indices`.
 * For any columns indicated by `columns_in_common`, only the corresponding
 * column in `left` will be included in the result. Final form would look like
 * `left(including common columns)+right(excluding common columns)`.
 */
template <join_kind JoinKind>
std::unique_ptr<table> construct_join_output_df(
  table_view const &left,
  table_view const &right,
  VectorPair &joined_indices,
  std::vector<std::pair<size_type, size_type>> const &columns_in_common,
  rmm::mr::device_memory_resource *mr,
  cudaStream_t stream)
{
  std::vector<size_type> left_common_col;
  left_common_col.reserve(columns_in_common.size());
  std::vector<size_type> right_common_col;
  right_common_col.reserve(columns_in_common.size());
  for (const auto &c : columns_in_common) {
    left_common_col.push_back(c.first);
    right_common_col.push_back(c.second);
  }
  std::vector<size_type> left_noncommon_col =
    non_common_column_indices(left.num_columns(), left_common_col);
  std::vector<size_type> right_noncommon_col =
    non_common_column_indices(right.num_columns(), right_common_col);

  bool const nullify_out_of_bounds{JoinKind != join_kind::INNER_JOIN};

  std::unique_ptr<table> common_table = std::make_unique<table>();
  // Construct the joined columns
  if (join_kind::FULL_JOIN == JoinKind) {
    auto complement_indices = get_left_join_indices_complement(
      joined_indices.second, left.num_rows(), right.num_rows(), stream);
    if (not columns_in_common.empty()) {
      auto common_from_right = detail::gather(right.select(right_common_col),
                                              complement_indices.second.begin(),
                                              complement_indices.second.end(),
                                              nullify_out_of_bounds,
                                              rmm::mr::get_default_resource(),
                                              stream);
      auto common_from_left  = detail::gather(left.select(left_common_col),
                                             joined_indices.first.begin(),
                                             joined_indices.first.end(),
                                             nullify_out_of_bounds,
                                             rmm::mr::get_default_resource(),
                                             stream);
      common_table           = cudf::detail::concatenate(
        {common_from_right->view(), common_from_left->view()}, mr, stream);
    }
    joined_indices = concatenate_vector_pairs(complement_indices, joined_indices);
  } else {
    if (not columns_in_common.empty()) {
      common_table = detail::gather(left.select(left_common_col),
                                    joined_indices.first.begin(),
                                    joined_indices.first.end(),
                                    nullify_out_of_bounds,
                                    mr,
                                    stream);
    }
  }

  // Construct the left non common columns
  std::unique_ptr<table> left_table = detail::gather(left.select(left_noncommon_col),
                                                     joined_indices.first.begin(),
                                                     joined_indices.first.end(),
                                                     nullify_out_of_bounds,
                                                     mr,
                                                     stream);

  std::unique_ptr<table> right_table = detail::gather(right.select(right_noncommon_col),
                                                      joined_indices.second.begin(),
                                                      joined_indices.second.end(),
                                                      nullify_out_of_bounds,
                                                      mr,
                                                      stream);

  return std::make_unique<table>(combine_join_columns(left_table->release(),
                                                      left_noncommon_col,
                                                      common_table->release(),
                                                      left_common_col,
                                                      right_table->release()));
}

hash_join_impl::hash_join_impl(cudf::table_view const &build,
                               std::vector<size_type> const &build_on)
  : _build(build),
    _build_selected(build.select(build_on)),
    _build_on(build_on),
    _hash_table(nullptr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != _build.num_columns(), "Hash join build table is empty");
  CUDF_EXPECTS(_build.num_rows() < MAX_JOIN_SIZE, "Build column size is too big for hash join");

  if (_build_on.empty() || 0 == build.num_rows()) { return; }

  auto build_table = cudf::table_device_view::create(_build_selected);
  _hash_table      = build_join_hash_table(*build_table, 0);
}

std::unique_ptr<cudf::table> hash_join_impl::inner_join(
  cudf::table_view const &probe,
  std::vector<size_type> const &probe_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const &columns_in_common,
  cudf::hash_join::probe_output_side probe_output_side,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource *mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join<join_kind::INNER_JOIN>(
    probe, probe_on, columns_in_common, probe_output_side, compare_nulls, mr);
}

std::unique_ptr<cudf::table> hash_join_impl::left_join(
  cudf::table_view const &probe,
  std::vector<size_type> const &probe_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const &columns_in_common,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource *mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join<join_kind::LEFT_JOIN>(probe,
                                                 probe_on,
                                                 columns_in_common,
                                                 cudf::hash_join::probe_output_side::LEFT,
                                                 compare_nulls,
                                                 mr);
}

std::unique_ptr<cudf::table> hash_join_impl::full_join(
  cudf::table_view const &probe,
  std::vector<size_type> const &probe_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const &columns_in_common,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource *mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join<join_kind::FULL_JOIN>(probe,
                                                 probe_on,
                                                 columns_in_common,
                                                 cudf::hash_join::probe_output_side::LEFT,
                                                 compare_nulls,
                                                 mr);
}

template <join_kind JoinKind>
std::unique_ptr<table> hash_join_impl::compute_hash_join(
  cudf::table_view const &probe,
  std::vector<size_type> const &probe_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const &columns_in_common,
  cudf::hash_join::probe_output_side probe_output_side,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource *mr,
  cudaStream_t stream) const
{
  CUDF_EXPECTS(0 != probe.num_columns(), "Hash join probe table is empty");
  CUDF_EXPECTS(probe.num_rows() < MAX_JOIN_SIZE, "Probe column size is too big for hash join");
  CUDF_EXPECTS(_build_on.size() == probe_on.size(),
               "Mismatch in number of columns to be joined on");

  CUDF_EXPECTS(std::all_of(columns_in_common.begin(),
                           columns_in_common.end(),
                           [this, &probe_on](auto pair) {
                             size_t b = std::find(_build_on.begin(), _build_on.end(), pair.first) -
                                        _build_on.begin();
                             size_t p = std::find(probe_on.begin(), probe_on.end(), pair.second) -
                                        probe_on.begin();
                             return (b != _build_on.size()) && (p != probe_on.size()) && (b == p);
                           }),
               "Invalid values passed to columns_in_common");

  if (is_trivial_join(probe, _build, probe_on, _build_on, JoinKind)) {
    return get_empty_joined_table(probe, _build, columns_in_common);
  }

  auto probe_selected = probe.select(probe_on);
  CUDF_EXPECTS(std::equal(std::cbegin(_build_selected),
                          std::cend(_build_selected),
                          std::cbegin(probe_selected),
                          std::cend(probe_selected),
                          [](const auto &b, const auto &p) { return b.type() == p.type(); }),
               "Mismatch in joining column data types");

  bool probe_output_left{probe_output_side == cudf::hash_join::probe_output_side::LEFT};

  constexpr join_kind ProbeJoinKind =
    (JoinKind == join_kind::FULL_JOIN) ? join_kind::LEFT_JOIN : JoinKind;
  auto joined_indices =
    probe_join_indices<ProbeJoinKind>(probe_selected, !probe_output_left, compare_nulls, stream);
  auto actual_columns_in_common = columns_in_common;
  if (!probe_output_left) {
    std::for_each(actual_columns_in_common.begin(), actual_columns_in_common.end(), [](auto &pair) {
      std::swap(pair.first, pair.second);
    });
  }
  return construct_join_output_df<JoinKind>(probe_output_left ? probe : _build,
                                            probe_output_left ? _build : probe,
                                            joined_indices,
                                            actual_columns_in_common,
                                            mr,
                                            stream);
}

template <join_kind JoinKind>
std::enable_if_t<JoinKind != join_kind::FULL_JOIN,
                 std::pair<rmm::device_vector<size_type>, rmm::device_vector<size_type>>>
hash_join_impl::probe_join_indices(cudf::table_view const &probe,
                                   bool flip_join_indices,
                                   null_equality compare_nulls,
                                   cudaStream_t stream) const
{
  // Trivial left join case - exit early
  if (!_hash_table && JoinKind == join_kind::LEFT_JOIN) {
    return get_trivial_left_join_indices(probe, flip_join_indices, stream);
  }

  CUDF_EXPECTS(_hash_table, "Hash table of hash join is null.");

  auto build_table = cudf::table_device_view::create(_build_selected, stream);
  auto probe_table = cudf::table_device_view::create(probe, stream);
  return probe_join_hash_table<JoinKind>(
    *build_table, *probe_table, *_hash_table, flip_join_indices, compare_nulls, stream);
}

}  // namespace detail
}  // namespace cudf
