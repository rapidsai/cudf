/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>

#include <join/hash_join.cuh>

#include <numeric>

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
  CUDF_EXPECTS(common_column_indices.size() <= static_cast<uint64_t>(num_columns),
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

std::pair<std::unique_ptr<table>, std::unique_ptr<table>> get_empty_joined_table(
  table_view const &probe,
  table_view const &build,
  std::vector<std::pair<size_type, size_type>> const &columns_in_common,
  cudf::hash_join::common_columns_output_side common_columns_output_side)
{
  std::vector<size_type> columns_to_exclude(columns_in_common.size());
  std::transform(columns_in_common.begin(),
                 columns_in_common.end(),
                 columns_to_exclude.begin(),
                 [common_columns_output_side](auto &col) {
                   return common_columns_output_side == hash_join::common_columns_output_side::PROBE
                            ? col.second
                            : col.first;
                 });
  std::vector<size_type> non_common_indices = non_common_column_indices(
    common_columns_output_side == hash_join::common_columns_output_side::PROBE
      ? build.num_columns()
      : probe.num_columns(),
    columns_to_exclude);
  std::unique_ptr<table> empty_probe = empty_like(probe);
  std::unique_ptr<table> empty_build = empty_like(build);
  if (common_columns_output_side == hash_join::common_columns_output_side::PROBE) {
    table_view empty_build_view = empty_build->select(non_common_indices);
    empty_build                 = std::make_unique<table>(empty_build_view);
  } else {
    table_view empty_probe_view = empty_probe->select(non_common_indices);
    empty_probe                 = std::make_unique<table>(empty_probe_view);
  }
  return std::make_pair(std::move(empty_probe), std::move(empty_build));
}

VectorPair concatenate_vector_pairs(VectorPair &a, VectorPair &b)
{
  CUDF_EXPECTS((a.first.size() == a.second.size()),
               "Mismatch between sizes of vectors in vector pair");
  CUDF_EXPECTS((b.first.size() == b.second.size()),
               "Mismatch between sizes of vectors in vector pair");
  if (a.first.empty()) {
    return b;
  } else if (b.first.empty()) {
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
  rmm::device_scalar<int> failure(0, stream);
  constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
  detail::grid_1d config(build_table_num_rows, block_size);
  build_hash_table<<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
    *hash_table, hash_build, build_table_num_rows, failure.data());
  // Check error code from the kernel
  if (failure.value(stream) == 1) { CUDF_FAIL("Hash Table insert failure."); }

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
    write_index.set_value(0, stream);

    row_hash hash_probe{probe_table};
    row_equality equality{probe_table, build_table, compare_nulls == null_equality::EQUAL};
    probe_hash_table<JoinKind, multimap_type, block_size, DEFAULT_JOIN_CACHE_SIZE>
      <<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(hash_table,
                                                                       build_table,
                                                                       probe_table,
                                                                       hash_probe,
                                                                       equality,
                                                                       left_indices.data().get(),
                                                                       right_indices.data().get(),
                                                                       write_index.data(),
                                                                       estimated_size);

    CHECK_CUDA(stream);

    join_size              = write_index.value(stream);
    current_estimated_size = estimated_size;
    estimated_size *= 2;
  } while ((current_estimated_size < join_size));

  left_indices.resize(join_size);
  right_indices.resize(join_size);
  return std::make_pair(std::move(left_indices), std::move(right_indices));
}

/**
 * @brief  Combines the non common probe, common probe, non common build and common build
 * columns in the correct order according to `common_columns_output_side` to form the joined
 * (`probe`, `build`) table pair.
 *
 * @param probe_noncommon_cols Columns obtained by gathering non common probe columns.
 * @param probe_noncommon_col_indices Output locations of non common probe columns in the probe
 * portion.
 * @param probe_common_col_indices Output locations of common probe columns in the probe portion.
 * @param build_noncommon_cols Columns obtained by gathering non common build columns.
 * @param build_noncommon_col_indices Output locations of non common build columns in the build
 * portion.
 * @param build_common_col_indices Output locations of common build columns in the build portion.
 * @param common_cols Columns obtained by gathering common columns from `probe` and `build` tables
 * in the build portion.
 * @param common_columns_output_side @see cudf::hash_join::common_columns_output_side.
 *
 * @return Table pair of (`probe`, `build`).
 */
std::pair<std::unique_ptr<table>, std::unique_ptr<table>> combine_join_columns(
  std::vector<std::unique_ptr<column>> &&probe_noncommon_cols,
  std::vector<size_type> const &probe_noncommon_col_indices,
  std::vector<size_type> const &probe_common_col_indices,
  std::vector<std::unique_ptr<column>> &&build_noncommon_cols,
  std::vector<size_type> const &build_noncommon_col_indices,
  std::vector<size_type> const &build_common_col_indices,
  std::vector<std::unique_ptr<column>> &&common_cols,
  cudf::hash_join::common_columns_output_side common_columns_output_side)
{
  if (common_columns_output_side == cudf::hash_join::common_columns_output_side::PROBE) {
    std::vector<std::unique_ptr<column>> probe_cols(probe_noncommon_cols.size() +
                                                    common_cols.size());
    for (size_t i = 0; i < probe_noncommon_cols.size(); ++i) {
      probe_cols.at(probe_noncommon_col_indices.at(i)) = std::move(probe_noncommon_cols.at(i));
    }
    for (size_t i = 0; i < common_cols.size(); ++i) {
      probe_cols.at(probe_common_col_indices.at(i)) = std::move(common_cols.at(i));
    }
    return std::make_pair(std::make_unique<cudf::table>(std::move(probe_cols)),
                          std::make_unique<cudf::table>(std::move(build_noncommon_cols)));
  } else {
    std::vector<std::unique_ptr<column>> build_cols(build_noncommon_cols.size() +
                                                    common_cols.size());
    for (size_t i = 0; i < build_noncommon_cols.size(); ++i) {
      build_cols.at(build_noncommon_col_indices.at(i)) = std::move(build_noncommon_cols.at(i));
    }
    for (size_t i = 0; i < common_cols.size(); ++i) {
      build_cols.at(build_common_col_indices.at(i)) = std::move(common_cols.at(i));
    }
    return std::make_pair(std::make_unique<cudf::table>(std::move(probe_noncommon_cols)),
                          std::make_unique<cudf::table>(std::move(build_cols)));
  }
}

/**
 * @brief  Gathers rows from `probe` and `build` table and returns a (`probe`, `build`) table pair,
 * which contains the probe and build portions of the logical joined table respectively.
 *
 * @tparam JoinKind The type of join to be performed
 *
 * @param probe Probe side table
 * @param build build side table
 * @param joined_indices Pair of vectors containing row indices from which
 * `probe` and `build` tables are gathered. If any row index is out of bounds,
 * the contribution in the output `table` will be NULL.
 * @param columns_in_common is a vector of pairs of column indices
 * from tables `probe` and `build` respectively, that are "in common".
 * For "common" columns, only a single output column will be produced.
 * For an inner or left join, the result will be gathered from the column in
 * `probe`. For a full join, the result will be gathered from both common
 * columns in `probe` and `build` and concatenated to form a single column.
 * @param common_columns_output_side @see cudf::hash_join::common_columns_output_side.
 *
 * @return Table pair of (`probe`, `build`) containing the rows from `probe` and
 * `build` specified by `joined_indices`.
 * Columns in `columns_in_common` will be included in either `probe` or `build` portion as
 * `common_columns_output_side` indicates. Final form would look like
 * (`probe(including common columns)`, `build(excluding common columns)`) if
 * `common_columns_output_side` is `PROBE`, or (`probe(excluding common columns)`,
 * `build(including common columns)`) if `common_columns_output_side` is `BUILD`.
 */
template <join_kind JoinKind>
std::pair<std::unique_ptr<table>, std::unique_ptr<table>> construct_join_output_df(
  table_view const &probe,
  table_view const &build,
  VectorPair &joined_indices,
  std::vector<std::pair<size_type, size_type>> const &columns_in_common,
  cudf::hash_join::common_columns_output_side common_columns_output_side,
  rmm::mr::device_memory_resource *mr,
  cudaStream_t stream)
{
  std::vector<size_type> probe_common_col;
  probe_common_col.reserve(columns_in_common.size());
  std::vector<size_type> build_common_col;
  build_common_col.reserve(columns_in_common.size());
  for (const auto &c : columns_in_common) {
    probe_common_col.push_back(c.first);
    build_common_col.push_back(c.second);
  }
  std::vector<size_type> probe_noncommon_col =
    non_common_column_indices(probe.num_columns(), probe_common_col);
  std::vector<size_type> build_noncommon_col =
    non_common_column_indices(build.num_columns(), build_common_col);

  bool const nullify_out_of_bounds{JoinKind != join_kind::INNER_JOIN};

  std::unique_ptr<table> common_table = std::make_unique<table>();
  // Construct the joined columns
  if (join_kind::FULL_JOIN == JoinKind) {
    auto complement_indices = get_left_join_indices_complement(
      joined_indices.second, probe.num_rows(), build.num_rows(), stream);
    if (not columns_in_common.empty()) {
      auto common_from_build = detail::gather(build.select(build_common_col),
                                              complement_indices.second.begin(),
                                              complement_indices.second.end(),
                                              nullify_out_of_bounds,
                                              rmm::mr::get_current_device_resource(),
                                              stream);
      auto common_from_probe = detail::gather(probe.select(probe_common_col),
                                              joined_indices.first.begin(),
                                              joined_indices.first.end(),
                                              nullify_out_of_bounds,
                                              rmm::mr::get_current_device_resource(),
                                              stream);
      common_table           = cudf::detail::concatenate(
        {common_from_build->view(), common_from_probe->view()}, mr, stream);
    }
    joined_indices = concatenate_vector_pairs(complement_indices, joined_indices);
  } else {
    if (not columns_in_common.empty()) {
      common_table = detail::gather(probe.select(probe_common_col),
                                    joined_indices.first.begin(),
                                    joined_indices.first.end(),
                                    nullify_out_of_bounds,
                                    mr,
                                    stream);
    }
  }

  // Construct the probe non common columns
  std::unique_ptr<table> probe_table = detail::gather(probe.select(probe_noncommon_col),
                                                      joined_indices.first.begin(),
                                                      joined_indices.first.end(),
                                                      nullify_out_of_bounds,
                                                      mr,
                                                      stream);

  std::unique_ptr<table> build_table = detail::gather(build.select(build_noncommon_col),
                                                      joined_indices.second.begin(),
                                                      joined_indices.second.end(),
                                                      nullify_out_of_bounds,
                                                      mr,
                                                      stream);

  return combine_join_columns(probe_table->release(),
                              probe_noncommon_col,
                              probe_common_col,
                              build_table->release(),
                              build_noncommon_col,
                              build_common_col,
                              common_table->release(),
                              common_columns_output_side);
}

std::unique_ptr<cudf::table> combine_table_pair(std::unique_ptr<cudf::table> &&left,
                                                std::unique_ptr<cudf::table> &&right)
{
  auto joined_cols = left->release();
  auto right_cols  = right->release();
  joined_cols.insert(joined_cols.end(),
                     std::make_move_iterator(right_cols.begin()),
                     std::make_move_iterator(right_cols.end()));
  return std::make_unique<cudf::table>(std::move(joined_cols));
}

}  // namespace detail

hash_join::hash_join_impl::~hash_join_impl() = default;

hash_join::hash_join_impl::hash_join_impl(cudf::table_view const &build,
                                          std::vector<size_type> const &build_on,
                                          cudaStream_t stream)
  : _build(build),
    _build_selected(build.select(build_on)),
    _build_on(build_on),
    _hash_table(nullptr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != _build.num_columns(), "Hash join build table is empty");
  CUDF_EXPECTS(_build.num_rows() < cudf::detail::MAX_JOIN_SIZE,
               "Build column size is too big for hash join");

  if (_build_on.empty() || 0 == build.num_rows()) { return; }

  auto build_table = cudf::table_device_view::create(_build_selected, stream);
  _hash_table      = build_join_hash_table(*build_table, stream);
}

std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
hash_join::hash_join_impl::inner_join(
  cudf::table_view const &probe,
  std::vector<size_type> const &probe_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const &columns_in_common,
  common_columns_output_side common_columns_output_side,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource *mr,
  cudaStream_t stream) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join<cudf::detail::join_kind::INNER_JOIN>(
    probe, probe_on, columns_in_common, common_columns_output_side, compare_nulls, mr, stream);
}

std::unique_ptr<cudf::table> hash_join::hash_join_impl::left_join(
  cudf::table_view const &probe,
  std::vector<size_type> const &probe_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const &columns_in_common,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource *mr,
  cudaStream_t stream) const
{
  CUDF_FUNC_RANGE();
  auto probe_build_pair =
    compute_hash_join<cudf::detail::join_kind::LEFT_JOIN>(probe,
                                                          probe_on,
                                                          columns_in_common,
                                                          common_columns_output_side::PROBE,
                                                          compare_nulls,
                                                          mr,
                                                          stream);
  return cudf::detail::combine_table_pair(std::move(probe_build_pair.first),
                                          std::move(probe_build_pair.second));
}

std::unique_ptr<cudf::table> hash_join::hash_join_impl::full_join(
  cudf::table_view const &probe,
  std::vector<size_type> const &probe_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const &columns_in_common,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource *mr,
  cudaStream_t stream) const
{
  CUDF_FUNC_RANGE();
  auto probe_build_pair =
    compute_hash_join<cudf::detail::join_kind::FULL_JOIN>(probe,
                                                          probe_on,
                                                          columns_in_common,
                                                          common_columns_output_side::PROBE,
                                                          compare_nulls,
                                                          mr,
                                                          stream);
  return cudf::detail::combine_table_pair(std::move(probe_build_pair.first),
                                          std::move(probe_build_pair.second));
}

template <cudf::detail::join_kind JoinKind>
std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
hash_join::hash_join_impl::compute_hash_join(
  cudf::table_view const &probe,
  std::vector<size_type> const &probe_on,
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const &columns_in_common,
  common_columns_output_side common_columns_output_side,
  null_equality compare_nulls,
  rmm::mr::device_memory_resource *mr,
  cudaStream_t stream) const
{
  CUDF_EXPECTS(0 != probe.num_columns(), "Hash join probe table is empty");
  CUDF_EXPECTS(probe.num_rows() < cudf::detail::MAX_JOIN_SIZE,
               "Probe column size is too big for hash join");
  CUDF_EXPECTS(_build_on.size() == probe_on.size(),
               "Mismatch in number of columns to be joined on");

  CUDF_EXPECTS(std::all_of(columns_in_common.begin(),
                           columns_in_common.end(),
                           [this, &probe_on](auto pair) {
                             size_t p = std::find(probe_on.begin(), probe_on.end(), pair.first) -
                                        probe_on.begin();
                             size_t b = std::find(_build_on.begin(), _build_on.end(), pair.second) -
                                        _build_on.begin();
                             return (p != probe_on.size()) && (b != _build_on.size()) && (p == b);
                           }),
               "Invalid values passed to columns_in_common");

  if (is_trivial_join(probe, _build, probe_on, _build_on, JoinKind)) {
    return get_empty_joined_table(probe, _build, columns_in_common, common_columns_output_side);
  }

  auto probe_selected = probe.select(probe_on);
  CUDF_EXPECTS(std::equal(std::cbegin(_build_selected),
                          std::cend(_build_selected),
                          std::cbegin(probe_selected),
                          std::cend(probe_selected),
                          [](const auto &b, const auto &p) { return b.type() == p.type(); }),
               "Mismatch in joining column data types");

  constexpr cudf::detail::join_kind ProbeJoinKind = (JoinKind == cudf::detail::join_kind::FULL_JOIN)
                                                      ? cudf::detail::join_kind::LEFT_JOIN
                                                      : JoinKind;
  auto joined_indices = probe_join_indices<ProbeJoinKind>(probe_selected, compare_nulls, stream);
  return cudf::detail::construct_join_output_df<JoinKind>(
    probe, _build, joined_indices, columns_in_common, common_columns_output_side, mr, stream);
}

template <cudf::detail::join_kind JoinKind>
std::enable_if_t<JoinKind != cudf::detail::join_kind::FULL_JOIN,
                 std::pair<rmm::device_vector<size_type>, rmm::device_vector<size_type>>>
hash_join::hash_join_impl::probe_join_indices(cudf::table_view const &probe,
                                              null_equality compare_nulls,
                                              cudaStream_t stream) const
{
  // Trivial left join case - exit early
  if (!_hash_table && JoinKind == cudf::detail::join_kind::LEFT_JOIN) {
    return get_trivial_left_join_indices(probe, stream);
  }

  CUDF_EXPECTS(_hash_table, "Hash table of hash join is null.");

  auto build_table = cudf::table_device_view::create(_build_selected, stream);
  auto probe_table = cudf::table_device_view::create(probe, stream);
  return cudf::detail::probe_join_hash_table<JoinKind>(
    *build_table, *probe_table, *_hash_table, compare_nulls, stream);
}

}  // namespace cudf
