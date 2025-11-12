/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"
#include "join_common_utils.hpp"
#include "mixed_join_kernel.hpp"
#include "mixed_join_size_kernel.hpp"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/mixed_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/scan.h>

#include <optional>
#include <utility>

namespace cudf {
namespace detail {

namespace {
/**
 * @brief Precomputes double hashing indices and row hash values for mixed join operations.
 *
 * This function exists as a performance optimization to work around the register spilling issue
 * reported in https://github.com/NVIDIA/cuCollections/issues/761. The new cuco hash table
 * implementation suffers from register spilling due to longer register live ranges, which can
 * cause up to 20x performance degradation.
 *
 * By precomputing the double hashing indices (initial slot and step size) and row hash values
 * in a separate pass, we reduce register pressure in the subsequent count and retrieve kernels.
 * This approach yields approximately 20% speedup compared to the legacy multimap-based
 * implementation.
 *
 * The tradeoff is that we cannot use cuco's device APIs directly in mixed join operations.
 * Instead, we must reimplement the entire double hashing probing logic in cudf without relying
 * on cuco's device APIs. This should be revisited and potentially removed once issue #761 is
 * fully resolved.
 *
 * @param hash_table The cuco multiset hash table
 * @param hash_probe Hash function for computing row hashes
 * @param probe_table_num_rows Number of rows in the probe table
 * @param stream CUDA stream for operations
 * @param mr Memory resource for allocations
 * @return A pair of device vectors: (input_pairs, hash_indices) where input_pairs contains
 *         (row_hash, row_index) pairs and hash_indices contains (initial_slot, step_size) pairs
 */
template <typename HashProbe>
std::pair<rmm::device_uvector<cuco::pair<hash_value_type, size_type>>,
          rmm::device_uvector<cuda::std::pair<hash_value_type, hash_value_type>>>
precompute_mixed_join_data(mixed_multiset_type const& hash_table,
                           HashProbe const& hash_probe,
                           size_type probe_table_num_rows,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto input_pairs =
    rmm::device_uvector<cuco::pair<hash_value_type, size_type>>(probe_table_num_rows, stream, mr);
  auto hash_indices = rmm::device_uvector<cuda::std::pair<hash_value_type, hash_value_type>>(
    probe_table_num_rows, stream, mr);

  auto const capacity                      = hash_table.capacity();
  auto const probe_hash_fn                 = hash_table.hash_function();
  static constexpr std::size_t bucket_size = mixed_multiset_type::bucket_size;

  auto const num_buckets           = capacity / bucket_size;
  auto const num_buckets_minus_one = num_buckets - 1;

  // Functor to pre-compute both input pairs and initial slots and step sizes for double hashing.
  auto precompute_fn = [=] __device__(size_type i) {
    auto const probe_key = cuco::pair<hash_value_type, size_type>{hash_probe(i), i};

    // Use the probing scheme's hash functions for proper double hashing
    auto const hash1_val = cuda::std::get<0>(probe_hash_fn)(probe_key);
    auto const hash2_val = cuda::std::get<1>(probe_hash_fn)(probe_key);

    auto const init_idx = static_cast<hash_value_type>(
      (static_cast<std::size_t>(hash1_val) % num_buckets) * bucket_size);
    auto const step_val = static_cast<hash_value_type>(
      ((static_cast<std::size_t>(hash2_val) % num_buckets_minus_one) + 1) * bucket_size);

    return cuda::std::pair{probe_key, cuda::std::pair{init_idx, step_val}};
  };

  // Single transform to fill both arrays using zip iterator
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    thrust::counting_iterator<size_type>(0),
    thrust::counting_iterator<size_type>(probe_table_num_rows),
    thrust::make_zip_iterator(thrust::make_tuple(input_pairs.begin(), hash_indices.begin())),
    precompute_fn);

  return std::make_pair(std::move(input_pairs), std::move(hash_indices));
}

struct mixed_join_setup_data {
  bool swap_tables;
  size_type outer_num_rows;
  cudf::nullate::DYNAMIC has_nulls;
  ast::detail::expression_parser parser;
  mixed_multiset_type hash_table;
  std::shared_ptr<detail::row::equality::preprocessed_table> preprocessed_build;
  std::shared_ptr<detail::row::equality::preprocessed_table> preprocessed_probe;
  std::unique_ptr<table_device_view, std::function<void(table_device_view*)>> left_conditional_view;
  std::unique_ptr<table_device_view, std::function<void(table_device_view*)>>
    right_conditional_view;
  detail::grid_1d config;
  thread_index_type shmem_size_per_block;
  row_equality equality_probe;
  cudf::device_span<cuco::pair<hash_value_type, size_type>> hash_table_storage;
  rmm::device_uvector<cuco::pair<hash_value_type, size_type>> input_pairs;
  rmm::device_uvector<cuda::std::pair<hash_value_type, hash_value_type>> hash_indices;
};

mixed_join_setup_data setup_mixed_join_common(table_view const& left_equality,
                                              table_view const& right_equality,
                                              table_view const& left_conditional,
                                              table_view const& right_conditional,
                                              ast::expression const& binary_predicate,
                                              null_equality compare_nulls,
                                              join_kind join_type,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  auto const right_num_rows = right_conditional.num_rows();
  auto const left_num_rows  = left_conditional.num_rows();
  auto const swap_tables = (join_type == join_kind::INNER_JOIN) && (right_num_rows > left_num_rows);
  auto const outer_num_rows = swap_tables ? right_num_rows : left_num_rows;

  // If evaluating the expression may produce null outputs we create a nullable
  // output column and follow the null-supporting expression evaluation code path.
  auto const has_nulls = cudf::nullate::DYNAMIC{
    cudf::has_nulls(left_equality) || cudf::has_nulls(right_equality) ||
    binary_predicate.may_evaluate_null(left_conditional, right_conditional, stream)};

  auto parser = ast::detail::expression_parser{
    binary_predicate, left_conditional, right_conditional, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.",
               cudf::data_type_error);

  // TODO: The non-conditional join impls start with a dictionary matching,
  // figure out what that is and what it's needed for (and if conditional joins
  // need to do the same).
  auto& probe = swap_tables ? right_equality : left_equality;
  auto& build = swap_tables ? left_equality : right_equality;

  // Create hash table with load factor following hash join pattern
  mixed_multiset_type hash_table{
    cuco::extent{static_cast<std::size_t>(build.num_rows())},
    cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
    cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(), cudf::JoinNoMatch}},
    {},
    {},
    {},
    {},
    rmm::mr::polymorphic_allocator<char>{},
    stream.value()};

  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  auto const row_bitmask =
    cudf::detail::bitmask_and(build, stream, cudf::get_current_device_resource_ref()).first;
  auto preprocessed_build = detail::row::equality::preprocessed_table::create(build, stream);
  build_join_hash_table(build,
                        preprocessed_build,
                        hash_table,
                        has_nulls,
                        compare_nulls,
                        static_cast<bitmask_type const*>(row_bitmask.data()),
                        stream);

  auto left_conditional_view  = table_device_view::create(left_conditional, stream);
  auto right_conditional_view = table_device_view::create(right_conditional, stream);

  // For inner joins we support optimizing the join by launching one thread for
  // whichever table is larger rather than always using the left table.
  detail::grid_1d const config(outer_num_rows, DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  auto preprocessed_probe = detail::row::equality::preprocessed_table::create(probe, stream);
  auto const row_hash     = cudf::detail::row::hash::row_hasher{preprocessed_probe};
  auto const hash_probe   = row_hash.device_hasher(has_nulls);
  auto const row_comparator =
    cudf::detail::row::equality::two_table_comparator{preprocessed_probe, preprocessed_build};
  auto const equality_probe = row_comparator.equal_to<false>(has_nulls, compare_nulls);

  // Precompute hash table storage and input data
  auto hash_table_storage = cudf::device_span<cuco::pair<hash_value_type, size_type>>{
    hash_table.data(), hash_table.capacity()};
  CUDF_EXPECTS(reinterpret_cast<std::uintptr_t>(hash_table_storage.data()) %
                   (2 * sizeof(cuco::pair<hash_value_type, size_type>)) ==
                 0,
               "Hash table storage must be aligned to 2-element boundary");
  auto [input_pairs, hash_indices] =
    precompute_mixed_join_data(hash_table, hash_probe, outer_num_rows, stream, mr);

  return {swap_tables,
          outer_num_rows,
          has_nulls,
          std::move(parser),
          std::move(hash_table),
          std::move(preprocessed_build),
          std::move(preprocessed_probe),
          std::move(left_conditional_view),
          std::move(right_conditional_view),
          config,
          shmem_size_per_block,
          equality_probe,
          hash_table_storage,
          std::move(input_pairs),
          std::move(hash_indices)};
}

/**
 * @brief Helper function to compute the output size for mixed joins by launching count kernels.
 *
 * This function encapsulates the common logic needed by both mixed_join and
 * compute_mixed_join_output_size to count the number of matches per row.
 */
std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>>
compute_mixed_join_matches_per_row(
  cudf::nullate::DYNAMIC has_nulls,
  table_device_view const& left_conditional_view,
  table_device_view const& right_conditional_view,
  bool is_outer_join,
  bool swap_tables,
  row_equality const& equality_probe,
  cudf::device_span<cuco::pair<hash_value_type, size_type>> hash_table_storage,
  cuco::pair<hash_value_type, size_type> const* input_pairs,
  cuda::std::pair<hash_value_type, hash_value_type> const* hash_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  size_type outer_num_rows,
  detail::grid_1d config,
  thread_index_type shmem_size_per_block,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto matches_per_row = std::make_unique<rmm::device_uvector<size_type>>(
    static_cast<std::size_t>(outer_num_rows), stream, mr);
  auto matches_per_row_span = cudf::device_span<size_type>{
    matches_per_row->begin(), static_cast<std::size_t>(outer_num_rows)};

  if (has_nulls) {
    launch_mixed_join_count<true>(left_conditional_view,
                                  right_conditional_view,
                                  is_outer_join,
                                  swap_tables,
                                  equality_probe,
                                  hash_table_storage,
                                  input_pairs,
                                  hash_indices,
                                  device_expression_data,
                                  matches_per_row_span,
                                  config,
                                  shmem_size_per_block,
                                  stream);
  } else {
    launch_mixed_join_count<false>(left_conditional_view,
                                   right_conditional_view,
                                   is_outer_join,
                                   swap_tables,
                                   equality_probe,
                                   hash_table_storage,
                                   input_pairs,
                                   hash_indices,
                                   device_expression_data,
                                   matches_per_row_span,
                                   config,
                                   shmem_size_per_block,
                                   stream);
  }

  std::size_t const size = thrust::reduce(rmm::exec_policy_nosync(stream),
                                          matches_per_row_span.begin(),
                                          matches_per_row_span.end(),
                                          std::size_t{0});

  return {size, std::move(matches_per_row)};
}
}  // anonymous namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  join_kind join_type,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> const& output_size_data,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS((join_type != join_kind::LEFT_SEMI_JOIN) && (join_type != join_kind::LEFT_ANTI_JOIN),
               "Left semi and anti joins should use mixed_join_semi.");

  auto const right_num_rows = right_conditional.num_rows();
  auto const left_num_rows  = left_conditional.num_rows();

  // We can immediately filter out cases where the right table is empty. In
  // some cases, we return all the rows of the left table with a corresponding
  // null index for the right table; in others, we return an empty output.
  if (right_num_rows == 0) {
    switch (join_type) {
      // Left and full joins all return all the row indices from
      // left with a corresponding NULL from the right.
      case join_kind::LEFT_JOIN:
      case join_kind::FULL_JOIN: return get_trivial_left_join_indices(left_conditional, stream, mr);
      // Inner joins return empty output because no matches can exist.
      case join_kind::INNER_JOIN:
        return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                         std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left_num_rows == 0) {
    switch (join_type) {
      // Left and inner joins all return empty sets.
      case join_kind::LEFT_JOIN:
      case join_kind::INNER_JOIN:
        return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                         std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
      // Full joins need to return the trivial complement.
      case join_kind::FULL_JOIN: {
        auto ret_flipped = get_trivial_left_join_indices(right_conditional, stream, mr);
        return std::pair(std::move(ret_flipped.second), std::move(ret_flipped.first));
      }
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  }

  auto setup = setup_mixed_join_common(left_equality,
                                       right_equality,
                                       left_conditional,
                                       right_conditional,
                                       binary_predicate,
                                       compare_nulls,
                                       join_type,
                                       stream,
                                       mr);

  bool const is_outer_join =
    (join_type == join_kind::LEFT_JOIN || join_type == join_kind::FULL_JOIN);

  // If the join size data was not provided as an input, compute it here.
  std::size_t join_size = 0;
  // Using an optional because we only need to allocate a new vector if one was
  // not passed as input, and rmm::device_uvector is not default constructible
  std::optional<rmm::device_uvector<size_type>> matches_per_row{};
  device_span<size_type const> matches_per_row_span{};

  if (output_size_data.has_value()) {
    join_size            = output_size_data->first;
    matches_per_row_span = output_size_data->second;
  } else {
    auto [size, matches] = compute_mixed_join_matches_per_row(setup.has_nulls,
                                                              *setup.left_conditional_view,
                                                              *setup.right_conditional_view,
                                                              is_outer_join,
                                                              setup.swap_tables,
                                                              setup.equality_probe,
                                                              setup.hash_table_storage,
                                                              setup.input_pairs.data(),
                                                              setup.hash_indices.data(),
                                                              setup.parser.device_expression_data,
                                                              setup.outer_num_rows,
                                                              setup.config,
                                                              setup.shmem_size_per_block,
                                                              stream,
                                                              mr);
    join_size            = size;
    matches_per_row      = std::move(*matches);
    matches_per_row_span = cudf::device_span<size_type const>{
      matches_per_row->begin(), static_cast<std::size_t>(setup.outer_num_rows)};
  }

  // Given the number of matches per row, we need to compute the offsets for insertion.
  auto join_result_offsets =
    rmm::device_uvector<size_type>{static_cast<std::size_t>(setup.outer_num_rows), stream, mr};
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                         matches_per_row_span.begin(),
                         matches_per_row_span.end(),
                         join_result_offsets.begin());

  // Get total count from scan result: last offset + last matches_per_row
  if (setup.outer_num_rows > 0 && !output_size_data.has_value()) {
    auto const last_offset  = join_result_offsets.element(setup.outer_num_rows - 1, stream);
    auto const last_matches = matches_per_row->element(setup.outer_num_rows - 1, stream);
    join_size               = last_offset + last_matches;
  }

  // The initial early exit clauses guarantee that we will not reach this point
  // unless both the left and right tables are non-empty. Under that
  // constraint, neither left nor full joins can return an empty result since
  // at minimum we are guaranteed null matches for all non-matching rows. In
  // all other cases (inner, left semi, and left anti joins) if we reach this
  // point we can safely return an empty result.
  if (join_size == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const& join_output_l = left_indices->data();
  auto const& join_output_r = right_indices->data();

  if (setup.has_nulls) {
    launch_mixed_join<true>(*setup.left_conditional_view,
                            *setup.right_conditional_view,
                            is_outer_join,
                            setup.swap_tables,
                            setup.equality_probe,
                            setup.hash_table_storage,
                            setup.input_pairs.data(),
                            setup.hash_indices.data(),
                            setup.parser.device_expression_data,
                            join_output_l,
                            join_output_r,
                            join_result_offsets.data(),
                            setup.config,
                            setup.shmem_size_per_block,
                            stream);
  } else {
    launch_mixed_join<false>(*setup.left_conditional_view,
                             *setup.right_conditional_view,
                             is_outer_join,
                             setup.swap_tables,
                             setup.equality_probe,
                             setup.hash_table_storage,
                             setup.input_pairs.data(),
                             setup.hash_indices.data(),
                             setup.parser.device_expression_data,
                             join_output_l,
                             join_output_r,
                             join_result_offsets.data(),
                             setup.config,
                             setup.shmem_size_per_block,
                             stream);
  }

  auto join_indices = std::pair(std::move(left_indices), std::move(right_indices));

  // For full joins, get the indices in the right table that were not joined to
  // by any row in the left table.
  if (join_type == join_kind::FULL_JOIN) {
    auto complement_indices = detail::get_left_join_indices_complement(
      join_indices.second, left_num_rows, right_num_rows, stream, mr);
    join_indices = detail::concatenate_vector_pairs(join_indices, complement_indices, stream);
  }
  return join_indices;
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>>
compute_mixed_join_output_size(table_view const& left_equality,
                               table_view const& right_equality,
                               table_view const& left_conditional,
                               table_view const& right_conditional,
                               ast::expression const& binary_predicate,
                               null_equality compare_nulls,
                               join_kind join_type,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(join_type != join_kind::FULL_JOIN,
               "Size estimation is not available for full joins.");

  CUDF_EXPECTS(
    (join_type != join_kind::LEFT_SEMI_JOIN) && (join_type != join_kind::LEFT_ANTI_JOIN),
    "Left semi and anti join size estimation should use compute_mixed_join_output_size_semi.");

  auto const right_num_rows = right_conditional.num_rows();
  auto const left_num_rows  = left_conditional.num_rows();

  // Handle empty table cases early
  if (right_num_rows == 0 || left_num_rows == 0) {
    auto const outer_num_rows =
      ((join_type == join_kind::INNER_JOIN) && (right_num_rows > left_num_rows)) ? right_num_rows
                                                                                 : left_num_rows;
    auto matches_per_row = std::make_unique<rmm::device_uvector<size_type>>(
      static_cast<std::size_t>(outer_num_rows), stream, mr);
    auto matches_per_row_span = cudf::device_span<size_type>{
      matches_per_row->begin(), static_cast<std::size_t>(outer_num_rows)};

    if (right_num_rows == 0 && join_type == join_kind::LEFT_JOIN) {
      thrust::fill(rmm::exec_policy_nosync(stream),
                   matches_per_row_span.begin(),
                   matches_per_row_span.end(),
                   1);
      return {left_num_rows, std::move(matches_per_row)};
    } else {
      thrust::fill(rmm::exec_policy_nosync(stream),
                   matches_per_row_span.begin(),
                   matches_per_row_span.end(),
                   0);
      return {0, std::move(matches_per_row)};
    }
  }

  auto setup = setup_mixed_join_common(left_equality,
                                       right_equality,
                                       left_conditional,
                                       right_conditional,
                                       binary_predicate,
                                       compare_nulls,
                                       join_type,
                                       stream,
                                       mr);

  bool const is_outer_join = (join_type == join_kind::LEFT_JOIN);

  // Use the helper function to compute matches per row
  return compute_mixed_join_matches_per_row(setup.has_nulls,
                                            *setup.left_conditional_view,
                                            *setup.right_conditional_view,
                                            is_outer_join,
                                            setup.swap_tables,
                                            setup.equality_probe,
                                            setup.hash_table_storage,
                                            setup.input_pairs.data(),
                                            setup.hash_indices.data(),
                                            setup.parser.device_expression_data,
                                            setup.outer_num_rows,
                                            setup.config,
                                            setup.shmem_size_per_block,
                                            stream,
                                            mr);
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_inner_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> const output_size_data,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join(left_equality,
                            right_equality,
                            left_conditional,
                            right_conditional,
                            binary_predicate,
                            compare_nulls,
                            detail::join_kind::INNER_JOIN,
                            output_size_data,
                            stream,
                            mr);
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_inner_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_mixed_join_output_size(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                detail::join_kind::INNER_JOIN,
                                                stream,
                                                mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_left_join(table_view const& left_equality,
                table_view const& right_equality,
                table_view const& left_conditional,
                table_view const& right_conditional,
                ast::expression const& binary_predicate,
                null_equality compare_nulls,
                output_size_data_type const output_size_data,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join(left_equality,
                            right_equality,
                            left_conditional,
                            right_conditional,
                            binary_predicate,
                            compare_nulls,
                            detail::join_kind::LEFT_JOIN,
                            output_size_data,
                            stream,
                            mr);
}

std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_left_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_mixed_join_output_size(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                detail::join_kind::LEFT_JOIN,
                                                stream,
                                                mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_full_join(table_view const& left_equality,
                table_view const& right_equality,
                table_view const& left_conditional,
                table_view const& right_conditional,
                ast::expression const& binary_predicate,
                null_equality compare_nulls,
                output_size_data_type const output_size_data,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mixed_join(left_equality,
                            right_equality,
                            left_conditional,
                            right_conditional,
                            binary_predicate,
                            compare_nulls,
                            detail::join_kind::FULL_JOIN,
                            output_size_data,
                            stream,
                            mr);
}

}  // namespace cudf
