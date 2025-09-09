/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "join_common_utils.cuh"
#include "join_common_utils.hpp"
#include "mixed_join_common_utils.cuh"
#include "mixed_join_kernel.hpp"
#include "mixed_join_size_kernel.hpp"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/hashing/detail/helper_functions.cuh>
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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <optional>
#include <utility>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Precompute input pairs and hash indices for mixed join operations
 *
 * Precomputes input pairs and hash indices in a single pass to reduce code duplication
 * between mixed_join and compute_mixed_join_output_size functions.
 *
 * Precomputation reduces register pressure in probing kernels by avoiding expensive
 * on-the-fly calculations of iterator transforms and hash table indices.
 *
 * @tparam HashProbe Type of the device hasher for computing probe keys
 * @param hash_table Hash table for probing
 * @param hash_probe Device hasher for probe keys
 * @param probe_table_num_rows Number of rows in probe table
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Pair of device vectors: precomputed input pairs and hash indices
 */
template <typename HashProbe>
std::pair<rmm::device_uvector<cuco::pair<hash_value_type, size_type>>,
          rmm::device_uvector<cuda::std::pair<size_type, size_type>>>
precompute_mixed_join_data(mixed_join_hash_table_t const& hash_table,
                           HashProbe const& hash_probe,
                           size_type probe_table_num_rows,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto input_pairs =
    rmm::device_uvector<cuco::pair<hash_value_type, size_type>>(probe_table_num_rows, stream, mr);
  auto hash_indices =
    rmm::device_uvector<cuda::std::pair<size_type, size_type>>(probe_table_num_rows, stream, mr);

  auto const extent                        = hash_table.capacity();
  auto const probe_hash_fn                 = hash_table.hash_function();
  static constexpr std::size_t bucket_size = mixed_join_hash_table_t::bucket_size;

  // Functor to pre-compute both input pairs and initial slots and step sizes for double hashing.
  auto precompute_fn = [=] __device__(size_type i) {
    auto const probe_key = cuco::pair<hash_value_type, size_type>{hash_probe(i), i};

    // Use the probing scheme's hash functions for proper double hashing
    auto const hash1_val = cuda::std::get<0>(probe_hash_fn)(probe_key);
    auto const hash2_val = cuda::std::get<1>(probe_hash_fn)(probe_key);

    // Double hashing logic: initial position and step size
    auto const init_idx = (hash1_val % (extent / bucket_size)) * bucket_size;
    auto const step_val =
      ((hash2_val % (extent / bucket_size - std::size_t{1})) + std::size_t{1}) * bucket_size;

    return cuda::std::pair{
      probe_key,
      cuda::std::pair{static_cast<size_type>(init_idx), static_cast<size_type>(step_val)}};
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

}  // anonymous namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_join(table_view const& left_equality,
           table_view const& right_equality,
           table_view const& left_conditional,
           table_view const& right_conditional,
           ast::expression const& binary_predicate,
           null_equality compare_nulls,
           join_kind join_type,
           std::optional<std::size_t> const& output_size,
           rmm::cuda_stream_view stream,
           rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  CUDF_EXPECTS((join_type != join_kind::LEFT_SEMI_JOIN) && (join_type != join_kind::LEFT_ANTI_JOIN),
               "Left semi and anti joins should use mixed_join_semi.");

  auto const right_num_rows{right_conditional.num_rows()};
  auto const left_num_rows{left_conditional.num_rows()};
  auto const swap_tables = (join_type == join_kind::INNER_JOIN) && (right_num_rows > left_num_rows);

  // The "probe" table is the table we iterate over during the join operation.
  // For performance optimization, we choose the larger table as the probe table.
  // The kernels are launched with one thread per row of the probe table.
  auto const probe_table_num_rows{swap_tables ? right_num_rows : left_num_rows};

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

  // If evaluating the expression may produce null outputs we create a nullable
  // output column and follow the null-supporting expression evaluation code
  // path.
  auto const has_nulls = cudf::nullate::DYNAMIC{
    cudf::has_nulls(left_equality) || cudf::has_nulls(right_equality) ||
    binary_predicate.may_evaluate_null(left_conditional, right_conditional, stream)};

  auto const parser = ast::detail::expression_parser{
    binary_predicate, left_conditional, right_conditional, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.",
               cudf::data_type_error);

  // TODO: The non-conditional join impls start with a dictionary matching,
  // figure out what that is and what it's needed for (and if conditional joins
  // need to do the same).
  auto& probe     = swap_tables ? right_equality : left_equality;
  auto& build     = swap_tables ? left_equality : right_equality;
  auto probe_view = table_device_view::create(probe, stream);
  auto build_view = table_device_view::create(build, stream);

  mixed_join_hash_table_t hash_table{
    cuco::extent{static_cast<std::size_t>(build.num_rows())},
    cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
    cuco::empty_key{
      cuco::pair{std::numeric_limits<hash_value_type>::max(), cudf::detail::JoinNoneValue}},
    {},
    {},
    {},
    {},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream.value()},
    stream.value()};

  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  auto const row_bitmask =
    cudf::detail::bitmask_and(build, stream, cudf::get_current_device_resource_ref()).first;
  auto const preprocessed_build = detail::row::equality::preprocessed_table::create(build, stream);
  build_join_hash_table(build,
                        preprocessed_build,
                        hash_table,
                        has_nulls,
                        compare_nulls,
                        static_cast<bitmask_type const*>(row_bitmask.data()),
                        stream);
  auto hash_table_storage = cudf::device_span<cuco::pair<hash_value_type, size_type>>{
    hash_table.data(), hash_table.capacity()};

  auto left_conditional_view  = table_device_view::create(left_conditional, stream);
  auto right_conditional_view = table_device_view::create(right_conditional, stream);

  // For inner joins we support optimizing the join by launching one thread for
  // whichever table is larger rather than always using the left table.
  detail::grid_1d const config(probe_table_num_rows, DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;
  join_kind const kernel_join_type =
    join_type == join_kind::FULL_JOIN ? join_kind::LEFT_JOIN : join_type;

  // If the join size was not provided as an input, compute it here.
  std::size_t join_size;

  auto const preprocessed_probe = detail::row::equality::preprocessed_table::create(probe, stream);
  auto const row_hash           = cudf::detail::row::hash::row_hasher{preprocessed_probe};
  auto const hash_probe         = row_hash.device_hasher(has_nulls);
  auto const row_comparator =
    cudf::detail::row::equality::two_table_comparator{preprocessed_probe, preprocessed_build};
  auto const equality_probe = row_comparator.equal_to<false>(has_nulls, compare_nulls);

  auto [input_pairs, hash_indices] =
    precompute_mixed_join_data(hash_table, hash_probe, probe_table_num_rows, stream, mr);

  if (output_size.has_value()) {
    join_size = output_size.value();
  } else {
    if (has_nulls) {
      join_size = launch_compute_mixed_join_output_size<true>(*left_conditional_view,
                                                              *right_conditional_view,
                                                              kernel_join_type,
                                                              equality_probe,
                                                              hash_table_storage,
                                                              input_pairs.data(),
                                                              hash_indices.data(),
                                                              parser.device_expression_data,
                                                              swap_tables,
                                                              config,
                                                              shmem_size_per_block,
                                                              stream);
    } else {
      join_size = launch_compute_mixed_join_output_size<false>(*left_conditional_view,
                                                               *right_conditional_view,
                                                               kernel_join_type,
                                                               equality_probe,
                                                               hash_table_storage,
                                                               input_pairs.data(),
                                                               hash_indices.data(),
                                                               parser.device_expression_data,
                                                               swap_tables,
                                                               config,
                                                               shmem_size_per_block,
                                                               stream);
    }
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

  if (has_nulls) {
    launch_mixed_join<true>(*left_conditional_view,
                            *right_conditional_view,
                            kernel_join_type,
                            equality_probe,
                            hash_table_storage,
                            input_pairs.data(),
                            hash_indices.data(),
                            join_output_l,
                            join_output_r,
                            parser.device_expression_data,
                            swap_tables,
                            config,
                            shmem_size_per_block,
                            stream);
  } else {
    launch_mixed_join<false>(*left_conditional_view,
                             *right_conditional_view,
                             kernel_join_type,
                             equality_probe,
                             hash_table_storage,
                             input_pairs.data(),
                             hash_indices.data(),
                             join_output_l,
                             join_output_r,
                             parser.device_expression_data,
                             swap_tables,
                             config,
                             shmem_size_per_block,
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

std::size_t compute_mixed_join_output_size(table_view const& left_equality,
                                           table_view const& right_equality,
                                           table_view const& left_conditional,
                                           table_view const& right_conditional,
                                           ast::expression const& binary_predicate,
                                           null_equality compare_nulls,
                                           join_kind join_type,
                                           rmm::cuda_stream_view stream)
{
  // Until we add logic to handle the number of non-matches in the right table,
  // full joins are not supported in this function. Note that this does not
  // prevent actually performing full joins since we do that by calculating the
  // left join and then concatenating the complementary right indices.
  CUDF_EXPECTS(join_type != join_kind::FULL_JOIN,
               "Size estimation is not available for full joins.");

  CUDF_EXPECTS(
    (join_type != join_kind::LEFT_SEMI_JOIN) && (join_type != join_kind::LEFT_ANTI_JOIN),
    "Left semi and anti join size estimation should use compute_mixed_join_output_size_semi.");

  CUDF_EXPECTS(left_conditional.num_rows() == left_equality.num_rows(),
               "The left conditional and equality tables must have the same number of rows.");
  CUDF_EXPECTS(right_conditional.num_rows() == right_equality.num_rows(),
               "The right conditional and equality tables must have the same number of rows.");

  auto const right_num_rows{right_conditional.num_rows()};
  auto const left_num_rows{left_conditional.num_rows()};
  auto const swap_tables = (join_type == join_kind::INNER_JOIN) && (right_num_rows > left_num_rows);

  // The "probe" table is the table we iterate over during the join operation.
  // For performance optimization, we choose the larger table as the probe table.
  // The kernels are launched with one thread per row of the probe table.
  auto const probe_table_num_rows{swap_tables ? right_num_rows : left_num_rows};

  // We can immediately filter out cases where one table is empty.
  if (right_num_rows == 0) {
    switch (join_type) {
      // Left joins return all the row indices from left with a corresponding NULL from the right.
      case join_kind::LEFT_JOIN: {
        return left_num_rows;
      }
      // Inner joins return empty output because no matches can exist.
      case join_kind::INNER_JOIN: {
        return 0;
      }
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  } else if (left_num_rows == 0) {
    switch (join_type) {
      // Left and inner joins all return empty sets.
      case join_kind::LEFT_JOIN:
      case join_kind::INNER_JOIN: {
        return 0;
      }
      default: CUDF_FAIL("Invalid join kind."); break;
    }
  }

  auto mr = cudf::get_current_device_resource_ref();

  // If evaluating the expression may produce null outputs we create a nullable
  // output column and follow the null-supporting expression evaluation code
  // path.
  auto const has_nulls = cudf::nullate::DYNAMIC{
    cudf::has_nulls(left_equality) || cudf::has_nulls(right_equality) ||
    binary_predicate.may_evaluate_null(left_conditional, right_conditional, stream)};

  auto const parser = ast::detail::expression_parser{
    binary_predicate, left_conditional, right_conditional, has_nulls, stream, mr};
  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.",
               cudf::data_type_error);

  // TODO: The non-conditional join impls start with a dictionary matching,
  // figure out what that is and what it's needed for (and if conditional joins
  // need to do the same).
  auto& probe     = swap_tables ? right_equality : left_equality;
  auto& build     = swap_tables ? left_equality : right_equality;
  auto probe_view = table_device_view::create(probe, stream);
  auto build_view = table_device_view::create(build, stream);

  mixed_join_hash_table_t hash_table{
    cuco::extent{static_cast<size_t>(build.num_rows())},
    cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
    cuco::empty_key{
      cuco::pair{std::numeric_limits<hash_value_type>::max(), cudf::detail::JoinNoneValue}},
    {},
    {},
    {},
    {},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream.value()},
    stream.value()};

  // TODO: To add support for nested columns we will need to flatten in many
  // places. However, this probably isn't worth adding any time soon since we
  // won't be able to support AST conditions for those types anyway.
  auto const row_bitmask        = cudf::detail::bitmask_and(build, stream, mr).first;
  auto const preprocessed_build = detail::row::equality::preprocessed_table::create(build, stream);
  build_join_hash_table(build,
                        preprocessed_build,
                        hash_table,
                        has_nulls,
                        compare_nulls,
                        static_cast<bitmask_type const*>(row_bitmask.data()),
                        stream);
  auto hash_table_storage = cudf::device_span<cuco::pair<hash_value_type, size_type>>{
    hash_table.data(), hash_table.capacity()};

  auto left_conditional_view  = table_device_view::create(left_conditional, stream);
  auto right_conditional_view = table_device_view::create(right_conditional, stream);

  // For inner joins we support optimizing the join by launching one thread for
  // whichever table is larger rather than always using the left table.
  detail::grid_1d const config(probe_table_num_rows, DEFAULT_JOIN_BLOCK_SIZE);
  auto const shmem_size_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  auto const preprocessed_probe = detail::row::equality::preprocessed_table::create(probe, stream);
  auto const row_hash           = cudf::detail::row::hash::row_hasher{preprocessed_probe};
  auto const hash_probe         = row_hash.device_hasher(has_nulls);
  auto const row_comparator =
    cudf::detail::row::equality::two_table_comparator{preprocessed_probe, preprocessed_build};
  auto const equality_probe = row_comparator.equal_to<false>(has_nulls, compare_nulls);

  // Precompute input pairs and hash indices using common utility function
  auto [input_pairs, hash_indices] =
    precompute_mixed_join_data(hash_table, hash_probe, probe_table_num_rows, stream, mr);

  // Determine number of output rows without actually building the output to simply
  // find what the size of the output will be.
  std::size_t const size = [&]() {
    if (has_nulls) {
      return launch_compute_mixed_join_output_size<true>(*left_conditional_view,
                                                         *right_conditional_view,
                                                         join_type,
                                                         equality_probe,
                                                         hash_table_storage,
                                                         input_pairs.data(),
                                                         hash_indices.data(),
                                                         parser.device_expression_data,
                                                         swap_tables,
                                                         config,
                                                         shmem_size_per_block,
                                                         stream);
    } else {
      return launch_compute_mixed_join_output_size<false>(*left_conditional_view,
                                                          *right_conditional_view,
                                                          join_type,
                                                          equality_probe,
                                                          hash_table_storage,
                                                          input_pairs.data(),
                                                          hash_indices.data(),
                                                          parser.device_expression_data,
                                                          swap_tables,
                                                          config,
                                                          shmem_size_per_block,
                                                          stream);
    }
  }();

  return size;
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_inner_join(table_view const& left_equality,
                 table_view const& right_equality,
                 table_view const& left_conditional,
                 table_view const& right_conditional,
                 ast::expression const& binary_predicate,
                 null_equality compare_nulls,
                 std::optional<std::size_t> const output_size,
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
                            output_size,
                            stream,
                            mr);
}

std::size_t mixed_inner_join_size(table_view const& left_equality,
                                  table_view const& right_equality,
                                  table_view const& left_conditional,
                                  table_view const& right_conditional,
                                  ast::expression const& binary_predicate,
                                  null_equality compare_nulls,
                                  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::compute_mixed_join_output_size(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                detail::join_kind::INNER_JOIN,
                                                stream);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_left_join(table_view const& left_equality,
                table_view const& right_equality,
                table_view const& left_conditional,
                table_view const& right_conditional,
                ast::expression const& binary_predicate,
                null_equality compare_nulls,
                std::optional<std::size_t> const output_size,
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
                            output_size,
                            stream,
                            mr);
}

std::size_t mixed_left_join_size(table_view const& left_equality,
                                 table_view const& right_equality,
                                 table_view const& left_conditional,
                                 table_view const& right_conditional,
                                 ast::expression const& binary_predicate,
                                 null_equality compare_nulls,
                                 rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::compute_mixed_join_output_size(left_equality,
                                                right_equality,
                                                left_conditional,
                                                right_conditional,
                                                binary_predicate,
                                                compare_nulls,
                                                detail::join_kind::LEFT_JOIN,
                                                stream);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_full_join(table_view const& left_equality,
                table_view const& right_equality,
                table_view const& left_conditional,
                table_view const& right_conditional,
                ast::expression const& binary_predicate,
                null_equality compare_nulls,
                std::optional<std::size_t> const output_size,
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
                            output_size,
                            stream,
                            mr);
}

}  // namespace cudf
