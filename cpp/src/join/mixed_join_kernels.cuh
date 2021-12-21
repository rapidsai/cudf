/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#pragma once

#include <join/join_common_utils.cuh>
#include <join/join_common_utils.hpp>

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cooperative_groups.h>

#include <cub/cub.cuh>
#include <cuco/detail/pair.cuh>
#include <thrust/equal.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cudf {
namespace detail {
namespace cg = cooperative_groups;

/**
 * @brief Device functor to determine if two pairs are identical.
 *
 * This equality comparator is designed for use with cuco::static_multimap's
 * pair* APIs, which will compare equality based on comparing (key, value)
 * pairs. In the context of joins, these pairs are of the form
 * (row_hash, row_id). A hash probe hit indicates that hash of a probe row's hash is
 * equal to the hash of the hash of some row in the multimap, at which point we need an
 * equality comparator that will check whether the contents of the rows are
 * identical. This comparator does so by verifying key equality (i.e. that
 * probe_row_hash == build_row_hash) and then using a row_equality_comparator
 * to compare the contents of the row indices that are stored as the payload in
 * the hash map.
 *
 * This particular comparator is a specialized version of the pair_equality used in hash joins. This
 * version also checks the expression_evaluator.
 */
template <bool has_nulls>
class pair_expression_equality {
 public:
  __device__ pair_expression_equality(
    table_device_view build,
    table_device_view probe,
    bool swap_tables,
    cudf::ast::detail::expression_evaluator<has_nulls> evaluator,
    cudf::ast::detail::IntermediateDataType<has_nulls>* thread_intermediate_storage,
    null_equality nulls_are_equal = null_equality::EQUAL)
    : build(build),
      probe{probe},
      swap_tables{swap_tables},
      nulls_are_equal{nulls_are_equal},
      evaluator{evaluator},
      thread_intermediate_storage{thread_intermediate_storage}
  {
  }

  // The parameters are build/probe rather than left/right because the operator
  // is called by cuco's kernels with parameters in this order (note that this
  // is an implementation detail that we should eventually stop relying on by
  // defining operators with suitable heterogeneous typing). Rather than
  // converting to left/right semantics, we can operate directly on build/probe
  // until we get to the expression evaluator, which needs to convert back to
  // left/right semantics because the conditional expression need not be
  // commutative.
  __device__ __forceinline__ bool operator()(const pair_type& build_row,
                                             const pair_type& probe_row) const noexcept
  {
    // TODO: I've inlined the logic from cudf's row_equality_comparator because
    // that comparator's constructor is not visible on device, and changing it
    // would require removing an assertion-raising test. I don't think we want
    // to do that, but should verify before finalizing.
    auto equal_elements = [=](column_device_view b, column_device_view p) {
      // Note: we could use nullate::DYNAMIC to avoid the extra template
      // instantiation, but in this case the performance impact of making that
      // decision at runtime is substantial since this functor is used within
      // a complex kernel.
      auto has_nulls_nullate = []() {
        if constexpr (has_nulls) {
          return nullate::YES{};
        } else if constexpr (!has_nulls) {
          return nullate::NO{};
        }
      }();
      return cudf::type_dispatcher(
        b.type(),
        element_equality_comparator{has_nulls_nullate, b, p, nulls_are_equal},
        build_row.second,
        probe_row.second);
    };

    auto output_dest = cudf::ast::detail::value_expression_result<bool, has_nulls>();
    // Three levels of checks:
    // 1. Row hashes of the columns involved in the equality condition are equal.
    // 2. The contents of the columns involved in the equality condition are equal.
    // 3. The predicate evaluated on the relevant columns (already encoded in the evaluator)
    // evaluates to true.
    if ((probe_row.first == build_row.first) &&
        (thrust::equal(thrust::seq, build.begin(), build.end(), probe.begin(), equal_elements))) {
      auto const lrow_idx = swap_tables ? build_row.second : probe_row.second;
      auto const rrow_idx = swap_tables ? probe_row.second : build_row.second;
      evaluator.evaluate(output_dest, lrow_idx, rrow_idx, 0, thread_intermediate_storage);
      return (output_dest.is_valid() && output_dest.value());
    }
    return false;
  }

 private:
  table_device_view const build;
  table_device_view const probe;
  bool const swap_tables;
  null_equality const nulls_are_equal;
  cudf::ast::detail::IntermediateDataType<has_nulls>* thread_intermediate_storage;
  cudf::ast::detail::expression_evaluator<has_nulls> const evaluator;
};

/**
 * @brief Computes the output size of joining the left table to the right table.
 *
 * This method uses a nested loop to iterate over the left and right tables and count the number of
 * matches according to a boolean expression.
 *
 * @tparam block_size The number of threads per block for this kernel
 * @tparam has_nulls Whether or not the inputs may contain nulls.
 *
 * @param[in] left_table The left table
 * @param[in] right_table The right table
 * @param[in] join_type The type of join to be performed
 * @param[in] device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param[in] swap_tables If true, the kernel was launched with one thread per right row and
 * the kernel needs to internally loop over left rows. Otherwise, loop over right rows.
 * @param[out] output_size The resulting output size
 */
template <int block_size, bool has_nulls>
__global__ void compute_mixed_join_output_size(
  table_device_view left_table,
  table_device_view right_table,
  table_device_view probe,
  table_device_view build,
  null_equality compare_nulls,
  join_kind join_type,
  cudf::detail::mixed_multimap_type::device_view hash_table_view,
  ast::detail::expression_device_view device_expression_data,
  bool const swap_tables,
  std::size_t* output_size,
  cudf::size_type* matches_per_row)
{
  // The (required) extern storage of the shared memory array leads to
  // conflicting declarations between different templates. The easiest
  // workaround is to declare an arbitrary (here char) array type then cast it
  // after the fact to the appropriate type.
  extern __shared__ char raw_intermediate_storage[];
  cudf::ast::detail::IntermediateDataType<has_nulls>* intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    intermediate_storage + (threadIdx.x * device_expression_data.num_intermediates);

  std::size_t thread_counter{0};
  cudf::size_type const start_idx      = threadIdx.x + blockIdx.x * blockDim.x;
  cudf::size_type const stride         = blockDim.x * gridDim.x;
  cudf::size_type const left_num_rows  = left_table.num_rows();
  cudf::size_type const right_num_rows = right_table.num_rows();
  auto const outer_num_rows            = (swap_tables ? right_num_rows : left_num_rows);

  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls>(
    left_table, right_table, device_expression_data);

  // TODO: The hash join code assumes that nulls exist here, so I'm doing the
  // same but at some point we may want to benchmark that.
  row_hash hash_probe{nullate::DYNAMIC{has_nulls}, probe};
  auto const empty_key_sentinel = hash_table_view.get_empty_key_sentinel();
  make_pair_function pair_func{hash_probe, empty_key_sentinel};

  for (cudf::size_type outer_row_index = start_idx; outer_row_index < outer_num_rows;
       outer_row_index += stride) {
    // Figure out the number of elements for this key.
    cg::thread_block_tile<1> this_thread = cg::this_thread();
    auto query_pair                      = pair_func(outer_row_index);
    // TODO: Address asymmetry in operator.
    auto count_equality = pair_expression_equality<has_nulls>{
      build, probe, swap_tables, evaluator, thread_intermediate_storage, compare_nulls};
    // TODO: This entire kernel probably won't work for left anti joins since I
    // need to use a normal map (not a multimap), so this condition is probably
    // overspecified at the moment.
    if (join_type == join_kind::LEFT_JOIN || join_type == join_kind::LEFT_ANTI_JOIN ||
        join_type == join_kind::FULL_JOIN) {
      matches_per_row[outer_row_index] =
        hash_table_view.pair_count_outer(this_thread, query_pair, count_equality);
    } else {
      matches_per_row[outer_row_index] =
        hash_table_view.pair_count(this_thread, query_pair, count_equality);
    }
    thread_counter += matches_per_row[outer_row_index];
  }

  using BlockReduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t block_counter = BlockReduce(temp_storage).Sum(thread_counter);

  // Add block counter to global counter
  if (threadIdx.x == 0) atomicAdd(output_size, block_counter);
}

/**
 * @brief Performs a join conditioned on a predicate to find all matching rows
 * between the left and right tables and generate the output for the desired
 * Join operation.
 *
 * @tparam block_size The number of threads per block for this kernel
 * @tparam output_cache_size The side of the shared memory buffer to cache join
 * output results
 * @tparam has_nulls Whether or not the inputs may contain nulls.
 *
 * @param[in] left_table The left table
 * @param[in] right_table The right table
 * @param[in] join_type The type of join to be performed
 * @param[out] join_output_l The left result of the join operation
 * @param[out] join_output_r The right result of the join operation
 * writes to the global output
 * @param device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param[in] swap_tables If true, the kernel was launched with one thread per right row and
 * the kernel needs to internally loop over left rows. Otherwise, loop over right rows.
 */
template <cudf::size_type block_size,
          cudf::size_type output_cache_size,
          bool has_nulls,
          typename OutputIt1,
          typename OutputIt2>
__global__ void mixed_join(table_device_view left_table,
                           table_device_view right_table,
                           table_device_view probe,
                           table_device_view build,
                           null_equality compare_nulls,
                           join_kind join_type,
                           cudf::detail::mixed_multimap_type::device_view hash_table_view,
                           OutputIt1 join_output_l,
                           OutputIt2 join_output_r,
                           cudf::ast::detail::expression_device_view device_expression_data,
                           cudf::size_type const* matches_per_row,
                           cudf::size_type const* join_result_offsets,
                           bool const swap_tables)
{
  // Normally the casting of a shared memory array is used to create multiple
  // arrays of different types from the shared memory buffer, but here it is
  // used to circumvent conflicts between arrays of different types between
  // different template instantiations due to the extern specifier.
  extern __shared__ char raw_intermediate_storage[];
  cudf::ast::detail::IntermediateDataType<has_nulls>* intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];

  cudf::size_type const left_num_rows  = left_table.num_rows();
  cudf::size_type const right_num_rows = right_table.num_rows();
  auto const outer_num_rows            = (swap_tables ? right_num_rows : left_num_rows);

  cudf::size_type outer_row_index = threadIdx.x + blockIdx.x * blockDim.x;

  unsigned int const activemask = __ballot_sync(0xffffffff, outer_row_index < outer_num_rows);

  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls>(
    left_table, right_table, device_expression_data);

  // TODO: The hash join code assumes that nulls exist here, so I'm doing the
  // same but at some point we may want to benchmark that.
  row_hash hash_probe{nullate::DYNAMIC{has_nulls}, probe};
  auto const empty_key_sentinel = hash_table_view.get_empty_key_sentinel();
  make_pair_function pair_func{hash_probe, empty_key_sentinel};

  if (outer_row_index < outer_num_rows) {
    // Figure out the number of elements for this key.
    cg::thread_block_tile<1> this_thread = cg::this_thread();
    // Figure out the number of elements for this key.
    auto query_pair = pair_func(outer_row_index);
    auto equality   = pair_expression_equality<has_nulls>{
      build, probe, swap_tables, evaluator, thread_intermediate_storage, compare_nulls};

    auto probe_key_begin       = thrust::make_discard_iterator();
    auto probe_value_begin     = swap_tables ? join_output_r + join_result_offsets[outer_row_index]
                                             : join_output_l + join_result_offsets[outer_row_index];
    auto contained_key_begin   = thrust::make_discard_iterator();
    auto contained_value_begin = swap_tables ? join_output_l + join_result_offsets[outer_row_index]
                                             : join_output_r + join_result_offsets[outer_row_index];

    // TODO: This entire kernel probably won't work for left anti joins since I
    // need to use a normal map (not a multimap), so this condition is probably
    // overspecified at the moment.
    if (join_type == join_kind::LEFT_JOIN || join_type == join_kind::LEFT_ANTI_JOIN ||
        join_type == join_kind::FULL_JOIN) {
      hash_table_view.pair_retrieve_outer(this_thread,
                                          query_pair,
                                          probe_key_begin,
                                          probe_value_begin,
                                          contained_key_begin,
                                          contained_value_begin,
                                          equality);
    } else {
      hash_table_view.pair_retrieve(this_thread,
                                    query_pair,
                                    probe_key_begin,
                                    probe_value_begin,
                                    contained_key_begin,
                                    contained_value_begin,
                                    equality);
    }
  }
}

}  // namespace detail

}  // namespace cudf
