/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/join/direct_join.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/join/join.hpp>

#include <cuda/functional>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>

// Apples-to-apples comparison of inner join implementations on input that satisfies
// `direct_inner_join`'s preconditions: a single UINT32 key column per side, distinct right keys,
// and all key values in [0, capacity) with capacity = 2 * right_size. Right keys are the shuffled
// even values; matching left keys draw from the even values and non-matching ones from the odd
// values, so `selectivity` is exact and identical for all algorithms.
void nvbench_direct_inner_join(nvbench::state& state)
{
  if (should_skip_large_sizes(state)) { return; }

  auto const right_size    = static_cast<cudf::size_type>(state.get_int64("right_size"));
  auto const left_size     = static_cast<cudf::size_type>(state.get_int64("left_size"));
  auto const algorithm     = state.get_string("algorithm");
  auto const capacity      = static_cast<std::size_t>(right_size) * 2;
  double const selectivity = 0.3;

  auto right = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT32}, right_size, cudf::mask_state::UNALLOCATED);
  thrust::tabulate(thrust::device,
                   right->mutable_view().begin<std::uint32_t>(),
                   right->mutable_view().end<std::uint32_t>(),
                   cuda::proclaim_return_type<std::uint32_t>([] __device__(cudf::size_type idx) {
                     return static_cast<std::uint32_t>(idx) * 2;  // distinct evens in [0, capacity)
                   }));
  thrust::shuffle(thrust::device,
                  right->mutable_view().begin<std::uint32_t>(),
                  right->mutable_view().end<std::uint32_t>(),
                  thrust::default_random_engine{12345});

  auto const num_matching = static_cast<cudf::size_type>(selectivity * left_size);
  auto left               = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT32}, left_size, cudf::mask_state::UNALLOCATED);
  thrust::tabulate(thrust::device,
                   left->mutable_view().begin<std::uint32_t>(),
                   left->mutable_view().end<std::uint32_t>(),
                   cuda::proclaim_return_type<std::uint32_t>(
                     [num_matching, right_size] __device__(cudf::size_type idx) {
                       thrust::default_random_engine rng(idx);
                       thrust::uniform_int_distribution<std::uint32_t> dist(0, right_size - 1);
                       auto const even = dist(rng) * 2;
                       return (idx < num_matching) ? even : even + 1;  // odd keys never match
                     }));
  thrust::shuffle(thrust::device,
                  left->mutable_view().begin<std::uint32_t>(),
                  left->mutable_view().end<std::uint32_t>(),
                  thrust::default_random_engine{67890});

  auto const left_view  = left->view();
  auto const right_view = right->view();
  auto const left_keys  = cudf::table_view{{left_view}};
  auto const right_keys = cudf::table_view{{right_view}};

  auto const join_input_size = estimate_size(left_keys) + estimate_size(right_keys);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.add_element_count(join_input_size, "join_input_size");  // number of bytes
  state.add_global_memory_reads<nvbench::int8_t>(join_input_size);

  if (algorithm == "hash") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto result = cudf::inner_join(left_keys, right_keys, cudf::null_equality::UNEQUAL);
    });
  } else if (algorithm == "distinct_hash") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto hj_obj = cudf::distinct_hash_join{right_keys, cudf::null_equality::UNEQUAL, 0.5};
      auto result = hj_obj.inner_join(left_keys);
    });
  } else if (algorithm == "direct") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto result = cudf::direct_inner_join(left_view, right_view, capacity);
    });
  } else {
    state.skip("unknown algorithm");
  }
}

NVBENCH_BENCH(nvbench_direct_inner_join)
  .set_name("direct_inner_join")
  .add_string_axis("algorithm", {"hash", "distinct_hash", "direct"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("skip_large_sizes", {1});
