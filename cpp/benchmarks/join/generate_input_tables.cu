/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "generate_input_tables.cuh"

#include <benchmarks/common/generate_input.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cassert>
#include <iostream>
#include <vector>

namespace {

CUDF_KERNEL void init_curand(curandState* state, int const nstates)
{
  int ithread = cudf::detail::grid_1d::global_thread_id();

  if (ithread < nstates) { curand_init(1234ULL, ithread, 0, state + ithread); }
}

template <typename key_type, typename size_type>
CUDF_KERNEL void init_build_tbl(key_type* const build_tbl,
                                size_type const build_tbl_size,
                                int const multiplicity,
                                curandState* state,
                                int const num_states)
{
  auto const start_idx = cudf::detail::grid_1d::global_thread_id();
  auto const stride    = cudf::detail::grid_1d::grid_stride();
  assert(start_idx < num_states);

  curandState localState = state[start_idx];

  for (cudf::thread_index_type tidx = start_idx; tidx < build_tbl_size; tidx += stride) {
    auto const idx = static_cast<size_type>(tidx);
    double const x = curand_uniform_double(&localState);

    build_tbl[idx] = static_cast<key_type>(x * (build_tbl_size / multiplicity));
  }

  state[start_idx] = localState;
}

template <typename key_type, typename size_type>
CUDF_KERNEL void init_probe_tbl(key_type* const probe_tbl,
                                size_type const probe_tbl_size,
                                size_type const build_tbl_size,
                                key_type const rand_max,
                                double const selectivity,
                                int const multiplicity,
                                curandState* state,
                                int const num_states)
{
  auto const start_idx = cudf::detail::grid_1d::global_thread_id();
  auto const stride    = cudf::detail::grid_1d::grid_stride();
  assert(start_idx < num_states);

  curandState localState = state[start_idx];

  for (cudf::thread_index_type tidx = start_idx; tidx < probe_tbl_size; tidx += stride) {
    auto const idx = static_cast<size_type>(tidx);
    key_type val;
    double x = curand_uniform_double(&localState);

    if (x <= selectivity) {
      // x <= selectivity means this key in the probe table should be present in the build table, so
      // we pick a key from [0, build_tbl_size / multiplicity]
      x   = curand_uniform_double(&localState);
      val = static_cast<key_type>(x * (build_tbl_size / multiplicity));
    } else {
      // This key in the probe table should not be present in the build table, so we pick a key from
      // [build_tbl_size, rand_max].
      x   = curand_uniform_double(&localState);
      val = static_cast<key_type>(x * (rand_max - build_tbl_size) + build_tbl_size);
    }
    probe_tbl[idx] = val;
  }

  state[start_idx] = localState;
}

}  // namespace

template <bool Nullable>
std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> generate_input_tables(
  std::vector<cudf::type_id> const& key_types,
  cudf::size_type build_table_numrows,
  cudf::size_type probe_table_numrows,
  cudf::size_type num_payload_cols,
  int multiplicity,
  double selectivity)
{
  // Construct build and probe tables
  // Unique table has build_table_numrows / multiplicity numrows
  auto unique_rows_build_table_numrows =
    static_cast<cudf::size_type>(build_table_numrows / multiplicity);

  double const null_probability = Nullable ? 0.3 : 0;
  auto const profile            = data_profile{data_profile_builder()
                                      .null_probability(null_probability)
                                      .cardinality(unique_rows_build_table_numrows + 1)};
  auto unique_rows_build_table =
    create_random_table(key_types, row_count{unique_rows_build_table_numrows + 1}, profile, 1);

#if 0
  std::cerr << "unique_rows_build_table_numrows: " << unique_rows_build_table_numrows << "\n";
  std::cerr << "unique_rows_build_table num_rows: " << unique_rows_build_table->num_rows() << "\n";
  // Print key values from unique_rows_build_table to check for duplicates
  {
    auto host_data = cudf::detail::make_host_vector(
      cudf::device_span<int32_t const>(
        unique_rows_build_table->view().column(0).data<int32_t>(),
        unique_rows_build_table->num_rows()),
      cudf::get_default_stream());
    std::cerr << "unique_rows_build_table keys: [";
    for (size_t i = 0; i < host_data.size(); ++i) {
      if (i > 0) std::cerr << ", ";
      std::cerr << host_data[i];
    }
    std::cerr << "]\n";
  }
#endif

  constexpr int BLOCK_SIZE = 128;

  // Maximize exposed parallelism while minimizing storage for curand state
  int num_blocks_init_build_tbl{-1};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_init_build_tbl, init_build_tbl<cudf::size_type, cudf::size_type>, BLOCK_SIZE, 0));

  int num_blocks_init_probe_tbl{-1};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_init_probe_tbl, init_probe_tbl<cudf::size_type, cudf::size_type>, BLOCK_SIZE, 0));

  auto const num_sms = cudf::detail::num_multiprocessors();
  auto const num_states =
    num_sms * std::max(num_blocks_init_build_tbl, num_blocks_init_probe_tbl) * BLOCK_SIZE;
  rmm::device_uvector<curandState> devStates(num_states, cudf::get_default_stream());

  init_curand<<<(num_states - 1) / BLOCK_SIZE + 1,
                BLOCK_SIZE,
                0,
                cudf::get_default_stream().value()>>>(devStates.data(), num_states);

  CUDF_CHECK_CUDA(0);

  auto build_table_gather_map = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, build_table_numrows, cudf::mask_state::ALL_VALID);
  thrust::tabulate(
    rmm::exec_policy_nosync(cudf::get_default_stream()),
    build_table_gather_map->mutable_view().begin<cudf::size_type>(),
    build_table_gather_map->mutable_view().end<cudf::size_type>(),
    cuda::proclaim_return_type<cudf::size_type>(
      [multiplicity] __device__(cudf::size_type idx) { return idx / multiplicity; }));

  // auto const num_matching    = static_cast<cudf::size_type>(selectivity * probe_table_numrows);
  auto const num_matching     = static_cast<cudf::size_type>(1);
  auto probe_table_gather_map = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, probe_table_numrows, cudf::mask_state::ALL_VALID);
  thrust::tabulate(
    rmm::exec_policy_nosync(cudf::get_default_stream()),
    probe_table_gather_map->mutable_view().begin<cudf::size_type>(),
    probe_table_gather_map->mutable_view().end<cudf::size_type>(),
    cuda::proclaim_return_type<cudf::size_type>(
      [unique_rows_build_table_numrows, num_matching] __device__(cudf::size_type idx) {
        if (idx < num_matching) {
          // Matching key: cycle through unique build keys
          return idx % unique_rows_build_table_numrows;
        } else {
          // Non-matching key: use values beyond unique key range
          return unique_rows_build_table_numrows;
        }
      }));

  // Shuffle gather maps to avoid cache effects
  thrust::shuffle(rmm::exec_policy_nosync(cudf::get_default_stream()),
                  build_table_gather_map->mutable_view().begin<cudf::size_type>(),
                  build_table_gather_map->mutable_view().end<cudf::size_type>(),
                  thrust::default_random_engine{12345});
  thrust::shuffle(rmm::exec_policy_nosync(cudf::get_default_stream()),
                  probe_table_gather_map->mutable_view().begin<cudf::size_type>(),
                  probe_table_gather_map->mutable_view().end<cudf::size_type>(),
                  thrust::default_random_engine{67890});

  // Debug: print gather maps
#if 0
  auto print_gather_map = [](cudf::column_view const& col, std::string const& name) {
    auto host_data = cudf::detail::make_host_vector(
      cudf::device_span<cudf::size_type const>(col.data<cudf::size_type>(), col.size()),
      cudf::get_default_stream());
    std::cerr << name << " (" << col.size() << " elements): [";
    for (size_t i = 0; i < std::min(host_data.size(), size_t{20}); ++i) {
      if (i > 0) std::cerr << ", ";
      std::cerr << host_data[i];
    }
    if (host_data.size() > 20) std::cerr << ", ...";
    std::cerr << "]\n";
  };
  print_gather_map(build_table_gather_map->view(), "build_gather_map");
  print_gather_map(probe_table_gather_map->view(), "probe_gather_map");
#endif

  auto build_table = cudf::gather(unique_rows_build_table->view(),
                                  build_table_gather_map->view(),
                                  cudf::out_of_bounds_policy::DONT_CHECK);

  auto probe_table = cudf::gather(unique_rows_build_table->view(),
                                  probe_table_gather_map->view(),
                                  cudf::out_of_bounds_policy::DONT_CHECK);

  auto init       = cudf::make_fixed_width_scalar<cudf::size_type>(static_cast<cudf::size_type>(0));
  auto build_cols = build_table->release();
  auto probe_cols = probe_table->release();
  for (auto i = 0; i < num_payload_cols; i++) {
    build_cols.emplace_back(cudf::sequence(build_table_numrows, *init));
    probe_cols.emplace_back(cudf::sequence(probe_table_numrows, *init));
  }

  return std::pair{std::make_unique<cudf::table>(std::move(build_cols)),
                   std::make_unique<cudf::table>(std::move(probe_cols))};
}

// Explicit instantiations
template std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
generate_input_tables<true>(std::vector<cudf::type_id> const& key_types,
                            cudf::size_type build_table_numrows,
                            cudf::size_type probe_table_numrows,
                            cudf::size_type num_payload_cols,
                            int multiplicity,
                            double selectivity);

template std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
generate_input_tables<false>(std::vector<cudf::type_id> const& key_types,
                             cudf::size_type build_table_numrows,
                             cudf::size_type probe_table_numrows,
                             cudf::size_type num_payload_cols,
                             int multiplicity,
                             double selectivity);
