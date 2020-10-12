#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <cudf/detail/utilities/cuda.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include "cudf/detail/utilities/cuda.cuh"
#include "cudf/utilities/span.hpp"
#include "rmm/thrust_rmm_allocator.h"

// ===== Row + Column Count ========================================================================

struct csv_dim {
  uint32_t num_columns;
  uint32_t num_rows;

  inline constexpr csv_dim operator+(csv_dim other) const
  {
    return {num_columns + other.num_columns, num_rows + other.num_rows};
  }
};

struct csv_dimensional_sum {
  uint8_t c;
  bool toggle;
  csv_dim dimensions[2];

  static inline constexpr csv_dimensional_sum identity() { return {'_'}; }

  static inline constexpr void incorperate_next_char(  //
    csv_dimensional_sum& value,
    bool& toggle,
    char prev,
    char next)
  {
    if (prev != '\\') {
      if (next == '"') { toggle = not toggle; }
      if (next == ',') { value.dimensions[value.toggle].num_columns++; }
      if (next == '\n') { value.dimensions[value.toggle].num_rows++; };
    }
  }

  inline constexpr csv_dimensional_sum operator+(csv_dimensional_sum rhs) const
  {
    auto lhs = *this;

    auto toggle = rhs.toggle;

    incorperate_next_char(rhs, toggle, lhs.c, rhs.c);

    auto lhs_map_0 = lhs.toggle != rhs.toggle;
    auto lhs_map_1 = lhs.toggle == rhs.toggle;

    return {
      rhs.c,
      toggle,
      {
        lhs.dimensions[lhs_map_0] + rhs.dimensions[0],
        lhs.dimensions[lhs_map_1] + rhs.dimensions[1]  //
      }                                                //
    };
  }
};

struct csv_dimensional_sum_factory {
  csv_dimensional_sum operator()(char c) const { return {static_cast<uint8_t>(c)}; }
};

// ===== Type Deduction ============================================================================

enum class csv_column_type : uint8_t { unknown = 0, string, integer, identity };

inline constexpr csv_column_type decay_csv_column_type(csv_column_type lhs, csv_column_type rhs)
{
  auto const lhs_raw = static_cast<std::underlying_type_t<csv_column_type>>(lhs);
  auto const rhs_raw = static_cast<std::underlying_type_t<csv_column_type>>(rhs);
  auto const raw     = std::min(lhs_raw, rhs_raw);
  return static_cast<csv_column_type>(raw);
}

inline constexpr csv_column_type operator+(csv_column_type lhs, csv_column_type rhs)
{
  return decay_csv_column_type(lhs, rhs);
}

inline constexpr csv_column_type get_csv_column_type(char c)
{
  return c - '0' < 10  //
           ? csv_column_type::integer
           : csv_column_type::string;
}

struct csv_type_deduction_sum {
  csv_column_type type;

  static inline constexpr csv_type_deduction_sum identity() { return {csv_column_type::identity}; }

  inline constexpr csv_type_deduction_sum operator+(csv_type_deduction_sum const& other) const
  {
    auto const& lhs = *this;
    auto const& rhs = other;

    return {
      rhs.type + lhs.type,
    };
  }
};

struct csv_type_deduction_sum_factory {
  csv_type_deduction_sum operator()(char c) const { return {get_csv_column_type(c)}; }
};

// ===== PARSER ACCESS PATTERN =====================================================================

using cudf::detail::device_span;

template <int BLOCK_DIM_X, int ITEMS_PER_THREAD>
__global__ void reduce_kernel(device_span<uint32_t> input, device_span<uint32_t> output)
{
  using BlockLoad =  //
    typename cub::BlockLoad<uint32_t, BLOCK_DIM_X, ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT>;
  using BlockReduce =  //
    typename cub::BlockReduce<uint32_t, BLOCK_DIM_X, cub::BLOCK_REDUCE_RAKING>;

  __shared__ typename BlockLoad::TempStorage block_load_temp_storage;
  __shared__ typename BlockReduce::TempStorage block_reduce_temp_storage;

  // __shared__ union {
  //   typename BlockLoad::TempStorage block_load;
  //   typename BlockReduce::TempStorage block_reduce;
  // } temp_storage;

  uint32_t block_offset = blockIdx.x * blockDim.x;
  uint32_t num_valid    = input.size() - block_offset;
  uint32_t thread_data[ITEMS_PER_THREAD];

  BlockLoad(block_load_temp_storage).Load(input, thread_data, num_valid, 0);

  auto block_result = BlockReduce(block_reduce_temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) { output[blockIdx.x] = block_result; }
}

uint32_t reduce(device_span<uint32_t> d_input)
{
  enum { BLOCK_DIM_X = 32, ITEMS_PER_THREAD = 8 };

  cudf::detail::grid_1d grid(d_input.size(), BLOCK_DIM_X, ITEMS_PER_THREAD);

  rmm::device_vector<uint32_t> d_temp_blockstorage(grid.num_blocks);

  auto kernel = reduce_kernel<BLOCK_DIM_X, ITEMS_PER_THREAD>;

  kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, 0>>>(  //
    d_input,
    d_temp_blockstorage);

  thrust::host_vector<uint32_t> h_temp_block_storage = d_temp_blockstorage;

  uint32_t res = 0;

  for (auto item : h_temp_block_storage) { res += item; }

  return h_temp_block_storage[0];
}
