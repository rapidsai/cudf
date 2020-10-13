#pragma once

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_scalar.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <cstdint>
#include <type_traits>

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
  uint32_t valid_items  = input.size() - block_offset;
  uint32_t thread_data[ITEMS_PER_THREAD];

  BlockLoad(block_load_temp_storage).Load(input, thread_data, valid_items, 0);

  auto block_result = BlockReduce(block_reduce_temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) { output[blockIdx.x] = block_result; }
}

inline constexpr int div_round_up(int dividend, int divsor)
{
  return dividend / divsor + (dividend % divsor) != 0;
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

template <int BLOCK_DIM_X, int ITEMS_PER_THREAD>
__global__ void find_kernel(device_span<uint8_t const> input,
                            device_span<uint32_t> temp_block_storage,
                            device_span<uint32_t> output,
                            uint8_t needle)
{
  using BlockLoad   = typename cub::BlockLoad<uint8_t,
                                            BLOCK_DIM_X,
                                            ITEMS_PER_THREAD,
                                            cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>;
  using BlockReduce = typename cub::BlockReduce<uint32_t, BLOCK_DIM_X>;
  using BlockScan   = typename cub::BlockScan<uint32_t, BLOCK_DIM_X>;

  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockReduce::TempStorage reduce;
    typename BlockScan::TempStorage scan;
  } temp_storage;

  uint8_t thread_data[ITEMS_PER_THREAD];

  uint32_t block_offset = (blockIdx.x * blockDim.x) * ITEMS_PER_THREAD;
  uint32_t valid_items  = input.size() - block_offset;

  BlockLoad(temp_storage.load).Load(input.data() + block_offset, thread_data, valid_items);

  uint32_t count_thread = 0;

  // incorperate, predicate, assign

  for (auto i = 0; i < valid_items && i < ITEMS_PER_THREAD; i++) {
    if (thread_data[i] == needle) { count_thread++; }
  }

  if (output.data() == nullptr) {
    // This is the first pass, so just return the block sums.

    uint32_t count_block = BlockReduce(temp_storage.reduce).Sum(count_thread);

    if (threadIdx.x == 0) { temp_block_storage[blockIdx.x + 1] = count_block; }

    return;
  }

  // This is the second pass.

  uint32_t block_output_offset = temp_block_storage[blockIdx.x];
  uint32_t thread_output_offset;

  BlockScan(temp_storage.scan).ExclusiveSum(count_thread, thread_output_offset);

  // printf("bidx(%i) tidx(%i): exc\n", blockIdx.x, threadIdx.x);

  for (auto i = 0; i < valid_items && i < ITEMS_PER_THREAD; i++) {
    if (thread_data[i] == needle) {
      output[block_output_offset + thread_output_offset++] =
        block_offset + threadIdx.x * ITEMS_PER_THREAD + i;
    }
  }
}

rmm::device_vector<uint32_t>  //
find(device_span<uint8_t const> d_input, uint8_t needle, cudaStream_t stream = 0)
{
  enum { BLOCK_DIM_X = 3, ITEMS_PER_THREAD = 7 };

  cudf::detail::grid_1d grid(d_input.size(), BLOCK_DIM_X, ITEMS_PER_THREAD);

  // include leading zero in offsets
  auto d_output_offsets = rmm::device_vector<uint32_t>(grid.num_blocks + 1);

  auto kernel = find_kernel<BLOCK_DIM_X, ITEMS_PER_THREAD>;

  kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
    d_input,
    d_output_offsets,
    device_span<uint32_t>(),
    needle);

  // convert block result sizes to block result offsets.
  thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                         d_output_offsets.begin(),
                         d_output_offsets.end(),
                         d_output_offsets.begin());

  // return d_output_offsets;

  auto d_results = rmm::device_vector<uint32_t>(d_output_offsets.back());

  kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(  //
    d_input,
    d_output_offsets,
    d_results,
    needle);

  cudaStreamSynchronize(stream);

  return d_results;
}
