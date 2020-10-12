#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include "cudf/detail/utilities/device_atomics.cuh"

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

template <cub::BlockLoadAlgorithm ALGORITHM, int BLOCK_DIM_X, int ITEMS_PER_THREAD = 128>
__global__ void supercool_access_pattern_kernel(uint8_t const* input,
                                                uint64_t input_size,
                                                csv_dimensional_sum* output)
{
  // each thread must know
  // - starting offset
  // - starting row
  // - starting column
  // - starting context (inside/outside quoted text)

  using BlockLoad   = typename cub::BlockLoad<uint8_t, BLOCK_DIM_X, ITEMS_PER_THREAD, ALGORITHM>;
  using BlockReduce = typename cub::
    BlockReduce<csv_dimensional_sum, BLOCK_DIM_X, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>;

  // __shared__ typename BlockLoad::TempStorage temp_storage;
  // Load relevant data

  __shared__ union TempStorage {
    typename BlockLoad::TempStorage block_load;
    typename BlockReduce::TempStorage block_reduce;
  } temp_storage;

  // BlockLoad(temp_storage).Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD]);
  // BlockLoad(temp_storage).Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int
  // valid_items); BlockLoad(temp_storage).Load(InputIteratorT block_itr, InputT
  // (&items)[ITEMS_PER_THREAD], int valid_items, DefaultT oob_default);

  // Thread Scan

  uint8_t thread_data[ITEMS_PER_THREAD];
  auto const block_offset = blockIdx.x * blockDim.x;
  // auto const thread_offset =  block_offset + threadIdx.x;
  // auto const input_size_remainder = input_size - block_offset;

  BlockLoad(temp_storage).Load(input, thread_data, input_size - block_offset);

  uint8_t prev = '\0';
  bool toggle;
  csv_dim dimensions[2];

  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    auto const current = thread_data[i];
    if (prev == '\\') {
      // this char is escaped, so skip it.
      prev = current;
      continue;
    }

    csv_dimensional_sum::incorperate_next_char(dimensions, toggle, prev, current);
  }

  auto result = csv_dimensional_sum{prev, toggle, {dimensions[0], dimensions[1]}};
}
