/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/hashing/detail/xxhash_64_for_hllpp.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/std/__algorithm/min.h>  // TODO #include <cuda/std/algorithm> once available
#include <cuda/std/bit>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reverse.h>
#include <thrust/tabulate.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

/**
 * The number of bits that is required for a HLLPP register value.
 *
 * This number is determined by the maximum number of leading binary zeros a hashcode can
 * produce. This is equal to the number of bits the hashcode returns. The current
 * implementation uses a 64-bit hashcode, this means 6-bits are (at most) needed to store the
 * number of leading zeros.
 */
constexpr int REGISTER_VALUE_BITS = 6;

// MASK binary 6 bits: 111-111
constexpr uint64_t MASK = (1L << REGISTER_VALUE_BITS) - 1L;

// This value is 10, one long stores 10 register values
constexpr int REGISTERS_PER_LONG = 64 / REGISTER_VALUE_BITS;

// XXHash seed, consistent with Spark
constexpr int64_t SEED = 42L;

// max precision, if require a precision bigger than 18, then use 18.
constexpr int MAX_PRECISION = 18;

/**
 *
 * Computes register values from hash values and partially groups register values.
 * It splits input into multiple segments with each segment has num_hashs_per_thread length.
 * The input is sorted by group labels, each segment contains several consecutive groups.
 * Each thread scans in its segment, find the max register values for all the register values
 * at the same register index at the same group, outputs gathered result of previous group
 * when meets a new group, and in the end each thread saves a buffer for the last group
 * in the segment.
 *
 * In this way, we can save memory usage, only need to cache
 * (num_hashs / num_hashs_per_thread) sketches.
 *
 * num_threads = div_round_up(num_hashs, num_hashs_per_thread).
 *
 *
 * e.g.: num_registers_per_sketch = 512 and num_hashs_per_thread = 4;
 *
 * Input:
 *    register_index register_value  group_lable
 * [
 * ------------------ segment 0 begin --------------------------------------
 *    (0,                1),          0
 *    (0,                2),          0
 *    (1,                1),          1   // meets a new group, outputs result for g0
 *    (1,                9),          1   // outputs for thread 0 when scan to here
 * ------------------ segment 1 begin --------------------------------------
 *    (1,                1),          1
 *    (1,                1),          1
 *    (1,                5),          1
 *    (1,                1),          1   // outputs for thread 1; Output result for g1
 * ]
 * Output e.g.:
 *
 * group_lables_thread_cache:
 * [
 *   g1
 *   g1
 * ]
 * Has num_threads rows.
 *
 * registers_thread_cache:
 * [
 *    512 values: [0, 9, 0, ... ] // register values for group 1
 *    512 values: [0, 5, 0, ... ] // register values for group 1
 * ]
 * Has num_threads rows, each row is corresponding to `group_lables_thread_cache`
 *
 * registers_output_cache:
 * [
 *    512 values: [2, 0, 0, ... ] // register values for group 0
 *    512 values: [0, 5, 0, ... ] // register values for group 1
 * ]
 * Has num_groups rows.
 *
 * The next kernel will merge the registers_output_cache and registers_thread_cache
 */
template <int num_hashs_per_thread>
CUDF_KERNEL void partial_group_sketches_from_hashs_kernel(
  column_device_view hashs,
  cudf::device_span<size_type const> group_lables,
  int64_t const precision,                    // num of bits for register addressing, e.g.: 9
  int* const registers_output_cache,          // num is num_groups * num_registers_per_sketch
  int* const registers_thread_cache,          // num is num_threads * num_registers_per_sketch
  size_type* const group_lables_thread_cache  // save the group lables for each thread
)
{
  auto const tid          = cudf::detail::grid_1d::global_thread_id();
  int64_t const num_hashs = hashs.size();
  if (tid * num_hashs_per_thread >= hashs.size()) { return; }

  // 2^precision = num_registers_per_sketch
  int64_t num_registers_per_sketch = 1L << precision;
  // e.g.: integer in binary: 1 0000 0000
  uint64_t const w_padding = 1ULL << (precision - 1);
  // e.g.: 64 - 9 = 55
  int const idx_shift = 64 - precision;

  auto const hash_first = tid * num_hashs_per_thread;
  auto const hash_end   = cuda::std::min((tid + 1) * num_hashs_per_thread, num_hashs);

  // init sketches for each thread
  int* const sketch_ptr = registers_thread_cache + tid * num_registers_per_sketch;
  for (auto i = 0; i < num_registers_per_sketch; i++) {
    sketch_ptr[i] = 0;
  }

  size_type prev_group = group_lables[hash_first];
  for (auto hash_idx = hash_first; hash_idx < hash_end; hash_idx++) {
    size_type curr_group = group_lables[hash_idx];

    // cast to unsigned, then >> will shift without preserve the sign bit.
    uint64_t const hash = static_cast<uint64_t>(hashs.element<int64_t>(hash_idx));
    auto const reg_idx  = hash >> idx_shift;
    int const reg_v =
      static_cast<int>(cuda::std::countl_zero((hash << precision) | w_padding) + 1ULL);

    if (curr_group == prev_group) {
      // still in the same group, update the max value
      if (reg_v > sketch_ptr[reg_idx]) { sketch_ptr[reg_idx] = reg_v; }
    } else {
      // meets new group, save output for the previous group and reset
      for (auto i = 0; i < num_registers_per_sketch; i++) {
        registers_output_cache[prev_group * num_registers_per_sketch + i] = sketch_ptr[i];
        sketch_ptr[i]                                                     = 0;
      }
      // save the result for current group
      sketch_ptr[reg_idx] = reg_v;
    }

    if (hash_idx == hash_end - 1) {
      // meets the last hash in the segment
      if (hash_idx == num_hashs - 1) {
        // meets the last segment, special logic: assume meets new group
        for (auto i = 0; i < num_registers_per_sketch; i++) {
          registers_output_cache[curr_group * num_registers_per_sketch + i] = sketch_ptr[i];
        }
      } else {
        // not the last segment, probe one item forward.
        if (curr_group != group_lables[hash_idx + 1]) {
          // meets a new group by checking the next item in the next segment
          for (auto i = 0; i < num_registers_per_sketch; i++) {
            registers_output_cache[curr_group * num_registers_per_sketch + i] = sketch_ptr[i];
          }
        }
      }
    }

    prev_group = curr_group;
  }

  // save the group lable for this thread
  group_lables_thread_cache[tid] = group_lables[hash_end - 1];
}

/*
 * Merge registers_output_cache and registers_thread_cache produced in the above kernel
 * Merge sketches vertically.
 *
 * For all register at the same index, starts a thread to merge the max value.
 * num_threads = num_registers_per_sketch.
 *
 * Input e.g.:
 *
 * group_lables_thread_cache:
 * [
 *   g0
 *   g0
 *   g1
 *   ...
 *   gN
 * ]
 * Has num_threads rows.
 *
 * registers_thread_cache:
 * [
 *    r0_g0, r1_g0, r2_g0, r3_g0, ... , r511_g0 // register values for group 0
 *    r0_g0, r1_g0, r2_g0, r3_g0, ... , r511_g0 // register values for group 0
 *    r0_g1, r1_g1, r2_g1, r3_g1, ... , r511_g1 // register values for group 1
 *    ...
 *    r0_gN, r1_gN, r2_gN, r3_gN, ... , r511_gN // register values for group N
 * ]
 * Has num_threads rows, each row is corresponding to `group_lables_thread_cache`
 *
 * registers_output_cache:
 * [
 *    r0_g0, r1_g0, r2_g0, r3_g0, ... , r511_g0 // register values for group 0
 *    r0_g1, r1_g1, r2_g1, r3_g1, ... , r511_g1 // register values for group 1
 *    ...
 *    r0_gN, r1_gN, r2_gN, r3_gN, ... , r511_gN // register values for group N
 * ]
 * Has num_groups rows.
 *
 * First find the max value in registers_thread_cache and then merge to registers_output_cache
 */
template <int block_size>
CUDF_KERNEL void merge_sketches_vertically(int64_t num_sketches,
                                           int64_t num_registers_per_sketch,
                                           int* const registers_output_cache,
                                           int const* const registers_thread_cache,
                                           size_type const* const group_lables_thread_cache)
{
  __shared__ int8_t shared_data[block_size];
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  int shared_idx = tid % block_size;

  // register idx is tid
  shared_data[shared_idx] = static_cast<int8_t>(0);
  int prev_group          = group_lables_thread_cache[0];
  for (auto i = 0; i < num_sketches; i++) {
    int curr_group = group_lables_thread_cache[i];
    int8_t curr_reg_v =
      static_cast<int8_t>(registers_thread_cache[i * num_registers_per_sketch + tid]);
    if (curr_group == prev_group) {
      if (curr_reg_v > shared_data[shared_idx]) { shared_data[shared_idx] = curr_reg_v; }
    } else {
      // meets a new group, store the result for previous group
      int64_t result_reg_idx = prev_group * num_registers_per_sketch + tid;
      int result_curr_reg_v  = registers_output_cache[result_reg_idx];
      if (shared_data[shared_idx] > result_curr_reg_v) {
        registers_output_cache[result_reg_idx] = shared_data[shared_idx];
      }

      shared_data[shared_idx] = curr_reg_v;
    }
    prev_group = curr_group;
  }

  // handles the last register in this thread
  int64_t reg_idx = prev_group * num_registers_per_sketch + tid;
  int curr_reg_v  = registers_output_cache[reg_idx];
  if (shared_data[shared_idx] > curr_reg_v) {
    registers_output_cache[reg_idx] = shared_data[shared_idx];
  }
}

/**
 * Compact register values, compact 10 registers values
 * (each register value is 6 bits) in to a long.
 * This is consistent with Spark.
 * Output: long columns which will be composed into a struct column
 *
 * Number of threads is num_groups * num_long_cols.
 *
 * e.g., num_registers_per_sketch is 512, precision is 9:
 * Input:
 * registers_output_cache:
 * [
 *    r0_g0, r1_g0, r2_g0, r3_g0, ... , r511_g0 // register values for group 0
 *    r0_g1, r1_g1, r2_g1, r3_g1, ... , r511_g1 // register values for group 1
 *    ...
 *    r0_gN, r1_gN, r2_gN, r3_gN, ... , r511_gN // register values for group N
 * ]
 * Has num_groups rows.
 *
 * Output:
 * 52 long columns
 *
 * r0 to r9 integers are all: 00000000-00000000-00000000-00100001, tailing 6 bits: 100-001
 * Compact to one long is: 100001-100001-100001-100001-100001-100001-100001-100001-100001-100001
 */
CUDF_KERNEL void compact_kernel(int64_t const num_groups,
                                int64_t const num_registers_per_sketch,
                                cudf::device_span<int64_t*> sketches_output,
                                // num_groups * num_registers_per_sketch integers
                                cudf::device_span<int> registers_output_cache)
{
  int64_t const tid           = cudf::detail::grid_1d::global_thread_id();
  int64_t const num_long_cols = num_registers_per_sketch / REGISTERS_PER_LONG + 1;
  if (tid >= num_groups * num_long_cols) { return; }

  int64_t const group_idx = tid / num_long_cols;
  int64_t const long_idx  = tid % num_long_cols;

  int64_t const reg_begin_idx =
    group_idx * num_registers_per_sketch + long_idx * REGISTERS_PER_LONG;
  int64_t num_regs = REGISTERS_PER_LONG;
  if (long_idx == num_long_cols - 1) { num_regs = num_registers_per_sketch % REGISTERS_PER_LONG; }

  int64_t ten_registers = 0;
  for (auto i = 0; i < num_regs; i++) {
    int64_t reg_v = registers_output_cache[reg_begin_idx + i];
    int64_t tmp   = reg_v << (REGISTER_VALUE_BITS * i);
    ten_registers |= tmp;
  }

  sketches_output[long_idx][group_idx] = ten_registers;
}

std::unique_ptr<column> group_hllpp(column_view const& input,
                                    int64_t const num_groups,
                                    cudf::device_span<size_type const> group_lables,
                                    int64_t const precision,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  int64_t num_registers_per_sketch = 1 << precision;

  // 1. compute all the hashs
  auto hash_col =
    make_numeric_column(data_type{type_id::INT64}, input.size(), mask_state::ALL_VALID, stream, mr);
  auto input_table   = cudf::table_view{{input}};
  auto d_input_table = cudf::table_device_view::create(input_table, stream);
  bool nullable      = has_nested_nulls(input_table);
  thrust::tabulate(
    rmm::exec_policy(stream),
    hash_col->mutable_view().begin<int64_t>(),
    hash_col->mutable_view().end<int64_t>(),
    cudf::hashing::detail::xxhash_64_hllpp_row_hasher<bool>(nullable, *d_input_table, SEED));
  auto d_hashs = cudf::column_device_view::create(hash_col->view(), stream);

  // 2. execute partial group by
  constexpr int64_t block_size           = 256;
  constexpr int64_t num_hashs_per_thread = 256;  // handles 32 items per thread
  int64_t total_threads_partial_group =
    cudf::util::div_rounding_up_safe(static_cast<int64_t>(input.size()), num_hashs_per_thread);
  int64_t num_blocks_p1 = cudf::util::div_rounding_up_safe(total_threads_partial_group, block_size);
  auto sketches_output =
    rmm::device_uvector<int32_t>(num_groups * num_registers_per_sketch, stream, mr);
  {
    auto registers_thread_cache = rmm::device_uvector<int32_t>(
      total_threads_partial_group * num_registers_per_sketch, stream, mr);
    auto group_lables_thread_cache =
      rmm::device_uvector<int32_t>(total_threads_partial_group, stream, mr);
    partial_group_sketches_from_hashs_kernel<num_hashs_per_thread>
      <<<num_blocks_p1, block_size, 0, stream.value()>>>(*d_hashs,
                                                         group_lables,
                                                         precision,
                                                         sketches_output.begin(),
                                                         registers_thread_cache.begin(),
                                                         group_lables_thread_cache.begin());

    // 3. merge the intermidate result
    auto num_merge_threads = num_registers_per_sketch;
    auto num_merge_blocks  = cudf::util::div_rounding_up_safe(num_merge_threads, block_size);
    merge_sketches_vertically<block_size>
      <<<num_merge_blocks, block_size, block_size, stream.value()>>>(
        total_threads_partial_group,  // num_sketches
        num_registers_per_sketch,
        sketches_output.begin(),
        registers_thread_cache.begin(),
        group_lables_thread_cache.begin());
  }

  // 4. create output columns
  auto num_long_cols      = num_registers_per_sketch / REGISTERS_PER_LONG + 1;
  auto const results_iter = cudf::detail::make_counting_transform_iterator(0, [&](int i) {
    return make_numeric_column(
      data_type{type_id::INT64}, num_groups, mask_state::ALL_VALID, stream, mr);
  });
  auto children  = std::vector<std::unique_ptr<column>>(results_iter, results_iter + num_long_cols);
  auto d_results = [&] {
    auto host_results_pointer_iter =
      thrust::make_transform_iterator(children.begin(), [](auto const& results_column) {
        return results_column->mutable_view().template data<int64_t>();
      });
    auto host_results_pointers =
      std::vector<int64_t*>(host_results_pointer_iter, host_results_pointer_iter + children.size());
    return cudf::detail::make_device_uvector_async(host_results_pointers, stream, mr);
  }();
  auto result = cudf::make_structs_column(num_groups,
                                          std::move(children),
                                          0,                     // null count
                                          rmm::device_buffer{},  // null mask
                                          stream);

  // 5. compact sketches
  auto num_phase3_threads = num_groups * num_long_cols;
  auto num_phase3_blocks  = cudf::util::div_rounding_up_safe(num_phase3_threads, block_size);
  compact_kernel<<<num_phase3_blocks, block_size, 0, stream.value()>>>(
    num_groups, num_registers_per_sketch, d_results, sketches_output);

  return result;
}

__device__ inline int get_register_value(int64_t const long_10_registers, int reg_idx)
{
  int64_t shift_mask = MASK << (REGISTER_VALUE_BITS * reg_idx);
  int64_t v          = (long_10_registers & shift_mask) >> (REGISTER_VALUE_BITS * reg_idx);
  return static_cast<int>(v);
}

/**
 * Partial groups sketches in long columns, similar to `partial_group_sketches_from_hashs_kernel`
 * It split longs into segments with each has `num_longs_per_threads` elements
 * e.g.: num_registers_per_sketch = 512.
 * Each sketch uses 52 (512 / 10 + 1) longs.
 *
 * Input:
 *           col_0  col_1      col_51
 * sketch_0: long,  long, ..., long
 * sketch_1: long,  long, ..., long
 * sketch_2: long,  long, ..., long
 *
 * num_threads = 52 * div_round_up(num_sketches_input, num_longs_per_threads)
 * Each thread scans and merge num_longs_per_threads longs,
 * and output the max register value when meets a new group.
 * For the last long in a thread, outputs the result into `registers_thread_cache`.
 *
 * By split inputs into segments like `partial_group_sketches_from_hashs_kernel` and
 * do partial merge, it will use less memory. Then the kernel merge_sketches_vertically
 * can be used to merge the intermidate results: registers_output_cache, registers_thread_cache
 */
template <int num_longs_per_threads>
CUDF_KERNEL void partial_group_long_sketches_kernel(
  cudf::device_span<int64_t const*> sketches_input,
  int64_t const num_sketches_input,
  int64_t const num_threads_per_col,
  int64_t const num_registers_per_sketch,
  int64_t const num_groups,
  cudf::device_span<size_type const> group_lables,
  // num_groups * num_registers_per_sketch integers
  int* const registers_output_cache,
  // num_threads * num_registers_per_sketch integers
  int* const registers_thread_cache,
  // num_threads integers
  size_type* const group_lables_thread_cache)
{
  auto const tid           = cudf::detail::grid_1d::global_thread_id();
  auto const num_long_cols = sketches_input.size();
  if (tid >= num_threads_per_col * num_long_cols) { return; }

  auto const long_idx            = tid / num_threads_per_col;
  auto const thread_idx_in_cols  = tid % num_threads_per_col;
  int64_t const* const longs_ptr = sketches_input[long_idx];

  int* const registers_thread_ptr =
    registers_thread_cache + thread_idx_in_cols * num_registers_per_sketch;

  auto const sketch_first = thread_idx_in_cols * num_longs_per_threads;
  auto const sketch_end = cuda::std::min(sketch_first + num_longs_per_threads, num_sketches_input);

  int num_regs = REGISTERS_PER_LONG;
  if (long_idx == num_long_cols - 1) { num_regs = num_registers_per_sketch % REGISTERS_PER_LONG; }

  for (auto i = 0; i < num_regs; i++) {
    size_type prev_group  = group_lables[sketch_first];
    int max_reg_v         = 0;
    int reg_idx_in_sketch = long_idx * REGISTERS_PER_LONG + i;
    for (auto sketch_idx = sketch_first; sketch_idx < sketch_end; sketch_idx++) {
      size_type curr_group = group_lables[sketch_idx];
      int curr_reg_v       = get_register_value(longs_ptr[sketch_idx], i);
      if (curr_group == prev_group) {
        // still in the same group, update the max value
        if (curr_reg_v > max_reg_v) { max_reg_v = curr_reg_v; }
      } else {
        // meets new group, save output for the previous group
        int64_t output_idx_prev = num_registers_per_sketch * prev_group + reg_idx_in_sketch;
        registers_output_cache[output_idx_prev] = max_reg_v;

        // reset
        max_reg_v = curr_reg_v;
      }

      if (sketch_idx == sketch_end - 1) {
        // last item in the segment
        int64_t output_idx_curr = num_registers_per_sketch * curr_group + reg_idx_in_sketch;
        if (sketch_idx == num_sketches_input - 1) {
          // last segment
          registers_output_cache[output_idx_curr] = max_reg_v;
          max_reg_v                               = curr_reg_v;
        } else {
          if (curr_group != group_lables[sketch_idx + 1]) {
            // look the first item in the next segment
            registers_output_cache[output_idx_curr] = max_reg_v;
            max_reg_v                               = curr_reg_v;
          }
        }
      }

      prev_group = curr_group;
    }

    // For each thread, output current max value
    registers_thread_ptr[reg_idx_in_sketch] = max_reg_v;
  }

  if (long_idx == 0) {
    group_lables_thread_cache[thread_idx_in_cols] = group_lables[sketch_end - 1];
  }
}

/**
 * Merge for struct<long, ..., long> column. Each long contains 10 register values.
 * Merge all rows in the same group.
 */
std::unique_ptr<column> merge_hyper_log_log(
  column_view const& hll_input,  // struct<long, ..., long> column
  int64_t const num_groups,
  cudf::device_span<size_type const> group_lables,
  int64_t const precision,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  int64_t num_registers_per_sketch        = 1 << precision;
  int64_t const num_sketches              = hll_input.size();
  int64_t const num_long_cols             = num_registers_per_sketch / REGISTERS_PER_LONG + 1;
  constexpr int64_t num_longs_per_threads = 256;
  constexpr int64_t block_size            = 256;

  int64_t num_threads_per_col_phase1 =
    cudf::util::div_rounding_up_safe(num_sketches, num_longs_per_threads);
  int64_t num_threads_phase1 = num_threads_per_col_phase1 * num_long_cols;
  int64_t num_blocks         = cudf::util::div_rounding_up_safe(num_threads_phase1, block_size);
  auto registers_output_cache =
    rmm::device_uvector<int32_t>(num_registers_per_sketch * num_groups, stream, mr);
  {
    auto registers_thread_cache =
      rmm::device_uvector<int32_t>(num_registers_per_sketch * num_threads_phase1, stream, mr);
    auto group_lables_thread_cache =
      rmm::device_uvector<int32_t>(num_threads_per_col_phase1, stream, mr);

    cudf::structs_column_view scv(hll_input);
    auto const input_iter = cudf::detail::make_counting_transform_iterator(
      0, [&](int i) { return scv.get_sliced_child(i, stream).begin<int64_t>(); });
    auto input_cols = std::vector<int64_t const*>(input_iter, input_iter + num_long_cols);
    auto d_inputs   = cudf::detail::make_device_uvector_async(input_cols, stream, mr);
    // 1st kernel: partially group
    partial_group_long_sketches_kernel<num_longs_per_threads>
      <<<num_blocks, block_size, 0, stream.value()>>>(d_inputs,
                                                      num_sketches,
                                                      num_threads_per_col_phase1,
                                                      num_registers_per_sketch,
                                                      num_groups,
                                                      group_lables,
                                                      registers_output_cache.begin(),
                                                      registers_thread_cache.begin(),
                                                      group_lables_thread_cache.begin());
    auto const num_phase2_threads = num_registers_per_sketch;
    auto const num_phase2_blocks = cudf::util::div_rounding_up_safe(num_phase2_threads, block_size);
    // 2nd kernel: vertical merge
    merge_sketches_vertically<block_size>
      <<<num_phase2_blocks, block_size, block_size, stream.value()>>>(
        num_threads_per_col_phase1,  // num_sketches
        num_registers_per_sketch,
        registers_output_cache.begin(),
        registers_thread_cache.begin(),
        group_lables_thread_cache.begin());
  }

  // create output columns
  auto const results_iter = cudf::detail::make_counting_transform_iterator(0, [&](int i) {
    return make_numeric_column(
      data_type{type_id::INT64}, num_groups, mask_state::ALL_VALID, stream, mr);
  });
  auto results = std::vector<std::unique_ptr<column>>(results_iter, results_iter + num_long_cols);
  auto d_sketches_output = [&] {
    auto host_results_pointer_iter =
      thrust::make_transform_iterator(results.begin(), [](auto const& results_column) {
        return results_column->mutable_view().template data<int64_t>();
      });
    auto host_results_pointers =
      std::vector<int64_t*>(host_results_pointer_iter, host_results_pointer_iter + results.size());
    return cudf::detail::make_device_uvector_async(host_results_pointers, stream, mr);
  }();

  // 3rd kernel: compact
  auto num_phase3_threads = num_groups * num_long_cols;
  auto num_phase3_blocks  = cudf::util::div_rounding_up_safe(num_phase3_threads, block_size);
  compact_kernel<<<num_phase3_blocks, block_size, 0, stream.value()>>>(
    num_groups, num_registers_per_sketch, d_sketches_output, registers_output_cache);

  return make_structs_column(num_groups, std::move(results), 0, rmm::device_buffer{});
}

/**
 * Launch only 1 block, uses max 1M(2^18 *sizeof(int)) shared memory.
 * For each hash, get a pair: (register index, register value).
 * Use shared memory to speedup the fetch max atomic operation.
 */
template <int block_size>
CUDF_KERNEL void reduce_hllpp_kernel(column_device_view hashs,
                                     cudf::device_span<int64_t*> output,
                                     int precision)
{
  __shared__ int32_t shared_data[block_size];

  auto const tid                          = cudf::detail::grid_1d::global_thread_id();
  auto const num_hashs                    = hashs.size();
  uint64_t const num_registers_per_sketch = 1L << precision;
  int const idx_shift                     = 64 - precision;
  uint64_t const w_padding                = 1ULL << (precision - 1);

  // init tmp data
  for (int i = tid; i < num_registers_per_sketch; i += block_size) {
    shared_data[i] = 0;
  }
  __syncthreads();

  // update max reg value for the reg index
  for (int i = tid; i < num_hashs; i += block_size) {
    uint64_t const hash = static_cast<uint64_t>(hashs.element<int64_t>(i));
    // use unsigned int to avoid insert 1 for the highest bit when do right shift
    uint64_t const reg_idx = hash >> idx_shift;
    // get the leading zeros
    int const reg_v =
      static_cast<int>(cuda::std::countl_zero((hash << precision) | w_padding) + 1ULL);
    cuda::atomic_ref<int32_t, cuda::thread_scope_block> register_ref(shared_data[reg_idx]);
    register_ref.fetch_max(reg_v, cuda::memory_order_relaxed);
  }
  __syncthreads();

  // compact from register values (int array) to long array
  // each long holds 10 integers, note reg value < 64 which means the bits from 7 to highest are all
  // 0.
  if (tid * REGISTERS_PER_LONG < num_registers_per_sketch) {
    int start = tid * REGISTERS_PER_LONG;
    int end   = (tid + 1) * REGISTERS_PER_LONG;
    if (end > num_registers_per_sketch) { end = num_registers_per_sketch; }

    int64_t ret = 0;
    for (int i = 0; i < end - start; i++) {
      int shift   = i * REGISTER_VALUE_BITS;
      int64_t reg = shared_data[start + i];
      ret |= (reg << shift);
    }

    output[tid][0] = ret;
  }
}

std::unique_ptr<scalar> reduce_hllpp(column_view const& input,
                                     int64_t const precision,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  int64_t num_registers_per_sketch = 1L << precision;
  // 1. compute all the hashs
  auto hash_col =
    make_numeric_column(data_type{type_id::INT64}, input.size(), mask_state::ALL_VALID, stream, mr);
  auto input_table   = cudf::table_view{{input}};
  auto d_input_table = cudf::table_device_view::create(input_table, stream);
  bool nullable      = has_nested_nulls(input_table);
  thrust::tabulate(
    rmm::exec_policy(stream),
    hash_col->mutable_view().begin<int64_t>(),
    hash_col->mutable_view().end<int64_t>(),
    cudf::hashing::detail::xxhash_64_hllpp_row_hasher<bool>(nullable, *d_input_table, SEED));
  auto d_hashs = cudf::column_device_view::create(hash_col->view(), stream);

  // 2. generate long columns, the size of each long column is 1
  auto num_long_cols      = num_registers_per_sketch / REGISTERS_PER_LONG + 1;
  auto const results_iter = cudf::detail::make_counting_transform_iterator(0, [&](int i) {
    return make_numeric_column(
      data_type{type_id::INT64}, 1 /**num_groups*/, mask_state::ALL_VALID, stream, mr);
  });
  auto children  = std::vector<std::unique_ptr<column>>(results_iter, results_iter + num_long_cols);
  auto d_results = [&] {
    auto host_results_pointer_iter =
      thrust::make_transform_iterator(children.begin(), [](auto const& results_column) {
        return results_column->mutable_view().template data<int64_t>();
      });
    auto host_results_pointers =
      std::vector<int64_t*>(host_results_pointer_iter, host_results_pointer_iter + children.size());
    return cudf::detail::make_device_uvector_async(host_results_pointers, stream, mr);
  }();

  // 2. reduce and generate compacted long values
  constexpr int64_t block_size = 256;
  // max shared memory is 2^18 * 4 = 1M
  auto const shared_mem_size = num_registers_per_sketch * sizeof(int32_t);
  reduce_hllpp_kernel<block_size>
    <<<1, block_size, shared_mem_size, stream.value()>>>(*d_hashs, d_results, precision);

  // 3. create struct scalar
  auto host_results_view_iter = thrust::make_transform_iterator(
    children.begin(), [](auto const& results_column) { return results_column->view(); });
  auto views =
    std::vector<column_view>(host_results_view_iter, host_results_view_iter + num_long_cols);
  auto table_view = cudf::table_view{views};
  auto table      = cudf::table(table_view);
  return std::make_unique<cudf::struct_scalar>(std::move(table), true, stream, mr);
}

CUDF_KERNEL void reduce_merge_hll_kernel_vertically(cudf::device_span<int64_t const*> sketch_longs,
                                                    size_type num_sketches,
                                                    int num_registers_per_sketch,
                                                    int* const output)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_registers_per_sketch) { return; }
  auto long_idx        = tid / REGISTERS_PER_LONG;
  auto reg_idx_in_long = tid % REGISTERS_PER_LONG;
  int max              = 0;
  for (auto row_idx = 0; row_idx < num_sketches; row_idx++) {
    int reg_v = get_register_value(sketch_longs[long_idx][row_idx], reg_idx_in_long);
    if (reg_v > max) { max = reg_v; }
  }
  output[tid] = max;
}

std::unique_ptr<scalar> reduce_merge_hllpp(column_view const& input,
                                           int64_t const precision,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  // create device input
  int64_t num_registers_per_sketch = 1 << precision;
  auto num_long_cols               = num_registers_per_sketch / REGISTERS_PER_LONG + 1;
  cudf::structs_column_view scv(input);
  auto const input_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](int i) { return scv.get_sliced_child(i, stream).begin<int64_t>(); });
  auto input_cols = std::vector<int64_t const*>(input_iter, input_iter + num_long_cols);
  auto d_inputs   = cudf::detail::make_device_uvector_async(input_cols, stream, mr);

  // create one row output
  auto const results_iter = cudf::detail::make_counting_transform_iterator(0, [&](int i) {
    return make_numeric_column(
      data_type{type_id::INT64}, 1 /** num_rows */, mask_state::ALL_VALID, stream, mr);
  });
  auto children  = std::vector<std::unique_ptr<column>>(results_iter, results_iter + num_long_cols);
  auto d_results = [&] {
    auto host_results_pointer_iter =
      thrust::make_transform_iterator(children.begin(), [](auto const& results_column) {
        return results_column->mutable_view().template data<int64_t>();
      });
    auto host_results_pointers =
      std::vector<int64_t*>(host_results_pointer_iter, host_results_pointer_iter + children.size());
    return cudf::detail::make_device_uvector_async(host_results_pointers, stream, mr);
  }();

  // execute merge kernel
  auto num_threads             = num_registers_per_sketch;
  constexpr int64_t block_size = 256;
  auto num_blocks              = cudf::util::div_rounding_up_safe(num_threads, block_size);
  auto output_cache            = rmm::device_uvector<int32_t>(num_registers_per_sketch, stream, mr);
  reduce_merge_hll_kernel_vertically<<<num_blocks, block_size, 0, stream.value()>>>(
    d_inputs, input.size(), num_registers_per_sketch, output_cache.begin());

  // compact to longs
  auto const num_compact_threads = num_long_cols;
  auto const num_compact_blocks = cudf::util::div_rounding_up_safe(num_compact_threads, block_size);
  compact_kernel<<<num_compact_blocks, block_size, 0, stream.value()>>>(
    1 /** num_groups **/, num_registers_per_sketch, d_results, output_cache);

  // create scalar
  auto host_results_view_iter = thrust::make_transform_iterator(
    children.begin(), [](auto const& results_column) { return results_column->view(); });
  auto views =
    std::vector<column_view>(host_results_view_iter, host_results_view_iter + num_long_cols);
  auto table_view = cudf::table_view{views};
  auto table      = cudf::table(table_view);
  return std::make_unique<cudf::struct_scalar>(std::move(table), true, stream, mr);
}

}  // namespace

/**
 * Compute hyper log log for the input values and merge the sketches in the same group.
 * Output is a struct column with multiple long columns which is consistent with Spark.
 */
std::unique_ptr<column> group_hyper_log_log_plus_plus(
  column_view const& input,
  int64_t const num_groups,
  cudf::device_span<size_type const> group_lables,
  int64_t const precision,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(precision >= 4, "HyperLogLogPlusPlus requires precision >= 4.");
  auto adjust_precision = precision > MAX_PRECISION ? MAX_PRECISION : precision;
  return group_hllpp(input, num_groups, group_lables, adjust_precision, stream, mr);
}

/**
 * Merge sketches in the same group.
 * Input is a struct column with multiple long columns which is consistent with Spark.
 */
std::unique_ptr<column> group_merge_hyper_log_log_plus_plus(
  column_view const& input,
  int64_t const num_groups,
  cudf::device_span<size_type const> group_lables,
  int64_t const precision,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(precision >= 4, "HyperLogLogPlusPlus requires precision >= 4.");
  CUDF_EXPECTS(input.type().id() == type_id::STRUCT,
               "HyperLogLogPlusPlus buffer type must be a STRUCT of long columns.");
  for (auto i = 0; i < input.num_children(); i++) {
    CUDF_EXPECTS(input.child(i).type().id() == type_id::INT64,
                 "HyperLogLogPlusPlus buffer type must be a STRUCT of long columns.");
  }
  auto adjust_precision   = precision > MAX_PRECISION ? MAX_PRECISION : precision;
  auto expected_num_longs = (1 << adjust_precision) / REGISTERS_PER_LONG + 1;
  CUDF_EXPECTS(input.num_children() == expected_num_longs,
               "The num of long columns in input is incorrect.");
  return merge_hyper_log_log(input, num_groups, group_lables, adjust_precision, stream, mr);
}

/**
 * Compute the hashs of the input column, then generate a sketch stored in a struct of long scalar.
 */
std::unique_ptr<scalar> reduce_hyper_log_log_plus_plus(column_view const& input,
                                                       int64_t const precision,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(precision >= 4, "HyperLogLogPlusPlus requires precision >= 4.");
  auto adjust_precision = precision > MAX_PRECISION ? MAX_PRECISION : precision;
  return reduce_hllpp(input, adjust_precision, stream, mr);
}

/**
 * Merge all sketches in the input column into one sketch.
 * Input is a struct column with multiple long columns which is consistent with Spark.
 */
std::unique_ptr<scalar> reduce_merge_hyper_log_log_plus_plus(column_view const& input,
                                                             int64_t const precision,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(precision >= 4, "HyperLogLogPlusPlus requires precision >= 4.");
  CUDF_EXPECTS(input.type().id() == type_id::STRUCT,
               "HyperLogLogPlusPlus buffer type must be a STRUCT of long columns.");
  for (auto i = 0; i < input.num_children(); i++) {
    CUDF_EXPECTS(input.child(i).type().id() == type_id::INT64,
                 "HyperLogLogPlusPlus buffer type must be a STRUCT of long columns.");
  }
  auto adjust_precision   = precision > MAX_PRECISION ? MAX_PRECISION : precision;
  auto expected_num_longs = (1 << adjust_precision) / REGISTERS_PER_LONG + 1;
  CUDF_EXPECTS(input.num_children() == expected_num_longs,
               "The num of long columns in input is incorrect.");
  return reduce_merge_hllpp(input, adjust_precision, stream, mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
