/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include "column_stats.h"

#include <io/utilities/block_utils.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <cub/cub.cuh>

#include <math_constants.h>

constexpr int block_size = 1024;

namespace cudf {
namespace io {
/**
 * @brief shared state for statistics gather kernel
 */
struct stats_state_s {
  stats_column_desc col;                 ///< Column information
  statistics_group group;                ///< Group description
  statistics_chunk ck;                   ///< Output statistics chunk
  volatile statistics_val warp_min[32];  ///< Min reduction scratch
  volatile statistics_val warp_max[32];  ///< Max reduction scratch
};

/**
 * @brief shared state for statistics merge kernel
 */
struct merge_state_s {
  stats_column_desc col;                 ///< Column information
  statistics_merge_group group;          ///< Group description
  statistics_chunk ck;                   ///< Resulting statistics chunk
  volatile statistics_val warp_min[32];  ///< Min reduction scratch
  volatile statistics_val warp_max[32];  ///< Max reduction scratch
};

/**
 * Custom addition functor to ignore NaN inputs
 */
struct IgnoreNaNSum {
  __device__ __forceinline__ double operator()(const double &a, const double &b)
  {
    double aval = isnan(a) ? 0 : a;
    double bval = isnan(b) ? 0 : b;
    return aval + bval;
  }
};

/**
 * Warp-wide Min reduction for string types
 */
inline __device__ string_stats WarpReduceMinString(const char *smin, uint32_t lmin)
{
  uint32_t len = shuffle_xor(lmin, 1);
  const char *ptr =
    reinterpret_cast<const char *>(shuffle_xor(reinterpret_cast<uintptr_t>(smin), 1));
  if (!smin || (ptr && nvstr_is_lesser(ptr, len, smin, lmin))) {
    smin = ptr;
    lmin = len;
  }
  len = shuffle_xor(lmin, 2);
  ptr = reinterpret_cast<const char *>(shuffle_xor(reinterpret_cast<uintptr_t>(smin), 2));
  if (!smin || (ptr && nvstr_is_lesser(ptr, len, smin, lmin))) {
    smin = ptr;
    lmin = len;
  }
  len = shuffle_xor(lmin, 4);
  ptr = reinterpret_cast<const char *>(shuffle_xor(reinterpret_cast<uintptr_t>(smin), 4));
  if (!smin || (ptr && nvstr_is_lesser(ptr, len, smin, lmin))) {
    smin = ptr;
    lmin = len;
  }
  len = shuffle_xor(lmin, 8);
  ptr = reinterpret_cast<const char *>(shuffle_xor(reinterpret_cast<uintptr_t>(smin), 8));
  if (!smin || (ptr && nvstr_is_lesser(ptr, len, smin, lmin))) {
    smin = ptr;
    lmin = len;
  }
  len = shuffle_xor(lmin, 16);
  ptr = reinterpret_cast<const char *>(shuffle_xor(reinterpret_cast<uintptr_t>(smin), 16));
  if (!smin || (ptr && nvstr_is_lesser(ptr, len, smin, lmin))) {
    smin = ptr;
    lmin = len;
  }
  return {smin, lmin};
}

/**
 * Warp-wide Max reduction for string types
 */
inline __device__ string_stats WarpReduceMaxString(const char *smax, uint32_t lmax)
{
  uint32_t len = shuffle_xor(lmax, 1);
  const char *ptr =
    reinterpret_cast<const char *>(shuffle_xor(reinterpret_cast<uintptr_t>(smax), 1));
  if (!smax || (ptr && nvstr_is_greater(ptr, len, smax, lmax))) {
    smax = ptr;
    lmax = len;
  }
  len = shuffle_xor(lmax, 2);
  ptr = reinterpret_cast<const char *>(shuffle_xor(reinterpret_cast<uintptr_t>(smax), 2));
  if (!smax || (ptr && nvstr_is_greater(ptr, len, smax, lmax))) {
    smax = ptr;
    lmax = len;
  }
  len = shuffle_xor(lmax, 4);
  ptr = reinterpret_cast<const char *>(shuffle_xor(reinterpret_cast<uintptr_t>(smax), 4));
  if (!smax || (ptr && nvstr_is_greater(ptr, len, smax, lmax))) {
    smax = ptr;
    lmax = len;
  }
  len = shuffle_xor(lmax, 8);
  ptr = reinterpret_cast<const char *>(shuffle_xor(reinterpret_cast<uintptr_t>(smax), 8));
  if (!smax || (ptr && nvstr_is_greater(ptr, len, smax, lmax))) {
    smax = ptr;
    lmax = len;
  }
  len = shuffle_xor(lmax, 16);
  ptr = reinterpret_cast<const char *>(shuffle_xor(reinterpret_cast<uintptr_t>(smax), 16));
  if (!smax || (ptr && nvstr_is_greater(ptr, len, smax, lmax))) {
    smax = ptr;
    lmax = len;
  }
  return {smax, lmax};
}

/**
 * @brief Gather statistics for integer-like columns
 *
 * @param s shared block state
 * @param dtype data type
 * @param t thread id
 * @param storage temporary storage for reduction
 */
template <typename Storage>
void __device__
gatherIntColumnStats(stats_state_s *s, statistics_dtype dtype, uint32_t t, Storage &storage)
{
  using block_reduce = cub::BlockReduce<int64_t, block_size>;
  int64_t vmin       = INT64_MAX;
  int64_t vmax       = INT64_MIN;
  int64_t vsum       = 0;
  int64_t v;
  uint32_t nn_cnt = 0;
  __shared__ volatile bool has_minmax;
  for (uint32_t i = 0; i < s->group.num_rows; i += block_size) {
    uint32_t r                = i + t;
    uint32_t row              = r + s->group.start_row;
    const uint32_t *valid_map = s->col.valid_map_base;
    uint32_t is_valid         = (r < s->group.num_rows && row < s->col.num_values)
                          ? (valid_map) ? (valid_map[(row + s->col.column_offset) / 32] >>
                                           ((row + s->col.column_offset) % 32)) &
                                            1
                                        : 1
                          : 0;
    if (is_valid) {
      switch (dtype) {
        case dtype_int32:
        case dtype_date32: v = static_cast<const int32_t *>(s->col.column_data_base)[row]; break;
        case dtype_int64:
        case dtype_decimal64: v = static_cast<const int64_t *>(s->col.column_data_base)[row]; break;
        case dtype_int16: v = static_cast<const int16_t *>(s->col.column_data_base)[row]; break;
        case dtype_timestamp64:
          v = static_cast<const int64_t *>(s->col.column_data_base)[row];
          if (s->col.ts_scale < -1) {
            v /= -s->col.ts_scale;
          } else if (s->col.ts_scale > 1) {
            v *= s->col.ts_scale;
          }
          break;
        default: v = static_cast<const int8_t *>(s->col.column_data_base)[row]; break;
      }
      vmin = min(vmin, v);
      vmax = max(vmax, v);
      vsum += v;
    }
    nn_cnt += __syncthreads_count(is_valid);
  }
  if (!t) {
    s->ck.non_nulls  = nn_cnt;
    s->ck.null_count = s->group.num_rows - nn_cnt;
  }
  vmin = block_reduce(storage.integer_stats).Reduce(vmin, cub::Min());
  __syncthreads();
  vmax = block_reduce(storage.integer_stats).Reduce(vmax, cub::Max());
  if (!t) { has_minmax = (vmin <= vmax); }
  __syncthreads();
  if (has_minmax) { vsum = block_reduce(storage.integer_stats).Sum(vsum); }
  if (!t) {
    if (has_minmax) {
      s->ck.min_value.i_val = vmin;
      s->ck.max_value.i_val = vmax;
      s->ck.sum.i_val       = vsum;
    }
    s->ck.has_minmax = has_minmax;
    // TODO: For now, don't set the sum flag with 64-bit values so we don't have to check for
    // 64-bit sum overflow
    s->ck.has_sum = (dtype <= dtype_int32 && has_minmax);
  }
}

/**
 * @brief Gather statistics for floating-point columns
 *
 * @param s shared block state
 * @param dtype data type
 * @param t thread id
 * @param storage temporary storage for reduction
 */
template <typename Storage>
void __device__
gatherFloatColumnStats(stats_state_s *s, statistics_dtype dtype, uint32_t t, Storage &storage)
{
  using block_reduce = cub::BlockReduce<double, block_size>;
  double vmin        = CUDART_INF;
  double vmax        = -CUDART_INF;
  double vsum        = 0;
  double v;
  uint32_t nn_cnt = 0;
  __shared__ volatile bool has_minmax;
  for (uint32_t i = 0; i < s->group.num_rows; i += block_size) {
    uint32_t r                = i + t;
    uint32_t row              = r + s->group.start_row;
    const uint32_t *valid_map = s->col.valid_map_base;
    uint32_t is_valid         = (r < s->group.num_rows && row < s->col.num_values)
                          ? (valid_map) ? (valid_map[(row + s->col.column_offset) >> 5] >>
                                           ((row + s->col.column_offset) & 0x1f)) &
                                            1
                                        : 1
                          : 0;
    if (is_valid) {
      if (dtype == dtype_float64) {
        v = static_cast<const double *>(s->col.column_data_base)[row];
      } else {
        v = static_cast<const float *>(s->col.column_data_base)[row];
      }
      if (v < vmin) { vmin = v; }
      if (v > vmax) { vmax = v; }
      if (!isnan(v)) { vsum += v; }
    }
    nn_cnt += __syncthreads_count(is_valid);
  }
  if (!t) {
    s->ck.non_nulls  = nn_cnt;
    s->ck.null_count = s->group.num_rows - nn_cnt;
  }
  vmin = block_reduce(storage.float_stats).Reduce(vmin, cub::Min());
  __syncthreads();
  vmax = block_reduce(storage.float_stats).Reduce(vmax, cub::Max());
  if (!t) { has_minmax = (vmin <= vmax); }
  __syncthreads();
  if (has_minmax) { vsum = block_reduce(storage.float_stats).Reduce(vsum, IgnoreNaNSum()); }
  if (!t) {
    if (has_minmax) {
      s->ck.min_value.fp_val = (vmin != 0.0) ? vmin : CUDART_NEG_ZERO;
      s->ck.max_value.fp_val = (vmax != 0.0) ? vmax : CUDART_ZERO;
      s->ck.sum.fp_val       = vsum;
    }
    s->ck.has_minmax = has_minmax;
    s->ck.has_sum    = has_minmax;  // Implies sum is valid as well
  }
}

// FIXME: Use native libcudf string type
struct nvstrdesc_s {
  const char *ptr;
  size_t count;
};

/**
 * @brief Gather statistics for string columns
 *
 * @param s shared block state
 * @param t thread id
 * @param storage temporary storage for reduction
 */
template <typename Storage>
void __device__ gatherStringColumnStats(stats_state_s *s, uint32_t t, Storage &storage)
{
  using block_reduce = cub::BlockReduce<uint32_t, block_size>;
  uint32_t len_sum   = 0;
  const char *smin   = nullptr;
  const char *smax   = nullptr;
  uint32_t lmin      = 0;
  uint32_t lmax      = 0;
  uint32_t nn_cnt    = 0;
  bool has_minmax;
  string_stats minval, maxval;

  for (uint32_t i = 0; i < s->group.num_rows; i += block_size) {
    uint32_t r                = i + t;
    uint32_t row              = r + s->group.start_row;
    const uint32_t *valid_map = s->col.valid_map_base;
    uint32_t is_valid         = (r < s->group.num_rows && row < s->col.num_values)
                          ? (valid_map) ? (valid_map[(row + s->col.column_offset) >> 5] >>
                                           ((row + s->col.column_offset) & 0x1f)) &
                                            1
                                        : 1
                          : 0;
    if (is_valid) {
      const nvstrdesc_s *str_col = static_cast<const nvstrdesc_s *>(s->col.column_data_base);
      uint32_t len               = (uint32_t)str_col[row].count;
      const char *ptr            = str_col[row].ptr;
      len_sum += len;
      if (!smin || nvstr_is_lesser(ptr, len, smin, lmin)) {
        lmin = len;
        smin = ptr;
      }
      if (!smax || nvstr_is_greater(ptr, len, smax, lmax)) {
        lmax = len;
        smax = ptr;
      }
    }
    nn_cnt += __syncthreads_count(is_valid);
  }
  if (!t) {
    s->ck.non_nulls  = nn_cnt;
    s->ck.null_count = s->group.num_rows - nn_cnt;
  }
  minval = WarpReduceMinString(smin, lmin);
  maxval = WarpReduceMaxString(smax, lmax);
  __syncwarp();
  if (!(t & 0x1f)) {
    s->warp_min[t >> 5].str_val.ptr    = minval.ptr;
    s->warp_min[t >> 5].str_val.length = minval.length;
    s->warp_max[t >> 5].str_val.ptr    = maxval.ptr;
    s->warp_max[t >> 5].str_val.length = maxval.length;
  }
  has_minmax = __syncthreads_or(smin != nullptr);
  if (has_minmax) { len_sum = block_reduce(storage.string_stats).Sum(len_sum); }
  if (t < 32 * 1) {
    minval = WarpReduceMinString(s->warp_min[t].str_val.ptr, s->warp_min[t].str_val.length);
    if (!(t & 0x1f)) {
      if (has_minmax) {
        s->ck.min_value.str_val.ptr    = minval.ptr;
        s->ck.min_value.str_val.length = minval.length;
        s->ck.sum.i_val                = len_sum;
      }
      s->ck.has_minmax = has_minmax;
      s->ck.has_sum    = has_minmax;
    }
  } else if (t < 32 * 2 and has_minmax) {
    maxval =
      WarpReduceMaxString(s->warp_max[t & 0x1f].str_val.ptr, s->warp_max[t & 0x1f].str_val.length);
    if (!(t & 0x1f)) {
      s->ck.max_value.str_val.ptr    = maxval.ptr;
      s->ck.max_value.str_val.length = maxval.length;
    }
  }
}

/**
 * @brief Gather column chunk statistics (min/max values, sum and null count)
 * for a group of rows.
 *
 * blockDim {1024,1,1}
 *
 * @param chunks Destination statistics results
 * @param groups Statistics source information
 */
template <int block_size>
__global__ void __launch_bounds__(block_size, 1)
  gpuGatherColumnStatistics(statistics_chunk *chunks, const statistics_group *groups)
{
  __shared__ __align__(8) stats_state_s state_g;
  __shared__ union {
    typename cub::BlockReduce<int64_t, block_size>::TempStorage integer_stats;
    typename cub::BlockReduce<double, block_size>::TempStorage float_stats;
    typename cub::BlockReduce<uint32_t, block_size>::TempStorage string_stats;
  } temp_storage;

  stats_state_s *const s = &state_g;
  uint32_t t             = threadIdx.x;
  statistics_dtype dtype;

  if (t < sizeof(statistics_group) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->group)[t] =
      reinterpret_cast<const uint32_t *>(&groups[blockIdx.x])[t];
  }
  if (t < sizeof(statistics_chunk) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->ck)[t] = 0;
  }
  __syncthreads();
  if (t < sizeof(stats_column_desc) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->col)[t] = reinterpret_cast<const uint32_t *>(s->group.col)[t];
  }
  __syncthreads();
  dtype = s->col.stats_dtype;
  if (dtype >= dtype_bool && dtype <= dtype_decimal64) {
    gatherIntColumnStats(s, dtype, t, temp_storage);
  } else if (dtype >= dtype_float32 && dtype <= dtype_float64) {
    gatherFloatColumnStats(s, dtype, t, temp_storage);
  } else if (dtype == dtype_string) {
    gatherStringColumnStats(s, t, temp_storage);
  }
  __syncthreads();
  if (t < sizeof(statistics_chunk) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&chunks[blockIdx.x])[t] = reinterpret_cast<uint32_t *>(&s->ck)[t];
  }
}

/**
 * @brief Merge statistics for integer-like columns
 *
 * @param s shared block state
 * @param dtype data type
 * @param ck_in pointer to first statistic chunk
 * @param num_chunks number of statistic chunks to merge
 * @param t thread id
 * @param storage temporary storage for reduction
 */
template <typename Storage>
void __device__ mergeIntColumnStats(merge_state_s *s,
                                    statistics_dtype dtype,
                                    const statistics_chunk *ck_in,
                                    uint32_t num_chunks,
                                    uint32_t t,
                                    Storage &storage)
{
  int64_t vmin        = INT64_MAX;
  int64_t vmax        = INT64_MIN;
  int64_t vsum        = 0;
  uint32_t non_nulls  = 0;
  uint32_t null_count = 0;
  __shared__ volatile bool has_minmax;
  for (uint32_t i = t; i < num_chunks; i += block_size) {
    const statistics_chunk *ck = &ck_in[i];
    if (ck->has_minmax) {
      vmin = min(vmin, ck->min_value.i_val);
      vmax = max(vmax, ck->max_value.i_val);
    }
    if (ck->has_sum) { vsum += ck->sum.i_val; }
    non_nulls += ck->non_nulls;
    null_count += ck->null_count;
  }
  vmin = cub::BlockReduce<int64_t, block_size>(storage.i64).Reduce(vmin, cub::Min());
  __syncthreads();
  vmax = cub::BlockReduce<int64_t, block_size>(storage.i64).Reduce(vmax, cub::Max());
  if (!t) { has_minmax = (vmin <= vmax); }
  __syncthreads();
  non_nulls = cub::BlockReduce<uint32_t, block_size>(storage.u32).Sum(non_nulls);
  __syncthreads();
  null_count = cub::BlockReduce<uint32_t, block_size>(storage.u32).Sum(null_count);
  __syncthreads();
  if (has_minmax) { vsum = cub::BlockReduce<int64_t, block_size>(storage.i64).Sum(vsum); }

  if (!t) {
    if (has_minmax) {
      s->ck.min_value.i_val = vmin;
      s->ck.max_value.i_val = vmax;
      s->ck.sum.i_val       = vsum;
    }
    s->ck.has_minmax = has_minmax;
    // TODO: For now, don't set the sum flag with 64-bit values so we don't have to check for
    // 64-bit sum overflow
    s->ck.has_sum    = (dtype <= dtype_int32 && has_minmax);
    s->ck.non_nulls  = non_nulls;
    s->ck.null_count = null_count;
  }
}

/**
 * @brief Merge statistics for floating-point columns
 *
 * @param s shared block state
 * @param dtype data type
 * @param ck_in pointer to first statistic chunk
 * @param num_chunks number of statistic chunks to merge
 * @param t thread id
 * @param storage temporary storage for reduction
 */
template <typename Storage>
void __device__ mergeFloatColumnStats(merge_state_s *s,
                                      const statistics_chunk *ck_in,
                                      uint32_t num_chunks,
                                      uint32_t t,
                                      Storage &storage)
{
  double vmin         = CUDART_INF;
  double vmax         = -CUDART_INF;
  double vsum         = 0;
  uint32_t non_nulls  = 0;
  uint32_t null_count = 0;
  __shared__ volatile bool has_minmax;
  for (uint32_t i = t; i < num_chunks; i += block_size) {
    const statistics_chunk *ck = &ck_in[i];
    if (ck->has_minmax) {
      double v0 = ck->min_value.fp_val;
      double v1 = ck->max_value.fp_val;
      if (v0 < vmin) { vmin = v0; }
      if (v1 > vmax) { vmax = v1; }
    }
    if (ck->has_sum) { vsum += ck->sum.fp_val; }
    non_nulls += ck->non_nulls;
    null_count += ck->null_count;
  }

  vmin = cub::BlockReduce<double, block_size>(storage.f64).Reduce(vmin, cub::Min());
  __syncthreads();
  vmax = cub::BlockReduce<double, block_size>(storage.f64).Reduce(vmax, cub::Max());
  if (!t) { has_minmax = (vmin <= vmax); }
  __syncthreads();
  non_nulls = cub::BlockReduce<uint32_t, block_size>(storage.u32).Sum(non_nulls);
  __syncthreads();
  null_count = cub::BlockReduce<uint32_t, block_size>(storage.u32).Sum(null_count);
  __syncthreads();
  if (has_minmax) {
    vsum = cub::BlockReduce<double, block_size>(storage.f64).Reduce(vsum, IgnoreNaNSum());
  }

  if (!t) {
    if (has_minmax) {
      s->ck.min_value.fp_val = (vmin != 0.0) ? vmin : CUDART_NEG_ZERO;
      s->ck.max_value.fp_val = (vmax != 0.0) ? vmax : CUDART_ZERO;
      s->ck.sum.fp_val       = vsum;
    }
    s->ck.has_minmax = has_minmax;
    s->ck.has_sum    = has_minmax;  // Implies sum is valid as well
    s->ck.non_nulls  = non_nulls;
    s->ck.null_count = null_count;
  }
}

/**
 * @brief Merge statistics for string columns
 *
 * @param s shared block state
 * @param ck_in pointer to first statistic chunk
 * @param num_chunks number of statistic chunks to merge
 * @param t thread id
 * @param storage temporary storage for reduction
 */
template <typename Storage>
void __device__ mergeStringColumnStats(merge_state_s *s,
                                       const statistics_chunk *ck_in,
                                       uint32_t num_chunks,
                                       uint32_t t,
                                       Storage &storage)
{
  uint32_t len_sum    = 0;
  const char *smin    = nullptr;
  const char *smax    = nullptr;
  uint32_t lmin       = 0;
  uint32_t lmax       = 0;
  uint32_t non_nulls  = 0;
  uint32_t null_count = 0;
  bool has_minmax;
  string_stats minval, maxval;

  for (uint32_t i = t; i < num_chunks; i += block_size) {
    const statistics_chunk *ck = &ck_in[i];
    if (ck->has_minmax) {
      uint32_t len0    = ck->min_value.str_val.length;
      const char *ptr0 = ck->min_value.str_val.ptr;
      uint32_t len1    = ck->max_value.str_val.length;
      const char *ptr1 = ck->max_value.str_val.ptr;
      if (!smin || (ptr0 && nvstr_is_lesser(ptr0, len0, smin, lmin))) {
        lmin = len0;
        smin = ptr0;
      }
      if (!smax || (ptr1 && nvstr_is_greater(ptr1, len1, smax, lmax))) {
        lmax = len1;
        smax = ptr1;
      }
    }
    if (ck->has_sum) { len_sum += (uint32_t)ck->sum.i_val; }
    non_nulls += ck->non_nulls;
    null_count += ck->null_count;
  }
  minval = WarpReduceMinString(smin, lmin);
  maxval = WarpReduceMaxString(smax, lmax);
  if (!(t & 0x1f)) {
    s->warp_min[t >> 5].str_val.ptr    = minval.ptr;
    s->warp_min[t >> 5].str_val.length = minval.length;
    s->warp_max[t >> 5].str_val.ptr    = maxval.ptr;
    s->warp_max[t >> 5].str_val.length = maxval.length;
  }
  has_minmax = __syncthreads_or(smin != nullptr);

  non_nulls = cub::BlockReduce<uint32_t, block_size>(storage.u32).Sum(non_nulls);
  __syncthreads();
  null_count = cub::BlockReduce<uint32_t, block_size>(storage.u32).Sum(null_count);
  __syncthreads();
  if (has_minmax) { len_sum = cub::BlockReduce<uint32_t, block_size>(storage.u32).Sum(len_sum); }
  if (t < 32 * 1) {
    minval = WarpReduceMinString(s->warp_min[t].str_val.ptr, s->warp_min[t].str_val.length);
    if (!(t & 0x1f)) {
      if (has_minmax) {
        s->ck.min_value.str_val.ptr    = minval.ptr;
        s->ck.min_value.str_val.length = minval.length;
        s->ck.sum.i_val                = len_sum;
      }
      s->ck.has_minmax = has_minmax;
      s->ck.has_sum    = has_minmax;
      s->ck.non_nulls  = non_nulls;
      s->ck.null_count = null_count;
    }
  } else if (t < 32 * 2) {
    maxval =
      WarpReduceMaxString(s->warp_max[t & 0x1f].str_val.ptr, s->warp_max[t & 0x1f].str_val.length);
    if (!((t & 0x1f) and has_minmax)) {
      s->ck.max_value.str_val.ptr    = maxval.ptr;
      s->ck.max_value.str_val.length = maxval.length;
    }
  }
}

/**
 * @brief Combine multiple statistics chunk together to form new statistics chunks
 *
 * blockDim {1024,1,1}
 *
 * @param chunks_out Destination statistic chunks
 * @param chunks_in Source statistic chunks
 * @param groups Statistic chunk grouping information
 */
template <int block_size>
__global__ void __launch_bounds__(block_size, 1)
  gpuMergeColumnStatistics(statistics_chunk *chunks_out,
                           const statistics_chunk *chunks_in,
                           const statistics_merge_group *groups)
{
  __shared__ __align__(8) merge_state_s state_g;
  __shared__ struct {
    typename cub::BlockReduce<uint32_t, block_size>::TempStorage u32;
    typename cub::BlockReduce<int64_t, block_size>::TempStorage i64;
    typename cub::BlockReduce<double, block_size>::TempStorage f64;
  } storage;

  merge_state_s *const s = &state_g;
  uint32_t t             = threadIdx.x;
  statistics_dtype dtype;

  if (t < sizeof(statistics_merge_group) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->group)[t] =
      reinterpret_cast<const uint32_t *>(&groups[blockIdx.x])[t];
  }
  __syncthreads();
  if (t < sizeof(stats_column_desc) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&s->col)[t] = reinterpret_cast<const uint32_t *>(s->group.col)[t];
  }
  __syncthreads();
  dtype = s->col.stats_dtype;

  if (dtype >= dtype_bool && dtype <= dtype_decimal64) {
    mergeIntColumnStats(
      s, dtype, chunks_in + s->group.start_chunk, s->group.num_chunks, t, storage);
  } else if (dtype >= dtype_float32 && dtype <= dtype_float64) {
    mergeFloatColumnStats(s, chunks_in + s->group.start_chunk, s->group.num_chunks, t, storage);
  } else if (dtype == dtype_string) {
    mergeStringColumnStats(s, chunks_in + s->group.start_chunk, s->group.num_chunks, t, storage);
  }

  __syncthreads();
  if (t < sizeof(statistics_chunk) / sizeof(uint32_t)) {
    reinterpret_cast<uint32_t *>(&chunks_out[blockIdx.x])[t] =
      reinterpret_cast<uint32_t *>(&s->ck)[t];
  }
}

/**
 * @brief Launches kernel to gather column statistics
 *
 * @param[out] chunks Statistics results [num_chunks]
 * @param[in] groups Statistics row groups [num_chunks]
 * @param[in] num_chunks Number of chunks & rowgroups
 * @param[in] stream CUDA stream to use, default 0
 */
void GatherColumnStatistics(statistics_chunk *chunks,
                            const statistics_group *groups,
                            uint32_t num_chunks,
                            rmm::cuda_stream_view stream)
{
  gpuGatherColumnStatistics<block_size>
    <<<num_chunks, block_size, 0, stream.value()>>>(chunks, groups);
}

/**
 * @brief Launches kernel to merge column statistics
 *
 * @param[out] chunks_out Statistics results [num_chunks]
 * @param[out] chunks_in Input statistics
 * @param[in] groups Statistics groups [num_chunks]
 * @param[in] num_chunks Number of chunks & groups
 * @param[in] stream CUDA stream to use, default 0
 */
void MergeColumnStatistics(statistics_chunk *chunks_out,
                           const statistics_chunk *chunks_in,
                           const statistics_merge_group *groups,
                           uint32_t num_chunks,
                           rmm::cuda_stream_view stream)
{
  gpuMergeColumnStatistics<block_size>
    <<<num_chunks, block_size, 0, stream.value()>>>(chunks_out, chunks_in, groups);
}

}  // namespace io
}  // namespace cudf
