/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <math_constants.h>
#include <io/utilities/block_utils.cuh>

namespace cudf {
namespace io {

struct stats_state_s
{
    stats_column_desc col;
    statistics_group group;
    statistics_chunk ck;
    volatile statistics_val warp_min[32];
    volatile statistics_val warp_max[32];
    volatile statistics_val warp_sum[32];
};

struct merge_state_s
{
    stats_column_desc col;
    statistics_merge_group group;
    statistics_chunk ck;
    volatile statistics_val warp_min[32];
    volatile statistics_val warp_max[32];
    volatile statistics_val warp_sum[32];
    volatile uint32_t warp_non_nulls[32];
    volatile uint32_t warp_nulls[32];
};


inline __device__ int64_t WarpReduceMinInt(int64_t vmin)
{
    int64_t v = SHFL_XOR(vmin, 1);
    vmin = min(vmin, v);
    v = SHFL_XOR(vmin, 2);
    vmin = min(vmin, v);
    v = SHFL_XOR(vmin, 4);
    vmin = min(vmin, v);
    v = SHFL_XOR(vmin, 8);
    vmin = min(vmin, v);
    v = SHFL_XOR(vmin, 16);
    return min(vmin, v);
}

inline __device__ int64_t WarpReduceMaxInt(int64_t vmax)
{
    int64_t v = SHFL_XOR(vmax, 1);
    vmax = max(vmax, v);
    v = SHFL_XOR(vmax, 2);
    vmax = max(vmax, v);
    v = SHFL_XOR(vmax, 4);
    vmax = max(vmax, v);
    v = SHFL_XOR(vmax, 8);
    vmax = max(vmax, v);
    v = SHFL_XOR(vmax, 16);
    return max(vmax, v);
}

inline __device__ double WarpReduceMinFloat(double vmin)
{
    double v = SHFL_XOR(vmin, 1);
    vmin = fmin(vmin, v);
    v = SHFL_XOR(vmin, 2);
    vmin = fmin(vmin, v);
    v = SHFL_XOR(vmin, 4);
    vmin = fmin(vmin, v);
    v = SHFL_XOR(vmin, 8);
    vmin = fmin(vmin, v);
    v = SHFL_XOR(vmin, 16);
    return fmin(vmin, v);
}

inline __device__ double WarpReduceMaxFloat(double vmax)
{
    double v = SHFL_XOR(vmax, 1);
    vmax = fmax(vmax, v);
    v = SHFL_XOR(vmax, 2);
    vmax = fmax(vmax, v);
    v = SHFL_XOR(vmax, 4);
    vmax = fmax(vmax, v);
    v = SHFL_XOR(vmax, 8);
    vmax = fmax(vmax, v);
    v = SHFL_XOR(vmax, 16);
    return fmax(vmax, v);
}

inline __device__ double WarpReduceSumFloat(double vsum)
{
    double v = SHFL_XOR(vsum, 1);
    if (!isnan(v)) vsum += v;
    v = SHFL_XOR(vsum, 2);
    if (!isnan(v)) vsum += v;
    v = SHFL_XOR(vsum, 4);
    if (!isnan(v)) vsum += v;
    v = SHFL_XOR(vsum, 8);
    if (!isnan(v)) vsum += v;
    v = SHFL_XOR(vsum, 16);
    if (!isnan(v)) vsum += v;
    return vsum;
}


inline __device__ string_stats WarpReduceMinString(const char *smin, uint32_t lmin)
{
    uint32_t len = SHFL_XOR(lmin, 1);
    const char *ptr = reinterpret_cast<const char *>(SHFL_XOR(reinterpret_cast<uintptr_t>(smin), 1));
    if (!smin || (ptr && nvstr_is_lesser(ptr, len, smin, lmin))) {
        smin = ptr;
        lmin = len;
    }
    len = SHFL_XOR(lmin, 2);
    ptr = reinterpret_cast<const char *>(SHFL_XOR(reinterpret_cast<uintptr_t>(smin), 2));
    if (!smin || (ptr && nvstr_is_lesser(ptr, len, smin, lmin))) {
        smin = ptr;
        lmin = len;
    }
    len = SHFL_XOR(lmin, 4);
    ptr = reinterpret_cast<const char *>(SHFL_XOR(reinterpret_cast<uintptr_t>(smin), 4));
    if (!smin || (ptr && nvstr_is_lesser(ptr, len, smin, lmin))) {
        smin = ptr;
        lmin = len;
    }
    len = SHFL_XOR(lmin, 8);
    ptr = reinterpret_cast<const char *>(SHFL_XOR(reinterpret_cast<uintptr_t>(smin), 8));
    if (!smin || (ptr && nvstr_is_lesser(ptr, len, smin, lmin))) {
        smin = ptr;
        lmin = len;
    }
    len = SHFL_XOR(lmin, 16);
    ptr = reinterpret_cast<const char *>(SHFL_XOR(reinterpret_cast<uintptr_t>(smin), 16));
    if (!smin || (ptr && nvstr_is_lesser(ptr, len, smin, lmin))) {
        smin = ptr;
        lmin = len;
    }
    return {smin, lmin};
}

inline __device__ string_stats WarpReduceMaxString(const char *smax, uint32_t lmax)
{
    uint32_t len = SHFL_XOR(lmax, 1);
    const char *ptr = reinterpret_cast<const char *>(SHFL_XOR(reinterpret_cast<uintptr_t>(smax), 1));
    if (!smax || (ptr && nvstr_is_greater(ptr, len, smax, lmax))) {
        smax = ptr;
        lmax = len;
    }
    len = SHFL_XOR(lmax, 2);
    ptr = reinterpret_cast<const char *>(SHFL_XOR(reinterpret_cast<uintptr_t>(smax), 2));
    if (!smax || (ptr && nvstr_is_greater(ptr, len, smax, lmax))) {
        smax = ptr;
        lmax = len;
    }
    len = SHFL_XOR(lmax, 4);
    ptr = reinterpret_cast<const char *>(SHFL_XOR(reinterpret_cast<uintptr_t>(smax), 4));
    if (!smax || (ptr && nvstr_is_greater(ptr, len, smax, lmax))) {
        smax = ptr;
        lmax = len;
    }
    len = SHFL_XOR(lmax, 8);
    ptr = reinterpret_cast<const char *>(SHFL_XOR(reinterpret_cast<uintptr_t>(smax), 8));
    if (!smax || (ptr && nvstr_is_greater(ptr, len, smax, lmax))) {
        smax = ptr;
        lmax = len;
    }
    len = SHFL_XOR(lmax, 16);
    ptr = reinterpret_cast<const char *>(SHFL_XOR(reinterpret_cast<uintptr_t>(smax), 16));
    if (!smax || (ptr && nvstr_is_greater(ptr, len, smax, lmax))) {
        smax = ptr;
        lmax = len;
    }
    return { smax, lmax };
}


void __device__ gatherIntColumnStats(stats_state_s *s, statistics_dtype dtype, uint32_t t)
{
    int64_t vmin = INT64_MAX;
    int64_t vmax = INT64_MIN;
    int64_t vsum = 0;
    int64_t v;
    uint32_t nn_cnt = 0;
    bool has_minmax;
    for (uint32_t i = 0; i < s->group.num_rows; i += 1024) {
        uint32_t r = i + t;
        uint32_t row = r + s->group.start_row;
        const uint32_t *valid_map = s->col.valid_map_base;
        uint32_t is_valid = (r < s->group.num_rows && row < s->col.num_rows) ? (valid_map) ? (valid_map[row >> 5] >> (row & 0x1f)) & 1 : 1 : 0;
        if (is_valid) {
            switch (dtype) {
            case dtype_int32:
                v = reinterpret_cast<const int32_t *>(s->col.column_data_base)[row];
                break;
            case dtype_int64:
                v = reinterpret_cast<const int64_t *>(s->col.column_data_base)[row];
                break;
            case dtype_int16:
                v = reinterpret_cast<const int16_t *>(s->col.column_data_base)[row];
                break;
            case dtype_timestamp64:
                v = reinterpret_cast<const int64_t *>(s->col.column_data_base)[row];
                if (s->col.ts_scale < -1) {
                    v /= -s->col.ts_scale;
                }
                else if (s->col.ts_scale > 1) {
                    v *= s->col.ts_scale;
                }
                break;
            default:
                v = reinterpret_cast<const int8_t *>(s->col.column_data_base)[row];
                break;
            }
            vmin = min(vmin, v);
            vmax = max(vmax, v);
            vsum += v;
        }
        nn_cnt += __syncthreads_count(is_valid);
    }
    if (!t) {
        s->ck.non_nulls = nn_cnt;
        s->ck.null_count = s->group.num_rows - nn_cnt;
    }
    vmin = WarpReduceMinInt(vmin);
    vmax = WarpReduceMaxInt(vmax);
    vsum = WarpReduceSum32(vsum);
    if (!(t & 0x1f)) {
        s->warp_min[t >> 5].i_val = vmin;
        s->warp_max[t >> 5].i_val = vmax;
        s->warp_sum[t >> 5].i_val = vsum;
    }
    has_minmax = __syncthreads_or(vmin <= vmax);
    if (t < 32 * 1) {
        vmin = WarpReduceMinInt(s->warp_min[t].i_val);
        if (!(t & 0x1f)) {
            s->ck.min_value.i_val = vmin;
            s->ck.has_minmax = (has_minmax);
        }
    }
    else if (t < 32 * 2) {
        vmax = WarpReduceMaxInt(s->warp_max[t & 0x1f].i_val);
        if (!(t & 0x1f)) {
            s->ck.max_value.i_val = vmax;
        }
    }
    else if (t < 32 * 3) {
        vsum = WarpReduceSum32(s->warp_sum[t & 0x1f].i_val);
        if (!(t & 0x1f)) {
            s->ck.sum.i_val = vsum;
            // TODO: For now, don't set the sum flag with 64-bit values so we don't have to check for 64-bit sum overflow
            s->ck.has_sum = (dtype <= dtype_int32 && has_minmax);
        }
    }
}


void __device__ gatherFloatColumnStats(stats_state_s *s, statistics_dtype dtype, uint32_t t)
{
    double vmin = CUDART_INF;
    double vmax = -CUDART_INF;
    double vsum = 0;
    double v;
    uint32_t nn_cnt = 0;
    bool has_minmax;
    for (uint32_t i = 0; i < s->group.num_rows; i += 1024) {
        uint32_t r = i + t;
        uint32_t row = r + s->group.start_row;
        const uint32_t *valid_map = s->col.valid_map_base;
        uint32_t is_valid = (r < s->group.num_rows && row < s->col.num_rows) ? (valid_map) ? (valid_map[row >> 5] >> (row & 0x1f)) & 1 : 1 : 0;
        if (is_valid) {
            if (dtype == dtype_float64) {
                v = reinterpret_cast<const double *>(s->col.column_data_base)[row];
            }
            else {
                v = reinterpret_cast<const float *>(s->col.column_data_base)[row];
            }
            if (v < vmin) {
                vmin = v;
            }
            if (v > vmax) {
                vmax = v;
            }
        }
        nn_cnt += __syncthreads_count(is_valid);
    }
    if (!t) {
        s->ck.non_nulls = nn_cnt;
        s->ck.null_count = s->group.num_rows - nn_cnt;
    }
    vmin = WarpReduceMinFloat(vmin);
    vmax = WarpReduceMaxFloat(vmax);
    vsum = WarpReduceSumFloat(vsum);
    if (!(t & 0x1f)) {
        s->warp_min[t >> 5].fp_val = vmin;
        s->warp_max[t >> 5].fp_val = vmax;
        s->warp_sum[t >> 5].fp_val = vsum;
    }
    has_minmax = __syncthreads_or(vmin <= vmax);
    if (t < 32 * 1) {
        vmin = WarpReduceMinFloat(s->warp_min[t].fp_val);
        if (!(t & 0x1f)) {
            s->ck.min_value.fp_val = (vmin != 0.0) ? vmin : CUDART_NEG_ZERO;
            s->ck.has_minmax = (has_minmax);
        }
    }
    else if (t < 32 * 2) {
        vmax = WarpReduceMaxFloat(s->warp_max[t & 0x1f].fp_val);
        if (!(t & 0x1f)) {
            s->ck.max_value.fp_val = (vmax != 0.0) ? vmax : CUDART_ZERO;
        }
    }
    else if (t < 32 * 3) {
        vsum = WarpReduceSumFloat(s->warp_sum[t & 0x1f].fp_val);
        if (!(t & 0x1f)) {
            s->ck.sum.fp_val = vsum;
            s->ck.has_sum = (has_minmax); // Implies sum is valid as well
        }
    }
}


// FIXME: Use native libcudf string type
struct nvstrdesc_s {
    const char *ptr;
    size_t count;
};


void __device__ gatherStringColumnStats(stats_state_s *s, uint32_t t)
{
    uint32_t len_sum = 0;
    const char *smin = nullptr;
    const char *smax = nullptr;
    uint32_t lmin = 0;
    uint32_t lmax = 0;
    uint32_t nn_cnt = 0;
    bool has_minmax;
    string_stats minval, maxval;

    for (uint32_t i = 0; i < s->group.num_rows; i += 1024) {
        uint32_t r = i + t;
        uint32_t row = r + s->group.start_row;
        const uint32_t *valid_map = s->col.valid_map_base;
        uint32_t is_valid = (r < s->group.num_rows && row < s->col.num_rows) ? (valid_map) ? (valid_map[row >> 5] >> (row & 0x1f)) & 1 : 1 : 0;
        if (is_valid) {
            const nvstrdesc_s *str_col = reinterpret_cast<const nvstrdesc_s *>(s->col.column_data_base);
            uint32_t len = (uint32_t)str_col[row].count;
            const char *ptr = str_col[row].ptr;
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
        s->ck.non_nulls = nn_cnt;
        s->ck.null_count = s->group.num_rows - nn_cnt;
    }
    minval = WarpReduceMinString(smin, lmin);
    maxval = WarpReduceMaxString(smax, lmax);
    len_sum = WarpReduceSum32(len_sum);
    if (!(t & 0x1f)) {
        s->warp_min[t >> 5].str_val.ptr = minval.ptr;
        s->warp_min[t >> 5].str_val.length = minval.length;
        s->warp_max[t >> 5].str_val.ptr = maxval.ptr;
        s->warp_max[t >> 5].str_val.length = maxval.length;
        s->warp_sum[t >> 5].str_val.length = len_sum;
    }
    has_minmax = __syncthreads_or(smin != nullptr);
    if (t < 32 * 1) {
        minval = WarpReduceMinString(s->warp_min[t].str_val.ptr, s->warp_min[t].str_val.length);
        if (!(t & 0x1f)) {
            s->ck.min_value.str_val.ptr = minval.ptr;
            s->ck.min_value.str_val.length = minval.length;
            s->ck.has_minmax = has_minmax;
        }
    }
    else if (t < 32 * 2) {
        maxval = WarpReduceMaxString(s->warp_max[t & 0x1f].str_val.ptr, s->warp_max[t & 0x1f].str_val.length);
        if (!(t & 0x1f)) {
            s->ck.max_value.str_val.ptr = maxval.ptr;
            s->ck.max_value.str_val.length = maxval.length;
        }
    }
    else if (t < 32 * 3) {
        len_sum = WarpReduceSum32(s->warp_sum[t & 0x1f].str_val.length);
        if (!(t & 0x1f)) {
            s->ck.sum.i_val = len_sum;
            s->ck.has_sum = has_minmax;
        }
    }
}

/**
 * @brief Gather column chunk statistics (min/max values, sum and null count)
 * for a group of rows.
 **/
// blockDim {1024,1,1}
__global__ void __launch_bounds__(1024, 1)
gpuGatherColumnStatistics(statistics_chunk *chunks, const statistics_group *groups)
{
    __shared__ __align__(8) stats_state_s state_g;

    stats_state_s *const s = &state_g;
    uint32_t t = threadIdx.x;
    statistics_dtype dtype;

    if (t < sizeof(statistics_group) / sizeof(uint32_t)) {
        reinterpret_cast<uint32_t *>(&s->group)[t] = reinterpret_cast<const uint32_t *>(&groups[blockIdx.x])[t];
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
    if (dtype >= dtype_bool8 && dtype <= dtype_timestamp64) {
        gatherIntColumnStats(s, dtype, t);
    }
    else if (dtype >= dtype_float32 && dtype <= dtype_float64) {
        gatherFloatColumnStats(s, dtype, t);
    }
    else if (dtype == dtype_string) {
        gatherStringColumnStats(s, t);
    }
    __syncthreads();
    if (t < sizeof(statistics_chunk) / sizeof(uint32_t)) {
        reinterpret_cast<uint32_t *>(&chunks[blockIdx.x])[t] = reinterpret_cast<uint32_t *>(&s->ck)[t];
    }
}


void __device__ mergeIntColumnStats(merge_state_s *s, statistics_dtype dtype, const statistics_chunk *ck_in, uint32_t num_chunks, uint32_t t)
{
    int64_t vmin = INT64_MAX;
    int64_t vmax = INT64_MIN;
    int64_t vsum = 0;
    uint32_t non_nulls = 0;
    uint32_t null_count = 0;
    bool has_minmax;
    for (uint32_t i = t; i < num_chunks; i += 1024) {
        const statistics_chunk *ck = &ck_in[i];
        if (ck->has_minmax) {
            vmin = min(vmin, ck->min_value.i_val);
            vmax = max(vmax, ck->max_value.i_val);
        }
        if (ck->has_sum) {
            vsum += ck->sum.i_val;
        }
        non_nulls += ck->non_nulls;
        null_count += ck->null_count;
    }
    non_nulls = WarpReduceSum32(non_nulls);
    null_count = WarpReduceSum32(null_count);
    vmin = WarpReduceMinInt(vmin);
    vmax = WarpReduceMaxInt(vmax);
    vsum = WarpReduceSum32(vsum);
    if (!(t & 0x1f)) {
        s->warp_non_nulls[t >> 5] = non_nulls;
        s->warp_nulls[t >> 5] = null_count;
        s->warp_min[t >> 5].i_val = vmin;
        s->warp_max[t >> 5].i_val = vmax;
        s->warp_sum[t >> 5].i_val = vsum;
    }
    has_minmax = __syncthreads_or(vmin <= vmax);
    if (t < 32 * 1) {
        vmin = WarpReduceMinInt(s->warp_min[t].i_val);
        if (!(t & 0x1f)) {
            s->ck.min_value.i_val = vmin;
            s->ck.has_minmax = (has_minmax);
        }
    }
    else if (t < 32 * 2) {
        vmax = WarpReduceMaxInt(s->warp_max[t & 0x1f].i_val);
        if (!(t & 0x1f)) {
            s->ck.max_value.i_val = vmax;
        }
    }
    else if (t < 32 * 3) {
        vsum = WarpReduceSum32(s->warp_sum[t & 0x1f].i_val);
        if (!(t & 0x1f)) {
            s->ck.sum.i_val = vsum;
            // TODO: For now, don't set the sum flag with 64-bit values so we don't have to check for 64-bit sum overflow
            s->ck.has_sum = (dtype <= dtype_int32 && has_minmax);
        }
    }
    else if (t < 32 * 4) {
        non_nulls = WarpReduceSum32(s->warp_non_nulls[t & 0x1f]);
        if (!(t & 0x1f)) {
            s->ck.non_nulls = non_nulls;
        }
    }
    else if (t < 32 * 5) {
        null_count = WarpReduceSum32(s->warp_nulls[t & 0x1f]);
        if (!(t & 0x1f)) {
            s->ck.null_count = null_count;
        }
    }
}


void __device__ mergeFloatColumnStats(merge_state_s *s, const statistics_chunk *ck_in, uint32_t num_chunks, uint32_t t)
{
    double vmin = CUDART_INF;
    double vmax = -CUDART_INF;
    double vsum = 0;
    uint32_t non_nulls = 0;
    uint32_t null_count = 0;
    bool has_minmax;
    for (uint32_t i = t; i < num_chunks; i += 1024) {
        const statistics_chunk *ck = &ck_in[i];
        if (ck->has_minmax) {
            double v0 = ck->min_value.fp_val;
            double v1 = ck->max_value.fp_val;
            if (v0 < vmin) {
                vmin = v0;
            }
            if (v1 > vmax) {
                vmax = v1;
            }
        }
        if (ck->has_sum) {
            vsum += ck->sum.fp_val;
        }
        non_nulls += ck->non_nulls;
        null_count += ck->null_count;
    }
    non_nulls = WarpReduceSum32(non_nulls);
    null_count = WarpReduceSum32(null_count);
    vmin = WarpReduceMinFloat(vmin);
    vmax = WarpReduceMaxFloat(vmax);
    vsum = WarpReduceSumFloat(vsum);
    if (!(t & 0x1f)) {
        s->warp_non_nulls[t >> 5] = non_nulls;
        s->warp_nulls[t >> 5] = null_count;
        s->warp_min[t >> 5].fp_val = vmin;
        s->warp_max[t >> 5].fp_val = vmax;
        s->warp_sum[t >> 5].fp_val = vsum;
    }
    has_minmax = __syncthreads_or(vmin <= vmax);
    if (t < 32 * 1) {
        vmin = WarpReduceMinFloat(s->warp_min[t].fp_val);
        if (!(t & 0x1f)) {
            s->ck.min_value.fp_val = (vmin != 0.0) ? vmin : CUDART_NEG_ZERO;
            s->ck.has_minmax = (has_minmax);
        }
    }
    else if (t < 32 * 2) {
        vmax = WarpReduceMaxFloat(s->warp_max[t & 0x1f].fp_val);
        if (!(t & 0x1f)) {
            s->ck.max_value.fp_val = (vmax != 0.0) ? vmax : CUDART_ZERO;
        }
    }
    else if (t < 32 * 3) {
        vsum = WarpReduceSumFloat(s->warp_sum[t & 0x1f].fp_val);
        if (!(t & 0x1f)) {
            s->ck.sum.fp_val = vsum;
            s->ck.has_sum = (has_minmax); // Implies sum is valid as well
        }
    }
    else if (t < 32 * 4) {
        non_nulls = WarpReduceSum32(s->warp_non_nulls[t & 0x1f]);
        if (!(t & 0x1f)) {
            s->ck.non_nulls = non_nulls;
        }
    }
    else if (t < 32 * 5) {
        null_count = WarpReduceSum32(s->warp_nulls[t & 0x1f]);
        if (!(t & 0x1f)) {
            s->ck.null_count = null_count;
        }
    }
}


void __device__ mergeStringColumnStats(merge_state_s *s, const statistics_chunk *ck_in, uint32_t num_chunks, uint32_t t)
{
    uint32_t len_sum = 0;
    const char *smin = nullptr;
    const char *smax = nullptr;
    uint32_t lmin = 0;
    uint32_t lmax = 0;
    uint32_t non_nulls = 0;
    uint32_t null_count = 0;
    bool has_minmax;
    string_stats minval, maxval;

    for (uint32_t i = t; i < num_chunks; i += 1024) {
        const statistics_chunk *ck = &ck_in[i];
        if (ck->has_minmax) {
            
            uint32_t len0 = ck->min_value.str_val.length;
            const char *ptr0 = ck->min_value.str_val.ptr;
            uint32_t len1 = ck->max_value.str_val.length;
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
        if (ck->has_sum) {
            len_sum += (uint32_t)ck->sum.i_val;
        }
        non_nulls += ck->non_nulls;
        null_count += ck->null_count;
    }
    non_nulls = WarpReduceSum32(non_nulls);
    null_count = WarpReduceSum32(null_count);
    minval = WarpReduceMinString(smin, lmin);
    maxval = WarpReduceMaxString(smax, lmax);
    len_sum = WarpReduceSum32(len_sum);
    if (!(t & 0x1f)) {
        s->warp_non_nulls[t >> 5] = non_nulls;
        s->warp_nulls[t >> 5] = null_count;
        s->warp_min[t >> 5].str_val.ptr = minval.ptr;
        s->warp_min[t >> 5].str_val.length = minval.length;
        s->warp_max[t >> 5].str_val.ptr = maxval.ptr;
        s->warp_max[t >> 5].str_val.length = maxval.length;
        s->warp_sum[t >> 5].str_val.length = len_sum;
    }
    has_minmax = __syncthreads_or(smin != nullptr);
    if (t < 32 * 1) {
        minval = WarpReduceMinString(s->warp_min[t].str_val.ptr, s->warp_min[t].str_val.length);
        if (!(t & 0x1f)) {
            s->ck.min_value.str_val.ptr = minval.ptr;
            s->ck.min_value.str_val.length = minval.length;
            s->ck.has_minmax = has_minmax;
        }
    }
    else if (t < 32 * 2) {
        maxval = WarpReduceMaxString(s->warp_max[t & 0x1f].str_val.ptr, s->warp_max[t & 0x1f].str_val.length);
        if (!(t & 0x1f)) {
            s->ck.max_value.str_val.ptr = maxval.ptr;
            s->ck.max_value.str_val.length = maxval.length;
        }
    }
    else if (t < 32 * 3) {
        len_sum = WarpReduceSum32(s->warp_sum[t & 0x1f].str_val.length);
        if (!(t & 0x1f)) {
            s->ck.sum.i_val = len_sum;
            s->ck.has_sum = has_minmax;
        }
    }
    else if (t < 32 * 4) {
        non_nulls = WarpReduceSum32(s->warp_non_nulls[t & 0x1f]);
        if (!(t & 0x1f)) {
            s->ck.non_nulls = non_nulls;
        }
    }
    else if (t < 32 * 5) {
        null_count = WarpReduceSum32(s->warp_nulls[t & 0x1f]);
        if (!(t & 0x1f)) {
            s->ck.null_count = null_count;
        }
    }
}

/**
 * @brief Combine multiple statistics chunk together to form new statistics chunks
 **/
// blockDim {1024,1,1}
__global__ void __launch_bounds__(1024, 1)
gpuMergeColumnStatistics(statistics_chunk *chunks_out, const statistics_chunk *chunks_in, const statistics_merge_group *groups)
{
    __shared__ __align__(8) merge_state_s state_g;

    merge_state_s *const s = &state_g;
    uint32_t t = threadIdx.x;
    statistics_dtype dtype;

    if (t < sizeof(statistics_merge_group) / sizeof(uint32_t)) {
        reinterpret_cast<uint32_t *>(&s->group)[t] = reinterpret_cast<const uint32_t *>(&groups[blockIdx.x])[t];
    }
    __syncthreads();
    if (t < sizeof(stats_column_desc) / sizeof(uint32_t)) {
        reinterpret_cast<uint32_t *>(&s->col)[t] = reinterpret_cast<const uint32_t *>(s->group.col)[t];
    }
    __syncthreads();
    dtype = s->col.stats_dtype;

    if (dtype >= dtype_bool8 && dtype <= dtype_timestamp64) {
        mergeIntColumnStats(s, dtype, chunks_in + s->group.start_chunk, s->group.num_chunks, t);
    }
    else if (dtype >= dtype_float32 && dtype <= dtype_float64) {
        mergeFloatColumnStats(s, chunks_in + s->group.start_chunk, s->group.num_chunks, t);
    }
    else if (dtype == dtype_string) {
        mergeStringColumnStats(s, chunks_in + s->group.start_chunk, s->group.num_chunks, t);
    }

    __syncthreads();
    if (t < sizeof(statistics_chunk) / sizeof(uint32_t)) {
        reinterpret_cast<uint32_t *>(&chunks_out[blockIdx.x])[t] = reinterpret_cast<uint32_t *>(&s->ck)[t];
    }
}


/**
 * @brief Launches kernel to gather column statistics
 *
 * @param[out] chunks Statistics results [num_chunks]
 * @param[in] groups Statistics row groups [num_chunks]
 * @param[in] num_chunks Number of chunks & rowgroups
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t GatherColumnStatistics(statistics_chunk *chunks, const statistics_group *groups, uint32_t num_chunks, cudaStream_t stream)
{
    gpuGatherColumnStatistics <<< num_chunks, 1024, 0, stream >>> (chunks, groups);
    return cudaSuccess;
}

/**
 * @brief Launches kernel to merge column statistics
 *
 * @param[out] chunks_out Statistics results [num_chunks]
 * @param[out] chunks_in Input statistics
 * @param[in] groups Statistics groups [num_chunks]
 * @param[in] num_chunks Number of chunks & groups
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t MergeColumnStatistics(statistics_chunk *chunks_out, const statistics_chunk *chunks_in,
                                  const statistics_merge_group *groups, uint32_t num_chunks, cudaStream_t stream)
{
    gpuMergeColumnStatistics <<< num_chunks, 1024, 0, stream >>> (chunks_out, chunks_in, groups);
    return cudaSuccess;
}


} // namespace io
} // namespace cudf

