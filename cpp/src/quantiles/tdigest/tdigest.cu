/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "quantiles/tdigest/tdigest_util.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

using namespace cudf::tdigest;

namespace cudf {
namespace tdigest {
namespace detail {

// https://developer.nvidia.com/blog/lerp-faster-cuda/
template <typename T>
__device__ inline T lerp(T v0, T v1, T t)
{
  return fma(t, v1, fma(-t, v0, v0));
}

struct centroid {
  double mean;
  double weight;
};

struct make_centroid {
  double const* means;
  double const* weights;
  __device__ centroid operator()(size_type i) { return {means[i], weights[i]}; }
};

// kernel for computing percentiles on input tdigest (mean, weight) centroid data.
template <bool CompactOutput, typename CentroidIter>
CUDF_KERNEL void compute_percentiles_kernel(device_span<int32_t const> tdigest_offsets,
                                            column_device_view percentiles,
                                            CentroidIter centroids_,
                                            double const* min_,
                                            double const* max_,
                                            double const* cumulative_weight_,
                                            int32_t const* output_offsets,
                                            double* output)
{
  auto const num_tdigests = tdigest_offsets.size() - 1;
  auto const num_pairs    = static_cast<thread_index_type>(num_tdigests) * percentiles.size();
  auto const stride       = cudf::detail::grid_1d::grid_stride();
  for (auto tid = cudf::detail::grid_1d::global_thread_id(); tid < num_pairs; tid += stride) {
    auto const tdigest_index = tid / percentiles.size();
    auto const pindex        = tid % percentiles.size();

    // Size of the digest we're querying.
    auto const tdigest_size = tdigest_offsets[tdigest_index + 1] - tdigest_offsets[tdigest_index];
    // No work to do. Values will be set to null.
    if (tdigest_size == 0 || !percentiles.is_valid(pindex)) { continue; }

    auto const output_index = [&]() {
      if constexpr (CompactOutput) {
        return output_offsets[tdigest_index] + pindex;
      } else {
        return tid;
      }
    }();

    output[output_index] = [&]() {
      double const percentage         = percentiles.element<double>(pindex);
      double const* cumulative_weight = cumulative_weight_ + tdigest_offsets[tdigest_index];

      // Centroids for this particular tdigest.
      CentroidIter centroids = centroids_ + tdigest_offsets[tdigest_index];

      // Min and max for the digest.
      double const* min_val = min_ + tdigest_index;
      double const* max_val = max_ + tdigest_index;

      double const total_weight = cumulative_weight[tdigest_size - 1];

      // The following Arrow code serves as a basis for this computation.
      // https://github.com/apache/arrow/blob/master/cpp/src/arrow/util/tdigest.cc#L280
      double const weighted_q = percentage * total_weight;
      if (weighted_q <= 1) {
        return *min_val;
      } else if (weighted_q > total_weight - 1) {
        return *max_val;
      }

      // Determine what centroid this weighted quantile falls within.
      size_type const centroid_index = static_cast<size_type>(cuda::std::distance(
        cumulative_weight,
        cuda::std::lower_bound(cumulative_weight, cumulative_weight + tdigest_size, weighted_q)));
      centroid c                     = centroids[centroid_index];

      // diff == how far from the "center" of the centroid we are,
      // in unit weights.
      // visually:
      //
      // centroid of weight 7
      //        C       <-- center of the centroid
      //    |-------|
      //      | |  |
      //      X Y  Z
      // X has a diff of -2 (2 units to the left of the center of the centroid)
      // Y has a diff of 0 (directly in the middle of the centroid)
      // Z has a diff of 3 (3 units to the right of the center of the centroid)
      double const diff = weighted_q + c.weight / 2 - cumulative_weight[centroid_index];

      // If we're completely within a centroid of weight 1, just return that.
      if (c.weight == 1 && cuda::std::abs(diff) <= 0.5) { return c.mean; }

      // Otherwise, interpolate between two centroids.

      // Get the two centroids we want to interpolate between.
      auto const look_left  = diff < 0;
      auto const [lhs, rhs] = [&]() {
        if (look_left) {
          // if we're at the first centroid, "left" of us is the min value
          auto const first_centroid = centroid_index == 0;
          auto const lhs = first_centroid ? centroid{*min_val, 0} : centroids[centroid_index - 1];
          auto const rhs = c;
          return cuda::std::pair<centroid, centroid>{lhs, rhs};
        } else {
          // if we're at the last centroid, "right" of us is the max value
          auto const last_centroid = (centroid_index == tdigest_size - 1);
          auto const lhs           = c;
          auto const rhs = last_centroid ? centroid{*max_val, 0} : centroids[centroid_index + 1];
          return cuda::std::pair<centroid, centroid>{lhs, rhs};
        }
      }();

      // Compute interpolation value t.

      // Total interpolation range. The total range of "space" between the lhs and rhs centroids.
      auto const tip = lhs.weight / 2 + rhs.weight / 2;
      // If we're looking left, diff is negative, so shift it so that we are interpolating
      // from lhs -> rhs.
      auto const t = (look_left) ? (diff + tip) / tip : diff / tip;

      // Interpolate.
      return lerp(lhs.mean, rhs.mean, t);
    }();
  }
}

/**
 * @brief Calculate approximate percentiles on a provided tdigest column.
 *
 * Produces a LIST column where each row `i` represents output from querying the
 * corresponding tdigest of from row `i` in `input`. The length of each output list
 * is the number of percentiles specified in `percentiles`
 *
 * @param input             tdigest input data. One tdigest per row.
 * @param percentiles       Desired percentiles in range [0, 1].
 * @param num_output_values Number of elements to allocate in the output column.
 * @param output_offsets    Optional compact output offsets for inputs containing empty digests.
 * @param stream            CUDA stream used for device memory operations and kernel launches.
 * @param mr                Device memory resource used for output allocation.
 *
 * @returns Column of doubles containing requested percentile values.
 */
std::unique_ptr<column> compute_approx_percentiles(tdigest_column_view const& input,
                                                   column_view const& percentiles,
                                                   size_type num_output_values,
                                                   int32_t const* output_offsets,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  tdigest_column_view tdv(input);

  // offsets, representing the size of each tdigest
  auto offsets = tdv.centroids().offsets();

  // compute summed weights
  auto weight             = tdv.weights();
  auto cumulative_weights = cudf::make_fixed_width_column(data_type{type_id::FLOAT64},
                                                          weight.size(),
                                                          mask_state::UNALLOCATED,
                                                          stream,
                                                          cudf::get_current_device_resource_ref());
  auto keys               = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<std::ptrdiff_t>(
      [offsets_begin = offsets.begin<int32_t>(),
       offsets_end   = offsets.end<int32_t>()] __device__(size_type i) {
        return cuda::std::distance(
          offsets_begin,
          cuda::std::prev(thrust::upper_bound(thrust::seq, offsets_begin, offsets_end, i)));
      }));
  thrust::inclusive_scan_by_key(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    keys,
    keys + weight.size(),
    weight.begin<double>(),
    cumulative_weights->mutable_view().begin<double>());

  auto percentiles_cdv = column_device_view::create(percentiles, stream);

  // null percentiles become null results.
  auto [null_mask, null_count] = [&]() {
    return percentiles.null_count() != 0
             ? cudf::detail::valid_if(
                 cuda::counting_iterator<size_type>{0},
                 cuda::counting_iterator<size_type>{0} + num_output_values,
                 [percentiles = *percentiles_cdv] __device__(size_type i) {
                   return percentiles.is_valid(i % percentiles.size());
                 },
                 stream,
                 mr)
             : std::pair<rmm::device_buffer, size_type>{rmm::device_buffer{}, 0};
  }();

  auto result = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, num_output_values, std::move(null_mask), null_count, stream, mr);

  auto centroids = cudf::detail::make_counting_transform_iterator(
    0, make_centroid{tdv.means().begin<double>(), tdv.weights().begin<double>()});

  constexpr size_type block_size  = 256;
  auto const logical_num_threads  = thread_index_type{percentiles.size()} * input.size();
  auto const launched_num_threads = std::min(
    logical_num_threads, static_cast<thread_index_type>(std::numeric_limits<size_type>::max()));
  cudf::detail::grid_1d const grid(launched_num_threads, block_size);
  auto const launch_kernel = [&]<bool CompactOutput>() {
    compute_percentiles_kernel<CompactOutput><<<grid.num_blocks, block_size, 0, stream.get()>>>(
      {offsets.begin<size_type>(), static_cast<size_t>(offsets.size())},
      *percentiles_cdv,
      centroids,
      tdv.min_begin(),
      tdv.max_begin(),
      cumulative_weights->view().begin<double>(),
      output_offsets,
      result->mutable_view().begin<double>());
  };
  if (output_offsets == nullptr) {
    launch_kernel.template operator()<false>();
  } else {
    launch_kernel.template operator()<true>();
  }
  CUDF_CUDA_TRY(cudaGetLastError());

  return result;
}

std::unique_ptr<column> make_tdigest_column(size_type num_rows,
                                            std::unique_ptr<column>&& centroid_means,
                                            std::unique_ptr<column>&& centroid_weights,
                                            std::unique_ptr<column>&& tdigest_offsets,
                                            std::unique_ptr<column>&& min_values,
                                            std::unique_ptr<column>&& max_values,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(tdigest_offsets->size() == num_rows + 1,
               "Encountered unexpected offset count in make_tdigest_column");
  CUDF_EXPECTS(centroid_means->size() == centroid_weights->size(),
               "Encountered unexpected centroid size mismatch in make_tdigest_column");
  CUDF_EXPECTS(min_values->size() == num_rows,
               "Encountered unexpected min value count in make_tdigest_column");
  CUDF_EXPECTS(max_values->size() == num_rows,
               "Encountered unexpected max value count in make_tdigest_column");

  // inner struct column
  auto const centroids_size = centroid_means->size();
  std::vector<std::unique_ptr<column>> inner_children;
  inner_children.push_back(std::move(centroid_means));
  inner_children.push_back(std::move(centroid_weights));
  auto tdigest_data =
    cudf::make_structs_column(centroids_size, std::move(inner_children), 0, {}, stream, mr);

  // grouped into lists
  auto tdigest =
    cudf::make_lists_column(num_rows, std::move(tdigest_offsets), std::move(tdigest_data), 0, {});

  // create the final column
  std::vector<std::unique_ptr<column>> children;
  children.push_back(std::move(tdigest));
  children.push_back(std::move(min_values));
  children.push_back(std::move(max_values));
  return make_structs_column(num_rows, std::move(children), 0, {}, stream, mr);
}

std::unique_ptr<column> make_empty_tdigests_column(size_type num_rows,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto offsets = cudf::make_fixed_width_column(
    data_type(type_id::INT32), num_rows + 1, mask_state::UNALLOCATED, stream, mr);
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               offsets->mutable_view().begin<int32_t>(),
               offsets->mutable_view().end<int32_t>(),
               0);

  auto min_col = cudf::make_numeric_column(
    data_type(type_id::FLOAT64), num_rows, mask_state::UNALLOCATED, stream, mr);
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               min_col->mutable_view().begin<double>(),
               min_col->mutable_view().end<double>(),
               0);
  auto max_col = cudf::make_numeric_column(
    data_type(type_id::FLOAT64), num_rows, mask_state::UNALLOCATED, stream, mr);
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               max_col->mutable_view().begin<double>(),
               max_col->mutable_view().end<double>(),
               0);

  return make_tdigest_column(num_rows,
                             cudf::make_empty_column(type_id::FLOAT64),
                             cudf::make_empty_column(type_id::FLOAT64),
                             std::move(offsets),
                             std::move(min_col),
                             std::move(max_col),
                             stream,
                             mr);
}

/**
 * @brief Create an empty tdigest scalar.
 *
 * An empty tdigest scalar is a struct_scalar that contains a single row of length 0
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 *
 * @returns An empty tdigest scalar.
 */
std::unique_ptr<scalar> make_empty_tdigest_scalar(cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  auto contents = make_empty_tdigests_column(1, stream, mr)->release();
  return std::make_unique<struct_scalar>(
    std::move(*std::make_unique<table>(std::move(contents.children))), true, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> percentile_approx(tdigest_column_view const& input,
                                          column_view const& percentiles,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  tdigest_column_view tdv(input);
  CUDF_EXPECTS(percentiles.type().id() == type_id::FLOAT64,
               "percentile_approx expects float64 percentile inputs");

  auto const make_offsets = [&](auto row_size_iter) {
    auto offsets = cudf::make_fixed_width_column(
      data_type{type_id::INT32}, input.size() + 1, mask_state::UNALLOCATED, stream, mr);
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           row_size_iter,
                           row_size_iter + input.size() + 1,
                           offsets->mutable_view().begin<int32_t>());
    return offsets;
  };

  auto const make_all_null_result = [&]() {
    auto offsets = make_offsets(cuda::make_constant_iterator(size_type{0}));
    return cudf::make_lists_column(
      input.size(),
      std::move(offsets),
      cudf::make_empty_column(type_id::FLOAT64),
      input.size(),
      cudf::detail::create_null_mask(
        input.size(), mask_state::ALL_NULL, cuda::stream_ref(stream), mr));
  };

  if (percentiles.size() == 0) { return make_all_null_result(); }

  auto const null_count = static_cast<size_type>(cudf::detail::count_if(
    detail::size_begin(input),
    detail::size_begin(input) + input.size(),
    [] __device__(auto const tdigest_size) -> bool { return tdigest_size == 0; },
    stream));

  if (null_count == input.size()) { return make_all_null_result(); }

  auto const output_size = thread_index_type{input.size() - null_count} * percentiles.size();
  CUDF_EXPECTS(output_size <= std::numeric_limits<size_type>::max(),
               "Size of output exceeds the column size limit",
               std::overflow_error);
  auto const num_output_values = static_cast<size_type>(output_size);

  // If any input digests are empty, nullify the corresponding output rows.
  auto bitmask = [stream, mr, &tdv, null_count]() {
    if (null_count == 0) { return rmm::device_buffer{}; }

    auto mask        = cudf::create_null_mask(tdv.size(), mask_state::UNINITIALIZED, stream, mr);
    auto valid_count = cudf::detail::make_zeroed_device_uvector_async<size_type>(
      1, stream, cudf::get_current_device_resource_ref());
    constexpr size_type block_size{256};
    auto const grid = cudf::detail::grid_1d{static_cast<thread_index_type>(tdv.size()), block_size};
    cudf::detail::valid_if_kernel<block_size>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
        static_cast<bitmask_type*>(mask.data()),
        detail::size_begin(tdv),
        tdv.size(),
        [] __device__(size_type tdigest_size) -> bool { return tdigest_size != 0; },
        valid_count.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    return mask;
  }();

  std::unique_ptr<column> offsets;
  if (null_count == 0) {
    offsets = make_offsets(cuda::make_constant_iterator(percentiles.size()));
  } else {
    auto const row_size_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<size_type>(
        [sizes_begin = detail::size_begin(tdv),
         num_rows    = input.size(),
         row_size    = percentiles.size()] __device__(size_type i) -> size_type {
          return i < num_rows && sizes_begin[i] != 0 ? row_size : 0;
        }));
    offsets = make_offsets(row_size_iter);
  }

  auto const output_offsets = null_count == 0 ? nullptr : offsets->view().begin<int32_t>();
  auto child                = detail::compute_approx_percentiles(
    input, percentiles, num_output_values, output_offsets, stream, mr);
  return cudf::make_lists_column(
    input.size(), std::move(offsets), std::move(child), null_count, std::move(bitmask));
}

}  // namespace tdigest

std::unique_ptr<column> percentile_approx(tdigest_column_view const& input,
                                          column_view const& percentiles,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return tdigest::percentile_approx(input, percentiles, stream, mr);
}

}  // namespace cudf
