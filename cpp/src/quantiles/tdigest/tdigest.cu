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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/detail/tdigest/tdigest_column_view.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>

namespace cudf {
namespace detail {
namespace tdigest {

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
template <typename CentroidIter>
__global__ void compute_percentiles_kernel(device_span<offset_type const> tdigest_offsets,
                                           column_device_view percentiles,
                                           CentroidIter centroids_,
                                           double const* min_,
                                           double const* max_,
                                           double const* cumulative_weight_,
                                           double* output)
{
  int const tid = threadIdx.x + blockIdx.x * blockDim.x;

  auto const num_tdigests  = tdigest_offsets.size() - 1;
  auto const tdigest_index = tid / percentiles.size();
  if (tdigest_index >= num_tdigests) { return; }
  auto const pindex = tid % percentiles.size();

  // size of the digest we're querying
  auto const tdigest_size = tdigest_offsets[tdigest_index + 1] - tdigest_offsets[tdigest_index];
  // no work to do. values will be set to null
  if (tdigest_size == 0 || !percentiles.is_valid(pindex)) { return; }

  output[tid] = [&]() {
    double const percentage         = percentiles.element<double>(pindex);
    double const* cumulative_weight = cumulative_weight_ + tdigest_offsets[tdigest_index];

    // centroids for this particular tdigest
    CentroidIter centroids = centroids_ + tdigest_offsets[tdigest_index];

    // min and max for the digest
    double const* min_val = min_ + tdigest_index;
    double const* max_val = max_ + tdigest_index;

    double const total_weight = cumulative_weight[tdigest_size - 1];

    // The following Arrow code serves as a basis for this computation
    // https://github.com/apache/arrow/blob/master/cpp/src/arrow/util/tdigest.cc#L280
    double const weighted_q = percentage * total_weight;
    if (weighted_q <= 1) {
      return *min_val;
    } else if (weighted_q >= total_weight - 1) {
      return *max_val;
    }

    // determine what centroid this weighted quantile falls within.
    size_type const centroid_index = static_cast<size_type>(thrust::distance(
      cumulative_weight,
      thrust::lower_bound(
        thrust::seq, cumulative_weight, cumulative_weight + tdigest_size, weighted_q)));
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

    // if we're completely within a centroid of weight 1, just return that.
    if (c.weight == 1 && std::abs(diff) < 0.5) { return c.mean; }

    // otherwise, interpolate between two centroids.

    // get the two centroids we want to interpolate between
    auto const look_left  = diff < 0;
    auto const [lhs, rhs] = [&]() {
      if (look_left) {
        // if we're at the first centroid, "left" of us is the min value
        auto const first_centroid = centroid_index == 0;
        auto const lhs = first_centroid ? centroid{*min_val, 0} : centroids[centroid_index - 1];
        auto const rhs = c;
        return std::pair<centroid, centroid>{lhs, rhs};
      } else {
        // if we're at the last centroid, "right" of us is the max value
        auto const last_centroid = (centroid_index == tdigest_size - 1);
        auto const lhs           = c;
        auto const rhs = last_centroid ? centroid{*max_val, 0} : centroids[centroid_index + 1];
        return std::pair<centroid, centroid>{lhs, rhs};
      }
    }();

    // compute interpolation value t

    // total interpolation range. the total range of "space" between the lhs and rhs centroids.
    auto const tip = lhs.weight / 2 + rhs.weight / 2;
    // if we're looking left, diff is negative, so shift it so that we are interpolating
    // from lhs -> rhs.
    auto const t = (look_left) ? (diff + tip) / tip : diff / tip;

    // interpolate
    return lerp(lhs.mean, rhs.mean, t);
  }();
}

/**
 * @brief Calculate approximate percentiles on a provided tdigest column.
 *
 * Produces a LIST column where each row `i` represents output from querying the
 * corresponding tdigest of from row `i` in `input`. The length of each output list
 * is the number of percentiles specified in `percentiles`
 *
 * @param input           tdigest input data. One tdigest per row.
 * @param percentiles     Desired percentiles in range [0, 1].
 * @param stream          CUDA stream used for device memory operations and kernel launches
 * @param mr              Device memory resource used to allocate the returned column's device
 * memory
 *
 * @returns Column of doubles containing requested percentile values.
 */
std::unique_ptr<column> compute_approx_percentiles(structs_column_view const& input,
                                                   column_view const& percentiles,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  tdigest_column_view tdv(input);

  // offsets, representing the size of each tdigest
  auto offsets = tdv.centroids().offsets();

  // means and weights internal to the digests
  auto mean   = tdv.means();
  auto weight = tdv.weights();

  // compute summed weights
  auto cumulative_weights = cudf::make_fixed_width_column(data_type{type_id::FLOAT64},
                                                          mean.size(),
                                                          mask_state::UNALLOCATED,
                                                          stream,
                                                          rmm::mr::get_current_device_resource());
  auto keys               = cudf::detail::make_counting_transform_iterator(
    0,
    [offsets_begin = offsets.begin<offset_type>(),
     offsets_end   = offsets.end<offset_type>()] __device__(size_type i) {
      return thrust::distance(
        offsets_begin,
        thrust::prev(thrust::upper_bound(thrust::seq, offsets_begin, offsets_end, i)));
    });
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                keys,
                                keys + weight.size(),
                                weight.begin<double>(),
                                cumulative_weights->mutable_view().begin<double>());

  auto percentiles_cdv = column_device_view::create(percentiles);

  // leaf is a column of size input.size() * percentiles.size()
  auto const num_output_values = input.size() * percentiles.size();

  // null percentiles become null results.
  auto [null_mask, null_count] = [&]() {
    return percentiles.null_count() != 0
             ? cudf::detail::valid_if(
                 thrust::make_counting_iterator<size_type>(0),
                 thrust::make_counting_iterator<size_type>(0) + num_output_values,
                 [percentiles = *percentiles_cdv] __device__(size_type i) {
                   return percentiles.is_valid(i % percentiles.size());
                 })
             : std::pair<rmm::device_buffer, size_type>{rmm::device_buffer{}, 0};
  }();

  auto result = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, num_output_values, std::move(null_mask), null_count, stream, mr);

  auto centroids = cudf::detail::make_counting_transform_iterator(
    0, make_centroid{mean.begin<double>(), weight.begin<double>()});

  column_view min_col = tdv.min_column();
  column_view max_col = tdv.max_column();

  constexpr size_type block_size = 256;
  cudf::detail::grid_1d const grid(percentiles.size() * input.size(), block_size);
  compute_percentiles_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    {offsets.begin<offset_type>(), static_cast<size_t>(offsets.size())},
    *percentiles_cdv,
    centroids,
    min_col.begin<double>(),
    max_col.begin<double>(),
    cumulative_weights->view().begin<double>(),
    result->mutable_view().begin<double>());

  return result;
}

std::unique_ptr<column> make_empty_tdigest_column(rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  // mean/weight columns
  std::vector<std::unique_ptr<column>> inner_children;
  inner_children.push_back(make_empty_column(data_type(type_id::FLOAT64)));
  inner_children.push_back(make_empty_column(data_type(type_id::FLOAT64)));

  auto offsets = cudf::make_fixed_width_column(
    data_type(type_id::INT32), 2, mask_state::UNALLOCATED, stream, mr);
  thrust::fill(rmm::exec_policy(stream),
               offsets->mutable_view().begin<offset_type>(),
               offsets->mutable_view().end<offset_type>(),
               0);
  auto list =
    make_lists_column(1,
                      std::move(offsets),
                      cudf::make_structs_column(0, std::move(inner_children), 0, {}, stream, mr),
                      0,
                      {});

  auto min_col =
    cudf::make_numeric_column(data_type(type_id::FLOAT64), 1, mask_state::UNALLOCATED, stream, mr);
  thrust::fill(rmm::exec_policy(stream),
               min_col->mutable_view().begin<double>(),
               min_col->mutable_view().end<double>(),
               0);
  auto max_col =
    cudf::make_numeric_column(data_type(type_id::FLOAT64), 1, mask_state::UNALLOCATED, stream, mr);
  thrust::fill(rmm::exec_policy(stream),
               max_col->mutable_view().begin<double>(),
               max_col->mutable_view().end<double>(),
               0);

  std::vector<std::unique_ptr<column>> children;
  children.push_back(std::move(list));
  children.push_back(std::move(min_col));
  children.push_back(std::move(max_col));

  return make_structs_column(1, std::move(children), 0, {}, stream, mr);
}

}  // namespace tdigest.

std::unique_ptr<column> percentile_approx(structs_column_view const& input,
                                          column_view const& percentiles,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  tdigest::tdigest_column_view tdv(input);
  CUDF_EXPECTS(percentiles.type().id() == type_id::FLOAT64,
               "percentile_approx expects float64 percentile inputs");

  // output is a list column with each row containing percentiles.size() percentile values
  auto offsets = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, input.size() + 1, mask_state::UNALLOCATED, stream, mr);
  auto row_size_iter = thrust::make_constant_iterator(percentiles.size());
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         row_size_iter,
                         row_size_iter + input.size() + 1,
                         offsets->mutable_view().begin<offset_type>());

  if (percentiles.size() == 0) {
    return cudf::make_lists_column(
      input.size(),
      std::move(offsets),
      cudf::make_empty_column(data_type{type_id::FLOAT64}),
      input.size(),
      cudf::detail::create_null_mask(
        input.size(), mask_state::ALL_NULL, rmm::cuda_stream_view(stream), mr));
  }

  // if any of the input digests are empty, nullify the corresponding output rows (values will be
  // uninitialized)
  auto [bitmask, null_count] = [stream, mr, &tdv]() {
    auto offsets = tdv.centroids().offsets();
    auto iter    = cudf::detail::make_counting_transform_iterator(
      0, [offsets = offsets.begin<offset_type>()] __device__(size_type index) {
        return offsets[index + 1] - offsets[index] == 0 ? 1 : 0;
      });
    auto const null_count = thrust::reduce(rmm::exec_policy(stream), iter, iter + tdv.size(), 0);
    if (null_count == 0) {
      return std::pair<rmm::device_buffer, size_type>{rmm::device_buffer{}, null_count};
    }
    return cudf::detail::valid_if(
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(0) + tdv.size(),
      [offsets = offsets.begin<offset_type>()] __device__(size_type index) {
        return offsets[index + 1] - offsets[index] == 0 ? 0 : 1;
      },
      stream,
      mr);
  }();

  return cudf::make_lists_column(
    input.size(),
    std::move(offsets),
    tdigest::compute_approx_percentiles(input, percentiles, stream, mr),
    null_count,
    std::move(bitmask),
    stream,
    mr);
}

}  // namespace detail

std::unique_ptr<column> percentile_approx(structs_column_view const& input,
                                          column_view const& percentiles,
                                          rmm::mr::device_memory_resource* mr)
{
  return percentile_approx(input, percentiles, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
