/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "quantiles/tdigest/tdigest_util.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/merge.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/advance.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cudf {
namespace tdigest {
namespace detail {

namespace {

// the most representative point within a cluster of similar
// values. {mean, weight}
// NOTE: Using a tuple here instead of a struct to take advantage of
// thrust zip iterators for output.
using centroid = thrust::tuple<double, double, bool>;

// make a centroid from a scalar with a weight of 1.
template <typename T>
struct make_centroid {
  column_device_view const col;

  centroid operator() __device__(size_type index) const
  {
    auto const is_valid = col.is_valid(index);
    auto const mean     = is_valid ? convert_to_floating<double>(col.element<T>(index)) : 0.0;
    auto const weight   = is_valid ? 1.0 : 0.0;
    return {mean, weight, is_valid};
  }
};

// make a centroid from a scalar with a weight of 1. this functor
// assumes any value index it is passed is not null
template <typename T>
struct make_centroid_no_nulls {
  column_device_view const col;

  centroid operator() __device__(size_type index) const
  {
    return {convert_to_floating<double>(col.element<T>(index)), 1.0, true};
  }
};

// make a centroid from an input stream of mean/weight values.
struct make_weighted_centroid {
  double const* mean;
  double const* weight;

  centroid operator() __device__(size_type index) { return {mean[index], weight[index], true}; }
};

// merge two centroids
struct merge_centroids {
  centroid operator() __device__(centroid const& lhs, centroid const& rhs) const
  {
    bool const lhs_valid = thrust::get<2>(lhs);
    bool const rhs_valid = thrust::get<2>(rhs);
    if (!lhs_valid && !rhs_valid) { return {0, 0, false}; }
    if (!lhs_valid) { return rhs; }
    if (!rhs_valid) { return lhs; }

    double const lhs_mean   = thrust::get<0>(lhs);
    double const rhs_mean   = thrust::get<0>(rhs);
    double const lhs_weight = thrust::get<1>(lhs);
    double const rhs_weight = thrust::get<1>(rhs);
    double const new_weight = lhs_weight + rhs_weight;
    return {(lhs_mean * lhs_weight + rhs_mean * rhs_weight) / new_weight, new_weight, true};
  }
};

/**
 * @brief A functor which returns the nearest cumulative weight in the grouped input stream prior to
 * the specified next weight limit.
 *
 * This functor assumes the weight for all scalars is simply 1. Under this assumption,
 * the nearest weight that will be <= the next limit is simply the nearest integer < the limit,
 * which we can get by just taking floor(next_limit).  For example if our next limit is 3.56, the
 * nearest whole number <= it is floor(3.56) == 3.
 */
struct nearest_value_scalar_weights_grouped {
  size_type const* group_offsets;

  thrust::pair<double, int> operator() __device__(double next_limit, size_type group_index) const
  {
    double const f                   = floor(next_limit);
    auto const relative_weight_index = max(0, static_cast<int>(next_limit) - 1);
    auto const group_size            = group_offsets[group_index + 1] - group_offsets[group_index];
    return {f, relative_weight_index < group_size ? relative_weight_index : group_size - 1};
  }
};

/**
 * @brief A functor which returns the nearest cumulative weight in the input stream prior to the
 * specified next weight limit.
 *
 * This functor assumes the weight for all scalars is simply 1. Under this assumption,
 * the nearest weight that will be <= the next limit is simply the nearest integer < the limit,
 * which we can get by just taking floor(next_limit).  For example if our next limit is 3.56, the
 * nearest whole number <= it is floor(3.56) == 3.
 */
struct nearest_value_scalar_weights {
  size_type const input_size;

  thrust::pair<double, int> operator() __device__(double next_limit, size_type) const
  {
    double const f                   = floor(next_limit);
    auto const relative_weight_index = max(0, static_cast<int>(next_limit) - 1);
    return {f, relative_weight_index < input_size ? relative_weight_index : input_size - 1};
  }
};

/**
 * @brief A functor which returns the nearest cumulative weight in the input stream prior to the
 * specified next weight limit.
 *
 * This functor assumes we are dealing with grouped, sorted, weighted centroids.
 */
template <typename GroupOffsetsIter>
struct nearest_value_centroid_weights {
  double const* cumulative_weights;  // cumulative weights of non-empty clusters
  GroupOffsetsIter group_offsets;    // groups
  size_type const* tdigest_offsets;  // tdigests within a group

  thrust::pair<double, int> operator() __device__(double next_limit, size_type group_index) const
  {
    auto const tdigest_begin = group_offsets[group_index];
    auto const tdigest_end   = group_offsets[group_index + 1];
    auto const num_weights   = tdigest_offsets[tdigest_end] - tdigest_offsets[tdigest_begin];
    // NOTE: as it is today, this functor will never be called for any digests that are empty, but
    // I'll leave this check here for safety.
    if (num_weights == 0) { return thrust::pair<double, int>{0, 0}; }
    double const* group_cumulative_weights = cumulative_weights + tdigest_offsets[tdigest_begin];

    auto const index = ((thrust::lower_bound(thrust::seq,
                                             group_cumulative_weights,
                                             group_cumulative_weights + num_weights,
                                             next_limit)) -
                        group_cumulative_weights);

    return index == 0 ? thrust::pair<double, int>{0, 0}
                      : thrust::pair<double, int>{group_cumulative_weights[index - 1],
                                                  static_cast<int>(index) - 1};
  }
};

/**
 * @brief A functor which returns the cumulative input weight for a given index in a
 * set of grouped input values.
 *
 * This functor assumes the weight for all scalars is simply 1. Under this assumption,
 * the cumulative weight for a given value index I is simply I+1.
 */
struct cumulative_scalar_weight_grouped {
  cudf::device_span<size_type const> group_offsets;
  cudf::device_span<size_type const> group_labels;
  std::tuple<size_type, size_type, double> operator() __device__(size_type value_index) const
  {
    auto const group_index          = group_labels[value_index];
    auto const relative_value_index = value_index - group_offsets[group_index];
    return {group_index, relative_value_index, relative_value_index + 1};
  }
};

/**
 * @brief A functor which returns the cumulative input weight for a given index in a
 * set of input values.
 *
 * This functor assumes the weight for all scalars is simply 1. Under this assumption,
 * the cumulative weight for a given value index I is simply I+1.
 */
struct cumulative_scalar_weight {
  std::tuple<size_type, size_type, double> operator() __device__(size_type value_index) const
  {
    return {0, value_index, value_index + 1};
  }
};

/**
 * @brief A functor which returns the cumulative input weight for a given index in a
 * set of grouped input centroids.
 *
 * This functor assumes we are dealing with grouped, weighted centroids.
 */
template <typename GroupLabelsIter, typename GroupOffsetsIter>
struct cumulative_centroid_weight {
  double const* cumulative_weights;  // cumulative weights of non-empty clusters
  GroupLabelsIter group_labels;      // group labels for each tdigest including empty ones
  GroupOffsetsIter group_offsets;    // groups
  cudf::device_span<size_type const> tdigest_offsets;  // tdigests with a group

  /**
   * @brief Returns the cumulative weight for a given value index. The index `n` is the index of
   * `n`-th non-empty cluster.
   */
  std::tuple<size_type, size_type, double> operator() __device__(size_type value_index) const
  {
    auto const tdigest_index =
      static_cast<size_type>(
        thrust::upper_bound(
          thrust::seq, tdigest_offsets.begin(), tdigest_offsets.end(), value_index) -
        tdigest_offsets.begin()) -
      1;
    auto const group_index                 = group_labels[tdigest_index];
    auto const first_tdigest_index         = group_offsets[group_index];
    auto const first_weight_index          = tdigest_offsets[first_tdigest_index];
    auto const relative_value_index        = value_index - first_weight_index;
    double const* group_cumulative_weights = cumulative_weights + first_weight_index;

    return {group_index, relative_value_index, group_cumulative_weights[relative_value_index]};
  }
};

// retrieve group info (total weight, size, start offset) of scalar inputs by group index.
struct scalar_group_info_grouped {
  size_type const* group_valid_counts;
  size_type const* group_offsets;

  __device__ thrust::tuple<double, size_type, size_type> operator()(size_type group_index) const
  {
    return {static_cast<double>(group_valid_counts[group_index]),
            group_offsets[group_index + 1] - group_offsets[group_index],
            group_offsets[group_index]};
  }
};

// retrieve group info (total weight, size, start offset) of scalar inputs
struct scalar_group_info {
  double const total_weight;
  size_type const size;

  __device__ thrust::tuple<double, size_type, size_type> operator()(size_type) const
  {
    return {total_weight, size, 0};
  }
};

// retrieve group info of centroid inputs by group index
template <typename GroupOffsetsIter>
struct centroid_group_info {
  double const* cumulative_weights;  // cumulative weights of non-empty clusters
  GroupOffsetsIter group_offsets;
  size_type const* tdigest_offsets;

  __device__ thrust::tuple<double, size_type, size_type> operator()(size_type group_index) const
  {
    // if there's no weights in this group of digests at all, return 0.
    auto const group_start       = tdigest_offsets[group_offsets[group_index]];
    auto const group_end         = tdigest_offsets[group_offsets[group_index + 1]];
    auto const num_weights       = group_end - group_start;
    auto const last_weight_index = group_end - 1;
    return num_weights == 0
             ? thrust::tuple<double, size_type, size_type>{0, num_weights, group_start}
             : thrust::tuple<double, size_type, size_type>{
                 cumulative_weights[last_weight_index], num_weights, group_start};
  }
};

struct tdigest_min {
  __device__ double operator()(thrust::tuple<double, size_type> const& t) const
  {
    auto const min  = thrust::get<0>(t);
    auto const size = thrust::get<1>(t);
    return size > 0 ? min : std::numeric_limits<double>::max();
  }
};

struct tdigest_max {
  __device__ double operator()(thrust::tuple<double, size_type> const& t) const
  {
    auto const max  = thrust::get<0>(t);
    auto const size = thrust::get<1>(t);
    return size > 0 ? max : std::numeric_limits<double>::lowest();
  }
};

// a monotonically increasing scale function which produces a distribution
// of centroids that is more densely packed in the middle of the input
// than at the ends.
__device__ double scale_func_k1(double quantile, double delta_norm)
{
  double k = delta_norm * asin(2.0 * quantile - 1.0);
  k += 1.0;
  double const q = (sin(k / delta_norm) + 1.0) / 2.0;
  return q;
}

// convert a single-row tdigest column to a scalar.
std::unique_ptr<scalar> to_tdigest_scalar(std::unique_ptr<column>&& tdigest,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(tdigest->size() == 1,
               "Encountered invalid tdigest column when converting to scalar");
  auto contents = tdigest->release();
  return std::make_unique<struct_scalar>(table(std::move(contents.children)), true, stream, mr);
}

/**
 * @brief Compute a set of cluster limits (brackets, essentially) for a
 * given tdigest based on the specified delta and the total weight of values
 * to be added.
 *
 * The number of clusters generated will always be <= delta_, where delta_ is
 * a reasonably small number likely << 10000.
 *
 * Each input group gets an independent set of clusters generated. 1 thread
 * per group.
 *
 * This kernel is called in a two-pass style.  Once to compute the per-group
 * cluster sizes and total # of clusters, and once to compute the actual
 * weight limits per cluster.
 *
 * @param delta               tdigest compression level
 * @param num_groups          The number of input groups
 * @param nearest_weight      A functor which returns the nearest weight in the input
 * stream that falls before our current cluster limit
 * @param group_info          A functor which returns the info for the specified group (total
 * weight, size and start offset)
 * @param group_cluster_wl    Output.  The set of cluster weight limits for each group.
 * @param group_num_clusters  Output.  The number of output clusters for each input group.
 * @param group_cluster_offsets  Offsets per-group to the start of it's clusters
 * @param has_nulls Whether or not the input contains nulls
 */

template <typename GroupInfo, typename NearestWeightFunc, typename CumulativeWeight>
CUDF_KERNEL void generate_cluster_limits_kernel(int delta,
                                                size_type num_groups,
                                                NearestWeightFunc nearest_weight,
                                                GroupInfo group_info,
                                                CumulativeWeight cumulative_weight,
                                                double* group_cluster_wl,
                                                size_type* group_num_clusters,
                                                size_type const* group_cluster_offsets,
                                                bool has_nulls)
{
  int const tid = threadIdx.x + blockIdx.x * blockDim.x;

  auto const group_index = tid;
  if (group_index >= num_groups) { return; }

  // we will generate at most delta clusters.
  double const delta_norm = static_cast<double>(delta) / (2.0 * M_PI);
  double total_weight;
  size_type group_size, group_start;
  thrust::tie(total_weight, group_size, group_start) = group_info(group_index);

  // start at the correct place based on our cluster offset.
  double* cluster_wl =
    group_cluster_wl ? group_cluster_wl + group_cluster_offsets[group_index] : nullptr;

  // a group with nothing in it.
  group_num_clusters[group_index] = 0;
  if (total_weight <= 0) {
    // if the input contains nulls we can potentially have a group that generates no
    // clusters because -all- of the input values are null.  in that case, the reduce_by_key call
    // in the tdigest generation step will need a location to store the unused reduction value for
    // that group of nulls. these "stubs" will be postprocessed out afterwards.
    if (has_nulls) { group_num_clusters[group_index] = 1; }
    return;
  }

  double cur_limit        = 0.0;
  double cur_weight       = 0.0;
  double next_limit       = -1.0;
  int last_inserted_index = -1;  // group-relative index into the input stream

  // compute the first cluster limit
  double nearest_w;
  int nearest_w_index;  // group-relative index into the input stream
  while (true) {
    cur_weight = next_limit < 0 ? 0 : max(cur_weight + 1, nearest_w);
    if (cur_weight >= total_weight) { break; }

    // based on where we are closing the cluster off (not including the incoming weight),
    // compute the next cluster limit
    double const quantile = cur_weight / total_weight;
    next_limit            = total_weight * scale_func_k1(quantile, delta_norm);

    // if the next limit is < the cur limit, we're past the end of the distribution, so we're done.
    if (next_limit <= cur_limit) {
      if (cluster_wl) { cluster_wl[group_num_clusters[group_index]] = total_weight; }
      group_num_clusters[group_index]++;
      break;
    }

    // compute the weight we will be at in the input values just before closing off the current
    // cluster (because adding the next value will cross the current limit).
    // NOTE: can't use structured bindings here.
    thrust::tie(nearest_w, nearest_w_index) = nearest_weight(next_limit, group_index);

    // because of the way the scale functions work, it is possible to generate clusters
    // in such a way that we end up with "gaps" where there are no input values that
    // fall into a given cluster.  An example would be this:
    //
    // cluster weight limits = 0.00003, 1.008, 3.008
    //
    // input values(weight) = A(1), B(2), C(3)
    //
    // naively inserting these values into the clusters simply by taking a lower_bound,
    // we would get the following distribution of input values into those 3 clusters.
    //  (), (A), (B,C)
    //
    // whereas what we really want is:
    //
    //  (A), (B), (C)
    //
    // to fix this, we will artificially adjust the output cluster limits to guarantee
    // at least 1 input value will be put in each cluster during the reduction step.
    // this does not affect final centroid results as we still use the "real" weight limits
    // to compute subsequent clusters - the purpose is only to allow cluster selection
    // during the reduction step to be trivial.
    //
    double adjusted_next_limit = next_limit;
    int adjusted_w_index       = nearest_w_index;
    if ((last_inserted_index < 0) ||  // if we haven't inserted anything yet
        (nearest_w_index ==
         last_inserted_index)) {  // if we land in the same bucket as the previous cap

      // force the value into this bucket
      adjusted_w_index = (last_inserted_index == group_size - 1)
                           ? last_inserted_index
                           : max(adjusted_w_index, last_inserted_index + 1);

      // the "adjusted" cluster limit must be high enough so that this value will fall in the
      // bucket. NOTE: cumulative_weight expects an absolute index into the input value stream, not
      // a group-relative index
      [[maybe_unused]] auto [r, i, adjusted_w] = cumulative_weight(adjusted_w_index + group_start);
      adjusted_next_limit                      = max(next_limit, adjusted_w);

      // update the weight with our adjusted value.
      nearest_w = adjusted_w;
    }
    if (cluster_wl) { cluster_wl[group_num_clusters[group_index]] = adjusted_next_limit; }
    last_inserted_index = adjusted_w_index;

    group_num_clusters[group_index]++;
    cur_limit = next_limit;
  }
}

/**
 * @brief Compute a set of cluster limits (brackets, essentially) for a
 * given tdigest based on the specified delta and the total weight of values
 * to be added.
 *
 * The number of clusters generated will always be <= delta_, where delta_ is
 * a reasonably small number likely << 10000.
 *
 * Each input group gets an independent set of clusters generated.
 *
 * @param delta_             tdigest compression level
 * @param num_groups         The number of input groups
 * @param nearest_weight     A functor which returns the nearest weight in the input
 * stream that falls before our current cluster limit
 * @param group_info         A functor which returns the info for the specified group (total weight,
 * size and start offset)
 * @param has_nulls          Whether or not the input data contains nulls
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A tuple containing the set of cluster weight limits for each group, a set of
 * list-style offsets indicating group sizes, and the total number of clusters
 */
template <typename GroupInfo, typename NearestWeight, typename CumulativeWeight>
std::tuple<rmm::device_uvector<double>, std::unique_ptr<column>, size_type>
generate_group_cluster_info(int delta,
                            size_type num_groups,
                            NearestWeight nearest_weight,
                            GroupInfo group_info,
                            CumulativeWeight cumulative_weight,
                            bool has_nulls,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  constexpr size_type block_size = 256;
  cudf::detail::grid_1d const grid(num_groups, block_size);

  // compute number of clusters per group
  // each thread computes 1 set of clusters (# of cluster sets == # of groups)
  rmm::device_uvector<size_type> group_num_clusters(num_groups, stream);
  generate_cluster_limits_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    delta,
    num_groups,
    nearest_weight,
    group_info,
    cumulative_weight,
    nullptr,
    group_num_clusters.begin(),
    nullptr,
    has_nulls);

  // generate group cluster offsets (where the clusters for a given group start and end)
  auto group_cluster_offsets = cudf::make_numeric_column(
    data_type{type_to_id<size_type>()}, num_groups + 1, mask_state::UNALLOCATED, stream, mr);
  auto cluster_size = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_type>(
      [group_num_clusters = group_num_clusters.begin(), num_groups] __device__(size_type index) {
        return index == num_groups ? 0 : group_num_clusters[index];
      }));
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         cluster_size,
                         cluster_size + num_groups + 1,
                         group_cluster_offsets->mutable_view().begin<size_type>(),
                         0);

  // total # of clusters
  size_type total_clusters =
    cudf::detail::get_value<size_type>(group_cluster_offsets->view(), num_groups, stream);

  // fill in the actual cluster weight limits
  rmm::device_uvector<double> group_cluster_wl(total_clusters, stream);
  generate_cluster_limits_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    delta,
    num_groups,
    nearest_weight,
    group_info,
    cumulative_weight,
    group_cluster_wl.begin(),
    group_num_clusters.begin(),
    group_cluster_offsets->view().begin<size_type>(),
    has_nulls);

  return {std::move(group_cluster_wl),
          std::move(group_cluster_offsets),
          static_cast<size_type>(total_clusters)};
}

std::unique_ptr<column> build_output_column(size_type num_rows,
                                            std::unique_ptr<column>&& means,
                                            std::unique_ptr<column>&& weights,
                                            std::unique_ptr<column>&& offsets,
                                            std::unique_ptr<column>&& min_col,
                                            std::unique_ptr<column>&& max_col,
                                            bool has_nulls,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  // whether or not this weight is a stub
  auto is_stub_weight = [weights = weights->view().begin<double>()] __device__(size_type i) {
    return weights[i] == 0;
  };
  // Whether or not this particular tdigest is a stub.
  // This should not be wrapped in `proclaim_return_type` as it will be used inside another
  // device lambda.
  auto is_stub_digest = [offsets = offsets->view().begin<size_type>(), is_stub_weight] __device__(
                          size_type i) { return is_stub_weight(offsets[i]) ? 1 : 0; };

  size_type const num_stubs = [&]() {
    if (!has_nulls) { return 0; }
    auto iter = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<size_type>(is_stub_digest));
    return thrust::reduce(rmm::exec_policy(stream), iter, iter + num_rows);
  }();

  // if there are no stub tdigests, we can return immediately.
  if (num_stubs == 0) {
    return cudf::tdigest::detail::make_tdigest_column(num_rows,
                                                      std::move(means),
                                                      std::move(weights),
                                                      std::move(offsets),
                                                      std::move(min_col),
                                                      std::move(max_col),
                                                      stream,
                                                      mr);
  }

  // otherwise we need to strip out the stubs.
  auto remove_stubs = [&](column_view const& col, size_type num_stubs) {
    auto result = cudf::make_numeric_column(
      data_type{type_id::FLOAT64}, col.size() - num_stubs, mask_state::UNALLOCATED, stream, mr);
    thrust::remove_copy_if(rmm::exec_policy(stream),
                           col.begin<double>(),
                           col.end<double>(),
                           thrust::make_counting_iterator(0),
                           result->mutable_view().begin<double>(),
                           is_stub_weight);
    return result;
  };
  // remove from the means and weights column
  auto _means   = remove_stubs(*means, num_stubs);
  auto _weights = remove_stubs(*weights, num_stubs);

  // adjust offsets.
  rmm::device_uvector<size_type> sizes(num_rows, stream);
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + num_rows,
    sizes.begin(),
    cuda::proclaim_return_type<size_type>([offsets = offsets->view().begin<size_type>()] __device__(
                                            size_type i) { return offsets[i + 1] - offsets[i]; }));
  auto iter = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_type>(
      [sizes = sizes.begin(), is_stub_digest, num_rows] __device__(size_type i) {
        return i == num_rows || is_stub_digest(i) ? 0 : sizes[i];
      }));
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         iter,
                         iter + num_rows + 1,
                         offsets->mutable_view().begin<size_type>(),
                         0);

  // assemble final column
  return cudf::tdigest::detail::make_tdigest_column(num_rows,
                                                    std::move(_means),
                                                    std::move(_weights),
                                                    std::move(offsets),
                                                    std::move(min_col),
                                                    std::move(max_col),
                                                    stream,
                                                    mr);
}

/**
 * @brief A functor which returns the cluster index within a group that the value at
 * the given value index falls into.
 */
template <typename CumulativeWeight>
struct compute_tdigests_keys_fn {
  int const delta;
  double const* group_cluster_wl;
  size_type const* group_cluster_offsets;
  CumulativeWeight group_cumulative_weight;

  __device__ size_type operator()(size_type value_index)
  {
    // get group index, relative value index within the group and cumulative weight.
    [[maybe_unused]] auto [group_index, relative_value_index, cumulative_weight] =
      group_cumulative_weight(value_index);

    auto const num_clusters =
      group_cluster_offsets[group_index + 1] - group_cluster_offsets[group_index];
    if (num_clusters == 0) { return group_cluster_offsets[group_index]; }

    // compute start of cluster weight limits for this group
    double const* weight_limits = group_cluster_wl + group_cluster_offsets[group_index];

    // local cluster index
    size_type const group_cluster_index =
      min(num_clusters - 1,
          static_cast<size_type>(
            thrust::lower_bound(
              thrust::seq, weight_limits, weight_limits + num_clusters, cumulative_weight) -
            weight_limits));

    // add the cluster offset to generate a globally unique key
    return group_cluster_index + group_cluster_offsets[group_index];
  }
};

/**
 * @brief Compute a column of tdigests.
 *
 * Assembles the output tdigest column based on the specified delta, a stream of
 * input values (either scalar or centroids), and an assortment of per-group
 * clustering information.
 *
 * This function is effectively just a reduce_by_key that performs a reduction
 * from input values -> centroid clusters as defined by the cluster weight
 * boundaries.
 *
 * @param delta              tdigest compression level
 * @param centroids_begin    Beginning of the range of centroids.
 * @param centroids_end      End of the range of centroids.
 * @param cumulative_weight  Functor which returns cumulative weight and group information for
 * an absolute input value index.
 * @param min_col            Column containing the minimum value per group.
 * @param max_col            Column containing the maximum value per group.
 * @param group_cluster_wl   Cluster weight limits for each group.
 * @param group_cluster_offsets R-value reference of offsets into the cluster weight limits.
 * @param total_clusters     Total number of clusters in all groups.
 * @param has_nulls          Whether or not the input contains nulls
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A tdigest column with 1 row per output tdigest.
 */
template <typename CentroidIter, typename CumulativeWeight>
std::unique_ptr<column> compute_tdigests(int delta,
                                         CentroidIter centroids_begin,
                                         CentroidIter centroids_end,
                                         CumulativeWeight group_cumulative_weight,
                                         std::unique_ptr<column>&& min_col,
                                         std::unique_ptr<column>&& max_col,
                                         rmm::device_uvector<double> const& group_cluster_wl,
                                         std::unique_ptr<column>&& group_cluster_offsets,
                                         size_type total_clusters,
                                         bool has_nulls,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  // the output for each group is a column of data that represents the tdigest. since we want 1 row
  // per group, each row will be a list the length of the tdigest for that group. so our output
  // column is of the form:
  // struct {
  //   centroids for the digest
  //   list {
  //     struct {
  //       double    // mean
  //       double    // weight
  //     }
  //   }
  //   double       // min
  //   double       // max
  // }
  //
  if (total_clusters == 0) {
    return cudf::tdigest::detail::make_empty_tdigests_column(1, stream, mr);
  }

  // each input group represents an individual tdigest.  within each tdigest, we want the keys
  // to represent cluster indices (for example, if a tdigest had 100 clusters, the keys should fall
  // into the range 0-99).  But since we have multiple tdigests, we need to keep the keys unique
  // between the groups, so we add our group start offset.
  auto keys = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    compute_tdigests_keys_fn<CumulativeWeight>{delta,
                                               group_cluster_wl.data(),
                                               group_cluster_offsets->view().begin<size_type>(),
                                               group_cumulative_weight});

  // mean and weight data
  auto centroid_means = cudf::make_numeric_column(
    data_type{type_id::FLOAT64}, total_clusters, mask_state::UNALLOCATED, stream, mr);
  auto centroid_weights = cudf::make_numeric_column(
    data_type{type_id::FLOAT64}, total_clusters, mask_state::UNALLOCATED, stream, mr);
  // reduce the centroids down by key.
  cudf::mutable_column_view mean_col(*centroid_means);
  cudf::mutable_column_view weight_col(*centroid_weights);

  // reduce the centroids into the clusters
  auto output = thrust::make_zip_iterator(thrust::make_tuple(
    mean_col.begin<double>(), weight_col.begin<double>(), thrust::make_discard_iterator()));

  auto const num_values = std::distance(centroids_begin, centroids_end);
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        keys,
                        keys + num_values,                // keys
                        centroids_begin,                  // values
                        thrust::make_discard_iterator(),  // key output
                        output,                           // output
                        thrust::equal_to{},               // key equality check
                        merge_centroids{});

  // create final tdigest column
  return build_output_column(group_cluster_offsets->size() - 1,
                             std::move(centroid_means),
                             std::move(centroid_weights),
                             std::move(group_cluster_offsets),
                             std::move(min_col),
                             std::move(max_col),
                             has_nulls,
                             stream,
                             mr);
}

// return the min/max value of scalar inputs by group index
template <typename T>
struct get_scalar_minmax_grouped {
  column_device_view const col;
  device_span<size_type const> group_offsets;
  size_type const* group_valid_counts;

  __device__ thrust::tuple<double, double> operator()(size_type group_index)
  {
    auto const valid_count = group_valid_counts[group_index];
    return valid_count > 0
             ? thrust::make_tuple(
                 convert_to_floating<double>(col.element<T>(group_offsets[group_index])),
                 convert_to_floating<double>(
                   col.element<T>(group_offsets[group_index] + valid_count - 1)))
             : thrust::make_tuple(0.0, 0.0);
  }
};

// return the min/max value of scalar inputs
template <typename T>
struct get_scalar_minmax {
  column_device_view const col;
  size_type const valid_count;

  __device__ thrust::tuple<double, double> operator()(size_type)
  {
    return valid_count > 0
             ? thrust::make_tuple(convert_to_floating<double>(col.element<T>(0)),
                                  convert_to_floating<double>(col.element<T>(valid_count - 1)))
             : thrust::make_tuple(0.0, 0.0);
  }
};

struct typed_group_tdigest {
  template <typename T,
            std::enable_if_t<cudf::is_numeric<T>() || cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& col,
                                     cudf::device_span<size_type const> group_offsets,
                                     cudf::device_span<size_type const> group_labels,
                                     cudf::device_span<size_type const> group_valid_counts,
                                     size_type num_groups,
                                     int delta,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    // first, generate cluster weight information for each input group
    auto [group_cluster_wl, group_cluster_offsets, total_clusters] = generate_group_cluster_info(
      delta,
      num_groups,
      nearest_value_scalar_weights_grouped{group_offsets.begin()},
      scalar_group_info_grouped{group_valid_counts.begin(), group_offsets.begin()},
      cumulative_scalar_weight_grouped{group_offsets, group_labels},
      col.null_count() > 0,
      stream,
      mr);

    // device column view. handy because the .element() function
    // automatically handles fixed-point conversions for us
    auto d_col = cudf::column_device_view::create(col, stream);

    // compute min and max columns
    auto min_col = cudf::make_numeric_column(
      data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, mr);
    auto max_col = cudf::make_numeric_column(
      data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, mr);
    thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(0) + num_groups,
      thrust::make_zip_iterator(thrust::make_tuple(min_col->mutable_view().begin<double>(),
                                                   max_col->mutable_view().begin<double>())),
      get_scalar_minmax_grouped<T>{*d_col, group_offsets, group_valid_counts.begin()});

    // for simple input values, the "centroids" all have a weight of 1.
    auto scalar_to_centroid =
      cudf::detail::make_counting_transform_iterator(0, make_centroid<T>{*d_col});

    // generate the final tdigest
    return compute_tdigests(delta,
                            scalar_to_centroid,
                            scalar_to_centroid + col.size(),
                            cumulative_scalar_weight_grouped{group_offsets, group_labels},
                            std::move(min_col),
                            std::move(max_col),
                            group_cluster_wl,
                            std::move(group_cluster_offsets),
                            total_clusters,
                            col.null_count() > 0,
                            stream,
                            mr);
  }

  template <typename T,
            typename... Args,
            std::enable_if_t<!cudf::is_numeric<T>() && !cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(Args&&...)
  {
    CUDF_FAIL("Non-numeric type in group_tdigest");
  }
};

struct typed_reduce_tdigest {
  // this function assumes col is sorted in ascending order with nulls at the end
  template <
    typename T,
    typename std::enable_if_t<cudf::is_numeric<T>() || cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& col,
                                     int delta,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    // treat this the same as the groupby path with a single group.  Note:  even though
    // there is only 1 group there are still multiple keys within the group that represent
    // the clustering of (N input values) -> (1 output centroid), so the final computation
    // remains a reduce_by_key() and not a reduce().
    //
    // additionally we get a few optimizations.
    // - since we only ever have 1 "group" that is sorted with nulls at the end,
    //   we can simply process just the non-null values and act as if the column
    //   is non-nullable, allowing us to process fewer values than if we were doing a groupby.
    //
    // - several of the functors used during the reduction are cheaper than during a groupby.

    auto const valid_count = col.size() - col.null_count();

    // first, generate cluster weight information for each input group
    auto [cluster_wl, cluster_offsets, total_clusters] =
      generate_group_cluster_info(delta,
                                  1,
                                  nearest_value_scalar_weights{valid_count},
                                  scalar_group_info{static_cast<double>(valid_count), valid_count},
                                  cumulative_scalar_weight{},
                                  false,
                                  stream,
                                  mr);

    // device column view. handy because the .element() function
    // automatically handles fixed-point conversions for us
    auto d_col = cudf::column_device_view::create(col, stream);

    // compute min and max columns
    auto min_col = cudf::make_numeric_column(
      data_type{type_id::FLOAT64}, 1, mask_state::UNALLOCATED, stream, mr);
    auto max_col = cudf::make_numeric_column(
      data_type{type_id::FLOAT64}, 1, mask_state::UNALLOCATED, stream, mr);
    thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(0) + 1,
      thrust::make_zip_iterator(thrust::make_tuple(min_col->mutable_view().begin<double>(),
                                                   max_col->mutable_view().begin<double>())),
      get_scalar_minmax<T>{*d_col, valid_count});

    // for simple input values, the "centroids" all have a weight of 1.
    auto scalar_to_centroid =
      cudf::detail::make_counting_transform_iterator(0, make_centroid_no_nulls<T>{*d_col});

    // generate the final tdigest and wrap it in a struct_scalar
    return to_tdigest_scalar(compute_tdigests(delta,
                                              scalar_to_centroid,
                                              scalar_to_centroid + valid_count,
                                              cumulative_scalar_weight{},
                                              std::move(min_col),
                                              std::move(max_col),
                                              cluster_wl,
                                              std::move(cluster_offsets),
                                              total_clusters,
                                              false,
                                              stream,
                                              mr),
                             stream,
                             mr);
  }

  template <
    typename T,
    typename... Args,
    typename std::enable_if_t<!cudf::is_numeric<T>() && !cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<scalar> operator()(Args&&...)
  {
    CUDF_FAIL("Non-numeric type in group_tdigest");
  }
};

/**
 * @brief Functor to compute the number of clusters in each group.
 *
 * Used in `merge_tdigests`.
 */
template <typename GroupOffsetsIter>
struct group_num_clusters_func {
  GroupOffsetsIter group_offsets;
  size_type const* tdigest_offsets;

  __device__ size_type operator()(size_type group_index)
  {
    auto const tdigest_begin = group_offsets[group_index];
    auto const tdigest_end   = group_offsets[group_index + 1];
    return tdigest_offsets[tdigest_end] - tdigest_offsets[tdigest_begin];
  }
};

/**
 * @brief Function to determine if a group is empty.
 *
 * Used in `merge_tdigests`.
 */
struct group_is_empty {
  __device__ bool operator()(size_type group_size) { return group_size == 0; }
};

/**
 * @brief Functor that returns the grouping key for each tdigest cluster.
 *
 * Used in `merge_tdigests`.
 */
template <typename GroupLabelsIter>
struct group_key_func {
  GroupLabelsIter group_labels;
  size_type const* tdigest_offsets;
  size_type num_tdigest_offsets;

  /**
   * @brief Returns the group index for an absolute cluster index. The index `n` is the index of the
   * `n`-th non-empty cluster.
   */
  __device__ size_type operator()(size_type index)
  {
    // what -original- tdigest index this absolute index corresponds to
    auto const iter          = thrust::prev(thrust::upper_bound(
      thrust::seq, tdigest_offsets, tdigest_offsets + num_tdigest_offsets, index));
    auto const tdigest_index = thrust::distance(tdigest_offsets, iter);

    // what group index the original tdigest belongs to
    return group_labels[tdigest_index];
  }
};

// merges all the tdigests within each group. returns a table containing 2 columns:
// the sorted means and weights.
template <typename GroupOffsetIter>
std::pair<rmm::device_uvector<double>, rmm::device_uvector<double>> generate_merged_centroids(
  tdigest_column_view const& tdv,
  GroupOffsetIter group_offsets,
  size_type num_groups,
  rmm::cuda_stream_view stream)
{
  auto temp_mr = cudf::get_current_device_resource_ref();

  auto const total_merged_centroids = tdv.means().size();

  // output is the merged centroids (means, weights)
  rmm::device_uvector<double> output_means(total_merged_centroids, stream, temp_mr);
  rmm::device_uvector<double> output_weights(total_merged_centroids, stream, temp_mr);

  // each group represents a collection of tdigest columns. each row is 1 tdigest.
  // within each group, we want to sort all the centroids within all the tdigests
  // in that group, using the means as the key. the "group offsets" represent the indices of the
  // tdigests, and the "tdigest offsets" represents the list of centroids for a particular tdigest.
  //
  //  rows
  //  ----        centroid 0 ---------
  //  tdigest 0   centroid 1
  //  ----        centroid 2  group 0
  //  tdigest 1   centroid 3
  //  ----        centroid 4 ---------
  //  tdigest 2   centroid 5
  //  ----        centroid 6  group 1
  //  tdigest 3   centroid 7
  //              centroid 8
  //  ----        centroid 9 --------
  auto tdigest_offsets  = tdv.centroids().offsets();
  auto centroid_offsets = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_type>(
      [group_offsets, tdigest_offsets = tdv.centroids().offsets().begin<size_type>()] __device__(
        size_type i) { return tdigest_offsets[group_offsets[i]]; }));

  // perform the sort using the means as the key
  size_t temp_size;
  CUDF_CUDA_TRY(cub::DeviceSegmentedSort::SortPairs(nullptr,
                                                    temp_size,
                                                    tdv.means().begin<double>(),
                                                    output_means.begin(),
                                                    tdv.weights().begin<double>(),
                                                    output_weights.begin(),
                                                    total_merged_centroids,
                                                    num_groups,
                                                    centroid_offsets,
                                                    centroid_offsets + 1,
                                                    stream.value()));

  rmm::device_buffer temp_mem(temp_size, stream, temp_mr);
  CUDF_CUDA_TRY(cub::DeviceSegmentedSort::SortPairs(temp_mem.data(),
                                                    temp_size,
                                                    tdv.means().begin<double>(),
                                                    output_means.begin(),
                                                    tdv.weights().begin<double>(),
                                                    output_weights.begin(),
                                                    total_merged_centroids,
                                                    num_groups,
                                                    centroid_offsets,
                                                    centroid_offsets + 1,
                                                    stream.value()));

  return {std::move(output_means), std::move(output_weights)};
}

/**
 * @brief Perform a merge aggregation of tdigests. This function usually takes the input as the
 * outputs of multiple `typed_group_tdigest` calls, and merges them.
 *
 * A tdigest can be empty in the input, which means that there was no valid input data to generate
 * it. These empty tdigests will have no centroids (means or weights) and will have a `min` and
 * `max` of 0.
 *
 * @param tdv input tdigests. The tdigests within this column are grouped by key.
 * @param group_offsets a device iterator of the offsets to the start of each group. A group is
 * counted as one even when the cluster is empty in it.
 * @param group_labels a device iterator of the the group label for each tdigest cluster including
 * empty clusters.
 * @param num_group_labels the number of unique group labels.
 * @param num_groups the number of groups.
 * @param max_centroids the maximum number of centroids (clusters) in the output (merged) tdigest.
 * @param stream CUDA stream
 * @param mr device memory resource
 *
 * @return A column containing the merged tdigests.
 */
template <typename GroupOffsetIter, typename GroupLabelIter>
std::unique_ptr<column> merge_tdigests(tdigest_column_view const& tdv,
                                       GroupOffsetIter group_offsets,
                                       GroupLabelIter group_labels,
                                       size_t num_group_labels,
                                       size_type num_groups,
                                       int max_centroids,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  // generate min and max values
  auto merged_min_col = cudf::make_numeric_column(
    data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, mr);
  auto min_iter =
    thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(
                                      tdv.min_begin(), cudf::tdigest::detail::size_begin(tdv))),
                                    tdigest_min{});
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        group_labels,
                        group_labels + num_group_labels,
                        min_iter,
                        thrust::make_discard_iterator(),
                        merged_min_col->mutable_view().begin<double>(),
                        thrust::equal_to{},  // key equality check
                        thrust::minimum{});

  auto merged_max_col = cudf::make_numeric_column(
    data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, mr);
  auto max_iter =
    thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(
                                      tdv.max_begin(), cudf::tdigest::detail::size_begin(tdv))),
                                    tdigest_max{});
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        group_labels,
                        group_labels + num_group_labels,
                        max_iter,
                        thrust::make_discard_iterator(),
                        merged_max_col->mutable_view().begin<double>(),
                        thrust::equal_to{},  // key equality check
                        thrust::maximum{});

  auto tdigest_offsets = tdv.centroids().offsets();

  // for any empty groups, set the min and max to be 0. not technically necessary but it makes
  // testing simpler.
  auto group_num_clusters = cudf::detail::make_counting_transform_iterator(
    0,
    group_num_clusters_func<decltype(group_offsets)>{group_offsets,
                                                     tdigest_offsets.begin<size_type>()});
  thrust::replace_if(rmm::exec_policy(stream),
                     merged_min_col->mutable_view().begin<double>(),
                     merged_min_col->mutable_view().end<double>(),
                     group_num_clusters,
                     group_is_empty{},
                     0);
  thrust::replace_if(rmm::exec_policy(stream),
                     merged_max_col->mutable_view().begin<double>(),
                     merged_max_col->mutable_view().end<double>(),
                     group_num_clusters,
                     group_is_empty{},
                     0);

  auto temp_mr = cudf::get_current_device_resource_ref();

  // merge the centroids
  auto [merged_means, merged_weights] =
    generate_merged_centroids(tdv, group_offsets, num_groups, stream);
  size_t const num_centroids = tdv.means().size();
  CUDF_EXPECTS(merged_means.size() == num_centroids,
               "Unexpected number of centroids in merged result");

  // generate cumulative weights
  rmm::device_uvector<double> cumulative_weights(merged_weights.size(), stream, temp_mr);

  // generate group keys for all centroids in the entire column
  rmm::device_uvector<size_type> group_keys(num_centroids, stream, temp_mr);
  auto iter = thrust::make_counting_iterator(0);
  thrust::transform(rmm::exec_policy(stream),
                    iter,
                    iter + num_centroids,
                    group_keys.begin(),
                    group_key_func<decltype(group_labels)>{
                      group_labels, tdigest_offsets.begin<size_type>(), tdigest_offsets.size()});
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                group_keys.begin(),
                                group_keys.begin() + num_centroids,
                                merged_weights.begin(),
                                cumulative_weights.begin());

  auto const delta = max_centroids;

  // TDigest merge takes the output of typed_group_tdigest as its input, which must not have
  // any nulls.
  auto const has_nulls = false;

  // generate cluster info
  auto [group_cluster_wl, group_cluster_offsets, total_clusters] = generate_group_cluster_info(
    delta,
    num_groups,
    nearest_value_centroid_weights<decltype(group_offsets)>{
      cumulative_weights.begin(), group_offsets, tdigest_offsets.begin<size_type>()},
    centroid_group_info<decltype(group_offsets)>{
      cumulative_weights.begin(), group_offsets, tdigest_offsets.begin<size_type>()},
    cumulative_centroid_weight<decltype(group_labels), decltype(group_offsets)>{
      cumulative_weights.begin(),
      group_labels,
      group_offsets,
      {tdigest_offsets.begin<size_type>(), static_cast<size_t>(tdigest_offsets.size())}},
    has_nulls,
    stream,
    mr);

  // input centroid values
  auto centroids = cudf::detail::make_counting_transform_iterator(
    0, make_weighted_centroid{merged_means.begin(), merged_weights.begin()});

  // compute the tdigest
  return compute_tdigests(
    delta,
    centroids,
    centroids + merged_means.size(),
    cumulative_centroid_weight<decltype(group_labels), decltype(group_offsets)>{
      cumulative_weights.begin(),
      group_labels,
      group_offsets,
      {tdigest_offsets.begin<size_type>(), static_cast<size_t>(tdigest_offsets.size())}},
    std::move(merged_min_col),
    std::move(merged_max_col),
    group_cluster_wl,
    std::move(group_cluster_offsets),
    total_clusters,
    has_nulls,
    stream,
    mr);
}

}  // anonymous namespace

std::unique_ptr<scalar> reduce_tdigest(column_view const& col,
                                       int max_centroids,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  if (col.size() == 0) { return cudf::tdigest::detail::make_empty_tdigest_scalar(stream, mr); }

  // since this isn't coming out of a groupby, we need to sort the inputs in ascending
  // order with nulls at the end.
  table_view t({col});
  auto sorted = cudf::detail::sort(
    t, {order::ASCENDING}, {null_order::AFTER}, stream, cudf::get_current_device_resource_ref());

  auto const delta = max_centroids;
  return cudf::type_dispatcher(
    col.type(), typed_reduce_tdigest{}, sorted->get_column(0), delta, stream, mr);
}

struct group_offsets_fn {
  size_type const size;
  CUDF_HOST_DEVICE size_type operator()(size_type i) const { return i == 0 ? 0 : size; }
};

std::unique_ptr<scalar> reduce_merge_tdigest(column_view const& input,
                                             int max_centroids,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  tdigest_column_view tdv(input);

  if (input.size() == 0) { return cudf::tdigest::detail::make_empty_tdigest_scalar(stream, mr); }

  auto group_offsets_ = group_offsets_fn{input.size()};
  auto group_offsets  = cudf::detail::make_counting_transform_iterator(0, group_offsets_);
  auto group_labels   = thrust::make_constant_iterator(0);
  return to_tdigest_scalar(
    merge_tdigests(tdv, group_offsets, group_labels, input.size(), 1, max_centroids, stream, mr),
    stream,
    mr);
}

std::unique_ptr<column> group_tdigest(column_view const& col,
                                      cudf::device_span<size_type const> group_offsets,
                                      cudf::device_span<size_type const> group_labels,
                                      cudf::device_span<size_type const> group_valid_counts,
                                      size_type num_groups,
                                      int max_centroids,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  if (col.size() == 0) { return cudf::tdigest::detail::make_empty_tdigests_column(1, stream, mr); }

  auto const delta = max_centroids;
  return cudf::type_dispatcher(col.type(),
                               typed_group_tdigest{},
                               col,
                               group_offsets,
                               group_labels,
                               group_valid_counts,
                               num_groups,
                               delta,
                               stream,
                               mr);
}

std::unique_ptr<column> group_merge_tdigest(column_view const& input,
                                            cudf::device_span<size_type const> group_offsets,
                                            cudf::device_span<size_type const> group_labels,
                                            size_type num_groups,
                                            int max_centroids,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  tdigest_column_view tdv(input);

  if (num_groups == 0 || input.size() == 0) {
    return cudf::tdigest::detail::make_empty_tdigests_column(1, stream, mr);
  }

  if (tdv.means().size() == 0) {
    // `group_merge_tdigest` takes the output of `typed_group_tdigest` as its input, which wipes
    // out the means and weights for empty clusters. Thus, no mean here indicates that all clusters
    // are empty in the input. Let's skip all complex computation in the below, but just return
    // an empty tdigest per group.
    return cudf::tdigest::detail::make_empty_tdigests_column(num_groups, stream, mr);
  }

  return merge_tdigests(tdv,
                        group_offsets.data(),
                        group_labels.data(),
                        group_labels.size(),
                        num_groups,
                        max_centroids,
                        stream,
                        mr);
}

}  // namespace detail
}  // namespace tdigest
}  // namespace cudf
