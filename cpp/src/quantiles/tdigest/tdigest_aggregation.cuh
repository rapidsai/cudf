/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "quantiles/tdigest/tdigest_util.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/merge.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/fixed_point/conv.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace cudf {
namespace tdigest {
namespace detail {

namespace {

// performance tunables.

// if we have <= 2x this many groups, run this many on the CPU. it represents the break
// even point where the CPU and GPU take about the same amount of time for large input groups.
// The theory is this:
// - the cpu and the gpu take about the same time to do this many group cluster computations
// (currently 32).
// - so up to 2x (eg 32 on the CPU + 32 on the GPU) the time should be about the same
// - above 32*2, do it all on the GPU because we can skip the step of copying to pinned and leave it
// all in device memory, and the GPU time will remain flat (or nearly so) as the group count goes up
// substantially.
constexpr size_type max_cpu_groups = 32;
bool use_cpu_for_cluster_computation(size_type num_groups)
{
  return (not is_cpu_cluster_computation_disabled) and (num_groups <= max_cpu_groups * 2);
}

// maximum temporary memory we will allow for using a worst-case allocation strategy that allows us
// to skip half of the cluster generation kernel calls
constexpr size_t max_simple_cluster_usage = 256 * 1024 * 1024;

// the most representative point within a cluster of similar
// values. {mean, weight}
// NOTE: Using a tuple here instead of a struct to take advantage of
// thrust zip iterators for output.
using centroid = cuda::std::tuple<double, double, bool>;

// merge two centroids
struct merge_centroids {
  centroid operator() CUDF_HOST_DEVICE(centroid const& lhs, centroid const& rhs) const
  {
    bool const lhs_valid = cuda::std::get<2>(lhs);
    bool const rhs_valid = cuda::std::get<2>(rhs);
    if (!lhs_valid && !rhs_valid) { return {0, 0, false}; }
    if (!lhs_valid) { return rhs; }
    if (!rhs_valid) { return lhs; }

    double const lhs_mean   = cuda::std::get<0>(lhs);
    double const rhs_mean   = cuda::std::get<0>(rhs);
    double const lhs_weight = cuda::std::get<1>(lhs);
    double const rhs_weight = cuda::std::get<1>(rhs);
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

  cuda::std::pair<double, int> operator()
    CUDF_HOST_DEVICE(double next_limit, size_type group_index) const
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

  cuda::std::pair<double, int> operator() CUDF_HOST_DEVICE(double next_limit, size_type) const
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

  cuda::std::pair<double, int> operator()
    CUDF_HOST_DEVICE(double next_limit, size_type group_index) const
  {
    auto const tdigest_begin = group_offsets[group_index];
    auto const tdigest_end   = group_offsets[group_index + 1];
    auto const num_weights   = tdigest_offsets[tdigest_end] - tdigest_offsets[tdigest_begin];
    // NOTE: as it is today, this functor will never be called for any digests that are empty, but
    // I'll leave this check here for safety.
    if (num_weights == 0) { return cuda::std::pair<double, int>{0, 0}; }
    double const* group_cumulative_weights = cumulative_weights + tdigest_offsets[tdigest_begin];

    auto const index = ((thrust::lower_bound(thrust::seq,
                                             group_cumulative_weights,
                                             group_cumulative_weights + num_weights,
                                             next_limit)) -
                        group_cumulative_weights);

    return index == 0 ? cuda::std::pair<double, int>{0, 0}
                      : cuda::std::pair<double, int>{group_cumulative_weights[index - 1],
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
  // Host-device span, as the offsets may reside in either device memory or pinned host memory
  cuda::std::span<size_type const> group_offsets;
  cuda::std::tuple<size_type, size_type, double> operator()
    CUDF_HOST_DEVICE(size_type value_index) const
  {
    auto const lb =
      thrust::lower_bound(thrust::seq, group_offsets.begin(), group_offsets.end(), value_index) -
      group_offsets.begin();
    auto const group_index          = group_offsets[lb] == value_index ? lb : lb - 1;
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
  cuda::std::tuple<size_type, size_type, double> operator()
    CUDF_HOST_DEVICE(size_type value_index) const
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
  // Host-device span, as the offsets may reside in either device memory or pinned host memory
  cuda::std::span<size_type const> tdigest_offsets;  // tdigests with a group

  /**
   * @brief Returns the cumulative weight for a given value index. The index `n` is the index of
   * `n`-th non-empty cluster.
   */
  cuda::std::tuple<size_type, size_type, double> operator()
    CUDF_HOST_DEVICE(size_type value_index) const
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

  CUDF_HOST_DEVICE cuda::std::tuple<double, size_type, size_type> operator()(
    size_type group_index) const
  {
    return {static_cast<double>(group_valid_counts[group_index]),
            group_offsets[group_index + 1] - group_offsets[group_index],
            group_offsets[group_index]};
  }
};

// retrieve group info (total weight, size) of scalar inputs
struct scalar_group_info {
  double const total_weight;
  size_type const size;

  CUDF_HOST_DEVICE cuda::std::tuple<double, size_type, size_type> operator()(size_type) const
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

  CUDF_HOST_DEVICE cuda::std::tuple<double, size_type, size_type> operator()(
    size_type group_index) const
  {
    // if there's no weights in this group of digests at all, return 0.
    auto const group_start       = tdigest_offsets[group_offsets[group_index]];
    auto const group_end         = tdigest_offsets[group_offsets[group_index + 1]];
    auto const num_weights       = group_end - group_start;
    auto const last_weight_index = group_end - 1;

    return num_weights == 0
             ? cuda::std::tuple<double, size_type, size_type>{0, num_weights, group_start}
             : cuda::std::tuple<double, size_type, size_type>{
                 cumulative_weights[last_weight_index], num_weights, group_start};
  }
};

struct tdigest_min {
  CUDF_HOST_DEVICE double operator()(cuda::std::tuple<double, size_type> const& t) const
  {
    auto const min  = cuda::std::get<0>(t);
    auto const size = cuda::std::get<1>(t);
    return size > 0 ? min : cuda::std::numeric_limits<double>::max();
  }
};

struct tdigest_max {
  CUDF_HOST_DEVICE double operator()(cuda::std::tuple<double, size_type> const& t) const
  {
    auto const max  = cuda::std::get<0>(t);
    auto const size = cuda::std::get<1>(t);
    return size > 0 ? max : cuda::std::numeric_limits<double>::lowest();
  }
};

//
struct cluster_info {
  // Important: there is an optimization in place that makes it o that the cluster weight limits
  // may be over-allocated per group. Therefore we need to separately track the start and count
  // separately, instead of using traditional offsets.

  // cluster weight limits
  rmm::device_uvector<double> cluster_wl{0, cudf::get_default_stream()};
  // start index of weight limits, per group
  rmm::device_uvector<size_type> cluster_start{0, cudf::get_default_stream()};
  // number of weight limits, per group
  rmm::device_uvector<size_type> num_clusters{0, cudf::get_default_stream()};
  bool requires_rescan =
    true;  // in the case of our worst-case memory optimization, this flag
           // is set to true to indicate that cluster_start needs to be rescanned
           // (using num_clusters) as an input to generate proper final output offsets.

  size_type total_clusters;  // total cluster count across all groups
};

// a monotonically increasing scale function which produces a distribution
// of centroids that is more densely packed in the middle of the input
// than at the ends.
CUDF_HOST_DEVICE constexpr inline double scale_func_k1(double quantile,
                                                       double sin_dn,
                                                       double cos_dn)
{
  double x      = 2.0 * quantile - 1.0;
  double result = x * cos_dn + sqrt(1.0 - x * x) * sin_dn;
  return (result + 1.0) * 0.5;
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
 * @param group_index         index of the group being processed
 * @param delta               tdigest compression level
 * @param nearest_weight      A functor which returns the nearest weight in the input
 * stream that falls before our current cluster limit
 * @param group_info          A functor which returns the info for the specified group (total
 * weight, size and start offset)
 * @param cumulative_weight   A functor which returns the cumulative wright for a given value index
 * @param group_cluster_wl    Output.  The set of cluster weight limits for each group.
 * @param group_num_clusters  Output.  The number of output clusters for each input group.
 * @param group_cluster_start Start pos per-group to the start of it's clusters
 * @param has_nulls Whether or not the input contains nulls
 */
template <typename GroupInfo, typename NearestWeightFunc, typename CumulativeWeight>
CUDF_HOST_DEVICE void generate_cluster_limit(int group_index,
                                             int delta,
                                             NearestWeightFunc nearest_weight,
                                             GroupInfo group_info,
                                             CumulativeWeight cumulative_weight,
                                             double* group_cluster_wl,
                                             size_type* group_num_clusters,
                                             size_type const* group_cluster_start,
                                             bool has_nulls)
{
  // we will generate at most delta clusters.
  double total_weight;
  size_type group_size, group_start;
  cuda::std::tie(total_weight, group_size, group_start) = group_info(group_index);

  // start at the correct place based on our cluster offset.
  double* cluster_wl =
    group_cluster_wl ? group_cluster_wl + group_cluster_start[group_index] : nullptr;

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

  double const delta_norm = static_cast<double>(delta) / (2.0 * M_PI);
  double sin_dn, cos_dn;
  sincos(1.0 / delta_norm, &sin_dn, &cos_dn);

  // compute the first cluster limit
  double nearest_w = 0.0;  // unnecessary, but compiler issues an incorrect warning otherwise
  int nearest_w_index;     // group-relative index into the input stream
  while (true) {
    cur_weight = next_limit < 0 ? 0 : max(cur_weight + 1, nearest_w);
    if (cur_weight >= total_weight) { break; }

    // based on where we are closing the cluster off (not including the incoming weight),
    // compute the next cluster limit
    double const quantile = cur_weight / total_weight;
    next_limit            = total_weight * scale_func_k1(quantile, sin_dn, cos_dn);

    // if the next limit is < the cur limit, we're past the end of the distribution, so we're done.
    if (next_limit <= cur_limit) {
      if (cluster_wl) { cluster_wl[group_num_clusters[group_index]] = total_weight; }
      group_num_clusters[group_index]++;
      break;
    }

    // compute the weight we will be at in the input values just before closing off the current
    // cluster (because adding the next value will cross the current limit).
    // NOTE: can't use structured bindings here
    cuda::std::tie(nearest_w, nearest_w_index) = nearest_weight(next_limit, group_index);

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

template <typename GroupIndex,
          typename GroupInfo,
          typename NearestWeightFunc,
          typename CumulativeWeight>
CUDF_KERNEL void generate_cluster_limits_kernel(int delta,
                                                size_type num_groups,
                                                NearestWeightFunc nearest_weight,
                                                GroupIndex group_index,
                                                GroupInfo group_info,
                                                CumulativeWeight cumulative_weight,
                                                double* group_cluster_wl,
                                                size_type* group_num_clusters,
                                                size_type const* group_cluster_offsets,
                                                bool has_nulls)
{
  int const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_groups) { return; }
  generate_cluster_limit(group_index[tid],
                         delta,
                         nearest_weight,
                         group_info,
                         cumulative_weight,
                         group_cluster_wl,
                         group_num_clusters,
                         group_cluster_offsets,
                         has_nulls);
}

/**
 * @brief Wrapper for the cluster computation kernel. Load balances the work between the CPU and the
 * GPU as appropriate.
 *
 * This function expects that if use_cpu_for_cluster_computation() returns true, that the memory
 * provided to it via group_cluster_wl, group_num_clusters and group_cluster_start are in pinned
 * memory (accessible by both CPU and GPU)
 *
 * @param delta               tdigest compression level
 * @param num_groups          Number of groups to be processed
 * @param nearest_weight      A functor which returns the nearest weight in the input
 * stream that falls before our current cluster limit
 * @param group_info          A functor which returns the info for the specified group (total
 * weight, size and start offset)
 * @param cumulative_weight   A functor which returns the cumulative wright for a given value index
 * @param group_cluster_wl    Output.  The set of cluster weight limits for each group.
 * @param group_num_clusters  Output.  The number of output clusters for each input group.
 * @param group_cluster_start Start pos per-group to the start of it's clusters
 * @param has_nulls Whether or not the input contains nulls
 * @param stream Stream to run kernels on
 */
template <typename GroupInfo, typename NearestWeightFunc, typename CumulativeWeight>
void generate_cluster_limits(int delta,
                             size_type num_groups,
                             NearestWeightFunc nearest_weight,
                             GroupInfo group_info,
                             CumulativeWeight cumulative_weight,
                             double* group_cluster_wl,
                             size_type* group_num_clusters,
                             size_type const* group_cluster_start,
                             bool has_nulls,
                             rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  // use the CPU to process the largest of the groups, in certain circumstances.
  // For large groups (those which generate a lot of clusters) the CPU is significantly
  // faster per-group at computing the clustering. At about 8 groups, the GPU starts to become
  // dominant. So, if we have <= 16 groups (where 8 would go to the CPU and 8 would go to the GPU,
  // since they take roughly the same amount of time in the worst case) send the 8 largest to the
  // CPU.  This specifically addresses customer use cases with large inputs and small numbers of
  // groups, such as just 1.
  int const num_cpu_groups =
    use_cpu_for_cluster_computation(num_groups) ? min(max_cpu_groups, num_groups) : 0;
  int const num_gpu_groups = num_groups - num_cpu_groups;

  // start GPU kernel first
  if (num_gpu_groups > 0) {
    constexpr size_type block_size = 256;
    cudf::detail::grid_1d const grid(num_gpu_groups, block_size);

    generate_cluster_limits_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
      delta,
      num_gpu_groups,
      nearest_weight,
      thrust::make_counting_iterator(num_cpu_groups),
      group_info,
      cumulative_weight,
      group_cluster_wl,
      group_num_clusters,
      group_cluster_start,
      has_nulls);
  }

  // overlap CPU work
  for (int idx = 0; idx < num_cpu_groups; idx++) {
    generate_cluster_limit(idx,
                           delta,
                           nearest_weight,
                           group_info,
                           cumulative_weight,
                           group_cluster_wl,
                           group_num_clusters,
                           group_cluster_start,
                           has_nulls);
  }
}

/**
 * @brief Computes the total number of clusters (one double each) for a worst-case allocation
 * strategy that allows us to bypass calling the cluster computation kernel twice.
 */
template <typename GroupInfo>
size_t compute_simple_cluster_count(int delta,
                                    GroupInfo group_info,
                                    cudf::device_span<size_type> group_num_clusters,
                                    rmm::cuda_stream_view stream)
{
  auto const num_groups = group_num_clusters.size();

  // worst-case sizes
  auto iter = thrust::make_counting_iterator(0);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    iter,
    iter + num_groups,
    group_num_clusters.begin(),
    cuda::proclaim_return_type<size_type>([group_info, delta] __device__(size_type i) {
      auto [_, group_size, __] = group_info(i);
      // delta is the largest number of clusters we'll ever generate for any given group.
      // but a group can be significantly smaller than delta as well, in which case we will never
      // generate more than the size of that group.
      return cuda::std::min(delta, group_size);
    }));

  // total size
  return thrust::reduce(
    rmm::exec_policy_nosync(stream), group_num_clusters.begin(), group_num_clusters.end());
}

/**
 * @brief Computes starts positions in the weight-limit buffer of each cluster. We are using
 * the terminology 'start' here instead of 'offsets' because our allocations strategy may
 * cause us to overallocate buffers within each group.
 */
void compute_cluster_starts(cluster_info& cinfo, rmm::cuda_stream_view stream)
{
  auto const num_groups = cinfo.num_clusters.size();
  auto cluster_size     = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_type>(
      [group_num_clusters = cinfo.num_clusters.begin(), num_groups] __device__(size_type index) {
        return index == num_groups ? 0 : group_num_clusters[index];
      }));
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                         cluster_size,
                         cluster_size + num_groups + 1,
                         cinfo.cluster_start.begin(),
                         0);
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
 * @param delta              tdigest compression level
 * @param num_groups         The number of input groups
 * @param nearest_weight     A functor which returns the nearest weight in the input
 * stream that falls before our current cluster limit
 * @param group_info         A functor which returns the info for the specified group (total weight,
 * size and start offset)
 * @param cumulative_weight  Cumulative weight column for computing cluster boundaries
 * @param has_nulls          Whether or not the input data contains nulls
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A tuple containing the set of cluster weight limits for each group, a set of
 * list-style offsets indicating group sizes, and the total number of clusters
 */
template <typename GroupInfo, typename NearestWeight, typename CumulativeWeight>
cluster_info generate_group_cluster_info(int delta,
                                         size_type num_groups,
                                         NearestWeight nearest_weight,
                                         GroupInfo group_info,
                                         CumulativeWeight cumulative_weight,
                                         bool has_nulls,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  bool const use_cpu = use_cpu_for_cluster_computation(num_groups);

  // use the CPU to process the largest of the groups, in certain circumstances.
  // For large groups (those which generate a lot of clusters) the CPU is significantly
  // faster per-group at computing the clustering. At about 8 groups, the GPU starts to become
  // dominant. So, if we have <= 16 groups (where 8 would go to the CPU and 8 would go to the GPU,
  // since they take roughly the same amount of time in the worst case) send the 8 largest to the
  // CPU.  This specifically addresses customer use cases with large inputs and small numbers of
  // groups, such as just 1. if we're going to be using the CPU, use pinned for a few of the temp
  // buffers
  rmm::device_async_resource_ref temp_mr =
    use_cpu ? rmm::device_async_resource_ref{cudf::get_pinned_memory_resource()}
            : cudf::get_current_device_resource_ref();

  // output from the function
  cluster_info cinfo;
  cinfo.num_clusters = rmm::device_uvector<size_type>(num_groups, stream, temp_mr);

  // compute the number of clusters we'd need to allocate for the fast path. the 'fast path' just
  // means using the worst case number of clusters instead of accurately computing the exact cluster
  // count via the kernel.
  size_t const simple_cluster_count =
    compute_simple_cluster_count(delta, group_info, cinfo.num_clusters, stream);
  size_t const simple_mem_usage = simple_cluster_count * sizeof(double);

  // if we can't go down the fast path, run the kernel to compute accurate cluster sizes
  if (simple_mem_usage > max_simple_cluster_usage) {
    cinfo.requires_rescan = false;

    // this kernel does not parallelize very well.
    // each thread computes 1 set of clusters (# of cluster sets == # of groups)
    generate_cluster_limits(delta,
                            num_groups,
                            nearest_weight,
                            group_info,
                            cumulative_weight,
                            nullptr,
                            cinfo.num_clusters.begin(),
                            nullptr,
                            has_nulls,
                            stream);
  }

  // generate group cluster start positions (where the clusters for a given group start. however
  // these are not traditional offsets because in the simple case, we will be over-allocating a
  // worse case estimate)
  cinfo.cluster_start = rmm::device_uvector<size_type>(num_groups + 1, stream, temp_mr);
  compute_cluster_starts(cinfo, stream);

  // because of the simple path, the number of allocated clusters may not end up being the same as
  // the number of real clusters.
  size_t const allocated_clusters = [&]() -> size_t {
    // if we can't go down the fast path, get the count from the result of the scan
    if (simple_mem_usage > max_simple_cluster_usage) {
      return cinfo.cluster_start.element(num_groups, stream);
    }
    // otherwise the simple count we've computed earlier is sufficient
    return simple_cluster_count;
  }();
  cinfo.cluster_wl = rmm::device_uvector<double>(allocated_clusters, stream, temp_mr);

  // sync required after compute_cluster_starts() and before generate_cluster_limits()
  stream.synchronize();

  // fill in the actual cluster weight limits.
  // if we are in the simple case, group_num_clusters will be updated here to reflect the accurate
  // number of clusters per group.
  generate_cluster_limits(delta,
                          num_groups,
                          nearest_weight,
                          group_info,
                          cumulative_weight,
                          cinfo.cluster_wl.begin(),
                          cinfo.num_clusters.begin(),
                          cinfo.cluster_start.begin(),
                          has_nulls,
                          stream);

  // if we used the cpu to do the computation, bring it back to the GPU. for large inputs,
  // leaving this info in pinned results in about a 1/3 increase in time do to the reduction.
  // Note that the size of this data will tend to be very small compared to the size of the
  // input columns themselves, so we are not doing huge memory transfers.
  if (use_cpu) {
    auto p_cluster_wl = std::move(cinfo.cluster_wl);
    cinfo.cluster_wl =
      rmm::device_uvector(p_cluster_wl, stream, cudf::get_current_device_resource_ref());
    auto p_num_clusters = std::move(cinfo.num_clusters);
    cinfo.num_clusters =
      rmm::device_uvector(p_num_clusters, stream, cudf::get_current_device_resource_ref());
    auto p_cluster_start = std::move(cinfo.cluster_start);
    // cluster_start is returned as part of the output, so make sure to use the user supplied mr
    // instead of the current resource.
    cinfo.cluster_start = rmm::device_uvector(p_cluster_start, stream, mr);
    stream.synchronize();
  }

  // if we are in the simple case we need to recompute the total clusters. allocated_cluster count
  // will not be accurate.
  // Note: group_cluster_start does not need to be updated.
  cinfo.total_clusters =
    (simple_mem_usage <= max_simple_cluster_usage)
      ? thrust::reduce(
          rmm::exec_policy_nosync(stream), cinfo.num_clusters.begin(), cinfo.num_clusters.end())
      : allocated_clusters;

  stream.synchronize();

  return cinfo;
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
    return thrust::reduce(rmm::exec_policy_nosync(stream), iter, iter + num_rows);
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
    thrust::remove_copy_if(rmm::exec_policy_nosync(stream),
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
    rmm::exec_policy_nosync(stream),
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
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
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
  size_type const* group_cluster_start;
  size_type const* group_cluster_size;
  CumulativeWeight group_cumulative_weight;

  __device__ size_type operator()(size_type value_index)
  {
    // get group index, relative value index within the group and cumulative weight.
    [[maybe_unused]] auto [group_index, relative_value_index, cumulative_weight] =
      group_cumulative_weight(value_index);

    auto const num_clusters = group_cluster_size[group_index];
    if (num_clusters == 0) { return group_cluster_start[group_index]; }

    // compute start of cluster weight limits for this group
    double const* weight_limits = group_cluster_wl + group_cluster_start[group_index];

    // local cluster index
    size_type const group_cluster_index =
      min(num_clusters - 1,
          static_cast<size_type>(
            thrust::lower_bound(
              thrust::seq, weight_limits, weight_limits + num_clusters, cumulative_weight) -
            weight_limits));

    // add the cluster offset to generate a globally unique key
    return group_cluster_index + group_cluster_start[group_index];
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
 * @param group_cumulative_weight Functor which returns cumulative weight and group information for
 * an absolute input value index.
 * @param min_col            Column containing the minimum value per group.
 * @param max_col            Column containing the maximum value per group.
 * @param cinfo              Clustering info per group.
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
                                         cluster_info& cinfo,
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
  if (cinfo.total_clusters == 0) {
    return cudf::tdigest::detail::make_empty_tdigests_column(1, stream, mr);
  }

  // each input group represents an individual tdigest.  within each tdigest, we want the keys
  // to represent cluster indices (for example, if a tdigest had 100 clusters, the keys should fall
  // into the range 0-99).  But since we have multiple tdigests, we need to keep the keys unique
  // between the groups, so we add our group start offset.
  auto keys = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    compute_tdigests_keys_fn<CumulativeWeight>{delta,
                                               cinfo.cluster_wl.begin(),
                                               cinfo.cluster_start.begin(),
                                               cinfo.num_clusters.begin(),
                                               group_cumulative_weight});

  // mean and weight data
  auto centroid_means = cudf::make_numeric_column(
    data_type{type_id::FLOAT64}, cinfo.total_clusters, mask_state::UNALLOCATED, stream, mr);
  auto centroid_weights = cudf::make_numeric_column(
    data_type{type_id::FLOAT64}, cinfo.total_clusters, mask_state::UNALLOCATED, stream, mr);
  // reduce the centroids down by key.
  cudf::mutable_column_view mean_col(*centroid_means);
  cudf::mutable_column_view weight_col(*centroid_weights);

  // reduce the centroids into the clusters
  auto output = thrust::make_zip_iterator(cuda::std::make_tuple(
    mean_col.begin<double>(), weight_col.begin<double>(), cuda::make_discard_iterator()));

  auto const num_values = std::distance(centroids_begin, centroids_end);
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream),
                        keys,
                        keys + num_values,              // keys
                        centroids_begin,                // values
                        cuda::make_discard_iterator(),  // key output
                        output,                         // output
                        cuda::std::equal_to{},          // key equality check
                        merge_centroids{});

  // generate offsets column. if we are running in the simple case, cinfo.cluster_start will not
  // be accurate, so we need to compute with a scan. in the non-simple case, cinfo.cluster_start
  // will already be accurate
  auto offsets = [&]() {
    if (cinfo.requires_rescan) { compute_cluster_starts(cinfo, stream); }
    return std::make_unique<cudf::column>(
      std::move(cinfo.cluster_start), rmm::device_buffer{0, stream, mr}, 0);
  }();

  // create final tdigest column
  return build_output_column(offsets->size() - 1,
                             std::move(centroid_means),
                             std::move(centroid_weights),
                             std::move(offsets),
                             std::move(min_col),
                             std::move(max_col),
                             has_nulls,
                             stream,
                             mr);
}

}  // anonymous namespace

}  // namespace detail
}  // namespace tdigest
}  // namespace cudf
