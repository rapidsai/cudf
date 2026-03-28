/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tdigest_aggregation.cuh"

namespace cudf {
namespace tdigest {
namespace detail {

namespace {

// make a centroid from an input stream of mean/weight values.
struct make_weighted_centroid {
  double const* mean;
  double const* weight;

  centroid operator() __device__(size_type index) { return {mean[index], weight[index], true}; }
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
    auto const iter          = cuda::std::prev(thrust::upper_bound(
      thrust::seq, tdigest_offsets, tdigest_offsets + num_tdigest_offsets, index));
    auto const tdigest_index = cuda::std::distance(tdigest_offsets, iter);

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
  CUDF_FUNC_RANGE();

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
  size_t temp_size = 0;
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
    thrust::make_transform_iterator(thrust::make_zip_iterator(cuda::std::make_tuple(
                                      tdv.min_begin(), cudf::tdigest::detail::size_begin(tdv))),
                                    tdigest_min{});
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream),
                        group_labels,
                        group_labels + num_group_labels,
                        min_iter,
                        cuda::make_discard_iterator(),
                        merged_min_col->mutable_view().begin<double>(),
                        cuda::std::equal_to{},  // key equality check
                        cuda::minimum{});

  auto merged_max_col = cudf::make_numeric_column(
    data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, mr);
  auto max_iter =
    thrust::make_transform_iterator(thrust::make_zip_iterator(cuda::std::make_tuple(
                                      tdv.max_begin(), cudf::tdigest::detail::size_begin(tdv))),
                                    tdigest_max{});
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream),
                        group_labels,
                        group_labels + num_group_labels,
                        max_iter,
                        cuda::make_discard_iterator(),
                        merged_max_col->mutable_view().begin<double>(),
                        cuda::std::equal_to{},  // key equality check
                        cuda::maximum{});

  auto tdigest_offsets = tdv.centroids().offsets();

  // for any empty groups, set the min and max to be 0. not technically necessary but it makes
  // testing simpler.
  auto group_num_clusters = cudf::detail::make_counting_transform_iterator(
    0,
    group_num_clusters_func<decltype(group_offsets)>{group_offsets,
                                                     tdigest_offsets.begin<size_type>()});
  thrust::replace_if(rmm::exec_policy_nosync(stream),
                     merged_min_col->mutable_view().begin<double>(),
                     merged_min_col->mutable_view().end<double>(),
                     group_num_clusters,
                     group_is_empty{},
                     0);
  thrust::replace_if(rmm::exec_policy_nosync(stream),
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
  thrust::transform(rmm::exec_policy_nosync(stream),
                    iter,
                    iter + num_centroids,
                    group_keys.begin(),
                    group_key_func<decltype(group_labels)>{
                      group_labels, tdigest_offsets.begin<size_type>(), tdigest_offsets.size()});
  thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(stream),
                                group_keys.begin(),
                                group_keys.begin() + num_centroids,
                                merged_weights.begin(),
                                cumulative_weights.begin());

  auto const delta = max_centroids;

  // TDigest merge takes the output of typed_group_tdigest as its input, which must not have
  // any nulls.
  auto const has_nulls = false;

  // generate cluster info
  auto cinfo = [&]() {
    // if we will be at least partially using the CPU here, move the important values into pinned
    // and reference those instead.
    if (use_cpu_for_cluster_computation(num_groups)) {
      auto pinned_mr = cudf::get_pinned_memory_resource();

      rmm::device_uvector<size_type> _p_group_offsets(num_groups + 1, stream, pinned_mr);
      thrust::copy(rmm::exec_policy_nosync(stream),
                   group_offsets,
                   group_offsets + _p_group_offsets.size(),
                   _p_group_offsets.begin());
      cudf::device_span<size_type const> p_group_offsets(_p_group_offsets);

      rmm::device_uvector<double> p_cumulative_weights(cumulative_weights, stream, pinned_mr);

      rmm::device_uvector<size_type> p_tdigest_offsets(tdigest_offsets.size(), stream, pinned_mr);
      thrust::copy(rmm::exec_policy_nosync(stream),
                   tdigest_offsets.begin<size_type>(),
                   tdigest_offsets.begin<size_type>() + p_tdigest_offsets.size(),
                   p_tdigest_offsets.begin());

      rmm::device_uvector<size_type> _p_group_labels(num_group_labels, stream, pinned_mr);
      thrust::copy(rmm::exec_policy_nosync(stream),
                   group_labels,
                   group_labels + num_group_labels,
                   _p_group_labels.begin());
      cudf::device_span<size_type const> p_group_labels(_p_group_labels);

      stream.synchronize();
      return generate_group_cluster_info(
        delta,
        num_groups,
        nearest_value_centroid_weights{
          p_cumulative_weights.begin(), p_group_offsets, p_tdigest_offsets.begin()},
        centroid_group_info{
          p_cumulative_weights.begin(), p_group_offsets, p_tdigest_offsets.begin()},
        cumulative_centroid_weight{
          p_cumulative_weights.begin(),
          p_group_labels,
          p_group_offsets,
          cuda::std::span<size_type const>{p_tdigest_offsets.begin(), p_tdigest_offsets.size()}},
        has_nulls,
        stream,
        mr);
    }

    // otherwise use the device values directly
    return generate_group_cluster_info(
      delta,
      num_groups,
      nearest_value_centroid_weights{
        cumulative_weights.begin(), group_offsets, tdigest_offsets.begin<size_type>()},
      centroid_group_info{
        cumulative_weights.begin(), group_offsets, tdigest_offsets.begin<size_type>()},
      cumulative_centroid_weight{
        cumulative_weights.begin(),
        group_labels,
        group_offsets,
        cuda::std::span<size_type const>{tdigest_offsets.begin<size_type>(),
                                         static_cast<size_t>(tdigest_offsets.size())}},
      has_nulls,
      stream,
      mr);
  }();

  // input centroid values
  auto centroids = cudf::detail::make_counting_transform_iterator(
    0, make_weighted_centroid{merged_means.begin(), merged_weights.begin()});

  // compute the tdigest
  return compute_tdigests(
    delta,
    centroids,
    centroids + merged_means.size(),
    cumulative_centroid_weight{
      cumulative_weights.begin(),
      group_labels,
      group_offsets,
      cuda::std::span<size_type const>{tdigest_offsets.begin<size_type>(),
                                       static_cast<size_t>(tdigest_offsets.size())}},
    std::move(merged_min_col),
    std::move(merged_max_col),
    cinfo,
    has_nulls,
    stream,
    mr);
}

}  // anonymous namespace

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
  auto group_labels   = cuda::make_constant_iterator(0);
  return to_tdigest_scalar(
    merge_tdigests(tdv, group_offsets, group_labels, input.size(), 1, max_centroids, stream, mr),
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
