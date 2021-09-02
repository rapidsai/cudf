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
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/merge.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/utilities/span.hpp>

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/discard_iterator.h>

std::unique_ptr<rmm::device_uvector<double>> weights_gpu;
std::unique_ptr<cudf::column> mean_gpu;
std::unique_ptr<cudf::column> weight_gpu;

namespace cudf {
namespace groupby {
namespace detail {

namespace {

// the most representative point within a cluster of similar
// values. {mean, weight}
typedef thrust::tuple<double, double, bool> centroid_tuple;

// make a centroid from a scalar with a weight of 1.
template <typename T>
struct make_centroid_tuple {
  column_device_view const col;
  centroid_tuple operator() __device__(size_type index)
  {
    return {col.element<T>(index), 1, col.is_valid(index)};
  }
};

// make a centroid from an input stream of mean/weight values.
struct make_weighted_centroid_tuple {
  double const* mean;
  double const* weight;

  centroid_tuple operator() __device__(size_type index)
  {
    return {mean[index], weight[index], true};
  }
};

// merge two centroids
struct merge_centroid_tuple {
  centroid_tuple operator() __device__(centroid_tuple const& lhs, centroid_tuple const& rhs)
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
 * @brief A functor which returns the nearest cumulative weight in the input stream prior to the
 * specified next weight limits.
 *
 * This functor assumes the weight for all scalars is simply 1. Under this assumption,
 * the nearest weight that will be <= the next limit is simply the nearest whole number, which
 * we can get by just taking floor(next_limit).  For example if our next limit is 3.56, the
 * nearest whole number <= it is floor(3.56) == 3.
 */
struct nearest_value_scalar_weights {
  double operator() __device__(double cur_weight, double next_limit, size_type)
  {
    return floor(next_limit);
  }
};

/**
 * @brief A functor which returns the cumulative input weight for a given index in a
 * set of grouped input values.
 *
 * This functor assumes the weight for all scalars is simply 1. Under this assumption,
 * the cumulative weight for a given value index I is simply I+1.
 */
struct cumulative_scalar_weight {
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
 * @brief A functor which returns the nearest cumulative weight in the input stream prior to the
 * specified next weight limit.
 *
 * This functor assumes we are dealing with grouped, weighted centroids.
 */
struct nearest_value_centroid_weights {
  double const* cumulative_weights;
  offset_type const* outer_offsets;  // groups
  offset_type const* inner_offsets;  // tdigests within a group

  double operator() __device__(double cur_weight, double next_limit, size_type group_index)
  {
    auto const tdigest_begin = outer_offsets[group_index];
    auto const tdigest_end   = outer_offsets[group_index + 1];
    auto const num_weights   = inner_offsets[tdigest_end] - inner_offsets[tdigest_begin];
    double const* group_cumulative_weights = cumulative_weights + inner_offsets[tdigest_begin];

    auto const index = ((thrust::lower_bound(thrust::seq,
                                             group_cumulative_weights,
                                             group_cumulative_weights + num_weights,
                                             next_limit)) -
                        group_cumulative_weights);
    return index == 0 ? 0 : group_cumulative_weights[index - 1];
  }
};

/**
 * @brief A functor which returns the cumulative input weight for a given index in a
 * set of grouped input centroids.
 *
 * This functor assumes we are dealing with grouped, weighted centroids.
 */
struct cumulative_centroid_weight {
  double const* cumulative_weights;
  cudf::device_span<size_type const> group_labels;
  offset_type const* outer_offsets;  // groups
  int num_outer_offsets;
  offset_type const* inner_offsets;  // tdigests with a group
  int num_inner_offsets;

  std::tuple<size_type, size_type, double> operator() __device__(size_type value_index) const
  {
    auto const tdigest_index =
      static_cast<size_type>(
        thrust::upper_bound(
          thrust::seq, inner_offsets, inner_offsets + num_inner_offsets, value_index) -
        inner_offsets) -
      1;
    auto const group_index                 = group_labels[tdigest_index];
    auto const first_tdigest_index         = outer_offsets[group_index];
    auto const first_weight_index          = inner_offsets[first_tdigest_index];
    auto const relative_value_index        = value_index - first_weight_index;
    double const* group_cumulative_weights = cumulative_weights + first_weight_index;

    return {group_index, relative_value_index, group_cumulative_weights[relative_value_index]};
  }
};

// a monotonically increasing scale function which produces a distribution
// of centroids that is more densely packed in the middle of the input
// than at the ends.
__device__ double scale_func_k1(double quantile, double delta_norm)
{
  double k = delta_norm * asin(2.0 * quantile - 1.0);
  k += 1.0;
  double q = (sin(k / delta_norm) + 1.0) / 2.0;
  return q;
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
 * Note that the return group cluster limits will be semi-sparse.  Since we can
 * assume that we will never generate > delta_ clusters per group, but we do
 * not know how many we will generate until we are complete, we simply write out
 * each set of clusters starting at group_cluster_wl + (group_index * delta_)
 *
 * @param delta_              tdigest compression level
 * @param num_groups          The number of input groups
 * @param nearest_weight_     A functor which returns the nearest weight in the input
 * stream that falls before our current cluster limit
 * @param total_weight_       A functor which returns the expected total weight for
 * the entire stream of input values for the specified group.
 * @param group_cluster_wl    Output.  The set of cluster weight limits for each group.
 * @param group_num_clusters  Output.  The number of output clusters for each input group.
 *
 */
template <typename TotalWeightIter, typename NearestWeightFunc>
__global__ void generate_cluster_limits_kernel(int delta_,
                                               size_type num_groups,
                                               NearestWeightFunc nearest_weight,
                                               TotalWeightIter total_weight_,
                                               double* group_cluster_wl,
                                               size_type* group_num_clusters)
{
  int const tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_groups) { return; }

  // we will generate at most delta clusters.
  double const delta        = static_cast<double>(delta_);
  double const delta_norm   = delta / (2.0 * M_PI);
  double const total_weight = total_weight_[tid];
  group_num_clusters[tid]   = 0;
  // a group with nothing in it.
  if (total_weight <= 0) { return; }

  // start each group of clusters at the nearest delta boundary.
  double* cluster_wl = group_cluster_wl + (tid * delta_);

  double next_limit = -1.0;
  double cur_limit  = 0.0;
  double cur_weight = 0.0;
  while (1) {
    // compute the weight we will be at just before closing off the current
    // cluster (because adding the next value will cross the current limit).
    cur_weight =
      next_limit < 0 ? 0 : max(cur_weight + 1, nearest_weight(cur_weight, next_limit, tid));

    if (cur_weight >= total_weight) { break; }
    // based on where we are closing the cluster off (not including the incoming weight),
    // compute the next cluster limit
    double const quantile = cur_weight / total_weight;
    next_limit            = total_weight * scale_func_k1(quantile, delta_norm);
    if (next_limit <= cur_limit) {
      cluster_wl[group_num_clusters[tid]++] = total_weight;
      break;
    } else {
      cluster_wl[group_num_clusters[tid]++] = next_limit;
      cur_limit                             = next_limit;
    }
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
 * Note that the return group cluster limits will be semi-sparse.  Since we can
 * assume that we will never generate > delta_ clusters per group, but we do
 * not know how many we will generate until we are complete, we simply write out
 * each set of clusters starting at group_cluster_wl + (group_index * delta_)
 *
 * @param delta_             tdigest compression level
 * @param num_groups         The number of input groups
 * @param nearest_weight     A functor which returns the nearest weight in the input
 * stream that falls before our current cluster limit
 * @param total_weight       A functor which returns the expected total weight for
 * the entire stream of input values for the specified group.
 *
 * @returns A pair containing The set of cluster weight limits for each group and the set
 * of associated sizes for each.
 */
template <typename TotalWeightIter, typename NearestWeight>
std::pair<rmm::device_uvector<double>, std::unique_ptr<column>> generate_group_cluster_limits(
  int delta,
  size_type num_groups,
  NearestWeight nearest_weight,
  TotalWeightIter total_weight,
  rmm::cuda_stream_view stream)
{
  // we will generate at most delta clusters per group but we don't know exactly how many
  // until after the computation is done, so reserve enough for the worse cast for each group.
  rmm::device_uvector<double> group_cluster_wl(
    delta * num_groups, stream, rmm::mr::get_current_device_resource());
  auto group_num_clusters = cudf::make_fixed_width_column(data_type{type_id::INT32},
                                                          num_groups,
                                                          mask_state::UNALLOCATED,
                                                          stream,
                                                          rmm::mr::get_current_device_resource());

  // each thread computes 1 set of clusters (# of cluster sets == # of groups)
  constexpr size_type block_size = 256;
  cudf::detail::grid_1d const grid(num_groups, block_size);
  generate_cluster_limits_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    delta,
    num_groups,
    nearest_weight,
    total_weight,
    group_cluster_wl.begin(),
    group_num_clusters->mutable_view().begin<size_type>());

  return {std::move(group_cluster_wl), std::move(group_num_clusters)};
}

/**
 * @brief Compute a column of tdigests.
 *
 * Assembles the output tdigest column based on the specified delta, a stream of
 * input values (either scalar or centroids), and an assortment of per-group
 * clustering information.
 *
 * This function is effectively just a reduce_by_key that performs a reduction
 * from input values -> centroid clusters as defined by the the cluster weight
 * boundaries.
 *
 * @param delta              tdigest compression level
 * @param values_begin       Beginning of the range of input values.
 * @param values_end         End of the range of input values.
 * @param cumulative_weight  Functor which returns cumulative weight and group information for
 * an absolute input value index.
 * @param group_cluster_wl   Cluster weight limits for each group
 * @param group_num_clusters The number of clusters for each group
 *
 * @returns A tdigest column with 1 row per output tdigest.
 */
template <typename CentroidIter, typename CumulativeWeight>
std::unique_ptr<column> compute_tdigests(int delta,
                                         CentroidIter centroids_begin,
                                         CentroidIter centroids_end,
                                         CumulativeWeight group_cumulative_weight,
                                         rmm::device_uvector<double> const& group_cluster_wl,
                                         cudf::column_view const& group_num_clusters,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  // the output for each group is column of data that represents the tdigest. since we want 1 row
  // per group, each row will be a list the length of the tdigest for that group. so our output
  // column is of the form:
  // list {
  //  struct {
  //    double    // mean
  //    double    // weight
  //  }
  // }
  //
  // or:  list<struct<double, double>>
  //
  size_type total_inner_size = thrust::reduce(rmm::exec_policy(stream),
                                              group_num_clusters.begin<size_type>(),
                                              group_num_clusters.end<size_type>());
  if (total_inner_size == 0) { return cudf::detail::make_empty_tdigest_column(stream, mr); }
  std::vector<std::unique_ptr<column>> children;
  // mean
  children.push_back(cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, total_inner_size, mask_state::UNALLOCATED, stream, mr));
  // weight
  children.push_back(cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, total_inner_size, mask_state::UNALLOCATED, stream, mr));
  // tdigest struct
  auto tdigests =
    cudf::make_structs_column(total_inner_size, std::move(children), 0, {}, stream, mr);
  auto const num_groups = group_num_clusters.size();
  // generate offsets.
  auto offsets = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, num_groups + 1, mask_state::UNALLOCATED, stream, mr);
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         group_num_clusters.begin<size_type>(),
                         group_num_clusters.begin<size_type>() + offsets->size(),
                         offsets->mutable_view().begin<offset_type>());

  // generate the clusters. this is tricky because groups of input values are getting reduced into
  // seperately sized groups of outputs.
  //
  // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  //   g0      |     g1    |             g2              |     <- incoming values
  //
  //
  // X, X, X,       Y, Y,             Z, Z, Z, Z               <- output tdigests
  //
  auto keys = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    [delta,
     group_cluster_wl   = group_cluster_wl.data(),
     group_num_clusters = group_num_clusters.begin<size_type>(),
     group_cumulative_weight] __device__(size_type value_index) {
      auto [group_index, relative_value_index, cumulative_weight] =
        group_cumulative_weight(value_index);

      // compute start of cluster weight limits for this group
      double const* weight_limits = group_cluster_wl + (group_index * delta);
      auto const num_clusters     = group_num_clusters[group_index];
      // value index relative to this group
      size_type const res =
        relative_value_index == 0
          ? 0
          : static_cast<size_type>(
              thrust::lower_bound(
                thrust::seq, weight_limits, weight_limits + num_clusters, cumulative_weight) -
              weight_limits);

      return res;
    });

  // reduce the centroids down by key.
  cudf::mutable_column_view mean_col =
    tdigests->child(cudf::detail::tdigest_mean_column_index).mutable_view();
  cudf::mutable_column_view weight_col =
    tdigests->child(cudf::detail::tdigest_weight_column_index).mutable_view();
  auto output           = thrust::make_zip_iterator(thrust::make_tuple(
    mean_col.begin<double>(), weight_col.begin<double>(), thrust::make_discard_iterator()));
  auto const num_values = std::distance(centroids_begin, centroids_end);
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        keys,
                        keys + num_values,                // keys
                        centroids_begin,                  // values
                        thrust::make_discard_iterator(),  // key output
                        output,                           // output
                        thrust::equal_to<size_type>{},    // key equality check
                        merge_centroid_tuple{});

  // wrap in the final list column.
  return cudf::make_lists_column(num_groups, std::move(offsets), std::move(tdigests), 0, {});
}

struct scalar_weight {
  size_type const* group_valid_counts;

  __device__ double operator()(size_type group_index) { return group_valid_counts[group_index]; }
};

struct typed_group_tdigest {
  template <typename T, typename std::enable_if_t<cudf::is_numeric<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& col,
                                     cudf::device_span<size_type const> group_offsets,
                                     cudf::device_span<size_type const> group_labels,
                                     column_view const& group_valid_counts,
                                     size_type num_groups,
                                     int delta,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    // first, generate cluster weight information for each input group
    auto total_weight = cudf::detail::make_counting_transform_iterator(
      0, scalar_weight{group_valid_counts.begin<size_type>()});
    auto [group_cluster_wl, group_num_clusters] = generate_group_cluster_limits(
      delta, num_groups, nearest_value_scalar_weights{}, total_weight, stream);

    // for simple input values, the "centroids" all have a weight of 1.
    auto d_col = cudf::column_device_view::create(col);
    auto sorted_centroids =
      cudf::detail::make_counting_transform_iterator(0, make_centroid_tuple<T>{*d_col});

    // generate the final tdigest
    return compute_tdigests(delta,
                            sorted_centroids,
                            sorted_centroids + col.size(),
                            cumulative_scalar_weight{group_offsets, group_labels},
                            group_cluster_wl,
                            *group_num_clusters,
                            stream,
                            mr);
  }

  template <typename T, typename std::enable_if_t<!cudf::is_numeric<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& col,
                                     cudf::device_span<size_type const> group_offsets,
                                     cudf::device_span<size_type const> group_labels,
                                     column_view const& group_valid_counts,
                                     size_type num_groups,
                                     int delta,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("Non-numeric type in group_tdigest");
  }
};

}  // anonymous namespace

std::unique_ptr<column> group_tdigest(column_view const& col,
                                      cudf::device_span<size_type const> group_offsets,
                                      cudf::device_span<size_type const> group_labels,
                                      column_view const& group_valid_counts,
                                      size_type num_groups,
                                      int delta,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  if (col.size() == 0) { return cudf::detail::make_empty_tdigest_column(stream, mr); }

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
                                            int delta,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  cudf::detail::check_is_valid_tdigest_column(input);

  if (num_groups == 0 || input.size() == 0) {
    return cudf::detail::make_empty_tdigest_column(stream, mr);
  }

  lists_column_view lcv(input);
  auto data = lcv.get_sliced_child(stream);
  structs_column_view scv(data);
  auto mean   = scv.get_sliced_child(cudf::detail::tdigest_mean_column_index);
  auto weight = scv.get_sliced_child(cudf::detail::tdigest_weight_column_index);

  // first step is to merge all the tdigests in each group. at the moment the only way to
  // make this work is to retrieve the group sizes (via group_offsets) and the individual digest
  // sizes (via input.offsets()) to the gpu and do the merges.  The scale problem is that while the
  // size of each group will likely be small (size of each group will typically map to # of batches
  // the input data was chopped into for tdigest generation), the -number- of groups can be
  // arbitrarily large.
  //
  // thrust::merge and thrust::merge_by_key don't provide what we need.  What we would need is an
  // algorithm like a super-merge that takes two layers of keys: one which identifies the outer
  // grouping of tdigests, and one which identifies the inner groupings of the tdigests within the
  // outer groups.

  // bring group offsets back to the host
  std::vector<size_type> h_outer_offsets(group_offsets.size());
  cudaMemcpyAsync(h_outer_offsets.data(),
                  group_offsets.data(),
                  sizeof(size_type) * group_offsets.size(),
                  cudaMemcpyDeviceToHost,
                  stream);

  // bring tdigest offsets back to the host
  auto tdigest_offsets = lcv.offsets();
  std::vector<size_type> h_inner_offsets(tdigest_offsets.size());
  cudaMemcpyAsync(h_inner_offsets.data(),
                  tdigest_offsets.begin<size_type>(),
                  sizeof(size_type) * tdigest_offsets.size(),
                  cudaMemcpyDeviceToHost,
                  stream);

  stream.synchronize();

  // extract all means and weights into a table
  cudf::table_view tdigests_unsliced({mean, weight});

  // generate the merged (but not yet compressed) tdigests for each group.
  std::vector<std::unique_ptr<table>> tdigests;
  tdigests.reserve(num_groups);
  auto iter = thrust::make_counting_iterator(0);
  std::transform(iter, iter + num_groups, std::back_inserter(tdigests), [&](int outer_index) {
    // the range of tdigests in this group
    auto const tdigest_start = h_outer_offsets[outer_index];
    auto const tdigest_end   = h_outer_offsets[outer_index + 1];
    auto const num_tdigests  = tdigest_end - tdigest_start;

    // slice each tdigest from the input
    std::vector<table_view> unmerged_tdigests;
    unmerged_tdigests.reserve(num_tdigests);
    auto offset_iter = thrust::make_counting_iterator(tdigest_start);
    std::transform(offset_iter,
                   offset_iter + num_tdigests,
                   std::back_inserter(unmerged_tdigests),
                   [&](int inner_index) {
                     std::vector<size_type> indices{h_inner_offsets[inner_index],
                                                    h_inner_offsets[inner_index + 1]};
                     return cudf::detail::slice(tdigests_unsliced, indices, stream);
                   });

    // merge
    return cudf::detail::merge(unmerged_tdigests, {0}, {order::ASCENDING}, {}, stream, mr);
  });

  // concatenate all the merged tdigests back into one table.
  std::vector<table_view> tdigest_views;
  tdigest_views.reserve(num_groups);
  std::transform(tdigests.begin(),
                 tdigests.end(),
                 std::back_inserter(tdigest_views),
                 [](std::unique_ptr<table> const& t) { return t->view(); });
  auto merged = cudf::detail::concatenate(tdigest_views, stream, mr);

  // generate cumulative weights
  auto cumulative_weights =
    std::make_unique<column>(merged->get_column(cudf::detail::tdigest_weight_column_index),
                             stream,
                             rmm::mr::get_current_device_resource());
  auto keys = cudf::detail::make_counting_transform_iterator(
    0,
    [group_labels      = group_labels.begin(),
     inner_offsets     = tdigest_offsets.begin<size_type>(),
     num_inner_offsets = tdigest_offsets.size()] __device__(int index) {
      // what -original- tdigest index this absolute index corresponds to
      auto const tdigest_index =
        static_cast<size_type>(
          thrust::upper_bound(
            thrust::seq, inner_offsets, inner_offsets + num_inner_offsets, index) -
          inner_offsets) -
        1;

      // what group index the original tdigest belongs to
      return group_labels[tdigest_index];
    });
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                keys,
                                keys + cumulative_weights->size(),
                                cumulative_weights->view().begin<double>(),
                                cumulative_weights->mutable_view().begin<double>());

  // generate cluster limits
  auto total_group_weight = cudf::detail::make_counting_transform_iterator(
    0,
    [outer_offsets = group_offsets.data(),
     inner_offsets = tdigest_offsets.begin<size_type>(),
     cumulative_weights =
       cumulative_weights->view().begin<double>()] __device__(size_type group_index) {
      auto const last_weight_index = inner_offsets[outer_offsets[group_index + 1]] - 1;
      return cumulative_weights[last_weight_index];
    });
  auto [group_cluster_wl, group_num_clusters] = generate_group_cluster_limits(
    delta,
    num_groups,
    nearest_value_centroid_weights{cumulative_weights->view().begin<double>(),
                                   group_offsets.data(),
                                   tdigest_offsets.begin<size_type>()},
    total_group_weight,
    stream);

  // compute the tdigest
  auto centroids = cudf::detail::make_counting_transform_iterator(
    0,
    make_weighted_centroid_tuple{
      merged->get_column(cudf::detail::tdigest_mean_column_index).view().begin<double>(),
      merged->get_column(cudf::detail::tdigest_weight_column_index).view().begin<double>()});
  return compute_tdigests(delta,
                          centroids,
                          centroids + merged->num_rows(),
                          cumulative_centroid_weight{cumulative_weights->view().begin<double>(),
                                                     group_labels,
                                                     group_offsets.data(),
                                                     static_cast<size_type>(group_offsets.size()),
                                                     tdigest_offsets.begin<size_type>(),
                                                     tdigest_offsets.size()},
                          group_cluster_wl,
                          *group_num_clusters,
                          stream,
                          mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
