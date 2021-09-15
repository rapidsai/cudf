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

namespace cudf {
namespace groupby {
namespace detail {

namespace {

// the most representative point within a cluster of similar
// values. {mean, weight}
using centroid_tuple = thrust::tuple<double, double, bool>;

// make a centroid from a scalar with a weight of 1.
template <typename T>
struct make_centroid_tuple {
  column_device_view const col;

  centroid_tuple operator() __device__(size_type index)
  {
    return {static_cast<double>(col.element<T>(index)), 1, col.is_valid(index)};
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
 * specified next weight limit.
 *
 * This functor assumes the weight for all scalars is simply 1. Under this assumption,
 * the nearest weight that will be <= the next limit is simply the nearest integer < the limit,
 * which we can get by just taking floor(next_limit).  For example if our next limit is 3.56, the
 * nearest whole number <= it is floor(3.56) == 3.
 */
struct nearest_value_scalar_weights {
  thrust::pair<double, int> operator() __device__(double next_limit, size_type)
  {
    double const f = floor(next_limit);
    return {f, max(0, static_cast<int>(next_limit) - 1)};
  }
};

/**
 * @brief A functor which returns the nearest cumulative weight in the input stream prior to the
 * specified next weight limit.
 *
 * This functor assumes we are dealing with grouped, sorted, weighted centroids.
 */
struct nearest_value_centroid_weights {
  double const* cumulative_weights;
  offset_type const* outer_offsets;  // groups
  offset_type const* inner_offsets;  // tdigests within a group

  thrust::pair<double, int> operator() __device__(double next_limit, size_type group_index)
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

    return index == 0 ? thrust::pair<double, int>{0, 0}
                      : thrust::pair<double, int>{group_cumulative_weights[index - 1], index - 1};
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
 * @brief A functor which returns the cumulative input weight for a given index in a
 * set of grouped input centroids.
 *
 * This functor assumes we are dealing with grouped, weighted centroids.
 */
struct cumulative_centroid_weight {
  double const* cumulative_weights;
  cudf::device_span<size_type const> group_labels;
  offset_type const* outer_offsets;                    // groups
  cudf::device_span<offset_type const> inner_offsets;  // tdigests with a group

  std::tuple<size_type, size_type, double> operator() __device__(size_type value_index) const
  {
    auto const tdigest_index =
      static_cast<size_type>(
        thrust::upper_bound(thrust::seq, inner_offsets.begin(), inner_offsets.end(), value_index) -
        inner_offsets.begin()) -
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
 * This kernel is called in a two-pass style.  Once to compute the per-group
 * cluster sizes and total # of clusters, and once to compute the actual
 * weight limits per cluster.
 *
 * @param delta_              tdigest compression level
 * @param num_groups          The number of input groups
 * @param nearest_weight_     A functor which returns the nearest weight in the input
 * stream that falls before our current cluster limit
 * @param total_weight_       A functor which returns the expected total weight for
 * the entire stream of input values for the specified group.
 * @param group_cluster_wl    Output.  The set of cluster weight limits for each group.
 * @param group_num_clusters  Output.  The number of output clusters for each input group.
 * @param group_cluster_offsets  Offsets per-group to the start of it's clusters
 *
 */
template <typename TotalWeightIter, typename NearestWeightFunc, typename CumulativeWeight>
__global__ void generate_cluster_limits_kernel(int delta_,
                                               size_type num_groups,
                                               NearestWeightFunc nearest_weight,
                                               TotalWeightIter total_weight_,
                                               CumulativeWeight cumulative_weight,
                                               double* group_cluster_wl,
                                               size_type* group_num_clusters,
                                               offset_type const* group_cluster_offsets)
{
  int const tid          = threadIdx.x + blockIdx.x * blockDim.x;
  auto const group_index = tid;
  if (group_index >= num_groups) { return; }

  // we will generate at most delta clusters.
  double const delta              = static_cast<double>(delta_);
  double const delta_norm         = delta / (2.0 * M_PI);
  double const total_weight       = total_weight_[group_index];
  group_num_clusters[group_index] = 0;
  // a group with nothing in it.
  if (total_weight <= 0) { return; }

  // start at the correct place based on our cluster offset.
  double* cluster_wl =
    group_cluster_wl ? group_cluster_wl + group_cluster_offsets[group_index] : nullptr;

  double cur_limit        = 0.0;
  double cur_weight       = 0.0;
  double next_limit       = -1.0;
  int last_inserted_index = -1;

  // compute the first cluster limit
  double nearest_w;
  int nearest_w_index;
  while (1) {
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

    if (cluster_wl) {
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
      if (nearest_w_index == last_inserted_index || last_inserted_index < 0) {
        nearest_w_index       = last_inserted_index + 1;
        auto [r, i, adjusted] = cumulative_weight(nearest_w_index);
        adjusted_next_limit   = max(next_limit, adjusted);
      }
      cluster_wl[group_num_clusters[group_index]] = adjusted_next_limit;
      last_inserted_index                         = nearest_w_index;
    }
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
 * @param total_weight       A functor which returns the expected total weight for
 * the entire stream of input values for the specified group.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A tuple containing the set of cluster weight limits for each group, a set of
 * list-style offsets indicating group sizes, and the total number of clusters
 */
template <typename TotalWeightIter, typename NearestWeight, typename CumulativeWeight>
std::tuple<rmm::device_uvector<double>, std::unique_ptr<column>, size_type>
generate_group_cluster_info(int delta,
                            size_type num_groups,
                            NearestWeight nearest_weight,
                            TotalWeightIter total_weight,
                            CumulativeWeight cumulative_weight,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr)
{
  constexpr size_type block_size = 256;
  cudf::detail::grid_1d const grid(num_groups, block_size);

  // compute number of clusters per group
  // each thread computes 1 set of clusters (# of cluster sets == # of groups)
  rmm::device_uvector<size_type> group_num_clusters(
    num_groups, stream, rmm::mr::get_current_device_resource());
  generate_cluster_limits_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    delta,
    num_groups,
    nearest_weight,
    total_weight,
    cumulative_weight,
    nullptr,
    group_num_clusters.begin(),
    nullptr);

  // generate group cluster offsets (where the clusters for a given group start and end)
  auto group_cluster_offsets = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, num_groups + 1, mask_state::UNALLOCATED, stream, mr);
  auto cluster_size = cudf::detail::make_counting_transform_iterator(
    0, [group_num_clusters = group_num_clusters.begin(), num_groups] __device__(size_type index) {
      return index == num_groups ? 0 : group_num_clusters[index];
    });
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         cluster_size,
                         cluster_size + num_groups + 1,
                         group_cluster_offsets->mutable_view().begin<offset_type>(),
                         0);

  // total # of clusters
  offset_type total_clusters;
  cudaMemcpyAsync(&total_clusters,
                  group_cluster_offsets->view().begin<offset_type>() + num_groups,
                  sizeof(offset_type),
                  cudaMemcpyDeviceToHost);
  stream.synchronize();

  // fill in the actual cluster weight limits
  rmm::device_uvector<double> group_cluster_wl(
    total_clusters, stream, rmm::mr::get_current_device_resource());
  generate_cluster_limits_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    delta,
    num_groups,
    nearest_weight,
    total_weight,
    cumulative_weight,
    group_cluster_wl.begin(),
    group_num_clusters.begin(),
    group_cluster_offsets->view().begin<offset_type>());

  return {std::move(group_cluster_wl),
          std::move(group_cluster_offsets),
          static_cast<size_type>(total_clusters)};
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
 * @param min_col            Column containing the minimum value per group.
 * @param max_col            Column containing the maximum value per group.
 * @param group_cluster_wl   Cluster weight limits for each group.
 * @param group_cluster_offsets R-value reference of offsets into the cluster weight limits.
 * @param total_clusters     Total number of clusters in all groups.
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
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  // the output for each group is column of data that represents the tdigest. since we want 1 row
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
  //
  if (total_clusters == 0) { return cudf::detail::make_empty_tdigest_column(stream, mr); }
  std::vector<std::unique_ptr<column>> inner_children;
  // mean
  inner_children.push_back(cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, total_clusters, mask_state::UNALLOCATED, stream, mr));
  // weight
  inner_children.push_back(cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, total_clusters, mask_state::UNALLOCATED, stream, mr));
  // tdigest struct
  auto tdigests =
    cudf::make_structs_column(total_clusters, std::move(inner_children), 0, {}, stream, mr);

  // each input group represents an individual tdigest.  within each tdigest, we want the keys
  // to represent cluster indices (for example, if a tdigest had 100 clusters, the keys should fall
  // into the range 0-99).  But since we have multiple tdigests, we need to keep the keys unique
  // between the groups, so we add our group start offset.
  auto keys = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    [delta,
     group_cluster_wl      = group_cluster_wl.data(),
     group_cluster_offsets = group_cluster_offsets->view().begin<offset_type>(),
     group_cumulative_weight] __device__(size_type value_index) -> size_type {
      auto [group_index, relative_value_index, cumulative_weight] =
        group_cumulative_weight(value_index);

      // compute start of cluster weight limits for this group
      double const* weight_limits = group_cluster_wl + group_cluster_offsets[group_index];
      auto const num_clusters =
        group_cluster_offsets[group_index + 1] - group_cluster_offsets[group_index];

      // local cluster index
      size_type const group_cluster_index =
        min(num_clusters - 1,
            static_cast<size_type>(
              thrust::lower_bound(
                thrust::seq, weight_limits, weight_limits + num_clusters, cumulative_weight) -
              weight_limits));

      // add the cluster offset to generate a globally unique key
      return group_cluster_index + group_cluster_offsets[group_index];
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

  // create the list
  auto const num_groups = group_cluster_offsets->size() - 1;
  auto list             = cudf::make_lists_column(
    num_groups, std::move(group_cluster_offsets), std::move(tdigests), 0, {});

  // create final tdigest column
  std::vector<std::unique_ptr<column>> children;
  children.push_back(std::move(list));
  children.push_back(std::move(min_col));
  children.push_back(std::move(max_col));
  return make_structs_column(num_groups, std::move(children), 0, {}, stream, mr);
}

// retrieve total weight of scalar inputs by group index
struct scalar_total_weight {
  size_type const* group_valid_counts;
  __device__ double operator()(size_type group_index) { return group_valid_counts[group_index]; }
};

// return the min/max value of scalar inputs by group index
template <typename T>
struct get_scalar_minmax {
  device_storage_type_t<T> const* values;
  device_span<size_type const> group_offsets;
  size_type const* group_valid_counts;

  __device__ thrust::tuple<double, double> operator()(size_type group_index)
  {
    return {static_cast<double>(values[group_offsets[group_index]]),
            static_cast<double>(
              values[group_offsets[group_index] + (group_valid_counts[group_index] - 1)])};
  }
};

struct typed_group_tdigest {
  template <
    typename T,
    typename std::enable_if_t<cudf::is_numeric<T>() || cudf::is_fixed_point<T>()>* = nullptr>
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
      0, scalar_total_weight{group_valid_counts.begin<size_type>()});
    auto [group_cluster_wl, group_cluster_offsets, total_clusters] =
      generate_group_cluster_info(delta,
                                  num_groups,
                                  nearest_value_scalar_weights{},
                                  total_weight,
                                  cumulative_scalar_weight{group_offsets, group_labels},
                                  stream,
                                  mr);

    // compute min and max columns
    auto min_col = cudf::make_fixed_width_column(
      data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, mr);
    auto max_col = cudf::make_fixed_width_column(
      data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, mr);
    thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(0) + num_groups,
      thrust::make_zip_iterator(thrust::make_tuple(min_col->mutable_view().begin<double>(),
                                                   max_col->mutable_view().begin<double>())),
      get_scalar_minmax<T>{col.begin<device_storage_type_t<T>>(),
                           group_offsets,
                           group_valid_counts.begin<size_type>()});

    // for simple input values, the "centroids" all have a weight of 1.
    auto d_col = cudf::column_device_view::create(col);
    auto scalar_to_centroid =
      cudf::detail::make_counting_transform_iterator(0, make_centroid_tuple<T>{*d_col});

    // generate the final tdigest
    return compute_tdigests(delta,
                            scalar_to_centroid,
                            scalar_to_centroid + col.size(),
                            cumulative_scalar_weight{group_offsets, group_labels},
                            std::move(min_col),
                            std::move(max_col),
                            group_cluster_wl,
                            std::move(group_cluster_offsets),
                            total_clusters,
                            stream,
                            mr);
  }

  template <
    typename T,
    typename std::enable_if_t<!cudf::is_numeric<T>() && !cudf::is_fixed_point<T>()>* = nullptr>
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

  structs_column_view scv(input);
  lists_column_view lcv(scv.child(cudf::detail::tdigest_centroid_column_index));
  // ideally, we would just call .parent().child() here because tdigests cannot be
  // sliced. however, lists_column_view() hides that particular interface. However,
  // for the same reason, get_sliced_child() should be just as cheap.
  auto data = lcv.get_sliced_child(stream);
  structs_column_view tdigest(data);
  auto mean   = tdigest.child(cudf::detail::tdigest_mean_column_index);
  auto weight = tdigest.child(cudf::detail::tdigest_weight_column_index);

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

  // generate min and max values
  auto min_col        = scv.child(cudf::detail::tdigest_min_column_index);
  auto merged_min_col = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, mr);
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        group_labels.begin(),
                        group_labels.end(),
                        min_col.begin<double>(),
                        thrust::make_discard_iterator(),
                        merged_min_col->mutable_view().begin<double>(),
                        thrust::equal_to<size_type>{},  // key equality check
                        thrust::minimum<double>{});

  auto max_col        = scv.child(cudf::detail::tdigest_max_column_index);
  auto merged_max_col = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, num_groups, mask_state::UNALLOCATED, stream, mr);
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        group_labels.begin(),
                        group_labels.end(),
                        max_col.begin<double>(),
                        thrust::make_discard_iterator(),
                        merged_max_col->mutable_view().begin<double>(),
                        thrust::equal_to<size_type>{},  // key equality check
                        thrust::maximum<double>{});

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

  // generate cluster info
  auto total_group_weight = cudf::detail::make_counting_transform_iterator(
    0,
    [outer_offsets = group_offsets.data(),
     inner_offsets = tdigest_offsets.begin<size_type>(),
     cumulative_weights =
       cumulative_weights->view().begin<double>()] __device__(size_type group_index) {
      auto const last_weight_index = inner_offsets[outer_offsets[group_index + 1]] - 1;
      return cumulative_weights[last_weight_index];
    });
  auto [group_cluster_wl, group_cluster_offsets, total_clusters] = generate_group_cluster_info(
    delta,
    num_groups,
    nearest_value_centroid_weights{cumulative_weights->view().begin<double>(),
                                   group_offsets.data(),
                                   tdigest_offsets.begin<size_type>()},
    total_group_weight,
    cumulative_centroid_weight{
      cumulative_weights->view().begin<double>(),
      group_labels,
      group_offsets.data(),
      {tdigest_offsets.begin<offset_type>(), static_cast<size_t>(tdigest_offsets.size())}},
    stream,
    mr);

  // input centroid values
  auto centroids = cudf::detail::make_counting_transform_iterator(
    0,
    make_weighted_centroid_tuple{
      merged->get_column(cudf::detail::tdigest_mean_column_index).view().begin<double>(),
      merged->get_column(cudf::detail::tdigest_weight_column_index).view().begin<double>()});

  // compute the tdigest
  return compute_tdigests(delta,
                          centroids,
                          centroids + merged->num_rows(),
                          cumulative_centroid_weight{cumulative_weights->view().begin<double>(),
                                                     group_labels,
                                                     group_offsets.data(),
                                                     {tdigest_offsets.begin<offset_type>(),
                                                      static_cast<size_t>(tdigest_offsets.size())}},
                          std::move(merged_min_col),
                          std::move(merged_max_col),
                          group_cluster_wl,
                          std::move(group_cluster_offsets),
                          total_clusters,
                          stream,
                          mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
