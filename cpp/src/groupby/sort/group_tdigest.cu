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
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/utilities/span.hpp>

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/detail/get_value.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/discard_iterator.h>

#include <cudf_test/column_utilities.hpp>

namespace cudf {
namespace groupby {
namespace detail {

namespace {

// the most representative point within a cluster of similar
// values. {mean, weight}
typedef thrust::tuple<double, double> centroid_tuple;

struct make_centroid_tuple {
  centroid_tuple operator() __device__ (double v)
  {
    return {v, 1};
  }
};

struct make_weighted_centroid_tuple {
  double const* mean;
  double const* weight;

  centroid_tuple operator() __device__ (size_type index)
  {
    return {mean[index], weight[index]};
  }
};

struct merge_centroid_tuple {
  centroid_tuple operator() __device__ (centroid_tuple const& lhs, centroid_tuple const& rhs)
  {
    double const lhs_mean = thrust::get<0>(lhs);
    double const rhs_mean = thrust::get<0>(rhs);
    double const lhs_weight = thrust::get<1>(lhs);
    double const rhs_weight = thrust::get<1>(rhs);

    double const new_weight = lhs_weight + rhs_weight;
    return {(lhs_mean * lhs_weight + rhs_mean * rhs_weight) / new_weight, new_weight};
  }
};

constexpr size_type tdigest_mean_column_index = 0;
constexpr size_type tdigest_weight_column_index = 1;

// functors for use with plain input values:  during the initial tdigest generation step
struct nearest_value_scalar_weights {
  // this functor assumes all incoming values have a weight of 1. under this assumption,
  // the nearest weight that will be <= the next limit is simply the nearest whole number, which 
  // we can get by just taking floor(next_limit).  For example if our next limit is 3.56, the 
  // nearest whole number <= it is floor(3.56) == 3.
  // 
  double operator() __device__ (double cur_weight, double next_limit, size_type)
  {
    return floor(next_limit);
  }
};
struct cumulative_scalar_weight {
    offset_type const* group_offsets;
  size_type num_group_offsets;
  /*
  offset_type const* group_offsets;
  size_type num_group_offsets;
  std::pair<size_type, double> operator() __device__ (size_type value_index) const
  {
    auto const group_index = static_cast<size_type>(thrust::upper_bound(thrust::seq, group_offsets, group_offsets + num_group_offsets, value_index) - group_offsets) - 1;
    return {group_index, value_index + 1};
  }
  */  
  std::tuple<size_type, size_type, double> operator() __device__ (size_type value_index) const
  {
    auto const group_index = static_cast<size_type>(thrust::upper_bound(thrust::seq, group_offsets, group_offsets + num_group_offsets, value_index) - group_offsets) - 1;
    auto const relative_value_index = value_index - group_offsets[group_index];
    return {group_index, relative_value_index, relative_value_index + 1};
  }
};

// functors for use with centroid values:  during the tdigest merge step
struct nearest_value_centroid_weights {
  double const* weights;
  size_type const num_weights;
  offset_type const* outer_offsets;
  offset_type const* inner_offsets;

  double operator() __device__ (double cur_weight, double next_limit, size_type group_index)
  { 
    auto const first_tdigest_index = outer_offsets[group_index];
    auto const first_weight_index = inner_offsets[first_tdigest_index];

    auto const last_tdigest_index = outer_offsets[group_index+1] - 1;    
    auto const last_weight_index = inner_offsets[last_tdigest_index+1] - 1;
    
    double const* group_weights = weights + first_weight_index;
    auto const num_weights = last_weight_index - first_weight_index;

    return *(thrust::lower_bound(thrust::seq, group_weights, group_weights + num_weights, next_limit) - 1);
  }
};
struct cumulative_centroid_weight {
  double const* cumulative_weights;
  offset_type const* outer_offsets;  
  int num_outer_offsets;
  offset_type const* inner_offsets;
  int num_inner_offsets;

  /*
  std::pair<size_type, double> operator() __device__ (size_type value_index) const
  { 
    auto const group_index = static_cast<size_type>(thrust::upper_bound(thrust::seq, outer_offsets, outer_offsets + num_outer_offsets, value_index) - outer_offsets) - 1;
    auto const first_tdigest_index = outer_offsets[group_index];
    auto const first_weight_index = inner_offsets[first_tdigest_index];
    double const* group_cumulative_weights = cumulative_weights + first_weight_index;

    printf("GCW(%d, %d, %d, %d): %lf\n", first_tdigest_index, first_weight_index, group_index, value_index, group_cumulative_weights[value_index]);

    return {group_index, group_cumulative_weights[value_index]};
  }
  */
  std::tuple<size_type, size_type, double> operator() __device__ (size_type value_index) const
  { 
    auto const tdigest_index = static_cast<size_type>(thrust::upper_bound(thrust::seq, inner_offsets, inner_offsets + num_inner_offsets, value_index) - inner_offsets) - 1;
    auto const group_index = static_cast<size_type>(thrust::upper_bound(thrust::seq, outer_offsets, outer_offsets + num_outer_offsets, tdigest_index) - outer_offsets) - 1;
    auto const first_tdigest_index = outer_offsets[group_index];
    auto const first_weight_index = inner_offsets[first_tdigest_index];
    auto const relative_value_index = value_index - first_weight_index;
    double const* group_cumulative_weights = cumulative_weights + first_weight_index;

    // printf("GCW(%d, %d, %d, %d): %lf\n", first_tdigest_index, first_weight_index, group_index, value_index, group_cumulative_weights[value_index]);

    return {group_index, relative_value_index, group_cumulative_weights[relative_value_index]};
  }
};

__device__ double scale_func_k1(double quantile, double delta_norm)
{
  double k = delta_norm * asin(2.0 * quantile - 1.0);
  k += 1.0;
  double q = (sin(k / delta_norm) + 1.0) / 2.0;
  return q;
}

// one thread per group of clusters
template<typename TotalWeightIter, typename NearestWeightFunc>
__global__ void generate_cluster_limits_kernel(int delta_, 
                                               size_type num_groups,
                                               NearestWeightFunc nearest_weight,
                                               TotalWeightIter total_weight_, 
                                               double *group_cluster_wl,
                                               size_type *group_num_clusters)
{
  int const tid = threadIdx.x + blockIdx.x * blockDim.x;  
  if(tid >= num_groups){
    return;
  }

  // we will generate at most delta clusters.
  double const delta = static_cast<double>(delta_);
  double const delta_norm = delta / (2.0 * M_PI);
  double const total_weight = total_weight_[tid];

  // start each group of clusters at the nearest delta boundary.
  double *cluster_wl = group_cluster_wl + (tid * delta_);

  double next_limit = -1.0;
  double cur_limit = 0.0;
  double cur_weight = 0.0;
  group_num_clusters[tid] = 0;
  while(1){
    // compute the weight we will be at just before closing off the current
    // cluster (because adding the next value will cross the current limit).
    cur_weight = next_limit < 0 ? 0 : max(cur_weight+1, nearest_weight(cur_weight, next_limit, tid));
    if(cur_weight >= total_weight){
      break;
    }
    // based on where we are closing the cluster off (not including the incoming weight), 
    // compute the next cluster limit
    double const quantile = cur_weight / total_weight;
    next_limit = total_weight * scale_func_k1(quantile, delta_norm);
    if(next_limit <= cur_limit){
      cluster_wl[group_num_clusters[tid]++] = total_weight;
      break;
    } else {   
      cluster_wl[group_num_clusters[tid]++] = next_limit;
      cur_limit = next_limit;      
    }
  }
}

template<typename TotalWeightIter, typename NearestWeight>
std::pair<rmm::device_uvector<double>, std::unique_ptr<column>> generate_group_cluster_limits(int delta, 
                                                                                              size_type num_groups,
                                                                                              NearestWeight nearest_weight,
                                                                                              TotalWeightIter total_weight,
                                                                                              rmm::cuda_stream_view stream)
{
  // we will generate at most delta clusters per group
  rmm::device_uvector<double> group_cluster_wl(delta * num_groups, stream, rmm::mr::get_current_device_resource());
  auto group_num_clusters = cudf::make_fixed_width_column(data_type{type_id::INT32}, num_groups, mask_state::UNALLOCATED, stream, rmm::mr::get_current_device_resource());

  // each thread computes 1 set of clusters (# of cluster sets == # of groups)
  constexpr size_type block_size = 256;  
  cudf::detail::grid_1d const grid(num_groups, block_size);
  generate_cluster_limits_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(delta, num_groups, nearest_weight, total_weight, group_cluster_wl.begin(), group_num_clusters->mutable_view().begin<size_type>());
  
  /*
  thrust::host_vector<double> _group_cluster_wl(group_cluster_wl.size());
  cudaMemcpyAsync(_group_cluster_wl.data(), group_cluster_wl.data(), sizeof(double) * group_cluster_wl.size(), cudaMemcpyDeviceToHost, rmm::cuda_stream_default);        
  rmm::cuda_stream_default.synchronize(); 
  std::vector<size_type> h_group_num_clusters(group_num_clusters->size());
  cudaMemcpy(h_group_num_clusters.data(), group_num_clusters->view().begin<size_type>(), sizeof(size_type) * group_num_clusters->size(), cudaMemcpyDeviceToHost);  
  for(size_t idx=0; idx<h_group_num_clusters.size(); idx++){
    printf("GPU GROUP %lu, num_clusters(%d)\n", idx, h_group_num_clusters[idx]);  
    
    for(int s_idx=0; s_idx<h_group_num_clusters[idx]; s_idx++){
      printf("WGPU(%d) : %lf\n", s_idx, _group_cluster_wl[idx * delta + s_idx]);
    }
  }  
  */

  return {std::move(group_cluster_wl), std::move(group_num_clusters)};
}

template<typename CentroidIter, typename CumulativeWeight>
std::unique_ptr<column> compute_tdigests(int delta, 
                                         CentroidIter values_begin, CentroidIter values_end, 
                                         CumulativeWeight group_cumulative_weight,
                                         cudf::device_span<size_type const> group_offsets,
                                         rmm::device_uvector<double> const& group_cluster_wl, 
                                         cudf::column_view const& group_num_clusters, 
                                         rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr)
{   
  // the output for each group is column of data that represents the tdigest. since we want 1 row per group, each row
  // will be a list the length of the tdigest for that group. so our output column is of the form:
  // list {
  //  struct {
  //    double    // mean
  //    double    // weight
  //  }
  // }
  //
  // or:  list<struct<double, double>>
  //
  size_type total_inner_size = thrust::reduce(rmm::exec_policy(stream), group_num_clusters.begin<size_type>(), group_num_clusters.end<size_type>());
    std::vector<std::unique_ptr<column>> children;
    // mean
    children.push_back(cudf::make_fixed_width_column(data_type{type_id::FLOAT64}, total_inner_size, mask_state::UNALLOCATED, stream, mr));
    // mweight
    children.push_back(cudf::make_fixed_width_column(data_type{type_id::FLOAT64}, total_inner_size, mask_state::UNALLOCATED, stream, mr));        
  // tdigest info    
  auto tdigests = cudf::make_structs_column(total_inner_size, std::move(children), 0, {}, stream, mr);
  auto const num_groups = group_num_clusters.size();
  // generate offsets. 
  auto offsets = cudf::make_fixed_width_column(data_type{type_id::INT32}, num_groups + 1, mask_state::UNALLOCATED, stream, mr);
  thrust::exclusive_scan(rmm::exec_policy(stream), group_num_clusters.begin<size_type>(), group_num_clusters.begin<size_type>() + offsets->size(), offsets->mutable_view().begin<offset_type>());

  //stream.synchronize();
  //cudf::test::print(*offsets);

  // generate the clusters. this is tricky because groups of input values are getting reduced into seperately sized groups of outputs.
  //
  // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  //   g0      |     g1    |             g2              |     <- incoming values
  //
  //
  // X, X, X,       Y, Y,             Z, Z, Z, Z               <- output tdigests
  //  
  auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator(0), [delta,
                                                                                  //group_offsets = group_offsets.begin(),
                                                                                  //num_offsets = group_offsets.size(), 
                                                                                  group_cluster_wl = group_cluster_wl.data(),
                                                                                  group_num_clusters = group_num_clusters.begin<size_type>(), 
                                                                                  group_cumulative_weight]
                                                                                  __device__ (size_type value_index) {
    auto [group_index, relative_value_index, cumulative_weight] = group_cumulative_weight(value_index);

    // compute start of cluster weight limits for this group
    double const* weight_limits = group_cluster_wl + (group_index * delta);
    auto const num_clusters = group_num_clusters[group_index];
    // value index relative to this group
    // auto const relative_value_index = value_index - group_offsets[group_index];
    size_type const res = relative_value_index == 0 ? 0 : static_cast<size_type>(thrust::lower_bound(thrust::seq, weight_limits, weight_limits + num_clusters, cumulative_weight) - weight_limits);
    return res;
    /*
    // compute which group this input value index represents
    auto const group_index = static_cast<size_type>(thrust::upper_bound(thrust::seq, group_offsets, group_offsets + num_offsets, value_index) - group_offsets) - 1;    
    // compute start of cluster weight limits for this group
    double const* weight_limits = group_cluster_wl + (group_index * delta);
    auto const num_clusters = group_num_clusters[group_index];
    // value index relative to this group
    auto const relative_value_index = value_index - group_offsets[group_index];
    double const cumulative_weight = group_cumulative_weight(group_index, relative_value_index);
    size_type const res = relative_value_index == 0 ? 0 : static_cast<size_type>(thrust::lower_bound(thrust::seq, weight_limits, weight_limits + num_clusters, cumulative_weight) - weight_limits);
    return res;
    */
  });
  
  cudf::mutable_column_view mean_col   = tdigests->child(tdigest_mean_column_index).mutable_view();
  cudf::mutable_column_view weight_col = tdigests->child(tdigest_weight_column_index).mutable_view();
  auto output = thrust::make_zip_iterator(thrust::make_tuple(mean_col.begin<double>(), weight_col.begin<double>()));
  auto const num_values = std::distance(values_begin, values_end);
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        keys, keys + num_values,                      // keys
                        values_begin,                                 // values
                        thrust::make_discard_iterator(),              // key output
                        output,                                       // output
                        thrust::equal_to<size_type>{},                // key equality check
                        merge_centroid_tuple{});

  // wrap in the final list column.
  auto result = cudf::make_lists_column(num_groups, std::move(offsets), std::move(tdigests), 0, {});
  
  /*
  // # of rows in the output column == # of groups == # of tdigests  
  lists_column_view lcv(*result);
  structs_column_view scv(lcv.get_sliced_child(stream));
  for(size_type idx=0; idx<lcv.size(); idx++){
    size_type row_start = cudf::detail::get_value<size_type>(lcv.offsets(), idx, stream);
    size_type row_end = cudf::detail::get_value<size_type>(lcv.offsets(), idx + 1, stream);
    size_type num_rows = row_end - row_start;
    
    std::vector<double> mean(num_rows);
    cudaMemcpy(mean.data(), scv.child(tdigest_mean_column_index).begin<double>() + row_start, sizeof(double) * num_rows, cudaMemcpyDeviceToHost);
    std::vector<double> weight(num_rows);
    cudaMemcpy(weight.data(), scv.child(tdigest_weight_column_index).begin<double>() + row_start, sizeof(double) * num_rows, cudaMemcpyDeviceToHost);  
    printf("GPU GROUP %d:\n", idx);
    for(size_type idx=0; idx<num_rows; idx++){
      printf("GC(%d): %lf, %lf\n", idx, mean[idx], weight[idx]);
    }    
  } 
  */ 

  return result;
}

} // anonymous namespace

std::unique_ptr<column> group_tdigest(column_view const& col,
                                      cudf::device_span<size_type const> group_offsets,
                                      size_type num_groups,
                                      int delta,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{   
  /*
  printf("Num groups: %d\n", num_groups);
  std::vector<size_type> h_group_offsets(group_offsets.size());
  cudaMemcpy(h_group_offsets.data(), group_offsets.data(), sizeof(size_type) * group_offsets.size(), cudaMemcpyDeviceToHost);
  printf("Offsets: ");
  for(size_t idx=0; idx<group_offsets.size(); idx++){
    printf("%d, ", h_group_offsets[idx]);
  }
  printf("\n");  
  */

  // generate group cluster and size information
  
  // total weight of a given group of simple values is just the # of values.
  auto total_group_weight = cudf::detail::make_counting_transform_iterator(0, [group_offsets = group_offsets.begin()] __device__ (size_type group_index){
    return group_offsets[group_index+1] - group_offsets[group_index];
  });
  auto [group_cluster_wl, group_num_clusters] = generate_group_cluster_limits(delta, num_groups, nearest_value_scalar_weights{}, total_group_weight, stream);    

  // we have to sort the input values.  
  table_view t({col});
  auto sorted = cudf::detail::sort(t, {}, {}, stream, rmm::mr::get_current_device_resource());
  // for simple input values, the "centroids" all have a weight of 1.
  auto sorted_values = thrust::make_transform_iterator(sorted->get_column(0).view().begin<float>(), make_centroid_tuple{});
  auto result = compute_tdigests(delta, sorted_values, sorted_values + col.size(), cumulative_scalar_weight{group_offsets.begin(), static_cast<size_type>(group_offsets.size())}, group_offsets, group_cluster_wl, *group_num_clusters, stream, mr);
  
  return result;
}

template<typename Data, typename OffsetIter>
auto slice_by_offsets(OffsetIter offsets_begin, OffsetIter offsets_end, Data const& values, rmm::cuda_stream_view stream)
{
  auto const num_offsets = std::distance(offsets_begin, offsets_end);
  if(num_offsets < 3){      
    return std::vector<Data>({values});
  }    
  std::vector<offset_type> splits(num_offsets - 2);
  cudaMemcpyAsync(splits.data(), offsets_begin + 1, num_offsets - 2, cudaMemcpyDeviceToHost, stream);
  stream.synchronize();
  return cudf::split(values, splits);
};

std::unique_ptr<column> group_merge_tdigest(column_view const& input,
                                            cudf::device_span<size_type const> group_offsets,
                                            size_type num_groups,
                                            int delta,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{  
  CUDF_EXPECTS(input.offset() == 0, "Encountered a sliced tdigest column");  
  /*
  printf("Num groups: %d\n", num_groups);
  std::vector<size_type> h_group_offsets(group_offsets.size());
  cudaMemcpy(h_group_offsets.data(), group_offsets.data(), sizeof(size_type) * group_offsets.size(), cudaMemcpyDeviceToHost);
  printf("Offsets: ");
  for(size_t idx=0; idx<group_offsets.size(); idx++){
    printf("%d, ", h_group_offsets[idx]);
  }
  printf("\n");  
  */
  
  // sanity check that this is actually tdigest data
  CUDF_EXPECTS(input.type().id() == type_id::LIST, "Encountered invalid tdigest column");
  CUDF_EXPECTS(input.offset() == 0, "Encountered a sliced tdigest column");
  CUDF_EXPECTS(input.nullable() == false, "Encountered nullable tdigest column");
  lists_column_view lcv(input);
  auto data = lcv.get_sliced_child(stream);
  CUDF_EXPECTS(data.type().id() == type_id::STRUCT, "Encountered invalid tdigest column");
  CUDF_EXPECTS(data.num_children() == 2, "Encountered invalid tdigest column");
  structs_column_view scv(data);    
  auto mean = scv.get_sliced_child(tdigest_mean_column_index);
  CUDF_EXPECTS(mean.type().id() == type_id::FLOAT64, "Encountered invalid tdigest column");
  auto weight = scv.get_sliced_child(tdigest_weight_column_index);
  CUDF_EXPECTS(weight.type().id() == type_id::FLOAT64, "Encountered invalid tdigest column"); 

  // first step is to merge all the tdigests in each group. at the moment the only way to
  // make this work is to retrieve the group sizes (via group_offsets) and the individual digest sizes 
  // (via input.offsets()) to the gpu and do the merges.  The scale problem is that while the 
  // size of each group will likely be small (size of each group will typically map to # of batches the
  // input data was chopped into for tdigest generation), the -number- of groups can be arbitrarily large.
  // 
  // thrust::merge and thrust::merge_by_key don't provide what we need.  What we would need is an algorithm
  // like a super-merge that takes two layers of keys: one which identifies the outer grouping of tdigests,
  // and one which identifies the inner groupings of the tdigests within the outer groups.
  
  // bring group offsets back to the host
  std::vector<size_type> h_outer_offsets(group_offsets.size());
  cudaMemcpyAsync(h_outer_offsets.data(), group_offsets.data(), sizeof(size_type) * group_offsets.size(), cudaMemcpyDeviceToHost, stream);

  // bring tdigest offsets back to the host  
  auto tdigest_offsets = lcv.offsets();
  std::vector<size_type> h_inner_offsets(tdigest_offsets.size());
  cudaMemcpyAsync(h_inner_offsets.data(), tdigest_offsets.begin<size_type>(), sizeof(size_type) * tdigest_offsets.size(), cudaMemcpyDeviceToHost, stream);
  
  stream.synchronize();

  // extract all means and weights into a table
  cudf::table_view tdigests_unsliced({mean, weight});
  
  std::vector<std::unique_ptr<table>> tdigests;
  tdigests.reserve(num_groups);
  auto iter = thrust::make_counting_iterator(0);
  std::transform(iter, iter + num_groups, std::back_inserter(tdigests), [&](int outer_index){    
    // the range of tdigests in this group
    auto const tdigest_start = h_outer_offsets[outer_index];
    auto const tdigest_end = h_outer_offsets[outer_index+1];
    auto const num_tdigests = tdigest_end - tdigest_start;

    // slice each tdigest from the input
    std::vector<table_view> unmerged_tdigests;
    unmerged_tdigests.reserve(num_tdigests);    
    auto offset_iter = thrust::make_counting_iterator(tdigest_start);
    std::transform(offset_iter, offset_iter + num_tdigests, std::back_inserter(unmerged_tdigests), [&](int inner_index){
      std::vector<size_type> indices{h_inner_offsets[inner_index], h_inner_offsets[inner_index+1]};
      return cudf::detail::slice(tdigests_unsliced, indices, stream);
    });

    // merge 
    return cudf::detail::merge(unmerged_tdigests, {0}, {order::ASCENDING}, {}, stream, mr);
  });

/*
  for(size_t idx=0; idx<tdigests.size(); idx++){
    table_view t = tdigests[idx]->view();
    // mean
    cudf::test::print(t.column(0));
    // weight
    cudf::test::print(t.column(1));
  }
  */

  // concatenate all the merged tdigests back into one table
  std::vector<table_view> tdigest_views;
  tdigest_views.reserve(num_groups);
  std::transform(tdigests.begin(), tdigests.end(), std::back_inserter(tdigest_views), [](std::unique_ptr<table> const& t){
    return t->view();
  });
  auto merged = cudf::detail::concatenate(tdigest_views, stream, mr);

  // generate cumulative weights
  auto cumulative_weights = std::make_unique<column>(merged->get_column(tdigest_weight_column_index), stream, rmm::mr::get_current_device_resource());
  auto keys = cudf::detail::make_counting_transform_iterator(0, [outer_offsets = group_offsets.data(),
                                                                 num_outer_offsets = group_offsets.size(),
                                                                 inner_offsets = tdigest_offsets.begin<size_type>(),
                                                                 num_inner_offsets = tdigest_offsets.size()] __device__ (int index){
    // what -original- tdigest index this absolute index corresponds to
    auto const tdigest_index = static_cast<size_type>(thrust::upper_bound(thrust::seq, inner_offsets, inner_offsets + num_inner_offsets, index) - inner_offsets) - 1;

    // what group index the original tdigest belongs to
    return static_cast<size_type>(thrust::upper_bound(thrust::seq, outer_offsets, outer_offsets + num_outer_offsets, tdigest_index) - outer_offsets) - 1;
  });
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream), 
                                keys, keys + cumulative_weights->size(),
                                cumulative_weights->view().begin<double>(),
                                cumulative_weights->mutable_view().begin<double>());

/*
  std::vector<double> cw(cumulative_weights->size());
  cudaMemcpy(cw.data(), cumulative_weights->view().begin<double>(), cw.size() * sizeof(double), cudaMemcpyDeviceToHost);
  for(size_t idx=0; idx<cw.size(); idx++){
    printf("CW(%lu): %lf\n", idx, cw[idx]);
  }
  */

  // generate cluster limits
  auto total_group_weight = cudf::detail::make_counting_transform_iterator(0, [outer_offsets = group_offsets.data(),
                                                                               inner_offsets = tdigest_offsets.begin<size_type>(),
                                                                               cumulative_weights = cumulative_weights->view().begin<double>()] __device__ (size_type group_index){
    // index of the last incoming tdigest that makes up this group
    auto const last_tdigest_index = outer_offsets[group_index+1] - 1;
    auto const last_weight_index = inner_offsets[last_tdigest_index+1] - 1;
    return cumulative_weights[last_weight_index];
  });
  auto [group_cluster_wl, group_num_clusters] = generate_group_cluster_limits(delta, 
                                                     num_groups, 
                                                     nearest_value_centroid_weights{cumulative_weights->view().begin<double>(), 
                                                                                    cumulative_weights->size(),
                                                                                    group_offsets.data(),
                                                                                    tdigest_offsets.begin<size_type>()},
                                                     total_group_weight, 
                                                     stream);

  // compute the tdigest
  auto values = cudf::detail::make_counting_transform_iterator(0, make_weighted_centroid_tuple{merged->get_column(tdigest_mean_column_index).view().begin<double>(),
                                                                                               merged->get_column(tdigest_weight_column_index).view().begin<double>()});
  auto result = compute_tdigests(delta, 
                                 values, values + merged->num_rows(), 
                                 cumulative_centroid_weight{cumulative_weights->view().begin<double>(),
                                                            group_offsets.data(),
                                                            static_cast<size_type>(group_offsets.size()),
                                                            tdigest_offsets.begin<size_type>(),
                                                            tdigest_offsets.size()},
                                 group_offsets, group_cluster_wl, *group_num_clusters, stream, mr);   

  return result;
}


} // namespace detail
} // namespace groupby
} // namespace cudf