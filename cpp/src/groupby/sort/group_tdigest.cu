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
#include <cudf/detail/iterator.cuh>
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

// #include <cudf_test/column_utilities.hpp>

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

struct nearest_value_unit_weights {
  // this functor assumes all incoming values have a weight of 1. under this assumption,
  // the nearest weight that will be <= the next limit is simply the nearest whole number, which 
  // we can get by just taking floor(next_limit).  For example if our next limit is 3.56, the 
  // nearest number <= it is floor(3.56) == 3.
  // 
  double operator() __device__ (double cur_weight, double next_limit)
  {
    return floor(next_limit);
  }
};

constexpr size_type tdigest_mean_column_index = 0;
constexpr size_type tdigest_weight_column_index = 1;

/*
struct nearest_value_real_weights {
  double const* weights;
  size_type const num_weights;

  double operator() __device__ (double cur_weight, double next_limit)
  { 
    return *(thrust::lower_bound(thrust::seq, weights, weights + num_weights, next_limit) - 1);
  }
};
*/

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
    cur_weight = next_limit < 0 ? 0 : max(cur_weight+1, nearest_weight(cur_weight, next_limit));
    if(cur_weight >= total_weight){
      break;
    }
    // based on where we are closing the cluster off, compute the next
    // cluster limit
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
  return {std::move(group_cluster_wl), std::move(group_num_clusters)};
}

template<typename CentroidIter, typename WeightIter>
std::unique_ptr<column> compute_tdigests(int delta, CentroidIter values_begin, CentroidIter values_end, WeightIter cumulative_weights,
                                        cudf::device_span<size_type const> group_offsets,
                                        rmm::device_uvector<double> const& group_cluster_wl, cudf::column_view const& group_num_clusters, 
                                        rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr)
{   
  // the output for each group is column of data that represents the tdigest. since we want 1 row per group, each row
  // will be a list the length of the tdigest for that group. so our output column is of the form:
  // list {
  //  struct {
  //    double                        // mean
  //    double                        // weight
  //  }
  // }
  //
  // or:  list<struct<double, double>>
  //
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
                                                                                  group_offsets = group_offsets.begin(),
                                                                                  num_offsets = group_offsets.size(), 
                                                                                  group_cluster_wl = group_cluster_wl.data(),
                                                                                  group_num_clusters = group_num_clusters.begin<size_type>(), 
                                                                                  cumulative_weights,
                                                                                  values_begin] __device__ (size_type value_index) {
    // compute which group this input value index represents
    auto const group_index = static_cast<size_type>(thrust::upper_bound(thrust::seq, group_offsets, group_offsets + num_offsets, value_index) - group_offsets) - 1;
    // compute start of cluster weight limits for this group
    double const* weight_limits = group_cluster_wl + (group_index * delta);
    auto const num_clusters = group_num_clusters[group_index];
    // value index relative to this group
    auto const relative_value_index = value_index - group_offsets[group_index];
    size_type const res = relative_value_index == 0 ? 0 : static_cast<size_type>(thrust::lower_bound(thrust::seq, weight_limits, weight_limits + num_clusters, cumulative_weights[relative_value_index]) - weight_limits);    
    /*
    if(relative_value_index < 10){
      printf("R: rvi(%d), cum_weight(%d), value(%lf), res(%d)\n", relative_value_index, cumulative_weights[relative_value_index], thrust::get<0>(values_begin[relative_value_index]), res);
    }
    */
    return res;
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
  return cudf::make_lists_column(num_groups, std::move(offsets), std::move(tdigests), 0, {});
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
  auto total_unit_weight = cudf::detail::make_counting_transform_iterator(0, [group_offsets = group_offsets.begin()] __device__ (size_type index){
    return group_offsets[index+1] - group_offsets[index];
  });
  auto [group_cluster_wl, group_num_clusters] = generate_group_cluster_limits(delta, num_groups, nearest_value_unit_weights{}, total_unit_weight, stream);

  /*    
  thrust::host_vector<double> _group_cluster_wl(group_cluster_wl.size());
  cudaMemcpyAsync(_group_cluster_wl.data(), group_cluster_wl.data(), sizeof(double) * group_cluster_wl.size(), cudaMemcpyDeviceToHost, rmm::cuda_stream_default);        
  rmm::cuda_stream_default.synchronize(); 
  std::vector<size_type> h_group_num_clusters(group_num_clusters->size());
  cudaMemcpy(h_group_num_clusters.data(), group_num_clusters->view().begin<size_type>(), sizeof(size_type) * group_num_clusters->size(), cudaMemcpyDeviceToHost);
  for(size_t idx=0; idx<h_group_num_clusters.size(); idx++){
    printf("Group %lu, num_clusters(%d)\n", idx, h_group_num_clusters[idx]);
    for(int s_idx=0; s_idx<h_group_num_clusters[idx]; s_idx++){
      printf("WGPU(%d) : %lf\n", s_idx, _group_cluster_wl[idx * delta + s_idx]);
    }   
  }
  */

  // we have to sort the input values.  
  table_view t({col});
  auto sorted = cudf::detail::sort(t, {}, {}, stream, rmm::mr::get_current_device_resource());
  // for simple input values, the "centroids" all have a weight of 1.
  auto sorted_values = thrust::make_transform_iterator(sorted->get_column(0).view().begin<float>(), make_centroid_tuple{});
  // for simple input values, the cumulative weight for value i is just i+1
  auto cumulative_weights = thrust::make_counting_iterator(1);
  auto tdigests = compute_tdigests(delta, sorted_values, sorted_values + col.size(), cumulative_weights, group_offsets, group_cluster_wl, *group_num_clusters, stream, mr);
    
  // # of rows in the output column == # of groups == # of tdigests
  /*
  lists_column_view lcv(*tdigests);
  structs_column_view scv(lcv.get_sliced_child(stream));
  for(size_type idx=0; idx<lcv.size(); idx++){
    size_type row_start = cudf::detail::get_value<size_type>(lcv.offsets(), idx, stream);
    size_type row_end = cudf::detail::get_value<size_type>(lcv.offsets(), idx + 1, stream);
    size_type num_rows = row_end - row_start;
    
    std::vector<double> mean(num_rows);
    cudaMemcpy(mean.data(), scv.child(tdigest_mean_column_index).begin<double>() + row_start, sizeof(double) * num_rows, cudaMemcpyDeviceToHost);
    std::vector<double> weight(num_rows);
    cudaMemcpy(weight.data(), scv.child(tdigest_weight_column_index).begin<double>() + row_start, sizeof(double) * num_rows, cudaMemcpyDeviceToHost);  
    printf("Group %d:\n", idx);
    for(size_type idx=0; idx<num_rows; idx++){
      printf("GC(%d): %lf, %lf\n", idx, mean[idx], weight[idx]);
    }    
  }  
  */
  
  return tdigests;
}

std::unique_ptr<column> group_merge_tdigest(column_view const& values,
                                            cudf::device_span<size_type const> group_offsets,
                                            size_type num_groups,
                                            int delta,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(values.offset() == 0, "Encountered a sliced tdigest column");

  // for now, just return the 0th tdigest
  if(values.size() > 0){    
    auto sliced = cudf::split(values, {1});
    return std::make_unique<column>(sliced[0], stream, mr);
  } 
  return std::make_unique<column>(values, stream, mr);

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
  
#if 0
// first step is to merge the centroids for each group of tdigests into ascending order
  lists_column_view lcv(values);
  cudf::column_view offsets = lcv.offsets();
  cudf::column_view tdigests = lcv.child();
  cudf::column_view mean_col = tdigests.child(tdigest_mean_column_index);
  cudf::column_view weight_col = tdigests.child(tdigest_weight_column_index);
  auto 

  // retrieve group offsets 
  std::vector<size_type> h_group_offsets(group_offsets.size());
  cudaMemcpyAsync(h_group_offsets.data(), group_offsets.data(), sizeof(size_type) * group_offsets.size(), cudaMemcpyDeviceToHost, stream);

  // retrieve offsets so we can compute individual tdigest sizes (each tdigest is 1 row in the list<struct<double, double>> column, 
  // so the size of each list is the size of each tdigest  
  std::vector<offset_type> h_tdigest_offsets(values.size() + 1);
  lists_column_view lcv(values);
  cudaMemcpyASync(h_tdigest_offsets.data(), lcv.offsets_begin(), sizeof(offset_type) * h_tdigest_offsets.size(), cudaMemcpyDeviceToHost, stream);

  // slice up the groups
  stream.synchronize();
  auto groups = cudf::split(values, h_group_offsets, stream);

  // for each group, merge centroids from all the tdigests, sorted by mean
  std::vector<std::unique_ptr<table>> merged_groups;
  merged_groups.reserve(num_groups);  
  std::transform(groups.begin(), groups.end(), std::back_inserter(merged_groups), [groups](column_view const& group){
    

    // each group consists of N tdigests. build table views for each one.    
    std::vector<table_view> tdigest_tables;
    std::transform(tdigests.begin(), tdigests.end(), std::back_inserter(tdigest_tables), [](cudf::column_view const& tdigest){
      cudf::column_view mean_col   = tdigest.child(tdigest_mean_column_index);
      cudf::column_view weight_col = tdigest.child(tdigest_weight_column_index);
      return table_view({mean_col, weight_col});
    });

    lists_column_view lcv(group);
    column_view tdigest = lcv.get_sliced_child(0);
    cudf::column_view mean_col   = tdigest.child(tdigest_mean_column_index);
    cudf::column_view weight_col = tdigest.child(tdigest_weight_column_index);
  });

  std::vector<table_view> tdigest_tables;
  std::transform(tdigests.begin(), tdigests.end(), std::back_inserter(tdigest_tables), [](cudf::column_view const& tdigest){
    cudf::column_view mean_col   = tdigest.child(tdigest_mean_column_index);
    cudf::column_view weight_col = tdigest.child(tdigest_weight_column_index);
    return table_view({mean_col, weight_col});
  });
  auto merged = detail::merge(tdigest_tables, {0}, {cudf::order::ASCENDING}, {}, stream, rmm::mr::get_current_device_resource());
  
  /*
  {
    std::vector<double> mean;
    mean.resize(merged->num_rows());
    std::vector<double> weight;
    weight.resize(merged->num_rows());
    stream.synchronize();
    cudaMemcpy(mean.data(), merged->get_column(tdigest_mean_column_index).view().begin<double>(), sizeof(double) * merged->num_rows(), cudaMemcpyDeviceToHost);
    cudaMemcpy(weight.data(), merged->get_column(tdigest_weight_column_index).view().begin<double>(), sizeof(double) * merged->num_rows(), cudaMemcpyDeviceToHost);
    for(size_t idx=0; idx<mean.size(); idx++){
      printf("GCTD(%lu) : %lf, %lf\n", idx, mean[idx], weight[idx]);
    }
  }
  */

  // generate cumulative weights and total weight
  auto cumulative_weights = std::make_unique<column>(merged->get_column(tdigest_weight_column_index), stream, rmm::mr::get_current_device_resource());
  thrust::inclusive_scan(rmm::exec_policy(stream), cumulative_weights->view().begin<double>(), cumulative_weights->view().end<double>(), 
                                                   cumulative_weights->mutable_view().begin<double>());
  double total_weight = cudf::detail::get_value<double>(cumulative_weights->view(), cumulative_weights->size() - 1, stream);

  // generate cluster limits
  auto cluster_limits = _generate_cluster_limits(delta, total_weight, nearest_value_real_weights{cumulative_weights->view().begin<double>(), cumulative_weights->size()}, stream);  

  // compute the tdigest
  auto values = cudf::detail::make_counting_transform_iterator(0, make_weighted_centroid_tuple{merged->get_column(tdigest_mean_column_index).view().begin<double>(),
                                                                                               merged->get_column(tdigest_weight_column_index).view().begin<double>()});
  auto tdigest = compute_tdigest(values, values + merged->num_rows(), cumulative_weights->view().begin<double>(), cluster_limits, stream, mr);

  {
    structs_column_view scv(*tdigest);    
    std::vector<double> mean(tdigest->size());
    cudaMemcpy(mean.data(), scv.child(tdigest_mean_column_index).begin<double>(), sizeof(double) * scv.size(), cudaMemcpyDeviceToHost);
    std::vector<double> weight(tdigest->size());
    cudaMemcpy(weight.data(), scv.child(tdigest_weight_column_index).begin<double>(), sizeof(double) * scv.size(), cudaMemcpyDeviceToHost);  
    for(size_type idx=0; idx<scv.size(); idx++){
      printf("GC(%d): %lf, %lf\n", idx, mean[idx], weight[idx]);
    }
  }
  
  return tdigest;
#endif
}


} // namespace detail
} // namespace groupby
} // namespace cudf