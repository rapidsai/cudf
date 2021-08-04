/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <quantiles/quantiles_util.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <vector>

namespace cudf {
namespace detail {
template <typename SortMapIterator>
std::unique_ptr<table> quantiles(table_view const& input,
                                 SortMapIterator sortmap,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  auto quantile_idx_lookup = [sortmap, interp, size = input.num_rows()] __device__(double q) {
    auto selector = [sortmap] __device__(auto idx) { return sortmap[idx]; };
    return detail::select_quantile<size_type>(selector, size, q, interp);
  };

  auto const q_device = cudf::detail::make_device_uvector_async(q, stream);

  auto quantile_idx_iter = thrust::make_transform_iterator(q_device.begin(), quantile_idx_lookup);

  return detail::gather(input,
                        quantile_idx_iter,
                        quantile_idx_iter + q.size(),
                        out_of_bounds_policy::DONT_CHECK,
                        stream,
                        mr);
}

__device__ double lerp(double a, double b, double t)
{
  return a + t * (b - a);
}

__global__ void compute_percentiles_kernel(offset_type const* tdigest_offsets,
                                           size_type num_tdigests,
                                           double const* percentages,
                                           size_type num_percentages,
                                           double const* mean_,
                                           double const* weight_,
                                           double const* cumulative_weight_,
                                           double *output)
{
  int const tid = threadIdx.x + blockIdx.x * blockDim.x;  

  auto const tdigest_index = tid / num_percentages;
  if(tdigest_index >= num_tdigests){
    return;
  }
  double const percentage = percentages[tid % num_percentages];
  
  // size of the digest we're querying
  auto const tdigest_size = tdigest_offsets[tdigest_index+1] - tdigest_offsets[tdigest_index];
  double const* cumulative_weight = cumulative_weight_ + tdigest_offsets[tdigest_index];
  
  // means and weights for the tdigest
  double const* mean = mean_ + tdigest_offsets[tdigest_index];
  double const* weight = weight_ + tdigest_offsets[tdigest_index];    

  // if we're at either end
  double const total_weight = cumulative_weight[tdigest_size - 1]; 
  double const cluster_q = percentage * total_weight;
  if(cluster_q <= 1){
    output[tid] = mean[0];
    return;
  } else if(cluster_q >= total_weight - 1){
    output[tid] = mean[tdigest_size-1];
    return;
  }

  // otherwise find the centroid we're in and interpolate
  size_type const centroid_index = static_cast<size_type>(thrust::lower_bound(thrust::seq, cumulative_weight, cumulative_weight + tdigest_size, cluster_q) - cumulative_weight);
  
  double diff = cluster_q + weight[centroid_index] / 2 - cumulative_weight[centroid_index];
  if(weight[centroid_index] == 1 && std::abs(diff) < 0.5){
    output[tid] = mean[centroid_index];
    return;
  }
  size_type left_index = centroid_index;
  size_type right_index = centroid_index;
  if(diff > 0){
    if(right_index == tdigest_size - 1){
      output[tid] = lerp(mean[right_index], mean[tdigest_size-1], diff / (weight[right_index] / 2));
      return;
    }
    right_index++;
  } else {
    if(left_index == 0){
      output[tid] = lerp(mean[0], mean[left_index], diff / (weight[left_index] / 2));
      return;
    }
    left_index--;
    diff += weight[left_index] / 2 + weight[right_index] / 2;
  }

  diff /= (weight[left_index] / 2 + weight[right_index] / 2);
  output[tid] = lerp(mean[left_index], mean[right_index], diff);
}

std::unique_ptr<column> compute_approx_percentiles(column_view const& input,
                                                   column_view const& percentages,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.type().id() == type_id::LIST, "Encountered invalid tdigest column");
  CUDF_EXPECTS(input.offset() == 0, "Encountered a sliced tdigest column");
  CUDF_EXPECTS(input.nullable() == false, "Encountered nullable tdigest column");
  CUDF_EXPECTS(percentages.type().id() == type_id::FLOAT64, "Encountered non-double percentages column in percentile_approx");
  lists_column_view lcv(input);

  // offsets, representing the size of each tdigest
  auto offsets = lcv.offsets();

  // extract means and weights
  auto data = lcv.get_sliced_child(stream);
  CUDF_EXPECTS(data.type().id() == type_id::STRUCT, "Encountered invalid tdigest column");
  CUDF_EXPECTS(data.num_children() == 2, "Encountered invalid tdigest column");
  structs_column_view scv(data);  
  auto mean = scv.get_sliced_child(0);
  CUDF_EXPECTS(mean.type().id() == type_id::FLOAT64, "Encountered invalid tdigest column");  
  auto weight = scv.get_sliced_child(1);
  CUDF_EXPECTS(weight.type().id() == type_id::FLOAT64, "Encountered invalid tdigest column");

  // compute summed weights
  auto cumulative_weights = cudf::make_fixed_width_column(data_type{type_id::FLOAT64}, mean.size(), mask_state::UNALLOCATED, stream, rmm::mr::get_current_device_resource());
  auto keys = cudf::detail::make_counting_transform_iterator(0, [offsets = offsets.begin<offset_type>(), num_offsets = offsets.size()] __device__ (size_type i){
    return static_cast<size_type>(thrust::upper_bound(thrust::seq, offsets, offsets + num_offsets, i) - offsets) - 1;
  });
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream), keys, keys + weight.size(), weight.begin<double>(), cumulative_weights->mutable_view().begin<double>());

  // output is a column of doubles of size input.size() * percentages.size()
  auto result = cudf::make_fixed_width_column(data_type{type_id::FLOAT64}, input.size() * percentages.size(), mask_state::UNALLOCATED, stream, mr);
  
  constexpr size_type block_size = 256;  
  cudf::detail::grid_1d const grid(percentages.size() * input.size(), block_size);
  compute_percentiles_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(offsets.begin<offset_type>(),
                                                                                 input.size(),
                                                                                 percentages.begin<double>(),
                                                                                 percentages.size(),
                                                                                 mean.begin<double>(),
                                                                                 weight.begin<double>(),
                                                                                 cumulative_weights->view().begin<double>(),
                                                                                 result->mutable_view().begin<double>());
  return result;
}

std::unique_ptr<column> percentile_approx(
  column_view const& input,
  column_view const& percentages,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  // output is a list column with each row containing percentages.size() percentile values
  auto offsets = cudf::make_fixed_width_column(data_type{type_id::INT32}, input.size() + 1, mask_state::UNALLOCATED, stream, mr);
  auto row_size_iter = thrust::make_constant_iterator(percentages.size());    
  thrust::exclusive_scan(rmm::exec_policy(stream), row_size_iter, row_size_iter + input.size() + 1, offsets->mutable_view().begin<offset_type>());

  return cudf::make_lists_column(input.size(), std::move(offsets), compute_approx_percentiles(input, percentages, stream, mr), 0, {}, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> quantiles(table_view const& input,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 cudf::sorted is_input_sorted,
                                 std::vector<order> const& column_order,
                                 std::vector<null_order> const& null_precedence,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  if (q.empty()) { return empty_like(input); }

  CUDF_EXPECTS(interp == interpolation::HIGHER || interp == interpolation::LOWER ||
                 interp == interpolation::NEAREST,
               "multi-column quantiles require a non-arithmetic interpolation strategy.");

  CUDF_EXPECTS(input.num_rows() > 0, "multi-column quantiles require at least one input row.");

  if (is_input_sorted == sorted::YES) {
    return detail::quantiles(
      input, thrust::make_counting_iterator<size_type>(0), q, interp, rmm::cuda_stream_default, mr);
  } else {
    auto sorted_idx = detail::sorted_order(input, column_order, null_precedence);
    return detail::quantiles(
      input, sorted_idx->view().data<size_type>(), q, interp, rmm::cuda_stream_default, mr);
  }
}

std::unique_ptr<column> percentile_approx(
  column_view const& input,
  column_view const& percentages,  
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource())
{
  return percentile_approx(input, percentages, rmm::cuda_stream_default, mr);  
}

}  // namespace cudf
