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

namespace {

// https://developer.nvidia.com/blog/lerp-faster-cuda/
template <typename T>
__device__ inline T lerp(T v0, T v1, T t)
{
  return fma(t, v1, fma(-t, v0, v0));
}

template <typename T>
struct cast_output {
  data_type t;

  __device__ device_storage_type_t<T> operator()(double v)
  {
    if constexpr (!cudf::is_fixed_point<T>()) { return static_cast<T>(v); }
    if constexpr (cudf::is_fixed_point<T>()) {
      return T{v, numeric::scale_type{t.scale()}}.value();
    }
  }
};

// kernel for computing percentiles on input tdigest (mean, weight) centroid data.
template <typename T>
__global__ void compute_percentiles_kernel(offset_type const* tdigest_offsets,
                                           size_type num_tdigests,
                                           device_span<double const> percentages,
                                           double const* mean_,
                                           double const* weight_,
                                           double const* min_,
                                           double const* max_,
                                           double const* cumulative_weight_,
                                           device_storage_type_t<T>* output,
                                           data_type output_type)
{
  int const tid = threadIdx.x + blockIdx.x * blockDim.x;

  auto const tdigest_index = tid / percentages.size();
  if (tdigest_index >= num_tdigests) { return; }
  double const percentage = percentages[tid % percentages.size()];

  // size of the digest we're querying
  auto const tdigest_size = tdigest_offsets[tdigest_index + 1] - tdigest_offsets[tdigest_index];
  if (tdigest_size == 0) {
    // no work to do. values will be set to null
    return;
  }
  double const* cumulative_weight = cumulative_weight_ + tdigest_offsets[tdigest_index];

  // means and weights for the tdigest
  double const* mean   = mean_ + tdigest_offsets[tdigest_index];
  double const* weight = weight_ + tdigest_offsets[tdigest_index];

  // min and max for the digest
  double const* min_val = min_ + tdigest_index;
  double const* max_val = max_ + tdigest_index;

  double const total_weight = cumulative_weight[tdigest_size - 1];
  double const cluster_q    = percentage * total_weight;

  cast_output<T> cast_op{output_type};

  if (cluster_q <= 1) {
    output[tid] = cast_op(*min_val);
    return;
  } else if (cluster_q >= total_weight - 1) {
    output[tid] = cast_op(*max_val);
    return;
  }

  // otherwise find the centroid we're in and interpolate
  size_type const centroid_index = static_cast<size_type>(
    thrust::lower_bound(
      thrust::seq, cumulative_weight, cumulative_weight + tdigest_size, cluster_q) -
    cumulative_weight);

  double diff = cluster_q + weight[centroid_index] / 2 - cumulative_weight[centroid_index];
  if (weight[centroid_index] == 1 && std::abs(diff) < 0.5) {
    output[tid] = cast_op(mean[centroid_index]);
    return;
  }
  size_type left_index  = centroid_index;
  size_type right_index = centroid_index;
  if (diff > 0) {
    if (right_index == tdigest_size - 1) {
      output[tid] = cast_op(lerp(mean[right_index], *max_val, diff / (weight[right_index] / 2)));
      return;
    }
    right_index++;
  } else {
    if (left_index == 0) {
      output[tid] = cast_op(lerp(*min_val, mean[left_index], diff / (weight[left_index] / 2)));
      return;
    }
    left_index--;
    diff += weight[left_index] / 2 + weight[right_index] / 2;
  }

  diff /= (weight[left_index] / 2 + weight[right_index] / 2);
  output[tid] = cast_op(lerp(mean[left_index], mean[right_index], diff));
}

/**
 * @brief Calculate approximate percentiles on a provided tdigest column.
 *
 * Produces a LIST column where each row N represents output from querying the
 * corresponding tdigest of from row N in `input`. The length of each output list
 * is the number of percentages specified in `percentages`
 *
 * @param input           tdigest input data. One tdigest per row.
 * @param percentages     Desired percentiles in range [0, 1].
 * @param stream          CUDA stream used for device memory operations and kernel launches
 * @param mr              Device memory resource used to allocate the returned column's device
 * memory
 *
 * @returns Column of doubles containing requested percentile values.
 */
template <typename T>
std::unique_ptr<column> compute_approx_percentiles(structs_column_view const& input,
                                                   column_view const& percentages,
                                                   data_type output_type,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  lists_column_view lcv(input.child(tdigest_centroid_column_index));
  column_view min_col = input.child(tdigest_min_column_index);
  column_view max_col = input.child(tdigest_max_column_index);

  // offsets, representing the size of each tdigest
  auto offsets = lcv.offsets();

  // extract means and weights
  auto data = lcv.get_sliced_child(stream);
  structs_column_view tdigest(data);
  auto mean   = tdigest.get_sliced_child(tdigest_mean_column_index);
  auto weight = tdigest.get_sliced_child(tdigest_weight_column_index);

  // compute summed weights
  auto cumulative_weights = cudf::make_fixed_width_column(data_type{type_id::FLOAT64},
                                                          mean.size(),
                                                          mask_state::UNALLOCATED,
                                                          stream,
                                                          rmm::mr::get_current_device_resource());
  auto keys               = cudf::detail::make_counting_transform_iterator(
    0,
    [offsets = offsets.begin<offset_type>(), num_offsets = offsets.size()] __device__(size_type i) {
      return static_cast<size_type>(
               thrust::upper_bound(thrust::seq, offsets, offsets + num_offsets, i) - offsets) -
             1;
    });
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                keys,
                                keys + weight.size(),
                                weight.begin<double>(),
                                cumulative_weights->mutable_view().begin<double>());

  // output is a column of size input.size() * percentages.size()
  auto result = cudf::make_fixed_width_column(
    output_type, input.size() * percentages.size(), mask_state::UNALLOCATED, stream, mr);

  constexpr size_type block_size = 256;
  cudf::detail::grid_1d const grid(percentages.size() * input.size(), block_size);
  compute_percentiles_kernel<T><<<grid.num_blocks, block_size, 0, stream.value()>>>(
    offsets.begin<offset_type>(),
    input.size(),
    {percentages.begin<double>(), static_cast<size_t>(percentages.size())},
    mean.begin<double>(),
    weight.begin<double>(),
    min_col.begin<double>(),
    max_col.begin<double>(),
    cumulative_weights->view().begin<double>(),
    result->mutable_view().begin<device_storage_type_t<T>>(),
    output_type);

  return result;
}

struct compute_percentiles_dispatch {
  template <
    typename T,
    typename std::enable_if_t<cudf::is_numeric<T>() || cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(structs_column_view const& input,
                                     column_view const& percentages,
                                     data_type output_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    return compute_approx_percentiles<T>(input, percentages, output_type, stream, mr);
  }

  template <
    typename T,
    typename std::enable_if_t<!cudf::is_numeric<T>() && !cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(structs_column_view const& input,
                                     column_view const& percentages,
                                     data_type output_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("Invalid non-numeric output requested for percentile_approx");
  }
};

}  // anonymous namespace

void check_is_valid_tdigest_column(column_view const& col)
{
  // sanity check that this is actually tdigest data
  CUDF_EXPECTS(col.type().id() == type_id::STRUCT, "Encountered invalid tdigest column");
  CUDF_EXPECTS(col.size() > 0, "tdigest columns must have > 0 rows");
  CUDF_EXPECTS(col.offset() == 0, "Encountered a sliced tdigest column");
  CUDF_EXPECTS(col.nullable() == false, "Encountered nullable tdigest column");

  structs_column_view scv(col);
  CUDF_EXPECTS(scv.num_children() == 3, "Encountered invalid tdigest column");
  CUDF_EXPECTS(scv.child(tdigest_min_column_index).type().id() == type_id::FLOAT64,
               "Encountered invalid tdigest column");
  CUDF_EXPECTS(scv.child(tdigest_max_column_index).type().id() == type_id::FLOAT64,
               "Encountered invalid tdigest column");

  lists_column_view lcv(scv.child(tdigest_centroid_column_index));
  auto data = lcv.child();
  CUDF_EXPECTS(data.type().id() == type_id::STRUCT, "Encountered invalid tdigest column");
  CUDF_EXPECTS(data.num_children() == 2,
               "Encountered tdigest column with an invalid number of children");
  auto mean = data.child(tdigest_mean_column_index);
  CUDF_EXPECTS(mean.type().id() == type_id::FLOAT64, "Encountered invalid tdigest mean column");
  auto weight = data.child(tdigest_weight_column_index);
  CUDF_EXPECTS(weight.type().id() == type_id::FLOAT64, "Encountered invalid tdigest weight column");
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

  auto min_col = cudf::make_fixed_width_column(
    data_type(type_id::FLOAT64), 1, mask_state::UNALLOCATED, stream, mr);
  thrust::fill(rmm::exec_policy(stream),
               min_col->mutable_view().begin<double>(),
               min_col->mutable_view().end<double>(),
               0);
  auto max_col = cudf::make_fixed_width_column(
    data_type(type_id::FLOAT64), 1, mask_state::UNALLOCATED, stream, mr);
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

std::unique_ptr<column> percentile_approx(structs_column_view const& input,
                                          column_view const& percentages,
                                          cudf::data_type output_type,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  check_is_valid_tdigest_column(input);

  // output is a list column with each row containing percentages.size() percentile values
  auto offsets = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, input.size() + 1, mask_state::UNALLOCATED, stream, mr);
  auto row_size_iter = thrust::make_constant_iterator(percentages.size());
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         row_size_iter,
                         row_size_iter + input.size() + 1,
                         offsets->mutable_view().begin<offset_type>());

  // if any of the input digests are empty, nullify the corresponding output rows (values will be
  // uninitialized)
  auto [bitmask, null_count] = [stream, mr, input]() {
    lists_column_view lcv(input.child(tdigest_centroid_column_index));
    auto iter = cudf::detail::make_counting_transform_iterator(
      0, [offsets = lcv.offsets().begin<offset_type>()] __device__(size_type index) {
        return offsets[index + 1] - offsets[index] == 0 ? 1 : 0;
      });
    auto const null_count = thrust::reduce(rmm::exec_policy(stream), iter, iter + input.size(), 0);
    if (null_count == 0) {
      return std::pair<rmm::device_buffer, size_type>{rmm::device_buffer{}, null_count};
    }
    return cudf::detail::valid_if(
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(0) + input.size(),
      [offsets = lcv.offsets().begin<offset_type>()] __device__(size_type index) {
        return offsets[index + 1] - offsets[index] == 0 ? 0 : 1;
      },
      stream,
      mr);
  }();

  return cudf::make_lists_column(
    input.size(),
    std::move(offsets),
    cudf::type_dispatcher(
      output_type, compute_percentiles_dispatch{}, input, percentages, output_type, stream, mr),
    null_count,
    std::move(bitmask),
    stream,
    mr);
}

}  // namespace detail

std::unique_ptr<column> percentile_approx(structs_column_view const& input,
                                          column_view const& percentages,
                                          cudf::data_type output_type,
                                          rmm::mr::device_memory_resource* mr)
{
  return percentile_approx(input, percentages, output_type, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
