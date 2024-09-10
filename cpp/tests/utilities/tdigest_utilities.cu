/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/tdigest_utilities.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

// for use with groupby and reduction aggregation tests.

namespace cudf {
namespace test {

void tdigest_sample_compare(cudf::tdigest::tdigest_column_view const& tdv,
                            std::vector<expected_value> const& h_expected)
{
  column_view result_mean   = tdv.means();
  column_view result_weight = tdv.weights();

  auto expected_mean = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, h_expected.size(), mask_state::UNALLOCATED);
  auto expected_weight = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, h_expected.size(), mask_state::UNALLOCATED);
  auto sampled_result_mean = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, h_expected.size(), mask_state::UNALLOCATED);
  auto sampled_result_weight = cudf::make_fixed_width_column(
    data_type{type_id::FLOAT64}, h_expected.size(), mask_state::UNALLOCATED);

  auto h_expected_src    = std::vector<size_type>(h_expected.size());
  auto h_expected_mean   = std::vector<double>(h_expected.size());
  auto h_expected_weight = std::vector<double>(h_expected.size());

  {
    auto iter = thrust::make_counting_iterator(0);
    std::for_each_n(iter, h_expected.size(), [&](size_type const index) {
      h_expected_src[index]    = thrust::get<0>(h_expected[index]);
      h_expected_mean[index]   = thrust::get<1>(h_expected[index]);
      h_expected_weight[index] = thrust::get<2>(h_expected[index]);
    });
  }

  auto d_expected_src = cudf::detail::make_device_uvector_async(
    h_expected_src, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto d_expected_mean = cudf::detail::make_device_uvector_async(
    h_expected_mean, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto d_expected_weight = cudf::detail::make_device_uvector_async(
    h_expected_weight, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto iter = thrust::make_counting_iterator(0);
  thrust::for_each(
    rmm::exec_policy(cudf::get_default_stream()),
    iter,
    iter + h_expected.size(),
    [expected_src_in     = d_expected_src.data(),
     expected_mean_in    = d_expected_mean.data(),
     expected_weight_in  = d_expected_weight.data(),
     expected_mean       = expected_mean->mutable_view().begin<double>(),
     expected_weight     = expected_weight->mutable_view().begin<double>(),
     result_mean         = result_mean.begin<double>(),
     result_weight       = result_weight.begin<double>(),
     sampled_result_mean = sampled_result_mean->mutable_view().begin<double>(),
     sampled_result_weight =
       sampled_result_weight->mutable_view().begin<double>()] __device__(size_type index) {
      expected_mean[index]         = expected_mean_in[index];
      expected_weight[index]       = expected_weight_in[index];
      auto const src_index         = expected_src_in[index];
      sampled_result_mean[index]   = result_mean[src_index];
      sampled_result_weight[index] = result_weight[src_index];
    });

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected_mean, *sampled_result_mean);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_weight, *sampled_result_weight);
}

std::unique_ptr<column> make_expected_tdigest_column(std::vector<expected_tdigest> const& groups)
{
  std::vector<std::unique_ptr<column>> tdigests;

  // make an individual digest
  auto make_digest = [&](expected_tdigest const& tdigest) {
    std::vector<std::unique_ptr<column>> inner_children;
    inner_children.push_back(std::make_unique<cudf::column>(tdigest.mean));
    inner_children.push_back(std::make_unique<cudf::column>(tdigest.weight));
    // tdigest struct
    auto tdigests =
      cudf::make_structs_column(tdigest.mean.size(), std::move(inner_children), 0, {});

    std::vector<size_type> h_offsets{0, tdigest.mean.size()};
    auto offsets =
      cudf::make_fixed_width_column(data_type{type_id::INT32}, 2, mask_state::UNALLOCATED);
    CUDF_CUDA_TRY(cudaMemcpy(offsets->mutable_view().begin<size_type>(),
                             h_offsets.data(),
                             sizeof(size_type) * 2,
                             cudaMemcpyDefault));

    auto list = cudf::make_lists_column(1, std::move(offsets), std::move(tdigests), 0, {});

    auto min_col =
      cudf::make_fixed_width_column(data_type{type_id::FLOAT64}, 1, mask_state::UNALLOCATED);
    thrust::fill(rmm::exec_policy(cudf::get_default_stream()),
                 min_col->mutable_view().begin<double>(),
                 min_col->mutable_view().end<double>(),
                 tdigest.min);
    auto max_col =
      cudf::make_fixed_width_column(data_type{type_id::FLOAT64}, 1, mask_state::UNALLOCATED);
    thrust::fill(rmm::exec_policy(cudf::get_default_stream()),
                 max_col->mutable_view().begin<double>(),
                 max_col->mutable_view().end<double>(),
                 tdigest.max);

    std::vector<std::unique_ptr<column>> children;
    children.push_back(std::move(list));
    children.push_back(std::move(min_col));
    children.push_back(std::move(max_col));
    return make_structs_column(1, std::move(children), 0, {});
  };

  // build the individual digests
  std::transform(groups.begin(), groups.end(), std::back_inserter(tdigests), make_digest);

  // concatenate them
  std::vector<column_view> views;
  std::transform(tdigests.begin(),
                 tdigests.end(),
                 std::back_inserter(views),
                 [](std::unique_ptr<column> const& c) { return c->view(); });

  return cudf::concatenate(views);
}

}  // namespace test
}  // namespace cudf
